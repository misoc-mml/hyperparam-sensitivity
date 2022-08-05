import os
import pandas as pd

import utils as ut

def compare_metrics(meta_models, base_models, plugin_meta_models, plugin_base_models, rscore_base_models, base_dir, plugin_dir, rscore_dir):
    mode = 'metric'
    df = _process_test(meta_models, base_models, base_dir)
    df = _process_mse(df, meta_models, base_models, base_dir, mode)
    df = _process_plugins(df, plugin_meta_models, plugin_base_models, meta_models, base_models, base_dir, plugin_dir, mode)
    df = _process_rscores(df, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode)
    return df

def compare_risks(meta_models, base_models, plugin_meta_models, plugin_base_models, rscore_base_models, base_dir, plugin_dir, rscore_dir):
    mode = 'risk'
    df = _process_mse(None, meta_models, base_models, base_dir, mode)
    df = _process_plugins(df, plugin_meta_models, plugin_base_models, meta_models, base_models, base_dir, plugin_dir, mode)
    df = _process_rscores(df, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode)
    return df

def _process_test(meta_models, base_models, results_dir):
    test_list = []
    for mm in meta_models:
        for bm in base_models:
            est_name = f'{mm}_{bm}'

            # Test ATE and PEHE
            df_base_test = pd.read_csv(os.path.join(results_dir, est_name, f'{est_name}_test_metrics.csv'))
            ate_test_i = ut.get_best_metric(df_base_test, 'ate')
            pehe_test_i = ut.get_best_metric(df_base_test, 'pehe')
            test_list.append([est_name, ate_test_i, pehe_test_i])

    return pd.DataFrame(test_list, columns=['name', 'ate_test', 'pehe_test'])

def _process_mse(df_main, meta_models, base_models, results_dir, mode):
    mse_list = []
    for mm in meta_models:
        for bm in base_models:
            est_name = f'{mm}_{bm}'
            df_base_test = pd.read_csv(os.path.join(results_dir, est_name, f'{est_name}_test_metrics.csv'))

            # Val MSE
            df_base_val = pd.read_csv(os.path.join(results_dir, est_name, f'{est_name}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            if mm == 'tl':
                df_base_val_gr['mse'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
                df_base_test['mse'] = df_base_test[['mse_m0', 'mse_m1']].mean(axis=1)

            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            mse_i = ut.fn_by_best(df_base, 'mse_val', ['ate_test', 'pehe_test'], mode, True)
            mse_list.append([est_name] + mse_i)

    df_mse = pd.DataFrame(mse_list, columns=['name', 'ate_mse', 'pehe_mse'])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_plugins(df_main, plugin_meta_models, plugin_base_models, meta_models, base_models, base_dir, plugin_dir, mode):
    df_copy = df_main.copy()
    for plugin_mm in plugin_meta_models:
        for plugin_bm in plugin_base_models:
            plugin_name = f'{plugin_mm}_{plugin_bm}'
            plugin_ate_list = []
            plugin_pehe_list = []
            for mm in meta_models:
                for bm in base_models:
                    est_name = f'{mm}_{bm}'
                    df_base_test = pd.read_csv(os.path.join(base_dir, est_name, f'{est_name}_test_metrics.csv'))

                    # Plugin ATE and PEHE
                    df_plugin_val = pd.read_csv(os.path.join(plugin_dir, plugin_name, f'{est_name}_plugin_{plugin_name}.csv'))
                    df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
                    plugin_ate_i = ut.fn_by_best(df_plugin, 'ate_val', ['ate_test', 'pehe_test'], mode, True)
                    plugin_pehe_i = ut.fn_by_best(df_plugin, 'pehe_val', ['ate_test', 'pehe_test'], mode, True)
                    plugin_ate_list.append([est_name] + plugin_ate_i)
                    plugin_pehe_list.append([est_name] + plugin_pehe_i)

            df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name', f'ate_{plugin_name}_ate', f'pehe_{plugin_name}_ate'])
            df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name', f'ate_{plugin_name}_pehe', f'pehe_{plugin_name}_pehe'])
            df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
            df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_rscores(df_main, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        for mm in meta_models:
            for bm in base_models:
                est_name = f'{mm}_{bm}'
                df_base_test = pd.read_csv(os.path.join(base_dir, est_name, f'{est_name}_test_metrics.csv'))

                # R-Score
                df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{est_name}_r_score_{rs_name}.csv'))
                df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
                rscore_i = ut.fn_by_best(df_rscore_test, 'rscore', ['ate', 'pehe'], mode, False)
                scores_list.append([est_name] + rscore_i)

        df_rscore = pd.DataFrame(scores_list, columns=['name', f'ate_{rs_name}', f'pehe_{rs_name}'])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy