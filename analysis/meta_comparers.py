import os
import pandas as pd

import utils as ut
import comparers as comp

def compare_metrics_meta(cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics):
    mode = 'metric'
    df_meta = _process_test_meta(meta_models, base_models, base_dir, metrics)
    df_meta = _process_mse_meta(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_rscores_meta(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    df_cate = comp.compare_metrics(cate_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics)
    return pd.concat([df_meta, df_cate], ignore_index=True)

def compare_risks_meta(cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics):
    mode = 'risk'
    df_meta = _process_mse_meta(None, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_rscores_meta(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    df_cate = comp.compare_risks(cate_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics)
    return pd.concat([df_meta, df_cate], ignore_index=True)

def compare_correlations_meta(cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics):
    mode = 'corr'
    df_meta = _process_mse_meta(None, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_rscores_meta(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    df_cate = comp.compare_correlations(cate_models, plugin_models, rscore_base_models, base_dir, plugin_dir, rscore_dir, metrics)
    return pd.concat([df_meta, df_cate], ignore_index=True)

def _process_test_meta(meta_models, base_models, base_dir, metrics):
    test_list = []
    for mm in meta_models:
        df_mm = None
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            df_mm = pd.concat([df_mm, df_base_test], ignore_index=True)
        best_metrics = [ut.get_best_metric(df_mm, metric) for metric in metrics]
        test_list.append([mm] + best_metrics)
    
    metrics_test = [f'{metric}_test' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _process_mse_meta(df_main, meta_models, base_models, base_dir, mode, metrics):
    metrics_test = [f'{metric}_test' for metric in metrics]
    mse_list = []
    for mm in meta_models:
        df_mm = None
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))

            df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            if mm == 'tl':
                df_base_val_gr['mse'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
                df_base_test['mse'] = df_base_test[['mse_m0', 'mse_m1']].mean(axis=1)
            
            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            df_mm = pd.concat([df_mm, df_base], ignore_index=True)

        mse_i = ut.fn_by_best(df_mm, 'mse_val', metrics_test, mode, True)
        mse_list.append([mm] + mse_i)
    
    df_mse = pd.DataFrame(mse_list, columns=['name'] + [f'{metric}_mse' for metric in metrics])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_r2scores_meta(df_main, meta_models, base_models, base_dir, mode, metrics):
    metrics_test = [f'{metric}_test' for metric in metrics]
    r2_list = []
    for mm in meta_models:
        df_mm = None
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))

            df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            if mm == 'tl':
                df_base_val_gr['r2_score'] = df_base_val_gr[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
                df_base_test['r2_score'] = df_base_test[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
            
            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            df_mm = pd.concat([df_mm, df_base], ignore_index=True)

        r2_i = ut.fn_by_best(df_mm, 'r2_score_val', metrics_test, mode, False)
        r2_list.append([mm] + r2_i)
    
    df_r2 = pd.DataFrame(r2_list, columns=['name'] + [f'{metric}_r2' for metric in metrics])
    return df_main.merge(df_r2, on=['name'])

def _process_plugins_meta(df_main, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for pm in plugin_models:
        plugin_ate_list = []
        plugin_pehe_list = []
        for mm in meta_models:
            df_mm = None
            for bm in base_models:
                model_name = f'{mm}_{bm}'

                df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))

                df_plugin_val = pd.read_csv(os.path.join(plugin_dir, pm, f'{model_name}_plugin_{pm}.csv'))
                df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                df_mm = pd.concat([df_mm, df_plugin], ignore_index=True)

            # Suffixes are applied only to duplicated column names.
            if 'ate' in metrics: # assume ['ate', 'pehe']
                plugin_ate_i = ut.fn_by_best(df_mm, 'ate_val', metrics_test, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_mm, 'pehe_val', metrics_test, mode, True)
            else:
                plugin_ate_i = ut.fn_by_best(df_mm, 'ate', metrics, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_mm, 'pehe', metrics, mode, True)

            plugin_ate_list.append([mm] + plugin_ate_i)
            plugin_pehe_list.append([mm] + plugin_pehe_i)
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_{pm}_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_{pm}_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_rscores_meta(df_main, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        for mm in meta_models:
            df_mm = None
            for bm in base_models:
                model_name = f'{mm}_{bm}'

                df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))

                df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{model_name}_r_score_{rs_name}.csv'))
                df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
                df_mm = pd.concat([df_mm, df_rscore_test], ignore_index=True)

            rscore_i = ut.fn_by_best(df_mm, 'rscore', metrics, mode, False)
            scores_list.append([mm] + rscore_i)
        
        df_rscore = pd.DataFrame(scores_list, columns=['name'] + [f'{metric}_{rs_name}' for metric in metrics])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy