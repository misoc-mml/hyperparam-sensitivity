import os
import pandas as pd

import utils as ut

def compare_metrics(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'
    df = _process_test(cate_models, base_dir, metrics)
    df = _process_mse(df, cate_models, base_dir, mode, metrics)
    df = _process_r2scores(df, cate_models, base_dir, mode, metrics)
    df = _process_mixed_score(df, cate_models, base_dir, mode, metrics)
    df = _process_plugins(df, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df = _process_matching(df, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df = _process_rscores(df, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    return df

def compare_risks(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'risk'
    df = _process_mse(None, cate_models, base_dir, mode, metrics)
    df = _process_r2scores(df, cate_models, base_dir, mode, metrics)
    df = _process_mixed_score(df, cate_models, base_dir, mode, metrics)
    df = _process_plugins(df, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df = _process_matching(df, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df = _process_rscores(df, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    return df

def compare_correlations(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'corr'
    df = _process_mse(None, cate_models, base_dir, mode, metrics)
    df = _process_r2scores(df, cate_models, base_dir, mode, metrics)
    df = _process_mixed_score(df, cate_models, base_dir, mode, metrics)
    df = _process_plugins(df, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df = _process_matching(df, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df = _process_rscores(df, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    return df

def _process_test(cate_models, results_dir, metrics):
    test_list = []
    for cm in cate_models:
        # Test ATE and PEHE
        df_base_test = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_test_metrics.csv'))
        best_metrics = [ut.get_best_metric(df_base_test, metric) for metric in metrics]
        test_list.append([cm] + best_metrics)

    metrics_test = [f'{metric}_test' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _process_mse(df_main, cate_models, results_dir, mode, metrics):
    mse_list = []
    for cm in cate_models:
        df_base_test = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_test_metrics.csv'))

        meta_model = cm.split('_')[0]
        if meta_model in ('cf', 'xl', 'drs', 'dmls'):
            mse_i = ['-'] * 2
        else:
            has_mse = 'mse' in df_base_test.columns
            if has_mse:
                target_param = 'mse_val'
            else:
                target_param = 'mse'
            
            # Val MSE
            df_base_val = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            has_duplicates = [metric for metric in metrics if metric in df_base_val.columns]
            if has_duplicates:
                metrics_param = [f'{metric}_test' for metric in metrics]
            else:
                metrics_param = metrics

            if meta_model == 'tl':
                df_base_val_gr['mse'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
            elif meta_model == 'ipsws':
                df_base_val_gr['mse'] = df_base_val_gr[['mse_prop', 'mse_reg']].mean(axis=1)
            
            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            mse_i = ut.fn_by_best(df_base, target_param, metrics_param, mode, True)

        mse_list.append([cm] + mse_i)

    df_mse = pd.DataFrame(mse_list, columns=['name'] + [f'{metric}_mse' for metric in metrics])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_mixed_score(df_main, cate_models, results_dir, mode, metrics):
    score_list = []
    for cm in cate_models:
        df_base_test = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_test_metrics.csv'))

        meta_model = cm.split('_')[0]
        # No standard selection metrics on validation set for ['cf', 'xl'].
        # No mixed score for ['sl', 'tl', 'ipsw'].
        if meta_model in ('cf', 'xl', 'sl', 'tl', 'ipsws'):
            score_i = ['-'] * 2
        else:
            df_base_val = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            #if meta_model in ('drs', 'dmls'):
            # mixed = R^2 + ACC - MSE
            df_base_val_gr['mixed'] = df_base_val_gr['reg_score'] + df_base_val_gr['prop_score'] - df_base_val_gr['final_score']

            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            score_i = ut.fn_by_best(df_base, 'mixed', metrics, mode, False)            

        score_list.append([cm] + score_i)

    df_score = pd.DataFrame(score_list, columns=['name'] + [f'{metric}_mixed' for metric in metrics])
    return df_main.merge(df_score, on=['name'])

def _process_plugins(df_main, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for pm in plugin_models:
        plugin_ate_list = []
        plugin_pehe_list = []
        for cm in cate_models:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))

            # Plugin ATE and PEHE
            df_plugin_val = pd.read_csv(os.path.join(plugin_dir, pm, f'{cm}_plugin_{pm}.csv'))
            df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            # Suffixes are applied only to duplicated column names.
            if 'ate' in metrics: # assume ['ate', 'pehe']
                plugin_ate_i = ut.fn_by_best(df_plugin, 'ate_val', metrics_test, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_plugin, 'pehe_val', metrics_test, mode, True)
            else:
                plugin_ate_i = ut.fn_by_best(df_plugin, 'ate', metrics, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_plugin, 'pehe', metrics, mode, True)

            plugin_ate_list.append([cm] + plugin_ate_i)
            plugin_pehe_list.append([cm] + plugin_pehe_i)

        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_{pm}_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_{pm}_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_matching(df_main, ks, cate_models, base_dir, matching_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for k in ks:
        plugin_ate_list = []
        plugin_pehe_list = []
        for cm in cate_models:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))

            # ATE and PEHE
            df_plugin_val = pd.read_csv(os.path.join(matching_dir, f'match_{k}k', f'{cm}_matching_match_{k}k.csv'))
            df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            # Suffixes are applied only to duplicated column names.
            if 'ate' in metrics: # assume ['ate', 'pehe']
                plugin_ate_i = ut.fn_by_best(df_plugin, 'ate_val', metrics_test, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_plugin, 'pehe_val', metrics_test, mode, True)
            else:
                plugin_ate_i = ut.fn_by_best(df_plugin, 'ate', metrics, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_plugin, 'pehe', metrics, mode, True)

            plugin_ate_list.append([cm] + plugin_ate_i)
            plugin_pehe_list.append([cm] + plugin_pehe_i)

        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_match_{k}k_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_match_{k}k_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_rscores(df_main, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        for cm in cate_models:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))

            # R-Score
            df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{cm}_r_score_{rs_name}.csv'))
            df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
            rscore_i = ut.fn_by_best(df_rscore_test, 'rscore', metrics, mode, False)
            scores_list.append([cm] + rscore_i)

        df_rscore = pd.DataFrame(scores_list, columns=['name'] + [f'{metric}_{rs_name}' for metric in metrics])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy

def _process_r2scores(df_main, cate_models, results_dir, mode, metrics):
    r2_list = []
    for cm in cate_models:
        df_base_test = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_test_metrics.csv'))

        meta_model = cm.split('_')[0]
        if meta_model in ('cf', 'xl', 'drs', 'dmls'):
            r2_i = ['-'] * 2
        else:
            has_mse = 'r2_score' in df_base_test.columns
            if has_mse:
                target_param = 'r2_score_val'
            else:
                target_param = 'r2_score'

            # R^2 Score
            df_base_val = pd.read_csv(os.path.join(results_dir, cm, f'{cm}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            has_duplicates = [metric for metric in metrics if metric in df_base_val.columns]
            if has_duplicates:
                metrics_param = [f'{metric}_test' for metric in metrics]
            else:
                metrics_param = metrics

            if meta_model == 'tl':
                df_base_val_gr['r2_score'] = df_base_val_gr[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
            elif meta_model == 'ipsws':
                df_base_val_gr['r2_score'] = df_base_val_gr[['r2_score_prop', 'r2_score_reg']].mean(axis=1)

            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            r2_i = ut.fn_by_best(df_base, target_param, metrics_param, mode, False)

        r2_list.append([cm] + r2_i)

    df_r2 = pd.DataFrame(r2_list, columns=['name'] + [f'{metric}_r2' for metric in metrics])

    return df_main.merge(df_r2, on=['name'])