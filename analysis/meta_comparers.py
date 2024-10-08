import os
import pandas as pd
import numpy as np

import utils as ut
import comparers as comp

def compare_metrics_meta_est(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'
    df_meta = _process_test_meta_est(meta_models, base_models, base_dir, metrics)
    df_meta = _process_mse_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_mixed_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta_est(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_est(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_est(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_metrics(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_metrics_meta_base(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'
    df_meta = _process_test_meta_base(meta_models, base_models, base_dir, metrics)
    df_meta = _process_plugins_meta_base(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_base(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_base(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_metrics(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_metrics_all(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'
    df_meta = _process_test_all(cate_models, base_dir, metrics)
    df_meta = _process_mse_all(df_meta, cate_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_all(df_meta, cate_models, base_dir, mode, metrics)

    if 'policy' in metrics:
        df_meta = _process_policy_all(df_meta, cate_models, base_dir, mode, metrics)

    df_meta = _process_plugins_all(df_meta, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_all(df_meta, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_all(df_meta, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    
    return df_meta

def compare_metrics_all_val(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'
    df_meta = _process_val_all(cate_models, base_dir, metrics)
    df_meta = _process_mse_all(df_meta, cate_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_all(df_meta, cate_models, base_dir, mode, metrics)

    if 'policy' in metrics:
        df_meta = _process_policy_all(df_meta, cate_models, base_dir, mode, metrics)

    df_meta = _process_plugins_all(df_meta, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_all(df_meta, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_all(df_meta, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    
    return df_meta

def compare_risks_all(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'risk'
    df_meta = _process_test_all(cate_models, base_dir, metrics)
    df_meta = _process_mse_all(df_meta, cate_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_all(df_meta, cate_models, base_dir, mode, metrics)
    df_meta = _process_plugins_all(df_meta, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_all(df_meta, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_all(df_meta, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    
    return df_meta

def compare_correlations_all(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'corr'
    df_meta = _process_mse_all(None, cate_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_all(df_meta, cate_models, base_dir, mode, metrics)

    if 'policy' in metrics:
        df_meta = _process_policy_all(df_meta, cate_models, base_dir, mode, metrics)

    df_meta = _process_plugins_all(df_meta, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_all(df_meta, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_all(df_meta, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    
    return df_meta

def compare_risks_meta_est(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'risk'
    df_meta = _process_mse_meta_est(None, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_mixed_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta_est(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_est(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_est(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_risks(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_risks_meta_base(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'risk'
    df_meta = _process_test_meta_base(meta_models, base_models, base_dir, metrics)
    df_meta = _process_plugins_meta_base(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_base(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_base(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_risks(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_correlations_meta_est(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'corr'
    df_meta = _process_mse_meta_est(None, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_r2scores_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_mixed_meta_est(df_meta, meta_models, base_models, base_dir, mode, metrics)
    df_meta = _process_plugins_meta_est(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_est(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_est(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_correlations(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_correlations_meta_base(cate_models, meta_models, base_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'corr'
    df_meta = _process_test_meta_base(meta_models, base_models, base_dir, metrics)
    df_meta = _process_plugins_meta_base(df_meta, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics)
    df_meta = _process_matching_meta_base(df_meta, match_models, meta_models, base_models, base_dir, match_dir, mode, metrics)
    df_meta = _process_rscores_meta_base(df_meta, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics)

    if cate_models:
        df_cate = comp.compare_correlations(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics)
        df_meta = pd.concat([df_meta, df_cate], ignore_index=True)
    
    return df_meta

def compare_test_correlations_est(cate_models, meta_models, base_models, base_dir, metrics):
    res_list = []
    for mm in meta_models:
        df_mm = None
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            try:
                df_base = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            except:
                print(f'{model_name} is missing')
                continue
            df_mm = pd.concat([df_mm, df_base[['iter_id'] + metrics]], ignore_index=True)
        res_i = ut.fn_by_best(df_mm, metrics[0], metrics[1], 'corr', False)
        res_list.append([mm] + res_i)

    for cm in cate_models:
        df_base = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        res_i = ut.fn_by_best(df_base[['iter_id'] + metrics], metrics[0], metrics[1], 'corr', False)
        res_list.append([cm] + res_i)

    return pd.DataFrame(res_list, columns=['name', f"({','.join(metrics)})"])

def compare_test_correlations_base(meta_models, base_models, base_dir, metrics):
    res_list = []
    for bm in base_models:
        df_bm = None
        for mm in meta_models:
            model_name = f'{mm}_{bm}'
            try:
                df_base = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            except:
                print(f'{model_name} is missing')
                continue
            df_bm = pd.concat([df_bm, df_base[['iter_id'] + metrics]], ignore_index=True)
        res_i = ut.fn_by_best(df_bm, metrics[0], metrics[1], 'corr', False)
        res_list.append([bm] + res_i)
    
    return pd.DataFrame(res_list, columns=['name', f"({','.join(metrics)})"])

def compare_test_correlations_all(cate_models, base_dir, metrics):
    df_all = None
    res_list = []
    for cm in cate_models:
        try:
            df_base = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue
        
        df_all = pd.concat([df_all, df_base[['iter_id'] + metrics]], ignore_index=True)

    res_i = ut.fn_by_best(df_all, metrics[0], metrics[1], 'corr', False)
    res_list.append(['all'] + res_i)

    return pd.DataFrame(res_list, columns=['name', f"({','.join(metrics)})"])

def _process_test_meta_est(meta_models, base_models, base_dir, metrics):
    test_list = []
    for mm in meta_models:
        df_mm = None
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            except:
                print(f'{model_name} is missing')
                continue
            df_mm = pd.concat([df_mm, df_base_test], ignore_index=True)

        best_metrics = [ut.get_best_metric(df_mm, metric) for metric in metrics]
        test_list.append([mm] + best_metrics)
    
    metrics_test = [f'{metric}_test' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _process_test_meta_base(meta_models, base_models, base_dir, metrics):
    test_list = []
    for bm in base_models:
        df_bm = None
        for mm in meta_models:
            model_name = f'{mm}_{bm}'
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            except:
                print(f'{model_name} is missing')
                continue
            df_bm = pd.concat([df_bm, df_base_test], ignore_index=True)
        
        best_metrics = [ut.get_best_metric(df_bm, metric) for metric in metrics]
        test_list.append([bm] + best_metrics)
    
    metrics_test = [f'{metric}_test' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _process_test_all(cate_models, base_dir, metrics):
    test_list = []
    df_all = None
    for cm in cate_models:
        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue
        df_all = pd.concat([df_all, df_base_test], ignore_index=True)
        
    best_metrics = [ut.get_best_metric(df_all, metric) for metric in metrics]
    test_list.append(['all'] + best_metrics)
    
    metrics_test = [f'{metric}_test' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _mean_target_by_lowest(df, by, target):
    best = df.apply(lambda x: x.loc[x[by].idxmin(), [target]])
    return np.mean(best[target])

def _process_val_all(cate_models, base_dir, metrics):
    test_list = []
    df_all = None
    for cm in cate_models:
        try:
            df_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            df_val = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_val_metrics.csv'))
            df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        except:
            print(f'{cm} is missing')
            continue
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    iter_all = df_all.groupby(['iter_id'], as_index=False)
    best_metrics = [_mean_target_by_lowest(iter_all, f'{metric}_val', f'{metric}_test') for metric in metrics]
    test_list.append(['all'] + best_metrics)
    
    metrics_test = [f'{metric}_val' for metric in metrics]
    return pd.DataFrame(test_list, columns=['name'] + metrics_test)

def _process_mse_meta_est(df_main, meta_models, base_models, base_dir, mode, metrics):
    mse_list = []
    for mm in meta_models:
        df_mm = None

        if mm in ('cf', 'xl', 'drs', 'dmls'):
            mse_i = ['-'] * 2
        else:
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
                df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                if mm == 'tl':
                    df_base_val_gr['mse_target'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
                elif mm == 'ipsws':
                    df_base_val_gr['mse_target'] = df_base_val_gr[['mse_prop', 'mse_reg']].mean(axis=1)
                else:
                    df_base_val_gr['mse_target'] = df_base_val_gr['mse']

                for metric in metrics:
                    df_base_test[f'{metric}_target'] = df_base_test[metric]
            
                df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
                df_mm = pd.concat([df_mm, df_base], ignore_index=True)

            metrics_target = [f'{metric}_target' for metric in metrics]
            mse_i = ut.fn_by_best(df_mm, 'mse_target', metrics_target, mode, True)

        mse_list.append([mm] + mse_i)
    
    df_mse = pd.DataFrame(mse_list, columns=['name'] + [f'{metric}_mse' for metric in metrics])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_mse_all(df_main, cate_models, base_dir, mode, metrics):
    mse_list = []
    df_all = None
    for cm in cate_models:
        mm = cm.split('_')[0]

        # These don't support MSE.
        if mm in ('cf', 'xl', 'drs', 'dmls'):
            continue
        
        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue

        df_base_val = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_val_metrics.csv'))
        df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

        if mm == 'tl':
            df_base_val_gr['mse_target'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
        elif mm == 'ipsws':
            df_base_val_gr['mse_target'] = df_base_val_gr[['mse_prop', 'mse_reg']].mean(axis=1)
        else:
            df_base_val_gr['mse_target'] = df_base_val_gr['mse']

        for metric in metrics:
            df_base_test[f'{metric}_target'] = df_base_test[metric]
        metrics_target = [f'{metric}_target' for metric in metrics]
            
        df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    mse_i = ut.fn_by_best(df_all, 'mse_target', metrics_target, mode, True)
    mse_list.append(['all'] + mse_i)
    
    df_mse = pd.DataFrame(mse_list, columns=['name'] + [f'{metric}_mse' for metric in metrics])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_policy_meta_est(df_main, meta_models, base_models, base_dir, mode, metrics):
    mse_list = []
    for mm in meta_models:
        df_mm = None

    
        for bm in base_models:
            model_name = f'{mm}_{bm}'
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
            except:
                print(f'{model_name} is missing')
                continue

            df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
            df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            if mm == 'tl':
                df_base_val_gr['mse_target'] = df_base_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
            elif mm == 'ipsws':
                df_base_val_gr['mse_target'] = df_base_val_gr[['mse_prop', 'mse_reg']].mean(axis=1)
            else:
                df_base_val_gr['mse_target'] = df_base_val_gr['mse']

            for metric in metrics:
                df_base_test[f'{metric}_target'] = df_base_test[metric]
            
            df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
            df_mm = pd.concat([df_mm, df_base], ignore_index=True)

        metrics_target = [f'{metric}_target' for metric in metrics]
        mse_i = ut.fn_by_best(df_mm, 'mse_target', metrics_target, mode, True)

        mse_list.append([mm] + mse_i)
    
    df_mse = pd.DataFrame(mse_list, columns=['name'] + [f'{metric}_mse' for metric in metrics])

    if df_main is None:
        return df_mse
    else:
        return df_main.merge(df_mse, on=['name'])

def _process_policy_all(df_main, cate_models, base_dir, mode, metrics):
    pol_list = []
    df_all = None
    for cm in cate_models:
        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue

        df_base_val = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_val_metrics.csv'))
        df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

        df_base_val_gr['pol_target'] = df_base_val_gr['policy']

        for metric in metrics:
            df_base_test[f'{metric}_target'] = df_base_test[metric]
        metrics_target = [f'{metric}_target' for metric in metrics]
            
        df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    pol_i = ut.fn_by_best(df_all, 'pol_target', metrics_target, mode, True)
    pol_list.append(['all'] + pol_i)
    
    df_pol = pd.DataFrame(pol_list, columns=['name'] + [f'{metric}_pol' for metric in metrics])

    return df_main.merge(df_pol, on=['name'])

def _process_r2scores_meta_est(df_main, meta_models, base_models, base_dir, mode, metrics):
    r2_list = []
    for mm in meta_models:
        df_mm = None

        if mm in ('cf', 'xl', 'drs', 'dmls'):
            r2_i = ['-'] * 2
        else:
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
                df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                if mm == 'tl':
                    df_base_val_gr['r2_score_target'] = df_base_val_gr[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
                elif mm == 'ipsws':
                    df_base_val_gr['r2_score_target'] = df_base_val_gr[['r2_score_prop', 'r2_score_reg']].mean(axis=1)
                else:
                    df_base_val_gr['r2_score_target'] = df_base_val_gr['r2_score']

                for metric in metrics:
                    df_base_test[f'{metric}_target'] = df_base_test[metric]
                metrics_target = [f'{metric}_target' for metric in metrics]
            
                df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
                df_mm = pd.concat([df_mm, df_base], ignore_index=True)

            r2_i = ut.fn_by_best(df_mm, 'r2_score_target', metrics_target, mode, False)

        r2_list.append([mm] + r2_i)
    
    df_r2 = pd.DataFrame(r2_list, columns=['name'] + [f'{metric}_r2' for metric in metrics])
    return df_main.merge(df_r2, on=['name'])

def _process_r2scores_all(df_main, cate_models, base_dir, mode, metrics):
    r2_list = []
    df_all = None
    for cm in cate_models:
        mm = cm.split('_')[0]

        if mm in ('cf', 'xl', 'drs', 'dmls'):
            continue

        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue

        df_base_val = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_val_metrics.csv'))
        df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

        if mm == 'tl':
            df_base_val_gr['r2_score_target'] = df_base_val_gr[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
        elif mm == 'ipsws':
            df_base_val_gr['r2_score_target'] = df_base_val_gr[['r2_score_prop', 'r2_score_reg']].mean(axis=1)
        else:
            df_base_val_gr['r2_score_target'] = df_base_val_gr['r2_score']

        for metric in metrics:
            df_base_test[f'{metric}_target'] = df_base_test[metric]
        metrics_target = [f'{metric}_target' for metric in metrics]
            
        df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    r2_i = ut.fn_by_best(df_all, 'r2_score_target', metrics_target, mode, False)
    r2_list.append(['all'] + r2_i)
    
    df_r2 = pd.DataFrame(r2_list, columns=['name'] + [f'{metric}_r2' for metric in metrics])
    return df_main.merge(df_r2, on=['name'])

def _process_mixed_meta_est(df_main, meta_models, base_models, base_dir, mode, metrics):
    score_list = []
    for mm in meta_models:
        df_mm = None

        if mm in ('cf', 'xl', 'sl', 'tl', 'ipsws'):
            score_i = ['-'] * 2
        else:
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_base_val = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_val_metrics.csv'))
                df_base_val_gr = df_base_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                # mixed = R^2 + ACC - MSE
                df_base_val_gr['mixed'] = df_base_val_gr['reg_score'] + df_base_val_gr['prop_score'] - df_base_val_gr['final_score']
            
                for metric in metrics:
                    df_base_test[f'{metric}_target'] = df_base_test[metric]

                df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
                df_mm = pd.concat([df_mm, df_base], ignore_index=True)

            metrics_target = [f'{metric}_target' for metric in metrics]
            score_i = ut.fn_by_best(df_mm, 'mixed', metrics_target, mode, False)

        score_list.append([mm] + score_i)
    
    df_r2 = pd.DataFrame(score_list, columns=['name'] + [f'{metric}_mixed' for metric in metrics])
    return df_main.merge(df_r2, on=['name'])

def _process_plugins_meta_est(df_main, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for pm in plugin_models:
        plugin_ate_list = []
        plugin_pehe_list = []
        for mm in meta_models:
            df_mm = None
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

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

def _process_plugins_meta_base(df_main, plugin_models, meta_models, base_models, base_dir, plugin_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for pm in plugin_models:
        plugin_ate_list = []
        plugin_pehe_list = []
        for bm in base_models:
            df_bm = None
            for mm in meta_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue
                
                df_plugin_val = pd.read_csv(os.path.join(plugin_dir, pm, f'{model_name}_plugin_{pm}.csv'))
                df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                df_bm = pd.concat([df_bm, df_plugin], ignore_index=True)
            
            # Suffixes are applied only to duplicated column names.
            if 'ate' in metrics: # assume ['ate', 'pehe']
                plugin_ate_i = ut.fn_by_best(df_bm, 'ate_val', metrics_test, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_bm, 'pehe_val', metrics_test, mode, True)
            else:
                plugin_ate_i = ut.fn_by_best(df_bm, 'ate', metrics, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_bm, 'pehe', metrics, mode, True)
            
            plugin_ate_list.append([bm] + plugin_ate_i)
            plugin_pehe_list.append([bm] + plugin_pehe_i)
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_{pm}_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_{pm}_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_plugins_all(df_main, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics):
    df_copy = df_main.copy()
    for pm in plugin_models:
        plugin_ate_list = []
        plugin_pehe_list = []
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue
                
            df_plugin_val = pd.read_csv(os.path.join(plugin_dir, pm, f'{cm}_plugin_{pm}.csv'))
            df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            df_plugin_val_gr['ate_val_target'] = df_plugin_val_gr['ate']
            df_plugin_val_gr['pehe_val_target'] = df_plugin_val_gr['pehe']

            for metric in metrics:
                df_base_test[f'{metric}_test_target'] = df_base_test[metric]

            df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            df_all = pd.concat([df_all, df_plugin], ignore_index=True)
            
        metrics_target = [f'{metric}_test_target' for metric in metrics]
            
        plugin_ate_i = ut.fn_by_best(df_all, 'ate_val_target', metrics_target, mode, True)
        plugin_pehe_i = ut.fn_by_best(df_all, 'pehe_val_target', metrics_target, mode, True)

        plugin_ate_list.append(['all'] + plugin_ate_i)
        plugin_pehe_list.append(['all'] + plugin_pehe_i)
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_{pm}_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_{pm}_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_matching_meta_est(df_main, ks, meta_models, base_models, base_dir, matching_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for k in ks:
        plugin_ate_list = []
        plugin_pehe_list = []
        for mm in meta_models:
            df_mm = None
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_plugin_val = pd.read_csv(os.path.join(matching_dir, f'match_{k}k', f'{model_name}_matching_match_{k}k.csv'))
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
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_match_{k}k_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_match_{k}k_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_matching_meta_base(df_main, ks, meta_models, base_models, base_dir, matching_dir, mode, metrics):
    df_copy = df_main.copy()
    metrics_test = [f'{metric}_test' for metric in metrics]
    for k in ks:
        plugin_ate_list = []
        plugin_pehe_list = []
        for bm in base_models:
            df_bm = None
            for mm in meta_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue
                
                df_plugin_val = pd.read_csv(os.path.join(matching_dir, f'match_{k}k', f'{model_name}_matching_match_{k}k.csv'))
                df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                df_bm = pd.concat([df_bm, df_plugin], ignore_index=True)
            
            # Suffixes are applied only to duplicated column names.
            if 'ate' in metrics: # assume ['ate', 'pehe']
                plugin_ate_i = ut.fn_by_best(df_bm, 'ate_val', metrics_test, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_bm, 'pehe_val', metrics_test, mode, True)
            else:
                plugin_ate_i = ut.fn_by_best(df_bm, 'ate', metrics, mode, True)
                plugin_pehe_i = ut.fn_by_best(df_bm, 'pehe', metrics, mode, True)
            
            plugin_ate_list.append([bm] + plugin_ate_i)
            plugin_pehe_list.append([bm] + plugin_pehe_i)
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_match_{k}k_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_match_{k}k_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_matching_all(df_main, ks, cate_models, base_dir, matching_dir, mode, metrics):
    df_copy = df_main.copy()
    for k in ks:
        plugin_ate_list = []
        plugin_pehe_list = []
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue
                
            df_plugin_val = pd.read_csv(os.path.join(matching_dir, f'match_{k}k', f'{cm}_matching_match_{k}k.csv'))
            df_plugin_val_gr = df_plugin_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            df_plugin_val_gr['ate_val_target'] = df_plugin_val_gr['ate']
            df_plugin_val_gr['pehe_val_target'] = df_plugin_val_gr['pehe']

            for metric in metrics:
                df_base_test[f'{metric}_test_target'] = df_base_test[metric]

            df_plugin = df_plugin_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            df_all = pd.concat([df_all, df_plugin], ignore_index=True)
            
        metrics_target = [f'{metric}_test_target' for metric in metrics]
            
        plugin_ate_i = ut.fn_by_best(df_all, 'ate_val_target', metrics_target, mode, True)
        plugin_pehe_i = ut.fn_by_best(df_all, 'pehe_val_target', metrics_target, mode, True)

        plugin_ate_list.append(['all'] + plugin_ate_i)
        plugin_pehe_list.append(['all'] + plugin_pehe_i)
        
        df_plugin_ate = pd.DataFrame(plugin_ate_list, columns=['name'] + [f'{metric}_match_{k}k_ate' for metric in metrics])
        df_plugin_pehe = pd.DataFrame(plugin_pehe_list, columns=['name'] + [f'{metric}_match_{k}k_pehe' for metric in metrics])
        df_plugin = df_plugin_ate.merge(df_plugin_pehe, on=['name'])
        df_copy = df_copy.merge(df_plugin, on=['name'])
    
    return df_copy

def _process_rscores_meta_est(df_main, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        for mm in meta_models:
            df_mm = None
            for bm in base_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{model_name}_r_score_{rs_name}.csv'))
                df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
                df_mm = pd.concat([df_mm, df_rscore_test], ignore_index=True)

            rscore_i = ut.fn_by_best(df_mm, 'rscore', metrics, mode, False)
            scores_list.append([mm] + rscore_i)
        
        df_rscore = pd.DataFrame(scores_list, columns=['name'] + [f'{metric}_{rs_name}' for metric in metrics])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy

def _process_rscores_meta_base(df_main, rscore_base_models, meta_models, base_models, base_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        for bm in base_models:
            df_bm = None
            for mm in meta_models:
                model_name = f'{mm}_{bm}'
                try:
                    df_base_test = pd.read_csv(os.path.join(base_dir, model_name, f'{model_name}_test_metrics.csv'))
                except:
                    print(f'{model_name} is missing')
                    continue

                df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{model_name}_r_score_{rs_name}.csv'))
                df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
                df_bm = pd.concat([df_bm, df_rscore_test], ignore_index=True)
            
            rscore_i = ut.fn_by_best(df_bm, 'rscore', metrics, mode, False)
            scores_list.append([bm] + rscore_i)
        
        df_rscore = pd.DataFrame(scores_list, columns=['name'] + [f'{metric}_{rs_name}' for metric in metrics])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy

def _process_rscores_all(df_main, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        scores_list = []
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue

            df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{cm}_r_score_{rs_name}.csv'))
            df_rscore_val_gr = df_rscore_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            df_rscore_test = df_rscore_val_gr.merge(df_base_test, on=['iter_id', 'param_id'])
            df_all = pd.concat([df_all, df_rscore_test], ignore_index=True)
            
        rscore_i = ut.fn_by_best(df_all, 'rscore', metrics, mode, False)
        scores_list.append(['all'] + rscore_i)
        
        df_rscore = pd.DataFrame(scores_list, columns=['name'] + [f'{metric}_{rs_name}' for metric in metrics])
        df_copy = df_copy.merge(df_rscore, on=['name'])
    
    return df_copy

def _process_ensemble_all(df_main, cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, mode, metrics):
    df_copy = df_main.copy()
    df_all = None
    results = []
    for cm in cate_models:
        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue

        df_base_test['ensemble_all'] = 0.0
        df_base_test['ensemble_ate'] = 0.0
        df_base_test['ensemble_pehe'] = 0.0

        for pm in plugin_models:
            df_plugin = pd.read_csv(os.path.join(plugin_dir, pm, f'{cm}_plugin_{pm}.csv'))
            df_plugin_gr = df_plugin.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_base_test['ensemble_all'] += df_plugin_gr['ate'] + df_plugin_gr['pehe']
            df_base_test['ensemble_ate'] += df_plugin_gr['ate']
            df_base_test['ensemble_pehe'] += df_plugin_gr['pehe']
        
        for k in match_models:
            df_matching = pd.read_csv(os.path.join(match_dir, f'match_{k}k', f'{cm}_matching_match_{k}k.csv'))
            df_matching_gr = df_matching.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            df_base_test['ensemble_all'] += df_matching_gr['ate'] + df_matching_gr['pehe']
            df_base_test['ensemble_ate'] += df_matching_gr['ate']
            df_base_test['ensemble_pehe'] += df_matching_gr['pehe']
        
        for rs_bm in rscore_base_models:
            rs_name = f'rs_{rs_bm}'
            df_rscore = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{cm}_r_score_{rs_name}.csv'))
            df_rscore_gr = df_rscore.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
            # Subtract R-Score as we want all to be "smaller is better".
            df_base_test['ensemble_all'] -= df_rscore_gr['rscore']
            df_base_test['ensemble_ate'] -= df_rscore_gr['rscore']
            df_base_test['ensemble_pehe'] -= df_rscore_gr['rscore']

        df_all = pd.concat([df_all, df_base_test], ignore_index=True)
    
    result_all = ut.fn_by_best(df_all, 'ensemble_all', metrics, mode, True)
    result_ate = ut.fn_by_best(df_all, 'ensemble_ate', metrics, mode, True)
    result_pehe = ut.fn_by_best(df_all, 'ensemble_pehe', metrics, mode, True)

    results.append(['all'] + result_all + result_ate + result_pehe)

    cols = [f'{metric}_ensemble_{en_type}' for en_type in ['all', 'ate', 'pehe'] for metric in metrics]

    df_result = pd.DataFrame(results, columns=['name'] + cols)
    df_copy = df_copy.merge(df_result, on=['name'])

    return df_copy