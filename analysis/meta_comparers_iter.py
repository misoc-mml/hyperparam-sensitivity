import os
import pandas as pd
import numpy as np

import utils as ut
import comparers as comp

def compare_metrics_all_val(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    mode = 'metric'

    df_oracle = _process_val_all(cate_models, base_dir, metrics)
    df_mse = _process_mse_all(df_meta, cate_models, base_dir, mode, metrics)
    df_r2 = _process_r2scores_all(df_meta, cate_models, base_dir, mode, metrics)

    if 'policy' in metrics:
        df_pol = _process_policy_all(df_meta, cate_models, base_dir, mode, metrics)

    df_plug = _process_plugins_all(df_meta, plugin_models, cate_models, base_dir, plugin_dir, mode, metrics)
    df_match = _process_matching_all(df_meta, match_models, cate_models, base_dir, match_dir, mode, metrics)
    df_rscore = _process_rscores_all(df_meta, rscore_base_models, cate_models, base_dir, rscore_dir, mode, metrics)
    
    return df_meta

def _target_by_lowest(df, by, target):
    best = df.apply(lambda x: x.loc[x[by].idxmin(), [target]])
    return best.reset_index()[target].to_numpy()

def _process_val_all(cate_models, base_dir, metrics):
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
    best_metrics = [_target_by_lowest(iter_all, f'{metric}_val', f'{metric}_test') for metric in metrics]

    df = pd.DataFrame(best_metrics, columns=metrics)
    df['name_s'] = 'Oracle'
    df['name_l'] = 'Oracle'

    return df

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