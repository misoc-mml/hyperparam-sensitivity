import os
import numpy as np
import pandas as pd
import utils as ut

def compare_metrics_all_val(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    df_oracle = _process_val_all(cate_models, base_dir, metrics)
    df_mse = _process_mse_all(cate_models, base_dir, metrics)
    df_r2 = _process_r2scores_all(cate_models, base_dir, metrics)
    df_plug = _process_plugins_all(plugin_models, cate_models, base_dir, plugin_dir, metrics)
    df_match = _process_matching_all(match_models, cate_models, base_dir, match_dir, metrics)
    df_rscore = _process_rscores_all(rscore_base_models, cate_models, base_dir, rscore_dir, metrics)
    
    if 'policy' in metrics:
        df_pol = _process_policy_all(cate_models, base_dir, metrics)
        return pd.concat([df_oracle, df_mse, df_r2, df_plug, df_match, df_rscore, df_pol], axis=0)
    else:
        return pd.concat([df_oracle, df_mse, df_r2, df_plug, df_match, df_rscore], axis=0)

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

    df = pd.DataFrame(np.array(best_metrics).T, columns=metrics)
    df['name_s'] = 'Oracle'
    df['name_l'] = 'Oracle'

    return df

def _process_mse_all(cate_models, base_dir, metrics):
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

    mse_i = [ut.get_best_metric_by_iter(df_all, 'mse_target', metric, True) for metric in metrics_target]

    df = pd.DataFrame(np.array(mse_i).T, columns=metrics)
    df['name_s'] = 'MSE'
    df['name_l'] = 'MSE'

    return df
    
def _process_policy_all(cate_models, base_dir, metrics):
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

    pol_i = [ut.get_best_metric_by_iter(df_all, 'pol_target', metric, True) for metric in metrics_target]
    df_pol = pd.DataFrame(np.array(pol_i).T, columns=metrics)
    df_pol['name_s'] = 'policy'
    df_pol['name_l'] = 'policy'

    return df_pol

def _process_r2scores_all(cate_models, base_dir, metrics):
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

    r2_i = [ut.get_best_metric_by_iter(df_all, 'r2_score_target', metric, False) for metric in metrics_target]
    df_r2 = pd.DataFrame(np.array(r2_i).T, columns=metrics)
    df_r2['name_s'] = 'R2'
    df_r2['name_l'] = 'R2'

    return df_r2

def _process_plugins_all(plugin_models, cate_models, base_dir, plugin_dir, metrics):
    df_main = None
    for pm in plugin_models:
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
            
        plugin_ate_i = [ut.get_best_metric_by_iter(df_all, 'ate_val_target', metric, True) for metric in metrics_target]
        plugin_pehe_i = [ut.get_best_metric_by_iter(df_all, 'pehe_val_target', metric, True) for metric in metrics_target]

        df_ate = pd.DataFrame(np.array(plugin_ate_i).T, columns=metrics)
        df_ate['name_s'] = 'plugin_ate'
        df_ate['name_l'] = f'plugin_ate_{pm}'

        df_pehe = pd.DataFrame(np.array(plugin_pehe_i).T, columns=metrics)
        df_pehe['name_s'] = 'plugin_pehe'
        df_pehe['name_l'] = f'plugin_pehe_{pm}'

        df_main = pd.concat([df_main, df_ate, df_pehe], axis=0)
    
    return df_main

def _process_matching_all(ks, cate_models, base_dir, matching_dir, metrics):
    df_main = None
    for k in ks:
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

        plugin_ate_i = [ut.get_best_metric_by_iter(df_all, 'ate_val_target', metric, True) for metric in metrics_target]
        plugin_pehe_i = [ut.get_best_metric_by_iter(df_all, 'pehe_val_target', metric, True) for metric in metrics_target]

        df_ate = pd.DataFrame(np.array(plugin_ate_i).T, columns=metrics)
        df_ate['name_s'] = 'matching_ate'
        df_ate['name_l'] = f'matching_ate_{k}'

        df_pehe = pd.DataFrame(np.array(plugin_pehe_i).T, columns=metrics)
        df_pehe['name_s'] = 'plugin_pehe'
        df_pehe['name_l'] = f'plugin_pehe_{k}'

        df_main = pd.concat([df_main, df_ate, df_pehe], axis=0)
    
    return df_main

def _process_rscores_all(rscore_base_models, cate_models, base_dir, rscore_dir, metrics):
    df_main = None
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
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
            
        rscore_i = [ut.get_best_metric_by_iter(df_all, 'rscore', metric, False) for metric in metrics]
        
        df_rscore = pd.DataFrame(np.array(rscore_i).T, columns=metrics)
        df_rscore['name_s'] = 'rscore'
        df_rscore['name_l'] = f'rscore_{rs_bm}'

        df_main = pd.concat([df_main, df_rscore], axis=0)
        
    return df_main