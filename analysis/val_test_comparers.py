import os
import pandas as pd
import utils as ut

def compare_metrics_test(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    df = _process_potential_test(cate_models, base_dir, metrics)
    df = _process_mse_test(df, cate_models, base_dir, metrics)
    df = _process_r2_test(df, cate_models, base_dir, metrics)
    df = _process_plugin_test(df, plugin_models, cate_models, base_dir, plugin_dir, metrics)
    df = _process_matching_test(df, match_models, cate_models, base_dir, match_dir, metrics)
    df = _process_rscore_test(df, rscore_base_models, cate_models, base_dir, rscore_dir, metrics)

    return df

def compare_metrics_val(cate_models, plugin_models, match_models, rscore_base_models, base_dir, plugin_dir, match_dir, rscore_dir, metrics):
    df = _process_potential_val(cate_models, base_dir, metrics)
    df = _process_mse_val(df, cate_models, base_dir, metrics)
    df = _process_r2_val(df, cate_models, base_dir, metrics)
    df = _process_plugin_val(df, plugin_models, cate_models, base_dir, plugin_dir, metrics)
    df = _process_matching_val(df, match_models, cate_models, base_dir, match_dir, metrics)
    df = _process_rscore_val(df, rscore_base_models, cate_models, base_dir, rscore_dir, metrics)

    return df

def _process_potential_test(cate_models, base_dir, metrics):
    df_all = None
    for cm in cate_models:
        try:
            df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
        except:
            print(f'{cm} is missing')
            continue
        df_all = pd.concat([df_all, df_base_test], ignore_index=True)
    
    d_selected = {f'{metric}_test': ut.get_best_metric_iter(df_all, metric) for metric in metrics}

    return pd.DataFrame.from_dict(d_selected)

def _process_potential_val(cate_models, base_dir, metrics):
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

    d_selected = {f'{metric}_test': ut.get_best_metric_by_iter(df_all, f'{metric}_val', f'{metric}_test') for metric in metrics}

    return pd.DataFrame.from_dict(d_selected)

def _process_mse_test(df_main, cate_models, base_dir, metrics):
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

        if mm == 'tl':
            df_base_test['mse_target'] = df_base_test[['mse_m0', 'mse_m1']].mean(axis=1)
        elif mm == 'ipsws':
            df_base_test['mse_target'] = df_base_test[['mse_prop', 'mse_reg']].mean(axis=1)
        else:
            df_base_test['mse_target'] = df_base_test['mse']

        for metric in metrics:
            df_base_test[f'{metric}_target'] = df_base_test[metric]
        
        df_all = pd.concat([df_all, df_base_test], ignore_index=True)

    d_selected = {f'{metric}_mse': ut.get_best_metric_by_iter(df_all, 'mse_target', f'{metric}_target') for metric in metrics}
    df_mse = pd.DataFrame.from_dict(d_selected)

    return pd.concat([df_main, df_mse], axis=1)

def _process_mse_val(df_main, cate_models, base_dir, metrics):
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
            
        df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    d_selected = {f'{metric}_mse': ut.get_best_metric_by_iter(df_all, 'mse_target', f'{metric}_target') for metric in metrics}
    df_mse = pd.DataFrame.from_dict(d_selected)

    return pd.concat([df_main, df_mse], axis=1)

def _process_r2_test(df_main, cate_models, base_dir, metrics):
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

        if mm == 'tl':
            df_base_test['r2_score_target'] = df_base_test[['r2_score_m0', 'r2_score_m1']].mean(axis=1)
        elif mm == 'ipsws':
            df_base_test['r2_score_target'] = df_base_test[['r2_score_prop', 'r2_score_reg']].mean(axis=1)
        else:
            df_base_test['r2_score_target'] = df_base_test['r2_score']

        for metric in metrics:
            df_base_test[f'{metric}_target'] = df_base_test[metric]
            
        df_all = pd.concat([df_all, df_base_test], ignore_index=True)

    d_selected = {f'{metric}_r2': ut.get_best_metric_by_iter(df_all, 'r2_score_target', f'{metric}_target', False) for metric in metrics}
    df_r2 = pd.DataFrame.from_dict(d_selected)

    return pd.concat([df_main, df_r2], axis=1)

def _process_r2_val(df_main, cate_models, base_dir, metrics):
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
            
        df_base = df_base_val_gr.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])
        df_all = pd.concat([df_all, df_base], ignore_index=True)

    d_selected = {f'{metric}_r2': ut.get_best_metric_by_iter(df_all, 'r2_score_target', f'{metric}_target', False) for metric in metrics}
    df_r2 = pd.DataFrame.from_dict(d_selected)

    return pd.concat([df_main, df_r2], axis=1)

def _process_plugin_test(df_main, plugin_models, cate_models, base_dir, plugin_dir, metrics):
    df_copy = df_main.copy()
    for pm in plugin_models:
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue
                
            df_plugin_val = pd.read_csv(os.path.join(plugin_dir, pm, f'{cm}_plugin_{pm}_test.csv'))
            df_plugin_val['ate_val_target'] = df_plugin_val['ate']
            df_plugin_val['pehe_val_target'] = df_plugin_val['pehe']

            for metric in metrics:
                df_base_test[f'{metric}_test_target'] = df_base_test[metric]

            df_plugin = df_plugin_val.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            df_all = pd.concat([df_all, df_plugin], ignore_index=True)
            
        d_ate = {f'{metric}_{pm}_ate': ut.get_best_metric_by_iter(df_all, 'ate_val_target', f'{metric}_test_target') for metric in metrics}
        d_pehe = {f'{metric}_{pm}_pehe': ut.get_best_metric_by_iter(df_all, 'pehe_val_target', f'{metric}_test_target') for metric in metrics}
        
        df_ate = pd.DataFrame.from_dict(d_ate)
        df_pehe = pd.DataFrame.from_dict(d_pehe)
        
        df_copy = pd.concat([df_copy, df_ate, df_pehe], axis=1)
    
    return df_copy

def _process_plugin_val(df_main, plugin_models, cate_models, base_dir, plugin_dir, metrics):
    df_copy = df_main.copy()
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
            
        d_ate = {f'{metric}_{pm}_ate': ut.get_best_metric_by_iter(df_all, 'ate_val_target', f'{metric}_test_target') for metric in metrics}
        d_pehe = {f'{metric}_{pm}_pehe': ut.get_best_metric_by_iter(df_all, 'pehe_val_target', f'{metric}_test_target') for metric in metrics}
        
        df_ate = pd.DataFrame.from_dict(d_ate)
        df_pehe = pd.DataFrame.from_dict(d_pehe)
        
        df_copy = pd.concat([df_copy, df_ate, df_pehe], axis=1)
    
    return df_copy

def _process_matching_test(df_main, ks, cate_models, base_dir, matching_dir, metrics):
    df_copy = df_main.copy()
    for k in ks:
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue
                
            df_plugin_val = pd.read_csv(os.path.join(matching_dir, f'match_{k}k', f'{cm}_matching_match_{k}k_test.csv'))
            df_plugin_val['ate_val_target'] = df_plugin_val['ate']
            df_plugin_val['pehe_val_target'] = df_plugin_val['pehe']

            for metric in metrics:
                df_base_test[f'{metric}_test_target'] = df_base_test[metric]

            df_plugin = df_plugin_val.merge(df_base_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            df_all = pd.concat([df_all, df_plugin], ignore_index=True)
        
        d_ate = {f'{metric}_match_{k}k_ate': ut.get_best_metric_by_iter(df_all, 'ate_val_target', f'{metric}_test_target') for metric in metrics}
        d_pehe = {f'{metric}_match_{k}k_pehe': ut.get_best_metric_by_iter(df_all, 'pehe_val_target', f'{metric}_test_target') for metric in metrics}
        
        df_ate = pd.DataFrame.from_dict(d_ate)
        df_pehe = pd.DataFrame.from_dict(d_pehe)
        
        df_copy = pd.concat([df_copy, df_ate, df_pehe], axis=1)
    
    return df_copy

def _process_matching_val(df_main, ks, cate_models, base_dir, matching_dir, metrics):
    df_copy = df_main.copy()
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

        d_ate = {f'{metric}_match_{k}k_ate': ut.get_best_metric_by_iter(df_all, 'ate_val_target', f'{metric}_test_target') for metric in metrics}
        d_pehe = {f'{metric}_match_{k}k_pehe': ut.get_best_metric_by_iter(df_all, 'pehe_val_target', f'{metric}_test_target') for metric in metrics}
        
        df_ate = pd.DataFrame.from_dict(d_ate)
        df_pehe = pd.DataFrame.from_dict(d_pehe)
        
        df_copy = pd.concat([df_copy, df_ate, df_pehe], axis=1)
    
    return df_copy

def _process_rscore_test(df_main, rscore_base_models, cate_models, base_dir, rscore_dir, metrics):
    df_copy = df_main.copy()
    for rs_bm in rscore_base_models:
        rs_name = f'rs_{rs_bm}'
        df_all = None
        for cm in cate_models:
            try:
                df_base_test = pd.read_csv(os.path.join(base_dir, cm, f'{cm}_test_metrics.csv'))
            except:
                print(f'{cm} is missing')
                continue

            df_rscore_val = pd.read_csv(os.path.join(rscore_dir, rs_name, f'{cm}_r_score_{rs_name}_test.csv'))
            df_rscore_test = df_rscore_val.merge(df_base_test, on=['iter_id', 'param_id'])
            df_all = pd.concat([df_all, df_rscore_test], ignore_index=True)
            
        d_selected = {f'{metric}_{rs_name}': ut.get_best_metric_by_iter(df_all, 'rscore', metric, False) for metric in metrics}
        df_rscore = pd.DataFrame.from_dict(d_selected)

        df_copy = pd.concat([df_copy, df_rscore], axis=1)
    
    return df_copy

def _process_rscore_val(df_main, rscore_base_models, cate_models, base_dir, rscore_dir, metrics):
    df_copy = df_main.copy()
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
            
        d_selected = {f'{metric}_{rs_name}': ut.get_best_metric_by_iter(df_all, 'rscore', metric, False) for metric in metrics}
        df_rscore = pd.DataFrame.from_dict(d_selected)

        df_copy = pd.concat([df_copy, df_rscore], axis=1)
    
    return df_copy

