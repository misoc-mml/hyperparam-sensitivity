import numpy as np
import pandas as pd

iter_per_dataset = 10

def get_probs_all(models, metrics, datasets, thresholds):
    oracle = get_probs_oracle_all(models, metrics, datasets, thresholds)
    worst = get_probs_worst_all(models, metrics, datasets, thresholds)
    default = get_probs_default_all(models, metrics, datasets, thresholds)

    arr = [oracle, worst, default]
    return pd.DataFrame(arr, columns=['selection', 'prob', 'err'])

def get_probs_est(c_models, b_models, metrics, datasets, thresholds):
    oracle = get_probs_oracle_est(c_models, b_models, metrics, datasets, thresholds)
    worst = get_probs_worst_est(c_models, b_models, metrics, datasets, thresholds)
    default = get_probs_default_est(c_models, b_models, metrics, datasets, thresholds)

    cf_model = ['cf']
    oracle_cf = get_probs_oracle_est(cf_model, cf_model, metrics, datasets, thresholds)
    worst_cf = get_probs_worst_est(cf_model, cf_model, metrics, datasets, thresholds)
    default_cf = get_probs_default_est(cf_model, cf_model, metrics, datasets, thresholds)

    arr = [*oracle, *oracle_cf, *worst, *worst_cf, *default, *default_cf]
    return pd.DataFrame(arr, columns=['selection', 'model', 'prob', 'err'])

def get_probs_bl(c_models, b_models, metrics, datasets, thresholds):
    default = get_probs_default_bl(c_models, b_models, metrics, datasets, thresholds)
    oracle = get_probs_oracle_bl(c_models, b_models, metrics, datasets, thresholds)
    worst = get_probs_worst_bl(c_models, b_models, metrics, datasets, thresholds)

    arr = [*oracle, *worst, *default]
    return pd.DataFrame(arr, columns=['selection', 'model', 'prob', 'err'])

def select_by_mode(df, m_select, m_target, mode):
    if mode == 'best':
        result = df.apply(lambda x: x.loc[x[m_select].idxmin(), [m_target]])[m_target].to_numpy()
    elif mode == 'worst':
        result = df.apply(lambda x: x.loc[x[m_select].idxmax(), [m_target]])[m_target].to_numpy()
    else:
        raise ValueError(f"select_by_mode unrecognised mode value = '{mode}'")
    
    return result

def split_model_name(name):
    if name == 'cf':
        cm = bl = 'cf'
    else:
        cm, bl = name.split('_')

    return cm, bl

def create_model_name(c_name, b_name):
    if c_name == 'cf':
        result = 'cf'
    else:
        result = f'{c_name}_{b_name}'
    
    return result

def get_win_count_def(df, model, ds, m_target, threshold):
    cm, bl = split_model_name(model)
    iter_gr = df.groupby(['iter_id'], as_index=False)[['param_id', m_target]]
    def_id = get_default_id(cm, bl, ds)
    iter_perfs = iter_gr.apply(lambda x: x.loc[x['param_id'] == def_id, [m_target]])[m_target].to_numpy()
    return np.sum(iter_perfs <= threshold)

def get_win_count(df, m_select, m_target, threshold, mode):
    iter_gr = df.groupby(['iter_id'], as_index=False)[[m_select, m_target]]
    iter_perfs = select_by_mode(iter_gr, m_select, m_target, mode)
    return np.sum(iter_perfs <= threshold)

def get_p_err(wins, n):
    if n != 0:
        win_prob = wins / n
        var = ((1.0 - win_prob) * win_prob) / n
        err = 1.96 * np.sqrt(var)
    else:
        win_prob = err = 0.0
        
    return win_prob, err

def get_probs_oracle_all(models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    for m in models:
        for ds, metric, thr in zip(datasets, metrics, thresholds):
            try:
                df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'best')
                n_mc += iter_per_dataset
            except:
                continue

    p_win, err = get_p_err(n_wins, n_mc)
    return ['oracle', p_win, err]

def get_probs_oracle_est(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for cm in c_models:
        n_wins = 0
        n_mc = 0
        for bm in b_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'best')
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['oracle', cm, p_win, err])

    return result

def get_probs_worst_est(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for cm in c_models:
        n_wins = 0
        n_mc = 0
        for bm in b_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'worst')
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['worst', cm, p_win, err])
    
    return result

def get_probs_oracle_bl(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for bm in b_models:
        n_wins = 0
        n_mc = 0
        for cm in c_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'best')
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['oracle', bm, p_win, err])

    return result

def get_probs_default_est(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for cm in c_models:
        n_wins = 0
        n_mc = 0
        for bm in b_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count_def(df_base, m, ds, f'{metric}_test', thr)
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['default', cm, p_win, err])
    
    return result

def get_probs_default_bl(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for bm in b_models:
        n_wins = 0
        n_mc = 0
        for cm in c_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count_def(df_base, m, ds, f'{metric}_test', thr)
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['default', bm, p_win, err])
    
    return result

def get_probs_worst_bl(c_models, b_models, metrics, datasets, thresholds):
    result = []
    for bm in b_models:
        n_wins = 0
        n_mc = 0
        for cm in c_models:
            m = create_model_name(cm, bm)
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                    df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                    df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                    df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                    n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'worst')
                    n_mc += iter_per_dataset
                except:
                    continue

        p_win, err = get_p_err(n_wins, n_mc)
        result.append(['worst', bm, p_win, err])
    
    return result

def get_probs_default_all(models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    for m in models:
        for ds, metric, thr in zip(datasets, metrics, thresholds):
            try:
                df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins += get_win_count_def(df_base, m, ds, f'{metric}_test', thr)
                n_mc += iter_per_dataset
            except:
                continue

    p_win, err = get_p_err(n_wins, n_mc)
    return ['default', p_win, err]

def get_probs_worst_all(models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    for m in models:
        for ds, metric, thr in zip(datasets, metrics, thresholds):
            try:
                df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr, 'worst')
                n_mc += iter_per_dataset
            except:
                continue

    p_win, err = get_p_err(n_wins, n_mc)
    return ['worst', p_win, err]

def get_default_params(name, ds):
    if name == 'l1':
        return {'alpha': 1, 'max_iter': 1000}
    elif name == 'l2':
        return {'alpha': 1, 'max_iter': 1000}
    elif name == 'dt':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'rf':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'et':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'kr':
        return {"alpha": 1, "gamma": 1, "kernel": "poly", "degree": 3}
    elif name == 'cb':
        return {"depth": 10, "l2_leaf_reg": 1}
    elif name == 'lgbm':
        return {"max_depth": 10, "reg_lambda": 0.1}
    elif name == 'cf':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'mlp':
        return {'n_layers': 1, 'n_units': 128, 'learning_rate': 1e-4, 'activation': 'relu', 'dropout': 0.25, 'l2': 1e-2, 'batch_size': -1, 'epochs': 10000, 'epochs_are_steps': True}
    else:
        raise ValueError("Unrecognised 'get_default_params' key.")
    
def get_condition(params, defaults, name):
    if name == 'l1':
        cond = (params['alpha'] == defaults['alpha']) & (params['max_iter'] == defaults['max_iter'])
    elif name == 'l2':
        cond = (params['alpha'] == defaults['alpha']) & (params['max_iter'] == defaults['max_iter'])
    elif name == 'dt':
        cond = (params['max_depth'] == defaults['max_depth']) & (params['min_samples_leaf'] == defaults['min_samples_leaf'])
    elif name == 'rf':
        cond = (params['max_depth'] == defaults['max_depth']) & (params['min_samples_leaf'] == defaults['min_samples_leaf'])
    elif name == 'et':
        cond = (params['max_depth'] == defaults['max_depth']) & (params['min_samples_leaf'] == defaults['min_samples_leaf'])
    elif name == 'kr':
        cond = (params['alpha'] == defaults['alpha']) & (params['gamma'] == defaults['gamma']) & (params['kernel'] == defaults['kernel']) & (params['degree'] == defaults['degree'])
    elif name == 'cb':
        cond = (params['depth'] == defaults['depth']) & (params['l2_leaf_reg'] == defaults['l2_leaf_reg'])
    elif name == 'lgbm':
        cond = (params['max_depth'] == defaults['max_depth']) & (params['reg_lambda'] == defaults['reg_lambda'])
    elif name == 'cf':
        cond = (params['max_depth'] == defaults['max_depth']) & (params['min_samples_leaf'] == defaults['min_samples_leaf'])
    elif name == 'mlp':
        cond = (params['n_layers'] == defaults['n_layers']) & (params['n_units'] == defaults['n_units']) & (params['learning_rate'] == defaults['learning_rate']) & (params['activation'] == defaults['activation']) & (params['dropout'] == defaults['dropout']) & (params['l2'] == defaults['l2']) & (params['batch_size'] == defaults['batch_size']) & (params['epochs'] == defaults['epochs'])
    else:
        raise ValueError("Unrecognised 'get_condition' key.")
    
    return cond

def get_base_id(est, bl, ds, defaults, file_prefix):
    model_name = f'{est}_{bl}'
    params = pd.read_csv(f'../results/predictions/{ds}/{model_name}/{file_prefix}_params.csv')
    cond = get_condition(params, defaults, bl)
    return int(params.loc[cond, 'id'])

# Get param_id that links to default HPs
def get_default_id(est, bl, ds):
    defaults = get_default_params(bl, ds)
    model_name = f'{est}_{bl}'

    if est in ['sl', 'ipsws', 'dmls', 'drs', 'xl']:
        id = get_base_id(est, bl, ds, defaults, model_name)
    elif est == 'tl':
        m0_id = get_base_id(est, bl, ds, defaults, f'{model_name}_m0')
        m1_id = get_base_id(est, bl, ds, defaults, f'{model_name}_m1')
        params = pd.read_csv(f'../results/predictions/{ds}/{model_name}/{model_name}_cate_params.csv')
        id = int(params.loc[(params['m0'] == m0_id) & (params['m1'] == m1_id), 'id'])
    elif est == 'cf':
        params = pd.read_csv(f'../results/predictions/{ds}/{est}/{est}_params.csv')
        cond = get_condition(params, defaults, est)
        id = int(params.loc[cond, 'id'])
    else:
        id = -1
    
    return id

def parse_est_name(name):
    name = name.upper()
    if name == 'IPSWS':
        return 'IPSW'
    elif name == 'DRS':
        return 'DR'
    elif name == 'DMLS':
        return 'DML'
    else:
        return name
    
def parse_base_name(name):
    name = name.upper()
    if name == 'MLP':
        return 'NN'
    else:
        return name