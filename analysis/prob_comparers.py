import numpy as np
import pandas as pd

iter_per_dataset = 10

def get_win_count(df, m_select, m_target, threshold):
    iter_gr = df.groupby(['iter_id'], as_index=False)[[m_select, m_target]]
    iter_perfs = iter_gr.apply(lambda x: x.loc[x[m_select].idxmin(), [m_target]])[m_target].to_numpy()
    return np.sum(iter_perfs <= threshold)

def get_p_err(wins, n):
    win_prob = wins / n
    var = ((1.0 - win_prob) * win_prob) / n
    err = 1.96 * np.sqrt(var)
    return win_prob, err

def get_probs(models, plugin_models, ks, r_models, metrics, datasets, thresholds):
    oracle = get_prob_oracle(models, metrics, datasets, thresholds)
    mse = get_prob_mse(models, metrics, datasets, thresholds)
    plugin = get_prob_plugin(plugin_models, models, metrics, datasets, thresholds)
    match = get_prob_matching(ks, models, metrics, datasets, thresholds)
    rscore = get_prob_rscore(r_models, models, metrics, datasets, thresholds)

    arr = [oracle, mse, *plugin, *match, *rscore]
    return pd.DataFrame(arr, columns=['metric', 'prob', 'err'])

def get_prob_oracle(models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    for m in models:
        for ds, metric, thr in zip(datasets, metrics, thresholds):
            try:
                df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins += get_win_count(df_base, f'{metric}_val', f'{metric}_test', thr)
                n_mc += iter_per_dataset
            except:
                continue

    p_win, err = get_p_err(n_wins, n_mc)
    return ['oracle', p_win, err]

def get_prob_mse(models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    for m in models:
        cm = m.split('_')[0]
        # These don't support MSE.
        if cm in ('cf', 'xl', 'drs', 'dmls'):
            continue
        for ds, metric, thr in zip(datasets, metrics, thresholds):
            try:
                df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
            except:
                continue
            df_val = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_val_metrics.csv')
            df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

            if cm == 'tl':
                df_val_gr['mse_target'] = df_val_gr[['mse_m0', 'mse_m1']].mean(axis=1)
            elif cm == 'ipsws':
                df_val_gr['mse_target'] = df_val_gr[['mse_prop', 'mse_reg']].mean(axis=1)
            else:
                df_val_gr['mse_target'] = df_val_gr['mse']
            
            df_test[f'{metric}_target'] = df_test[metric]
            df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

            n_wins += get_win_count(df_base, 'mse_target', f'{metric}_target', thr)
            n_mc += iter_per_dataset
    
    p_win, err = get_p_err(n_wins, n_mc)
    return ['mse', p_win, err]

def get_prob_plugin(plugin_models, models, metrics, datasets, thresholds):
    n_wins_ate = 0
    n_wins_pehe = 0
    n_mc = 0
    results = []
    for pm in plugin_models:
        for m in models:
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                except:
                    continue
                df_val = pd.read_csv(f'../results/scores/{ds}/{pm}/{m}_plugin_{pm}.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_val_gr['ate_val_target'] = df_val_gr['ate']
                df_val_gr['pehe_val_target'] = df_val_gr['pehe']

                df_test[f'{metric}_test_target'] = df_test[metric]
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins_ate += get_win_count(df_base, 'ate_val_target', f'{metric}_test_target', thr)
                n_wins_pehe += get_win_count(df_base, 'pehe_val_target', f'{metric}_test_target', thr)
                n_mc += iter_per_dataset
        
        p_win_ate, err_ate = get_p_err(n_wins_ate, n_mc)
        p_win_pehe, err_pehe = get_p_err(n_wins_pehe, n_mc)
        results.append([f'{pm}_ate', p_win_ate, err_ate])
        results.append([f'{pm}_pehe', p_win_pehe, err_pehe])
    
    return results

def get_prob_matching(ks, models, metrics, datasets, thresholds):
    n_wins_ate = 0
    n_wins_pehe = 0
    n_mc = 0
    results = []
    for k in ks:
        for m in models:
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                except:
                    continue
                df_val = pd.read_csv(f'../results/scores/{ds}/match_{k}k/{m}_matching_match_{k}k.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])

                df_val_gr['ate_val_target'] = df_val_gr['ate']
                df_val_gr['pehe_val_target'] = df_val_gr['pehe']

                df_test[f'{metric}_test_target'] = df_test[metric]
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins_ate += get_win_count(df_base, 'ate_val_target', f'{metric}_test_target', thr)
                n_wins_pehe += get_win_count(df_base, 'pehe_val_target', f'{metric}_test_target', thr)
                n_mc += iter_per_dataset

        p_win_ate, err_ate = get_p_err(n_wins_ate, n_mc)
        p_win_pehe, err_pehe = get_p_err(n_wins_pehe, n_mc)
        results.append([f'match_{k}k_ate', p_win_ate, err_ate])
        results.append([f'match_{k}k_pehe', p_win_pehe, err_pehe])

    return results

def get_prob_rscore(r_models, models, metrics, datasets, thresholds):
    n_wins = 0
    n_mc = 0
    results = []
    for rm in r_models:
        rs_name = f'rs_{rm}'    
        for m in models:
            for ds, metric, thr in zip(datasets, metrics, thresholds):
                try:
                    df_test = pd.read_csv(f'../results/metrics_val/{ds}/{m}/{m}_test_metrics.csv')
                except:
                    continue
                df_val = pd.read_csv(f'../results/scores/{ds}/{rs_name}/{m}_r_score_{rs_name}.csv')
                df_val_gr = df_val.groupby(['iter_id', 'param_id'], as_index=False).mean().drop(columns=['fold_id'])
                df_base = df_val_gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

                n_wins += get_win_count(df_base, 'rscore', metric, thr)
                n_mc += iter_per_dataset

        p_win, err = get_p_err(n_wins, n_mc)
        results.append([rs_name, p_win, err])

    return results