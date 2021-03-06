import numpy as np
import pandas as pd

def _merge_scores(scores1, scores2):
    return [f'{s1:.3f} +/- {s2:.3f}' for s1, s2 in zip(scores1, scores2)]

def load_merge(path, method, cols=['mse']):
    val_scores = pd.read_csv(f'{path}/{method}/{method}_val_metrics.csv')
    test_scores = pd.read_csv(f'{path}/{method}/{method}_test_metrics.csv')

    gr = val_scores.groupby(['iter_id', 'param_id'], as_index=False).mean()
    gr = gr.drop(columns=['fold_id'])

    if len(cols) > 1:
        gr['mse'] = gr[cols].mean(axis=1)
        test_scores['mse'] = test_scores[cols].mean(axis=1)

    return gr.merge(test_scores, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

def get_mse_corr(df):
    corr_per_iter = df.groupby(['iter_id'], as_index=False).corrwith(df['mse_val'])
    corr_mean = np.mean(corr_per_iter[['ate_val', 'pehe_val', 'mse_test', 'ate_test', 'pehe_test']], axis=0)
    corr_std = np.std(corr_per_iter[['ate_val', 'pehe_val', 'mse_test', 'ate_test', 'pehe_test']], axis=0)

    return _merge_scores(corr_mean, corr_std)

def get_risk(df):
    iter_gr = df.groupby(['iter_id'], as_index=False)
    best_mse_iter = iter_gr.apply(lambda x: x.loc[x['mse_val'].idxmin(), ['ate_test', 'pehe_test']])
    best_ate_iter = iter_gr.apply(lambda x: x.loc[x['ate_test'].idxmin(), ['ate_test']])
    best_pehe_iter = iter_gr.apply(lambda x: x.loc[x['pehe_test'].idxmin(), ['pehe_test']])

    ate_risk_raw = (best_mse_iter['ate_test'] - best_ate_iter['ate_test'])
    ate_risk_mean = np.mean(ate_risk_raw)
    ate_risk_std = np.std(ate_risk_raw)
    ate_risk = [f'{ate_risk_mean:.3f} +/- {ate_risk_std:.3f}']

    pehe_risk_raw = (best_mse_iter['pehe_test'] - best_pehe_iter['pehe_test'])
    pehe_risk_mean = np.mean(pehe_risk_raw)
    pehe_risk_std = np.std(pehe_risk_raw)
    pehe_risk = [f'{pehe_risk_mean:.3f} +/- {pehe_risk_std:.3f}']

    return ate_risk, pehe_risk

def get_achieved_best(df):
    iter_gr = df.groupby(['iter_id'], as_index=False)
    best_mse_iter = iter_gr.apply(lambda x: x.loc[x['mse_val'].idxmin(), ['ate_test', 'pehe_test']])
    best_ate_iter = iter_gr.apply(lambda x: x.loc[x['ate_test'].idxmin(), ['ate_test']])
    best_pehe_iter = iter_gr.apply(lambda x: x.loc[x['pehe_test'].idxmin(), ['pehe_test']])

    d = {'ate_on_mse': best_mse_iter['ate_test'], 'ate_best': best_ate_iter['ate_test'],
        'pehe_on_mse': best_mse_iter['pehe_test'], 'pehe_best': best_pehe_iter['pehe_test']}
    df = pd.DataFrame(data=d)

    d_mean = np.mean(df[['ate_on_mse', 'ate_best', 'pehe_on_mse', 'pehe_best']], axis=0)
    d_std = np.std(df[['ate_on_mse', 'ate_best', 'pehe_on_mse', 'pehe_best']], axis=0)

    return _merge_scores(d_mean, d_std)