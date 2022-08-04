import os
import numpy as np
import pandas as pd

def _mean_std_str(mean, std):
    return f'{mean:.3f} +/- {std:.3f}'

def _merge_scores(scores1, scores2):
    return [_mean_std_str(s1, s2) for s1, s2 in zip(scores1, scores2)]

def load_scores(df_scores, df_test):
    gr = df_scores.groupby(['iter_id', 'param_id'], as_index=False).mean()
    gr = gr.drop(columns=['fold_id'])

    return gr.merge(df_test, on=['iter_id', 'param_id'], suffixes=['_val', '_test'])

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

def get_best_metric(df, col):
    iter_gr = df.groupby(['iter_id'], as_index=False)
    best_iter = iter_gr.apply(lambda x: x.loc[x[col].idxmin(), [col]])
    best_mean = np.mean(best_iter[col])
    best_std = np.std(best_iter[col])

    return _mean_std_str(best_mean, best_std)

def get_best_by(df, by, targets):
    iter_gr = df.groupby(['iter_id'], as_index=False)
    best_by_iter = iter_gr.apply(lambda x: x.loc[x[by].idxmin(), targets])
    best_mean = np.mean(best_by_iter[targets], axis=0)
    best_std = np.std(best_by_iter[targets], axis=0)

    return _merge_scores(best_mean, best_std)

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

def get_achieved_best_plugin(df):
    iter_gr = df.groupby(['iter_id'], as_index=False)

    plugin_ate_iter = iter_gr.apply(lambda x: x.loc[x['ate_val'].idxmin(), ['ate_test', 'pehe_test']])
    plugin_pehe_iter = iter_gr.apply(lambda x: x.loc[x['pehe_val'].idxmin(), ['ate_test', 'pehe_test']])

    best_ate_iter = iter_gr.apply(lambda x: x.loc[x['ate_test'].idxmin(), ['ate_test']])
    best_pehe_iter = iter_gr.apply(lambda x: x.loc[x['pehe_test'].idxmin(), ['pehe_test']])

    d = {'ate_on_plugin_ate': plugin_ate_iter['ate_test'],
        'ate_on_plugin_pehe': plugin_pehe_iter['ate_test'],
        'ate_best': best_ate_iter['ate_test'],
        'pehe_on_plugin_ate': plugin_ate_iter['pehe_test'],
        'pehe_on_plugin_pehe': plugin_pehe_iter['pehe_test'],
        'pehe_best': best_pehe_iter['pehe_test']}
    df = pd.DataFrame(data=d)

    d_mean = np.mean(df[['ate_on_plugin_ate', 'ate_on_plugin_pehe', 'ate_best', 'pehe_on_plugin_ate', 'pehe_on_plugin_pehe', 'pehe_best']], axis=0)
    d_std = np.std(df[['ate_on_plugin_ate', 'ate_on_plugin_pehe', 'ate_best', 'pehe_on_plugin_ate', 'pehe_on_plugin_pehe', 'pehe_best']], axis=0)

    return _merge_scores(d_mean, d_std)