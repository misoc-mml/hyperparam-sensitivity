import os
from prob_hyper_comparers import get_probs_all, get_probs_est, get_probs_bl

out_dir = './tables'

ihdp_ate_upper = 0.46
ihdp_ate_lower = 0.2
ihdp_pehe_upper = 2.6
ihdp_pehe_lower = 0.656

jobs_att_upper = 0.11
jobs_att_lower = 0.03
jobs_pol_upper = 0.26
jobs_pol_lower = 0.14

twins_pehe_upper = 0.315
twins_pehe_lower = 0.297

news_ate_upper = 0.6
news_ate_lower = 0.3
news_pehe_upper = 3.4
news_pehe_lower = 2.0

avg_upper = [ihdp_ate_upper, jobs_att_upper, news_ate_upper]
avg_lower = [ihdp_ate_lower, jobs_att_lower, news_ate_lower]

ind_upper = [ihdp_pehe_upper, jobs_pol_upper, twins_pehe_upper, news_pehe_upper]
ind_upper_notwins = [ihdp_pehe_upper, jobs_pol_upper, news_pehe_upper]
ind_lower = [ihdp_pehe_lower, jobs_pol_lower, twins_pehe_lower, news_pehe_lower]

thr_upper = [ihdp_ate_upper, ihdp_pehe_upper, jobs_att_upper, jobs_pol_upper, twins_pehe_upper, news_ate_upper, news_pehe_upper]

datasets = ['ihdp', 'ihdp', 'jobs', 'jobs', 'twins', 'news', 'news']
all_metrics = ['ate', 'pehe', 'att', 'policy', 'pehe', 'ate', 'pehe']

datasets_avg = ['ihdp', 'jobs', 'news']
datasets_ind = ['ihdp', 'jobs', 'twins', 'news']

metrics_avg = ['ate', 'att', 'ate']
metrics_ind = ['pehe', 'policy', 'pehe', 'pehe']
metrics_ind_notwins = ['pehe', 'policy', 'pehe']

meta_models = ['sl', 'tl', 'ipsws', 'drs', 'dmls', 'xl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm']
base_models_nn = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp']
standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp']
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + standalone_cate_models


get_probs_all(all_cate_models, all_metrics, datasets, thr_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_all.csv'), index=False)
get_probs_all(all_cate_models, metrics_avg, datasets_avg, avg_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_all_avg.csv'), index=False)
get_probs_all(all_cate_models, metrics_ind, datasets_ind, ind_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_all_ind.csv'), index=False)

get_probs_est(meta_models, base_models_nn, all_metrics, datasets, thr_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_est.csv'), index=False)
get_probs_est(meta_models, base_models_nn, metrics_avg, datasets_avg, avg_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_est_avg.csv'), index=False)
get_probs_est(meta_models, base_models_nn, metrics_ind, datasets_ind, ind_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_est_ind.csv'), index=False)

get_probs_bl(meta_models, base_models_nn, all_metrics, datasets, thr_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_bl.csv'), index=False)
get_probs_bl(meta_models, base_models_nn, metrics_avg, datasets_avg, avg_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_bl_avg.csv'), index=False)
get_probs_bl(meta_models, base_models_nn, metrics_ind, datasets_ind, ind_upper).to_csv(os.path.join(out_dir, 'compare_probs_hyper_bl_ind.csv'), index=False)