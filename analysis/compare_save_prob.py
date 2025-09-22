import os
import prob_comparers as pcomp

out_dir = './tables'

plugin_meta_models = ['sl', 'tl']
plugin_base_models = ['dt', 'lgbm', 'kr']
plugin_models = [f'{pmm}_{pbm}' for pmm in plugin_meta_models for pbm in plugin_base_models]

matching_ks = [1, 3, 5]
rscore_base_models = ['dt', 'lgbm', 'kr']

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
standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp']
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + standalone_cate_models


#pcomp.get_probs(all_cate_models, plugin_models, matching_ks, rscore_base_models, all_metrics, datasets, thr_upper).to_csv(os.path.join(out_dir, 'compare_probs_all.csv'), index=False)

pcomp.get_probs(all_cate_models, plugin_models, matching_ks, rscore_base_models, metrics_avg, datasets_avg, avg_upper).to_csv(os.path.join(out_dir, 'compare_probs_avg.csv'), index=False)

pcomp.get_probs(all_cate_models, plugin_models, matching_ks, rscore_base_models, metrics_ind, datasets_ind, ind_upper).to_csv(os.path.join(out_dir, 'compare_probs_ind.csv'), index=False)

pcomp.get_probs(all_cate_models, plugin_models, matching_ks, rscore_base_models, metrics_ind_notwins, datasets_avg, ind_upper_notwins).to_csv(os.path.join(out_dir, 'compare_probs_ind_notwins.csv'), index=False)