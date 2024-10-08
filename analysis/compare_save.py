import os
import comparers as comp
import meta_comparers as mcomp

out_dir = './tables'

base_metrics_dir = '../results/metrics/'

plugin_meta_models = ['sl', 'tl']
plugin_base_models = ['dt', 'lgbm', 'kr']
plugin_models = [f'{pmm}_{pbm}' for pmm in plugin_meta_models for pbm in plugin_base_models]
plugin_dir = '../results/scores/'

matching_ks = [1, 3, 5]
matching_dir = '../results/scores/'

rscore_base_models = ['dt', 'lgbm', 'kr']
rscore_dir = '../results/scores/'

datasets = ['ihdp', 'jobs', 'twins', 'news']
all_metrics = {'ihdp': ['ate', 'pehe'], 'jobs': ['att', 'policy'], 'news': ['ate', 'pehe'], 'twins': ['ate', 'pehe']}


# ==================== 'LATEX' OUTPUTS ====================
#
# make sure RESULTS in 'utils.py' is set to 'latex'
# use with:
# - metrics_meta_latex.ipynb
# - correlations_meta_latex.ipynb
# - test_correlations.ipynb
#

meta_models = ['sl', 'tl', 'ipsws', 'drs', 'dmls', 'xl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm']
standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp']
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + standalone_cate_models
for ds in datasets:
    base_dir_ds = os.path.join(base_metrics_dir, ds)
    plugin_dir_ds = os.path.join(plugin_dir, ds)
    matching_dir_ds = os.path.join(matching_dir, ds)
    rscore_dir_ds = os.path.join(rscore_dir, ds)

    mcomp.compare_metrics_meta_est(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_meta_est_sem_latex.csv'), index=False)
    mcomp.compare_metrics_all(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_all_sem_latex.csv'), index=False)

    mcomp.compare_correlations_all(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations_all.csv'), index=False)

    mcomp.compare_test_correlations_est(standalone_cate_models, meta_models, base_models, base_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_test_correlations_est.csv'), index=False)
    mcomp.compare_test_correlations_all(all_cate_models, base_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_test_correlations_all.csv'), index=False)


meta_models = ['sl', 'tl', 'ipsws', 'drs', 'dmls', 'xl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp']
standalone_cate_models = []
for ds in datasets:
    base_dir_ds = os.path.join(base_metrics_dir, ds)
    plugin_dir_ds = os.path.join(plugin_dir, ds)
    matching_dir_ds = os.path.join(matching_dir, ds)
    rscore_dir_ds = os.path.join(rscore_dir, ds)

    mcomp.compare_metrics_meta_base(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_meta_base_sem_latex.csv'), index=False)
    mcomp.compare_test_correlations_base(meta_models, base_models, base_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_test_correlations_base.csv'), index=False)

# END
# ========================================================


# ==================== 'MEAN' OUTPUTS ====================
#
# make sure RESULTS in 'utils.py' is set to 'mean'
# use with:
# - plot_estimators.ipynb
# - plot_hyperparams.ipynb
# - plot_metrics.ipynb
#

"""
for ds in datasets:
    base_dir_ds = os.path.join(base_metrics_dir, ds)
    plugin_dir_ds = os.path.join(plugin_dir, ds)
    matching_dir_ds = os.path.join(matching_dir, ds)
    rscore_dir_ds = os.path.join(rscore_dir, ds)

    mcomp.compare_metrics_all(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_all_raw.csv'), index=False)
"""

# END
# ========================================================




# Optional stuff (not included in the paper).

#comp.compare_metrics(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics.csv'), index=False)

#comp.compare_risks(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks.csv'), index=False)
#mcomp.compare_risks_meta_est(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks_meta_est.csv'), index=False)
#mcomp.compare_risks_meta_base(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks_meta_base.csv'), index=False)
#mcomp.compare_risks_all(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks_meta_base.csv'), index=False)

#comp.compare_correlations(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations.csv'), index=False)
#mcomp.compare_correlations_meta_est(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations_meta_est.csv'), index=False)
#mcomp.compare_correlations_meta_base(standalone_cate_models, meta_models, base_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations_meta_base.csv'), index=False)