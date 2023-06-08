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



# ==================== 'MEAN' OUTPUTS ====================
#
# make sure RESULTS in 'utils.py' is set to 'mean'
# use with:
# - plot_metrics.ipynb
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

    mcomp.compare_metrics_all(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_all_raw.csv'), index=False)


# END
# ========================================================
