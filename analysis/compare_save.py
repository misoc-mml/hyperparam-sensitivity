import os

import comparers as comp
import meta_comparers as mcomp

out_dir = './tables'

meta_models = ['sl', 'tl', 'ipsws', 'dmls', 'xl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm']
#extra_models = [f'{mm}_{bm}' for mm in ['drs', 'dmls', 'xl'] for bm in ['l1', 'l2', 'dt']]
drs_models = [f'{mm}_{bm}' for mm in ['drs'] for bm in ['l1', 'l2', 'dt']]
#standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp']
#standalone_cate_models = ['cf', 'sl_mlp']
standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp'] + drs_models
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + drs_models + standalone_cate_models
base_metrics_dir = '../results/metrics/'

plugin_meta_models = ['sl', 'tl']
plugin_base_models = ['dt', 'lgbm', 'cb']
#plugin_base_models = ['dt', 'lgbm']
plugin_models = [f'{pmm}_{pbm}' for pmm in plugin_meta_models for pbm in plugin_base_models]
plugin_dir = '../results/scores/'

rscore_base_models = ['dt', 'lgbm', 'cb']
rscore_dir = '../results/scores/'

datasets = ['ihdp']
all_metrics = {'ihdp': ['ate', 'pehe'], 'jobs': ['att', 'policy'], 'news': ['ate', 'pehe'], 'twins': ['ate', 'pehe']}

for ds in datasets:
    base_dir_ds = os.path.join(base_metrics_dir, ds)
    plugin_dir_ds = os.path.join(plugin_dir, ds)
    rscore_dir_ds = os.path.join(rscore_dir, ds)

    #comp.compare_metrics(all_cate_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics.csv'), index=False)
    #comp.compare_risks(all_cate_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks.csv'), index=False)
    #comp.compare_correlations(all_cate_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations.csv'), index=False)

    mcomp.compare_metrics_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_meta.csv'), index=False)
    #mcomp.compare_risks_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_risks_meta.csv'), index=False)
    #mcomp.compare_correlations_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_dir_ds, plugin_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_correlations_meta.csv'), index=False)