import os

import comparers as comp
import meta_comparers as mcomp

out_dir = './tables'

meta_models = ['sl', 'tl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm']
standalone_cate_models = ['cf']
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + standalone_cate_models
base_metrics_dir = '../results/metrics/run2/'

plugin_meta_models = ['sl', 'tl']
plugin_base_models = ['dt', 'lgbm', 'cb']
plugin_models = [f'{pmm}_{pbm}' for pmm in plugin_meta_models for pbm in plugin_base_models]
plugin_dir = '../results/scores/run1/'

rscore_base_models = ['dt', 'lgbm', 'cb']
rscore_dir = '../results/scores/run1/'


comp.compare_metrics(all_cate_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_metrics.csv'), index=False)
comp.compare_risks(all_cate_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_risks.csv'), index=False)
comp.compare_correlations(all_cate_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_correlations.csv'), index=False)

mcomp.compare_metrics_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_metrics_meta.csv'), index=False)
mcomp.compare_risks_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_risks_meta.csv'), index=False)
mcomp.compare_correlations_meta(standalone_cate_models, meta_models, base_models, plugin_models, rscore_base_models, base_metrics_dir, plugin_dir, rscore_dir).to_csv(os.path.join(out_dir, 'compare_correlations_meta.csv'), index=False)