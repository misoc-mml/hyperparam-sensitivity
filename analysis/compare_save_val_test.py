import os
import val_test_comparers as comp

out_dir = './tables'

base_metrics_dir = '../results/metrics/'
base_val_metrics_dir = '../results/metrics_val/'

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

meta_models = ['sl', 'tl', 'ipsws', 'drs', 'dmls', 'xl']
base_models = ['l1', 'l2', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm']
standalone_cate_models = ['cf', 'sl_mlp', 'tl_mlp']
all_cate_models = [f'{mm}_{bm}' for mm in meta_models for bm in base_models] + standalone_cate_models
for ds in datasets:
    print(ds)

    base_dir_ds = os.path.join(base_metrics_dir, ds)
    base_dir_val_ds = os.path.join(base_val_metrics_dir, ds)
    plugin_dir_ds = os.path.join(plugin_dir, ds)
    matching_dir_ds = os.path.join(matching_dir, ds)
    rscore_dir_ds = os.path.join(rscore_dir, ds)

    print('validation')
    comp.compare_metrics_val(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_val_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_iter_val.csv'), index=False)

    #print('test')
    #comp.compare_metrics_test(all_cate_models, plugin_models, matching_ks, rscore_base_models, base_dir_ds, plugin_dir_ds, matching_dir_ds, rscore_dir_ds, all_metrics[ds]).to_csv(os.path.join(out_dir, f'{ds}_compare_metrics_iter_test.csv'), index=False)