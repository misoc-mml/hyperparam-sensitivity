# Optional as all data splits can be downloaded along with datasets
# Uncomment to run anyway
#sh ./init_folds.sh
#sh ./init_train_test.sh

# Train estimators and predict CATEs
sh ./run_preds.sh
sh ./run_preds_tf.sh

# Create plugin, matching and rscorer metrics (learning part)
sh ./run_plugin.sh
sh ./run_matching.sh
sh ./run_rscore.sh

# Compute all metrics
sh ./run_metrics.sh
sh ./run_scores_plugin.sh
sh ./run_scores_matching.sh
sh ./run_scores_rscore.sh