ITERS=1
DEBUG_DIR="debug_1"
META_MODEL="sl"
BASE_MODEL="l1"

# With Tensorflow
#python ../make_predictions_tf.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${DEBUG_DIR} --scale_y --em $META_MODEL --bm $BASE_MODEL

# No Tensorflow
#python ../make_predictions.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${DEBUG_DIR} --scale_y --em $META_MODEL --bm $BASE_MODEL

#python ../make_predictions.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${DEBUG_DIR} --em $META_MODEL --bm $BASE_MODEL

#python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${DEBUG_DIR} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${DEBUG_DIR} --scale_y --em $META_MODEL --bm $BASE_MODEL

#python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${DEBUG_DIR} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${DEBUG_DIR} --em $META_MODEL --bm $BASE_MODEL

#python ../make_plugin.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${DEBUG_DIR} --em $META_MODEL --bm $BASE_MODEL --cv 5

#python ../make_plugin.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${DEBUG_DIR} --em $META_MODEL --bm $BASE_MODEL --cv 5

#python ../compute_scores.py --results_path ../results/predictions/run1/sl_mlp --scorer_path ../results/scorers/debug_3 --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/debug/rs_lgbm --em sl --bm mlp --st r_score --sn rs_lgbm

#python ../compute_scores.py --results_path ../results/predictions/debug_1 --scorer_path ../results/scorers/debug_2 --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/debug_1 --em sl --bm l1 --st r_score --sn rs_l1

#python ../make_rscorer.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${DEBUG_DIR} --bm $BASE_MODEL --cv 5

#python ../make_rscorer.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${DEBUG_DIR} --bm $BASE_MODEL --cv 5

#python ../debug.py --data_path ../datasets/IHDP --iters 1 -o ../results/debug --em sl --bm lgbm