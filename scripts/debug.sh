ITERS=1
DEBUG_DIR="debug_2"
META_MODEL="tl"
BASE_MODEL="dt"

python ../make_predictions.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${DEBUG_DIR} --scale_y --em $META_MODEL --bm $BASE_MODEL

python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/predictions/${DEBUG_DIR} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${DEBUG_DIR} --scale_y --em $META_MODEL --bm $BASE_MODEL