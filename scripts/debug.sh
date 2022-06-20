python ../main.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters 1 -o ../results/debug --scale_y --em sl --bm dt

python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/debug --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters 1 -o ../analysis/debug --scale_y --em sl --bm dt