ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm" "mlp")

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/predictions/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL
done