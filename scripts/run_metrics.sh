ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm" "mlp")
MID_DIR="run1"

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL
done

MODEL="cf"
echo $MODEL
python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL