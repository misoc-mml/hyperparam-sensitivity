ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    python ../make_predictions.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL
done