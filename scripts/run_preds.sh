ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")
MID_DIR="run1"

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    python ../make_predictions.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL
done

# No scaling of the target Y.
MODEL="cf"
echo $MODEL
python ../make_predictions.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL} --em $MODEL