ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("dt" "cb" "lgbm")
MID_DIR="run1"


for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}

    python ../make_plugin.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5
done


MODEL="tl"
for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}

    python ../make_plugin.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5
done