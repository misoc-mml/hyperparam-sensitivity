ITERS=10
BASE_MODELS=("dt" "cb" "lgbm")
MID_DIR="run1"

for MODEL in ${BASE_MODELS[@]}
do
    echo $MODEL
    python ../make_rscorer.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/rs_${MODEL} --bm $MODEL --cv 5
done