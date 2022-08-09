ITERS=10

MODEL="cf"

# [plugin, r_score]
SCORER_TYPE="plugin"
# [sl, tl, rs]
SCORER_MODELS=("sl" "tl")
SCORER_BASE_MODELS=("dt" "lgbm" "cb")

for SCORER_MODEL in ${SCORER_MODELS[@]}
do
    for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
    do
        echo ${MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

        python ../compute_scores.py --results_path ../results/predictions/run1/${MODEL} --scorer_path ../results/scorers/run1/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/run1/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
    done
done