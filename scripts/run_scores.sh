ITERS=10

MODELS=("sl" "tl")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")

SCORER_TYPE="plugin"
SCORER_MODELS=("sl" "tl")
SCORER_BASE_MODELS=("dt" "lgbm" "cb")

for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        for SCORER_MODEL in ${SCORER_MODELS[@]}
        do
            for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
            do
                echo ${MODEL}_${BASE_MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

                python ../compute_scores.py --results_path ../results/predictions/run1/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/run1/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/run1/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
            done
        done
    done
done