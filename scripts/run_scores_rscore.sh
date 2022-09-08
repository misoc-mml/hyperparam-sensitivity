ITERS=10

MODELS=("sl")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")

# [plugin, r_score]
SCORER_TYPE="r_score"
# [sl, tl, rs]
SCORER_MODELS=("rs")
SCORER_BASE_MODELS=("dt" "lgbm" "cb")

MID_DIR="jobs"

for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        for SCORER_MODEL in ${SCORER_MODELS[@]}
        do
            for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
            do
                echo ${MODEL}_${BASE_MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

                #python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
            done
        done
    done
done