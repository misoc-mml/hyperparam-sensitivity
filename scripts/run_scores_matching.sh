ITERS=10

MODELS=("sl")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")
#BASE_MODELS=("mlp")

SCORER_TYPE="matching"
N_VALS=(1 3 5)

#for MODEL in ${MODELS[@]}
for MODEL in "$@"
do
    #for BASE_MODEL in "$@"
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        for N_VAL in ${N_VALS[@]}
        do
            echo ${MODEL}_${BASE_MODEL}_${SCORER_TYPE}_match_${N_VAL}k

            MID_DIR="ihdp"
            python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/match_${N_VAL}k --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/match_${N_VAL}k --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn match_${N_VAL}k

            MID_DIR="jobs"
            python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/match_${N_VAL}k --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/match_${N_VAL}k --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn match_${N_VAL}k
        done        
    done
done