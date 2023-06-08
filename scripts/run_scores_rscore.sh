ITERS=10

# [plugin, r_score]
SCORER_TYPE="r_score"
# [sl, tl, rs]
SCORER_MODELS=("rs")
SCORER_BASE_MODELS=("dt" "kr" "lgbm")

# Standard estimators
MODELS=("sl" "tl" "xl" "dmls" "drs" "ipsws")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")
for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        for SCORER_MODEL in ${SCORER_MODELS[@]}
        do
            for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
            do
                echo ${MODEL}_${BASE_MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="ihdp"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="jobs"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="twins"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="news"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
            done
        done
    done
done

# Neural Networks
MODELS=("sl" "tl")
BASE_MODELS=("mlp")
for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        for SCORER_MODEL in ${SCORER_MODELS[@]}
        do
            for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
            do
                echo ${MODEL}_${BASE_MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="ihdp"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="jobs"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="twins"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

                MID_DIR="news"
                python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --bm $BASE_MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
            done
        done
    done
done

# Causal Forest
MODEL="cf"
for SCORER_MODEL in ${SCORER_MODELS[@]}
do
    for SCORER_BASE_MODEL in ${SCORER_BASE_MODELS[@]}
    do
        echo ${MODEL}_${SCORER_TYPE}_${SCORER_MODEL}_${SCORER_BASE_MODEL}

        MID_DIR="ihdp"
        python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

        MID_DIR="jobs"
        python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

        MID_DIR="twins"
        python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}

        MID_DIR="news"
        python ../compute_scores.py --results_path ../results/predictions/${MID_DIR}/${MODEL} --scorer_path ../results/scorers/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/scores/${MID_DIR}/${SCORER_MODEL}_${SCORER_BASE_MODEL} --em $MODEL --st $SCORER_TYPE --sn ${SCORER_MODEL}_${SCORER_BASE_MODEL}
    done
done