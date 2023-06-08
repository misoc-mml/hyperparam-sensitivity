ITERS=10

# Standard estimators
MODELS=("sl" "tl" "xl" "dmls" "drs" "ipsws")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")

for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        echo ${MODEL}_${BASE_MODEL}

        MID_DIR="ihdp"
        python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="jobs"
        python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="twins"
        python ../compute_metrics.py --data_path ../datasets/TWINS/csv --dtype twins --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    
        MID_DIR="news"
        python ../compute_metrics.py --data_path ../datasets/NEWS --dtype news --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    done
done

# Neural Networks
MODELS=("sl" "tl")
BASE_MODELS=("mlp")

for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        echo ${MODEL}_${BASE_MODEL}

        MID_DIR="ihdp"
        python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="jobs"
        python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="twins"
        python ../compute_metrics.py --data_path ../datasets/TWINS/csv --dtype twins --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    
        MID_DIR="news"
        python ../compute_metrics.py --data_path ../datasets/NEWS --dtype news --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    done
done


# Causal Forest
MODEL="cf"
echo $MODEL

MID_DIR="ihdp"
python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="jobs"
python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="twins"
python ../compute_metrics.py --data_path ../datasets/TWINS/csv --dtype twins --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="news"
python ../compute_metrics.py --data_path ../datasets/NEWS --dtype news --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL