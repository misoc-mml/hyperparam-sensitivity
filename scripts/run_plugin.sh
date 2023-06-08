ITERS=10
# Options: sl, tl
MODELS=("sl" "tl")
BASE_MODELS=("dt" "kr" "lgbm")


for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        echo ${MODEL}_${BASE_MODEL}

        MID_DIR="ihdp"
        python ../make_plugin.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5 --n_jobs 10

        MID_DIR="jobs"
        python ../make_plugin.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5 --n_jobs 10

        MID_DIR="twins"
        python ../make_plugin.py --data_path ../datasets/TWINS/csv --dtype twins --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5 --n_jobs 10

        MID_DIR="news"
        python ../make_plugin.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --cv 5 --n_jobs 10
    done
done