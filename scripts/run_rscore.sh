ITERS=10
BASE_MODELS=("dt" "kr" "lgbm")

for MODEL in ${BASE_MODELS[@]}
do
    echo $MODEL

    MID_DIR="ihdp"
    python ../make_rscorer.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/rs_${MODEL} --bm $MODEL --cv 5 --n_jobs 10

    MID_DIR="jobs"
    python ../make_rscorer.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/rs_${MODEL} --bm $MODEL --cv 5 --n_jobs 10

    MID_DIR="twins"
    python ../make_rscorer.py --data_path ../datasets/TWINS/csv --dtype twins --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/rs_${MODEL} --bm $MODEL --cv 5 --n_jobs 10

    MID_DIR="news"
    python ../make_rscorer.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/rs_${MODEL} --bm $MODEL --cv 5 --n_jobs 10
done