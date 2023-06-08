ITERS=10
# Options: sl, tl
#MODELS=("tl")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")
#BASE_MODELS=("mlp")
MID_DIR="news"

for MODEL in "$@"
#for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    #for BASE_MODEL in "$@"
    do
        echo ${MODEL}_${BASE_MODEL}
    
        #python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL

        #python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        #python ../compute_metrics.py --data_path ../datasets/TWINS/csv --dtype twins --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        python ../compute_metrics.py --data_path ../datasets/NEWS --dtype news --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    done
done