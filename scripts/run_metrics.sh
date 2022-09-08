ITERS=10
# Options: sl, tl
MODEL="sl"
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")
MID_DIR="jobs"

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    #python ../compute_metrics.py --data_path ../datasets/IHDP --dtype ihdp --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL

    python ../compute_metrics.py --data_path ../datasets/JOBS --dtype jobs --results_path ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
done

#MODEL="cf"
#echo $MODEL
#python ../compute_metrics.py --data_path ../datasets/IHDP --results_path ../results/predictions/${MID_DIR}/${MODEL} --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/metrics/${MID_DIR}/${MODEL} --em $MODEL