ITERS=10

MODELS=("sl" "tl" "xl" "dmls" "drs" "ipsws")
BASE_MODELS=("l1" "l2" "dt" "rf" "et" "kr" "cb" "lgbm")

# Standard estimators
for MODEL in ${MODELS[@]}
do
    for BASE_MODEL in ${BASE_MODELS[@]}
    do
        echo ${MODEL}_${BASE_MODEL}
    
        MID_DIR="ihdp"
        python ../make_predictions.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="jobs"
        python ../make_predictions.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="twins"
        python ../make_predictions.py --data_path ../datasets/TWINS/csv --dtype twins --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

        MID_DIR="news"
        python ../make_predictions.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
    done
done


# Causal Forest
MODEL="cf"
echo $MODEL

MID_DIR="ihdp"
python ../make_predictions.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="jobs"
python ../make_predictions.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="twins"
python ../make_predictions.py --data_path ../datasets/TWINS/csv --dtype twins --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL} --em $MODEL

MID_DIR="news"
python ../make_predictions.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL} --em $MODEL