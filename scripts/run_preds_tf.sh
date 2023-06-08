ITERS=10

# Neural Networks
MODELS=("sl" "tl")
BASE_MODEL="mlp"

for MODEL in ${MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}

    MID_DIR="ihdp"
    python ../make_predictions_tf.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

    MID_DIR="jobs"
    python ../make_predictions_tf.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

    MID_DIR="twins"
    python ../make_predictions_tf.py --data_path ../datasets/TWINS/csv --dtype twins --sf ../datasets/TWINS/csv/twins_splits_10iters_10folds.npz --iters $iter -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL

    MID_DIR="news"
    python ../make_predictions_tf.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $iter -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL
done
