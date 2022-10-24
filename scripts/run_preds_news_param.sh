MODEL=$1
BASE_MODEL=$2

ITERS=$3
skip=$((ITERS-1))

MID_DIR="news"


echo ${MODEL}_${BASE_MODEL}, ${ITERS}, $skip

python ../make_predictions.py --data_path ../datasets/NEWS --dtype news --sf ../datasets/NEWS/news_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MID_DIR}/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --skip_iter $skip
