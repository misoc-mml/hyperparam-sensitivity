ITERS=10
# Options: [sl, tl, two-head]
MODEL="sl"
BASE_MODEL="mlp"

echo ${MODEL}_${BASE_MODEL}
python ../make_predictions_tf.py --data_path ../datasets/IHDP --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/predictions/${MODEL}_${BASE_MODEL} --scale_y --em $MODEL --bm $BASE_MODEL