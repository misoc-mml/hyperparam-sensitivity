MODEL="sl"
BASE_MODELS=("l1" "l2" "kr" "dt" "rf" "et" "lgbm" "cb")

for BASE_MODEL in ${BASE_MODELS[@]}
do
    echo ${MODEL}_${BASE_MODEL}
    
    #python ../convert_preds.py --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters 10 --results_path ../results/predictions/run1/${MODEL}_${BASE_MODEL} -o ../results/predictions/ihdp/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --mt est

    python ../convert_preds.py --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters 10 --results_path ../results/predictions/jobs_old/${MODEL}_${BASE_MODEL} -o ../results/predictions/jobs/${MODEL}_${BASE_MODEL} --em $MODEL --bm $BASE_MODEL --mt est
done


#python ../convert_preds.py --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters 10 --results_path ../results/predictions/run1/${MODEL} -o ../results/predictions/ihdp/${MODEL} --em $MODEL --mt est