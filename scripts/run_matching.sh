ITERS=10
N_VALS=(1 3 5)

for N_VAL in ${N_VALS[@]}
do
    echo "knn = ${N_VAL}"

    MID_DIR="ihdp"
    python ../make_matching.py --data_path ../datasets/IHDP --dtype ihdp --sf ../datasets/IHDP/ihdp_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/match_${N_VAL}k --knn $N_VAL --n_jobs 10

    MID_DIR="jobs"
    python ../make_matching.py --data_path ../datasets/JOBS --dtype jobs --sf ../datasets/JOBS/jobs_splits_10iters_10folds.npz --iters $ITERS -o ../results/scorers/${MID_DIR}/match_${N_VAL}k --knn $N_VAL --n_jobs 10
done