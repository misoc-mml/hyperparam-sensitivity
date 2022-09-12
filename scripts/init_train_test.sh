# Keep train/test ratio consistent with the literature.

python ../init_train_test.py --data_path ../datasets/NEWS --dtype news --n_iters 10 --tr 0.1 -o ../datasets/NEWS

python ../init_train_test.py --data_path ../datasets/TWINS/csv --dtype twins --n_iters 10 --tr 0.2 -o ../datasets/TWINS/csv