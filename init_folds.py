# Initialise CV folds and save them to file(s) for reproducibility.

import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.data import IHDP

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('-o', type=str, dest='output_path', default='./')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    # Only IHDP atm.
    dataset = IHDP(options.data_path, options.n_iters)

    train_iters = []
    valid_iters = []
    for data_i in range(options.n_iters):
        (X, t, _), _ = dataset.get_train(data_i)

        kf_obj = StratifiedKFold(options.n_folds)

        train_folds = []
        valid_folds = []
        for train_idx, valid_idx in kf_obj.split(X, t):
            train_folds.append(train_idx)
            valid_folds.append(valid_idx)
        
        train_iters.append(np.array(train_folds, dtype=object))
        valid_iters.append(np.array(valid_folds, dtype=object))

    train_arr = np.array(train_iters, dtype=object)
    valid_arr = np.array(valid_iters, dtype=object)

    # Save iters to files
    # Structure:
    # (n_iters, n_folds, train_fold_size)
    # (n_iters, n_folds, valid_fold_size)
    np.savez(os.path.join(options.output_path, f'ihdp_splits_{options.n_iters}iters_{options.n_folds}folds'), train=train_arr, valid=valid_arr)