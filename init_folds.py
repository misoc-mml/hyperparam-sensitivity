# Initialise CV folds and save them to file(s) for reproducibility.

import argparse
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

from .models.data import IHDP

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--n_folds', type=int)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

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
        
        train_iters.append(train_folds)
        valid_iters.append(valid_folds)
        
    # save iters to files