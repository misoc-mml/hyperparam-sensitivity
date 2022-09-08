fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import logging
import argparse
import pandas as pd

from helpers.utils import init_logger
from helpers.data import get_scaler
from models.data import IHDP, JOBS, TWINS, NEWS
from models.scorers import RScorerWrapper

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=-1)

    # Estimation
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm'], default='l1')

    return parser

def get_dataset(name, path, iters):
    result = None
    if name == 'ihdp':
        result = IHDP(path, iters)
    elif name == 'jobs':
        result = JOBS(path, iters)
    elif name == 'twins':
        result = TWINS(path, iters)
    elif name == 'news':
        result = NEWS(path, iters)
    else:
        raise ValueError('Unknown dataset type selected.')
    return result

def scale_x(X_train, X_test, opt, cont_vars):
    scaler_x = get_scaler(opt.scaler)
    
    # Scale only continuous features.
    X_train[:, cont_vars] = scaler_x.fit_transform(X_train[:, cont_vars])
    X_test[:, cont_vars] = scaler_x.transform(X_test[:, cont_vars])
    
    return X_train, X_test

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    # Check if output folder exists and create if necessary.
    if not os.path.isdir(options.output_path):
        os.mkdir(options.output_path)

    # Initialise the logger (writes simultaneously to a file and the console).
    init_logger(options)
    logging.debug(options)

    # (iters, folds, idx)
    splits = np.load(options.splits_file, allow_pickle=True)
    n_iters = options.iters if options.iters > 0 else splits.shape[0]
    dataset = get_dataset(options.dtype, options.data_path, n_iters)

    scorer = RScorerWrapper(options)

    base_scores = []
    base_scores_cols = ['iter_id', 'fold_id', 'base_score']

    # Data iterations
    for i in range(n_iters):
        train, _ = dataset._get_train_test(i)

        X_tr, t_tr, y_tr = dataset.get_xty(train)

        # CV iterations
        for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
            train_idx = train_idx.astype(int)
            valid_idx = valid_idx.astype(int)

            X_tr_fold = X_tr[train_idx]
            X_val_fold, t_val_fold, y_val_fold = X_tr[valid_idx], t_tr[valid_idx], y_tr[valid_idx]

            # Scale train/val AFTER the split.
            X_tr_fold, X_val_fold = scale_x(X_tr_fold, X_val_fold, options, dataset.contfeats)

            (Y_res, T_res), base_score = scorer.run(X_val_fold, t_val_fold, y_val_fold)

            np.savez_compressed(os.path.join(options.output_path, f'rs_{options.base_model}_iter{i+1}_fold{k+1}'), y_res=Y_res, t_res=T_res)
            base_scores.append([i+1, k+1, base_score])
    
    pd.DataFrame(base_scores, columns=base_scores_cols).to_csv(os.path.join(options.output_path, f'rs_{options.base_model}_base_scores.csv'), index=False)