fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import logging
import argparse

from helpers.utils import init_logger
from helpers.data import get_scaler
from models.estimators import SSearch, TSearch, XSearch, DRSearch, DMLSearch, IPSWSearch, CausalForestSearch
from models.estimators import TSSearch, DRSSearch, DMLSSearch, IPSWSSearch
from models.data import IHDP, JOBS, TWINS, NEWS

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('--skip_iter', type=int, default=0)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_y', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--icv', dest='inner_cv', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=1)

    # Estimation
    # Models with 's' suffix refer to single-model/flat searches.
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'tls', 'xl', 'dr', 'drs', 'dml', 'dmls', 'ipsw', 'ipsws', 'cf'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm'], default='l1')

    return parser

def get_model(opt):
    if opt.estimation_model == 'sl':
        return SSearch(opt)
    elif opt.estimation_model == 'tl':
        return TSearch(opt)
    elif opt.estimation_model == 'tls':
        return TSSearch(opt)
    elif opt.estimation_model == 'xl':
        return XSearch(opt)
    elif opt.estimation_model == 'dr':
        return DRSearch(opt)
    elif opt.estimation_model == 'drs':
        return DRSSearch(opt)
    elif opt.estimation_model == 'dml':
        return DMLSearch(opt)
    elif opt.estimation_model == 'dmls':
        return DMLSSearch(opt)
    elif opt.estimation_model == 'ipsw':
        return IPSWSearch(opt)
    elif opt.estimation_model == 'ipsws':
        return IPSWSSearch(opt)
    elif opt.estimation_model == 'cf':
        return CausalForestSearch(opt)
    else:
        raise ValueError("Unrecognised 'get_model' key.")

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

def scale_xxy(X_train, X_test, y_train, opt, cont_vars):
    scaler_x = get_scaler(opt.scaler)
    # Scale only continuous features.
    X_train[:, cont_vars] = scaler_x.fit_transform(X_train[:, cont_vars])
    X_test[:, cont_vars] = scaler_x.transform(X_test[:, cont_vars])

    scaler_y = None
    if opt.scale_y:
        scaler_y = get_scaler(opt.scaler)
        y_train = scaler_y.fit_transform(y_train)
    
    return X_train, X_test, y_train, scaler_y

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
    skipped = 0
    dataset = get_dataset(options.dtype, options.data_path, n_iters)

    model = get_model(options)

    # Save the mapping of parameter combinations and IDs.
    model.save_params_info()

    # Data iterations
    for i in range(n_iters):
        if skipped < options.skip_iter:
            skipped += 1
            continue

        train, test = dataset._get_train_test(i)

        X_tr, t_tr, y_tr = dataset.get_xty(train)
        X_test, t_test, y_test = dataset.get_xty(test)

        # CV iterations
        for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
            train_idx = train_idx.astype(int)
            valid_idx = valid_idx.astype(int)
            
            X_tr_fold, t_tr_fold, y_tr_fold = X_tr[train_idx], t_tr[train_idx], y_tr[train_idx]
            X_val_fold, t_val_fold, y_val_fold = X_tr[valid_idx], t_tr[valid_idx], y_tr[valid_idx]

            # Scale train/val AFTER the split.
            X_tr_fold, X_val_fold, y_tr_fold, scaler_y = scale_xxy(X_tr_fold, X_val_fold, y_tr_fold, options, dataset.contfeats)

            # Fit on training set, predict on validation set.
            model.run((X_tr_fold, t_tr_fold, y_tr_fold), (X_val_fold, t_val_fold, y_val_fold), scaler_y, i+1, k+1)    

        # Scale train/test.
        X_tr_scaled, X_test_scaled, y_tr_scaled, scaler_y_test = scale_xxy(X_tr, X_test, y_tr, options, dataset.contfeats)

        # Fit on the *entire* training set, predict on test set.
        model.run((X_tr_scaled, t_tr, y_tr_scaled), (X_test_scaled, t_test, y_test), scaler_y_test, i+1, -1)