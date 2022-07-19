import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
n_threads = 8
os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ['TF_NUM_INTEROP_THREADS'] = str(n_threads)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(n_threads)

fixed_seed = 1
import random
import numpy as np
import tensorflow as tf
random.seed(fixed_seed)
np.random.seed(fixed_seed)
tf.random.set_seed(fixed_seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging
import argparse

from helpers.utils import init_logger
from helpers.data import get_scaler
from models.estimators_tf import SSearch, TSearch, TwoHeadSearch
from models.data import IHDP

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_y', action='store_true')
    parser.add_argument('--sr', dest='save_results', action='store_true')
    parser.add_argument('--sp', dest='save_preds', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # Estimation
    # Consider adding: XL, DR, DML, IPSW
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'two-head'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['mlp'], default='mlp')

    return parser

def get_model(opt):
    if opt.estimation_model == 'sl':
        return SSearch(opt)
    elif opt.estimation_model == 'tl':
        return TSearch(opt)
    elif opt.estimation_model == 'two-head':
        return TwoHeadSearch(opt)
    else:
        raise ValueError("Unrecognised 'get_model' key.")

def get_dataset(name, path, iters):
    result = None
    if name == 'ihdp':
        result = IHDP(path, iters)
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
    dataset = get_dataset('ihdp', options.data_path, n_iters)

    model = get_model(options)

    # Data iterations
    for i in range(n_iters):
        train, test = dataset._get_train_test(i)

        (X_tr, t_tr, y_tr), _ = train
        (X_test, t_test, y_test), _ = test

        # CV iterations
        for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
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

    # Save the mapping of parameter combinations and IDs.
    model.save_params_info()