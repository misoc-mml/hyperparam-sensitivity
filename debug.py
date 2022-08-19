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
from models.estimators import SDebug
from models.data import IHDP

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('--skip_iter', type=int, default=0)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')

    # Estimation
    # Consider adding: XL, DR, DML, IPSW
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'cf'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm'], default='l1')

    return parser

def get_model(opt):
    if opt.estimation_model == 'sl':
        return SDebug(opt)
    else:
        raise ValueError("Unrecognised 'get_model' key.")

def get_dataset(name, path, iters):
    result = None
    if name == 'ihdp':
        result = IHDP(path, iters)
    else:
        raise ValueError('Unknown dataset type selected.')
    return result

def scale_xxy(X_train, X_test, opt, cont_vars):
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

    n_iters = options.iters
    skipped = 0
    dataset = get_dataset('ihdp', options.data_path, n_iters)

    model = get_model(options)

    df_test = None

    # Data iterations
    for i in range(n_iters):
        if skipped < options.skip_iter:
            skipped += 1
            continue

        train, test = dataset._get_train_test(i)

        (X_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
        (X_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
        cate_test = mu1_test - mu0_test

        # No CV iterations here.

        # Scale train/test.
        X_tr_scaled, X_test_scaled = scale_xxy(X_tr, X_test, options, dataset.contfeats)

        # Fit on the *entire* training set, predict on test set.
        df_iter = model.run((X_tr_scaled, t_tr, y_tr), (X_test_scaled, t_test, y_test), i+1, cate_test)

        df_test = pd.concat([df_test, df_iter], ignore_index=True)
    
    df_test.to_csv(os.path.join(options.output_path, 'test_metrics.csv'), index=False)

    print(df_test)