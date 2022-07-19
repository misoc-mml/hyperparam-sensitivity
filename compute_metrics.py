import os
import logging
import argparse
import numpy as np
import pandas as pd

from models.data import IHDP
from models.estimators import SEvaluator, TEvaluator
from helpers.utils import init_logger

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_y', action='store_true')

    # Estimation
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp'], default='lr')

    return parser

def get_evaluator(opt):
    if opt.estimation_model == 'sl':
        return SEvaluator(opt)
    elif opt.estimation_model == 'tl':
        return TEvaluator(opt)
    else:
        raise ValueError("Unrecognised 'get_evaluator' key.")

def get_dataset(name, path, iters):
    result = None
    if name == 'ihdp':
        result = IHDP(path, iters)
    else:
        raise ValueError('Unknown dataset type selected.')
    return result

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

    # iter id, fold id, param id, mse, ate, pehe
    # iter id, param id, mse, ate, pehe
    df_val = None
    df_test = None

    evaluator = get_evaluator(options)

    # Data iterations
    for i in range(n_iters):
        train, test = dataset._get_train_test(i)

        (X_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
        (X_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
        cate_test = mu1_test - mu0_test

        # CV iterations
        for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
            y_tr_fold = y_tr[train_idx]
            t_val_fold = t_tr[valid_idx]
            y_val_fold = y_tr[valid_idx]
            cate_val_fold = mu1_tr[valid_idx] - mu0_tr[valid_idx]

            # *** CV metrics ***
            df_fold = evaluator.run(i+1, k+1, y_tr_fold, t_val_fold, y_val_fold, cate_val_fold)
            df_val = pd.concat([df_val, df_fold], ignore_index=True)
            # ***

        # *** Test set metrics ***
        df_iter = evaluator.run(i+1, -1, y_tr, t_test, y_test, cate_test)
        df_test = pd.concat([df_test, df_iter], ignore_index=True)
        # ***

    df_val.to_csv(os.path.join(options.output_path, f'{options.estimation_model}_{options.base_model}_val_metrics.csv'), index=False)

    df_test.to_csv(os.path.join(options.output_path, f'{options.estimation_model}_{options.base_model}_test_metrics.csv'), index=False)
