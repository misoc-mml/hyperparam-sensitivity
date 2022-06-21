import os
import logging
import argparse
import numpy as np
import pandas as pd

from models.data import IHDP
from helpers.utils import init_logger
from helpers.metrics import get_metrics

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
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'xl', 'dr', 'ipsw', 'dml'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['lr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm'], default='lr')

    return parser

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
    df_params = pd.read_csv(os.path.join(options.results_path, f'{options.estimation_model}_{options.base_model}_params.csv'))

    # iter id, fold id, param id, mse, ate, pehe
    # iter id, param id, mse, ate, pehe
    val_results = []
    test_results = []

    # Data iterations
    for i in range(n_iters):
        train, test = dataset._get_train_test(i)

        (X_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train

        # Hyperparam iterations
        for p in df_params['id']:

            # CV iterations
            for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
                y_tr_fold = y_tr[train_idx]
                X_val_fold, t_val_fold, y_val_fold = X_tr[valid_idx], t_tr[valid_idx], y_tr[valid_idx]
                y_cf_val_fold, mu0_val_fold, mu1_val_fold = y_cf_tr[valid_idx], mu0_tr[valid_idx], mu1_tr[valid_idx]
            
                # *** CV metrics ***
                df_val_results = pd.read_csv(os.path.join(options.results_path, f'{options.estimation_model}_{options.base_model}_iter{i+1}_fold{k+1}_param{p}.csv'))

                val_fold = (X_val_fold, t_val_fold, y_val_fold), (y_cf_val_fold, mu0_val_fold, mu1_val_fold)
                mse_val, ate_val, pehe_val = get_metrics(df_val_results, y_tr_fold, val_fold, options)

                val_results.append([i+1, k+1, p, mse_val, ate_val, pehe_val])
                # ***

            # *** Test set metrics ***
            df_test_results = pd.read_csv(os.path.join(options.results_path, f'{options.estimation_model}_{options.base_model}_iter{i+1}_param{p}.csv'))

            mse_test, ate_test, pehe_test = get_metrics(df_test_results, y_tr, test, options)

            test_results.append([i+1, p, mse_test, ate_test, pehe_test])
            # ***

    pd.DataFrame(val_results, columns=['iter_id', 'fold_id', 'param_id', 'mse', 'ate', 'pehe']).to_csv(os.path.join(options.output_path, f'{options.estimation_model}_{options.base_model}_val_metrics.csv'), index=False)

    pd.DataFrame(test_results, columns=['iter_id', 'param_id', 'mse', 'ate', 'pehe']).to_csv(os.path.join(options.output_path, f'{options.estimation_model}_{options.base_model}_test_metrics.csv'), index=False)
