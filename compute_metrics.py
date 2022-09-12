import os
import logging
import argparse
import numpy as np
import pandas as pd

from models.data import IHDP, JOBS, TWINS, NEWS
from models.estimators import SEvaluator, TEvaluator, XEvaluator, DREvaluator, DMLEvaluator, IPSWEvaluator, CausalForestEvaluator
from models.estimators import TSEvaluator, DRSEvaluator, DMLSEvaluator, IPSWSEvaluator
from helpers.utils import init_logger, get_model_name

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['ihdp', 'jobs', 'news', 'twins'])
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--scaler', type=str, choices=['minmax', 'std'], default='std')
    parser.add_argument('--scale_y', action='store_true')

    # Estimation
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'tls', 'xl', 'dr', 'drs', 'dml', 'dmls', 'ipsw', 'ipsws', 'two-head', 'cf'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp'], default='lr')

    return parser

def get_evaluator(opt):
    if opt.estimation_model in ('sl', 'two-head'):
        return SEvaluator(opt)
    elif opt.estimation_model == 'tl':
        return TEvaluator(opt)
    elif opt.estimation_model == 'tls':
        return TSEvaluator(opt)
    elif opt.estimation_model == 'xl':
        return XEvaluator(opt)
    elif opt.estimation_model == 'dr':
        return DREvaluator(opt)
    elif opt.estimation_model == 'drs':
        return DRSEvaluator(opt)
    elif opt.estimation_model == 'dml':
        return DMLEvaluator(opt)
    elif opt.estimation_model == 'dmls':
        return DMLSEvaluator(opt)
    elif opt.estimation_model == 'ipsw':
        return IPSWEvaluator(opt)
    elif opt.estimation_model == 'ipsws':
        return IPSWSEvaluator(opt)
    elif opt.estimation_model == 'cf':
        return CausalForestEvaluator(opt)
    else:
        raise ValueError("Unrecognised 'get_evaluator' key.")

def get_dataset(name, path, iters):
    result = None
    if name == 'ihdp':
        result = IHDP(path, iters)
    elif name == 'jobs':
        result = JOBS(path, iters)
    elif name == 'twins':
        result = TWINS(path, iters, static_splits=True)
    elif name == 'news':
        result = NEWS(path, iters, static_splits=True)
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
    dataset = get_dataset(options.dtype, options.data_path, n_iters)

    # iter id, fold id, param id, mse, ate, pehe
    # iter id, param id, mse, ate, pehe
    df_val = None
    df_test = None

    evaluator = get_evaluator(options)

    # Data iterations
    for i in range(n_iters):
        train, test = dataset._get_train_test(i)

        X_tr, t_tr, y_tr = dataset.get_xty(train)
        X_test, t_test, y_test = dataset.get_xty(test)
        eval_test = dataset.get_eval(test)

        # CV iterations
        for k, (train_idx, valid_idx) in enumerate(zip(splits['train'][i], splits['valid'][i])):
            train_idx = train_idx.astype(int)
            valid_idx = valid_idx.astype(int)

            y_tr_fold = y_tr[train_idx]
            t_val_fold = t_tr[valid_idx]
            y_val_fold = y_tr[valid_idx]
            eval_valid = dataset.get_eval_idx(train, valid_idx)

            # *** CV metrics ***
            df_fold = evaluator.run(i+1, k+1, y_tr_fold, t_val_fold, y_val_fold, eval_valid)
            df_val = pd.concat([df_val, df_fold], ignore_index=True)
            # ***

        # *** Test set metrics ***
        df_iter = evaluator.run(i+1, -1, y_tr, t_test, y_test, eval_test)
        df_test = pd.concat([df_test, df_iter], ignore_index=True)
        # ***

    model_name = get_model_name(options)
    df_val.to_csv(os.path.join(options.output_path, f'{model_name}_val_metrics.csv'), index=False)
    df_test.to_csv(os.path.join(options.output_path, f'{model_name}_test_metrics.csv'), index=False)
