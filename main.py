fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import logging
import argparse

from helpers.utils import init_logger
from models.estimators import SSearch
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
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'xl', 'dr', 'ipsw', 'dml'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['lr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm'], default='lr')

    return parser

def get_model(opt):
    if opt.estimation_model == 'sl':
        return SSearch(opt)
    else:
        raise ValueError("Unrecognised 'get_model' key.")

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

    model = get_model(options)

    for i in range(n_iters):
        train, test = dataset._get_train_test(i)

        model.run(train, test, splits[i], i+1)
