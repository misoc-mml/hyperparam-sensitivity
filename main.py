fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import time
import logging
import argparse
import pandas as pd

from helpers.utils import init_logger, get_dataset

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

    # load main search object
    # pass data, splits, iters and options
    # run the search
    # save the results (metrics, predictions)