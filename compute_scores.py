fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import logging
import argparse
import pandas as pd

from helpers.utils import init_logger, get_model_name
from models.estimators import SEvaluator, TEvaluator, DREvaluator, DMLEvaluator, IPSWEvaluator, CausalForestEvaluator
from models.scorers import PluginScorer, RScorerEvaluator

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--scorer_path', type=str)
    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)

    # Estimation
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'dr', 'dml', 'ipsw', 'two-head', 'cf'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp'], default='l1')
    parser.add_argument('--st', type=str, dest='scorer_type', choices=['plugin', 'r_score'], default='plugin')
    parser.add_argument('--sn', type=str, dest='scorer_name')

    return parser

def get_scorer(opt):
    if opt.scorer_type == 'plugin':
        return PluginScorer(opt)
    elif opt.scorer_type == 'r_score':
        return RScorerEvaluator(opt)
    else:
        raise ValueError("Unrecognised 'get_scorer' key.")

def get_evaluator(opt):
    if opt.estimation_model in ('sl', 'two-head'):
        return SEvaluator(opt)
    elif opt.estimation_model == 'tl':
        return TEvaluator(opt)
    elif opt.estimation_model == 'dr':
        return DREvaluator(opt)
    elif opt.estimation_model == 'dml':
        return DMLEvaluator(opt)
    elif opt.estimation_model == 'ipsw':
        return IPSWEvaluator(opt)
    elif opt.estimation_model == 'cf':
        return CausalForestEvaluator(opt)
    else:
        raise ValueError("Unrecognised 'get_evaluator' key.")

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

    scorer = get_scorer(options)
    evaluator = get_evaluator(options)

    df_all = None

    # Data iterations
    for i in range(n_iters):
        # CV iterations
        for k, _ in enumerate(splits['train'][i]):
            if options.scorer_type == 'plugin':
                df_fold = evaluator.score_cate(i+1, k+1, scorer)
            elif options.scorer_type == 'r_score':
                df_fold = evaluator.rscore(i+1, k+1, scorer)
            else:
                raise ValueError('Unrecognised scorer type.')

            df_all = pd.concat([df_all, df_fold], ignore_index=True)

    model_name = get_model_name(options)
    df_all.to_csv(os.path.join(options.output_path, f'{model_name}_{options.scorer_type}_{options.scorer_name}.csv'), index=False)