import os
import argparse
import numpy as np

from models.estimators import SConverter, TConverter, CausalForestConverter
from models.scorers import PluginConverter, RScorerConverter

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sf', dest='splits_file', type=str)
    parser.add_argument('--iters', type=int, default=-1)
    parser.add_argument('--skip_iter', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--em', dest='estimation_model', type=str, choices=['sl', 'tl', 'cf'], default='sl')
    parser.add_argument('--bm', dest='base_model', type=str, choices=['l1', 'l2', 'tr', 'dt', 'rf', 'et', 'kr', 'cb', 'lgbm', 'mlp'], default='l1')
    parser.add_argument('--mt', dest='model_type', type=str, choices=['est', 'plugin', 'rscorer'], default='est')

    return parser

def get_model(opt):
    if opt.model_type == 'est':
        if opt.estimation_model == 'sl':
            return SConverter(opt)
        elif opt.estimation_model == 'tl':
            return TConverter(opt)
        elif opt.estimation_model == 'cf':
            return CausalForestConverter(opt)
        else:
            raise ValueError('Unknown estimation model selected.')
    elif opt.model_type == 'plugin':
        return PluginConverter(opt)
    elif opt.model_type == 'rscorer':
        return RScorerConverter(opt)
    else:
        raise ValueError('Unknown model type selected.')

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    # Check if output folder exists and create if necessary.
    if not os.path.isdir(options.output_path):
        os.mkdir(options.output_path)
    
    # (iters, folds, idx)
    splits = np.load(options.splits_file, allow_pickle=True)
    n_iters = options.iters if options.iters > 0 else splits.shape[0]

    model = get_model(options)

    skipped = 0
    # Data iterations
    for i in range(n_iters):
        if skipped < options.skip_iter:
            skipped += 1
            continue

        # CV iterations
        for k, _ in enumerate(splits['train'][i]):
            model.convert(i+1, k+1)
        
        model.convert(i+1, -1)