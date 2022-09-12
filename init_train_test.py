import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from models.data import TWINS, NEWS

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dtype', type=str, choices=['news', 'twins'])
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tr', dest='test_ratio', type=float)
    parser.add_argument('-o', type=str, dest='output_path', default='./')

    return parser

def get_dataset(name, path, iters):
    result = None
    if name == 'twins':
        result = TWINS(path, iters)
    elif name == 'news':
        result = NEWS(path, iters)
    else:
        raise ValueError('Unknown dataset type selected.')
    return result

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    dataset = get_dataset(options.dtype, options.data_path, options.n_iters)

    train_iters = []
    test_iters = []
    for data_i in range(options.n_iters):
        n_rows = dataset.get_rows_count(data_i)

        itr, ite = train_test_split(np.arange(n_rows), test_size=options.test_ratio, random_state=options.seed)        
        
        train_iters.append(itr)
        test_iters.append(ite)

    train_arr = np.array(train_iters, dtype=object)
    test_arr = np.array(test_iters, dtype=object)

    # Save to files
    # Structure:
    # (n_iters, train_size)
    # (n_iters, test_size)
    np.savez(os.path.join(options.output_path, f'{options.dtype}_splits_{options.n_iters}iters'), train=train_arr, test=test_arr)