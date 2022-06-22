# Metrics related helper functions.

import numpy as np

def mse(a, b):
    return np.mean((a - b)**2)

def rmse(a, b):
    return np.sqrt(mse(a, b))

def pehe(true_cate, cate_hat):
    return rmse(true_cate, cate_hat)

def abs_ate(true_cate, cate_hat):
    return np.abs(np.mean(cate_hat) - np.mean(true_cate))