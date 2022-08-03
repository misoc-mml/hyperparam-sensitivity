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

def rscore(cate_hat, y_res, t_res, base_score):
    '''
    Based on: https://github.com/microsoft/EconML/blob/main/econml/score/rscorer.py .
    '''
    if y_res.ndim == 1:
        y_res = y_res.reshape((-1, 1))
    if t_res.ndim == 1:
        t_res = t_res.reshape((-1, 1))
    
    effects = cate_hat.reshape((-1, y_res.shape[1], t_res.shape[1]))
    y_res_pred = np.einsum('ijk,ik->ij', effects, t_res).reshape(y_res.shape)
    
    return 1 - np.mean((y_res - y_res_pred) ** 2) / base_score