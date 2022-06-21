# Data related helper functions.

import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def xt_from_x(x):
    xt0 = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    xt1 = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xt0, xt1

def get_scaler(name):
    result = None
    if name == 'minmax':
        result = MinMaxScaler(feature_range=(-1, 1))
    elif name == 'std':
        result = StandardScaler()
    else:
        raise ValueError('Unknown scaler type.')
    return result