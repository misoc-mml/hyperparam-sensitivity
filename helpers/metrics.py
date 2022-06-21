# Metrics related helper functions.

import numpy as np

from models.data import Evaluator
from utils import get_scaler

def mse(a, b):
    return np.mean((a - b)**2)

def get_metrics(df_preds, y_tr, test, opt):
    # Assumes df_preds includes at least y_hat and cate_hat columns.
    # y_hat - scaled
    # cate_hat - unscaled

    (X_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
    eval_test = Evaluator(y_test, t_test, y_cf=y_cf_test, mu0=mu0_test, mu1=mu1_test)

    if opt.scale_y:
        # Replicate the scaler
        scaler = get_scaler(opt.scaler)
        scaler.fit(y_tr)
        y_test_scaled = scaler.transform(y_test)
    else:
        y_test_scaled = y_test

    # y_hat is scaled
    test_mse = mse(df_preds['y_hat'].to_numpy(), y_test_scaled)

    # cate_hat is unscaled
    _, ate, pehe = eval_test.calc_stats_effect(df_preds['cate_hat'].to_numpy())

    return test_mse, ate, pehe