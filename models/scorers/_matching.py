import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

from helpers.metrics import abs_ate, pehe
from helpers.utils import get_model_name

class MatchingLearner():
    def __init__(self, opt):
        self.opt = opt
    
    def run(self, X, t, y):
        t = t.flatten()
        y = y.flatten()

        X0 = X[t < 1]
        X1 = X[t > 0]
        y0 = y[t < 1]
        y1 = y[t > 0]

        weights_mode = 'distance' if self.opt.knn > 1 else 'uniform'
        m0 = KNeighborsRegressor(n_neighbors=self.opt.knn, weights=weights_mode, p=2, n_jobs=self.opt.n_jobs)
        m1 = KNeighborsRegressor(n_neighbors=self.opt.knn, weights=weights_mode, p=2, n_jobs=self.opt.n_jobs)

        # Store control and treated units as separate models.
        m0.fit(X0, y0)
        m1.fit(X1, y1)

        # Find matching units.
        y0_hat = m0.predict(X1)
        y1_hat = m1.predict(X0)

        # Get imputed CATEs.
        cate_t = y1 - y0_hat
        cate_c = y1_hat - y0

        # Go back to the original ordering.
        cate_hat = np.zeros_like(y)
        cate_hat[t < 1] = cate_c
        cate_hat[t > 0] = cate_t

        return cate_hat

class MatchingEvaluator():
    def __init__(self, opt):
        self.opt = opt
    
    def get_cate(self, iter, fold):
        arr = np.load(os.path.join(self.opt.scorer_path, f'{self.opt.scorer_name}_iter{iter}_fold{fold}.npz'), allow_pickle=True)
        return arr['cate_hat'].reshape(-1, 1).astype(float)

    def score(self, est, iter_id, fold_id):
        filename = f'{get_model_name(self.opt)}_iter{iter_id}_fold{fold_id}.npz'

        preds = np.load(os.path.join(self.opt.results_path, filename), allow_pickle=True)
        cate_test = self.get_cate(iter_id, fold_id)

        cate_hats = preds['cate_hat'].astype(float)

        test_results = []
        for p_id in est.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)

            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            result = [iter_id, fold_id, p_id, test_ate, test_pehe]
            test_results.append(result)
        
        results_cols = ['iter_id', 'fold_id', 'param_id', 'ate', 'pehe']
        return pd.DataFrame(test_results, columns=results_cols)