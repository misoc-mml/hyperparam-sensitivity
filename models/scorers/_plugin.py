import os
import numpy as np
import pandas as pd

from econml.metalearners import SLearner, TLearner
from models.estimators._common import get_params, get_regressor

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import Parallel

from helpers.fixes import delayed
from helpers.metrics import abs_ate, pehe
from helpers.utils import get_model_name

def _fit_and_predict(estimator, X, t, y, train, test):
    estimator.fit(y[train], t[train], X=X[train])
    return estimator.effect(X[test])

class PluginLearner():
    def __init__(self, opt):
        self.opt = opt

    def run(self, X, t, y):
        pass

class SPlugin(PluginLearner):
    def run(self, X, t, y):
        reg = get_regressor(self.opt.base_model, self.opt.seed, n_jobs=1)
        params = get_params(self.opt.base_model)
        grid = GridSearchCV(reg, params, n_jobs=self.opt.n_jobs, cv=self.opt.cv)
        grid.fit(np.concatenate([X, t.reshape(-1, 1)], axis=1), y.flatten())
        best_params = grid.best_params_

        cv = StratifiedKFold(self.opt.cv)
        splits = list(cv.split(X, t.flatten()))
        test_indices = np.concatenate([test for _, test in splits])

        reg_clean = get_regressor(self.opt.base_model, self.opt.seed, n_jobs=1)
        reg_clean.set_params(**best_params)

        parallel = Parallel(self.opt.n_jobs)
        predictions = parallel(
            delayed(_fit_and_predict)(
                SLearner(overall_model=reg_clean), X, t.flatten(), y.flatten(), train, test
            )
            for train, test in splits
        )

        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        predictions = np.concatenate(predictions)
        return predictions[inv_test_indices]

class TPlugin(PluginLearner):
    def run(self, X, t, y):
        m0_grid = GridSearchCV(get_regressor(self.opt.base_model, self.opt.seed, n_jobs=1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)
        m1_grid = GridSearchCV(get_regressor(self.opt.base_model, self.opt.seed, n_jobs=1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)
        tl_grid = TLearner(models=[m0_grid, m1_grid])
        tl_grid.fit(y.flatten(), t.flatten(), X=X)
        best_params = [m.best_params_ for m in tl_grid.models]

        mx_clean = []
        for params in best_params:
            m = get_regressor(self.opt.base_model, self.opt.seed, n_jobs=1)
            m.set_params(**params)
            mx_clean.append(m)
        
        cv = StratifiedKFold(self.opt.cv)
        splits = list(cv.split(X, t.flatten()))
        test_indices = np.concatenate([test for _, test in splits])

        parallel = Parallel(self.opt.n_jobs)
        predictions = parallel(
            delayed(_fit_and_predict)(
                TLearner(models=mx_clean), X, t.flatten(), y.flatten(), train, test
            )
            for train, test in splits
        )

        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        predictions = np.concatenate(predictions)
        return predictions[inv_test_indices]

class PluginScorer():
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

class PluginConverter():
    def __init__(self, opt):
        self.opt = opt
    
    def convert(self, iter_id, fold_id):
        df = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}.csv'))
        np.savez_compressed(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'), cate_hat=df['cate_hat'].to_numpy().reshape(-1, 1))