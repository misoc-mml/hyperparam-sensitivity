import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from econml.metalearners import XLearner

from ._common import get_params, get_regressor, get_classifier
from helpers.utils import get_params_df

class XSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.m_prop = get_classifier(opt.base_model, n_jobs=self.opt.n_jobs)
        self.params_base = get_params(opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        X_test = test[0]

        if fold_id > 0:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        cate_hats = []
        for params in ParameterGrid(self.params_base):
            model_reg = clone(self.m_reg)
            model_reg.set_params(**params)

            model_prop = clone(self.m_prop)
            model_prop.set_params(**params)

            # Pass the same regression model.
            # Internally cloned into 4 new models - T/C first stage + T/C second stage.
            # ALL 5 models (reg + clf) use the same parameter set for simplicity.
            xl = XLearner(models=model_reg, propensity_model=model_prop)
            xl.fit(y_tr, t_tr, X=X_tr)

            cate_hat = xl.effect(X_test)
            cate_hats.append(cate_hat)
        
        cate_hats_arr = np.array(cate_hats, dtype=object)
        np.savez_compressed(os.path.join(self.opt.output_path, base_filename), cate_hat=cate_hats_arr)
    
    def save_params_info(self):
        df_params = get_params_df(self.params_base)
        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class XEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'))

    def _run_valid(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'fold_id', 'param_id'] + eval.metrics
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)
        cate_hats = preds['cate_hat'].astype(float)

        test_results = []
        for p_id in self.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, fold_id, p_id] + test_metrics
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def _run_test(self, iter_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'param_id', 'ate_hat'] + eval.metrics
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)
        cate_hats = preds['cate_hat'].astype(float)

        test_results = []
        for p_id in self.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            ate_hat = np.mean(cate_hat)
            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id, ate_hat] + test_metrics
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if fold_id > 0:
            return self._run_valid(iter_id, fold_id, y_tr, t_test, y_test, eval)
        else:
            return self._run_test(iter_id, y_tr, t_test, y_test, eval)