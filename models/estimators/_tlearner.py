import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from ._common import get_params, get_regressor
from helpers.data import get_scaler
from helpers.metrics import mse, pehe, abs_ate
from helpers.utils import get_params_df

class TSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model0 = get_regressor(opt.base_model)
        self.model1 = get_regressor(opt.base_model)
        self.params0 = get_params(opt.base_model)
        self.params1 = get_params(opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()

        X0_tr = X_tr[t_tr < 1]
        X1_tr = X_tr[t_tr > 0]
        y0_tr = y_tr[t_tr < 1]
        y1_tr = y_tr[t_tr > 0]

        X_test = test[0]
        t_test = test[1].flatten()
        X0_test = X_test[t_test < 1]
        X1_test = X_test[t_test > 0]

        # *** Model y0 ***
        y0_hat_cates = []
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            m0 = clone(self.model0)
            m0.set_params(**p0)

            m0.fit(X0_tr, y0_tr)

            # Factual predictions for model selection purposes (predict X[t==0]).
            y0_hat = m0.predict(X0_test)
            self._save_predictions(y0_hat, ['y_hat'], iter_id, fold_id, p0_id+1, 'm0')

            # For CATE prediction purposes (predict ALL X).
            y0_hat_cate = m0.predict(X_test)

            if self.opt.scale_y:
                y0_hat_cate = scaler.inverse_transform(y0_hat_cate)

            y0_hat_cates.append(y0_hat_cate)
        # ***

        # *** Model y1 ***
        y1_hat_cates = []
        for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
            m1 = clone(self.model1)
            m1.set_params(**p1)

            m1.fit(X1_tr, y1_tr)
            # Factual predictions for model selection purposes (predict X[t==1]).
            y1_hat = m1.predict(X1_test)
            self._save_predictions(y1_hat, ['y_hat'], iter_id, fold_id, p1_id+1, 'm1')

            # For CATE prediction purposes (predict ALL X).
            y1_hat_cate = m1.predict(X_test)

            if self.opt.scale_y:
                y1_hat_cate = scaler.inverse_transform(y1_hat_cate)

            y1_hat_cates.append(y1_hat_cate)
        # ***

        # *** CATE estimator ***
        p_global_id = 1
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
                cate_hat = y1_hat_cates[p1_id] - y0_hat_cates[p0_id]
                self._save_predictions(cate_hat, ['cate_hat'], iter_id, fold_id, p_global_id, 'cate')
                p_global_id += 1
        # ***

    def save_params_info(self):
        # Individual (id, params) pairs per model.
        df_p0 = get_params_df(self.params0)
        df_p1 = get_params_df(self.params1)

        # Keep the same order as in 'run()'.
        p_global_id = 1
        params_mapping = []
        for p0_id, _ in enumerate(ParameterGrid(self.params0)):
            for p1_id, _ in enumerate(ParameterGrid(self.params1)):
                params_mapping.append([p_global_id, p0_id+1, p1_id+1])
                p_global_id += 1
        df_all = pd.DataFrame(params_mapping, columns=['id', 'm0', 'm1'])

        df_p0.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m0_params.csv'), index=False)
        df_p1.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m1_params.csv'), index=False)
        df_all.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'), index=False)

    def _save_predictions(self, preds, cols, iter_id, fold_id, param_id, model):
        filename = f'{self.opt.estimation_model}_{self.opt.base_model}_{model}_iter{iter_id}'

        if fold_id > 0:
            filename += f'_fold{fold_id}'

        filename += f'_param{param_id}.csv'

        pd.DataFrame(preds, columns=cols).to_csv(os.path.join(self.opt.output_path, filename), index=False)

class TEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_m0_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m0_params.csv'))
        self.df_m1_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m1_params.csv'))
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))
    
    def score_cate(self, iter_id, fold_id, cate_test):
        results_cols = ['iter_id', 'fold_id', 'param_id' 'ate', 'pehe']
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}_fold{fold_id}'

        test_results = []
        for p_id in self.df_params['id']:
            preds_cate_filename = f'{preds_cate_filename_base}_param{p_id}.csv'
            df_cate = pd.read_csv(os.path.join(self.opt.results_path, preds_cate_filename))

            cate_hat = df_cate['cate_hat'].to_numpy()

            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            result = [iter_id, fold_id, p_id, test_ate, test_pehe]
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, cate_test):
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
        else:
            y_test_scaled = y_test
        
        # Split y_test into y0 and y1
        y0_test = y_test_scaled[t_test < 1]
        y1_test = y_test_scaled[t_test > 0]

        results_cols = ['iter_id', 'param_id', 'mse_m0', 'mse_m1', 'ate', 'pehe', 'ate_hat']
        preds_m0_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_m0_iter{iter_id}'
        preds_m1_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_m1_iter{iter_id}'
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}'

        if fold_id > 0:
            preds_m0_filename_base += f'_fold{fold_id}'
            preds_m1_filename_base += f'_fold{fold_id}'
            preds_cate_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')

        m0_mse = {}
        for p0_id in self.df_m0_params['id']:
            preds_m0_filename = f'{preds_m0_filename_base}_param{p0_id}.csv'
            df_m0 = pd.read_csv(os.path.join(self.opt.results_path, preds_m0_filename))
            m0_mse[p0_id] = mse(df_m0['y_hat'].to_numpy(), y0_test)
        
        m1_mse = {}
        for p1_id in self.df_m1_params['id']:
            preds_m1_filename = f'{preds_m1_filename_base}_param{p1_id}.csv'
            df_m1 = pd.read_csv(os.path.join(self.opt.results_path, preds_m1_filename))
            m1_mse[p1_id] = mse(df_m1['y_hat'].to_numpy(), y1_test)

        test_results = []
        for p_id, p0_id, p1_id in zip(self.df_params['id'], self.df_params['m0'], self.df_params['m1']):
            preds_cate_filename = f'{preds_cate_filename_base}_param{p_id}.csv'
            df_cate = pd.read_csv(os.path.join(self.opt.results_path, preds_cate_filename))

            cate_hat = df_cate['cate_hat'].to_numpy()
            ate_hat = np.mean(cate_hat)

            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            result = [iter_id, p_id, m0_mse[p0_id], m1_mse[p1_id], test_ate, test_pehe, ate_hat]
            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)