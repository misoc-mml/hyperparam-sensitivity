import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

from ._common import get_params, get_regressor
from helpers.data import xt_from_x, get_scaler
from helpers.metrics import mse, pehe, abs_ate
from helpers.utils import get_params_df, get_model_name

class SSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model = get_regressor(self.opt.base_model)
        self.params_grid = get_params(self.opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        for param_id, params in enumerate(ParameterGrid(self.params_grid)):
            model1 = clone(self.model)
            model1.set_params(**params)

            model1.fit(Xt_tr, y_tr)

            y_hat = model1.predict(Xt_test)

            y0_hat = model1.predict(xt0)
            y1_hat = model1.predict(xt1)

            if self.opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            
            cate_hat = y1_hat - y0_hat

            cols = ['y_hat', 'y0_hat', 'y1_hat', 'cate_hat']
            results = np.concatenate([y_hat.reshape(-1, 1), y0_hat.reshape(-1, 1), y1_hat.reshape(-1, 1), cate_hat.reshape(-1, 1)], axis=1)

            if fold_id > 0:
                filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}_param{param_id+1}.csv'
            else:
                filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_param{param_id+1}.csv'

            pd.DataFrame(results, columns=cols).to_csv(os.path.join(self.opt.output_path, filename), index=False)

    def save_params_info(self):
        df_params = get_params_df(self.params_grid)

        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class SEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.model_name = get_model_name(opt)
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.model_name}_params.csv'))
    
    def score_cate(self, iter_id, fold_id, cate_test):
        results_cols = ['iter_id', 'fold_id', 'param_id', 'ate', 'pehe']
        preds_filename_base = f'{self.model_name}_iter{iter_id}_fold{fold_id}'

        test_results = []
        for p_id in self.df_params['id']:
            preds_filename = f'{preds_filename_base}_param{p_id}.csv'
            df_preds = pd.read_csv(os.path.join(self.opt.results_path, preds_filename))

            cate_hat = df_preds['cate_hat'].to_numpy()

            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            result = [iter_id, fold_id, p_id, test_ate, test_pehe]
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, cate_test):
        results_cols = ['iter_id', 'param_id', 'mse', 'ate', 'pehe', 'ate_hat', 'r2_score']
        
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
            results_cols.insert(3, 'mse_inv')
        else:
            y_test_scaled = y_test

        preds_filename_base = f'{self.model_name}_iter{iter_id}'
        if fold_id > 0:
            preds_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')

        test_results = []
        for p_id in self.df_params['id']:
            preds_filename = f'{preds_filename_base}_param{p_id}.csv'
            df_preds = pd.read_csv(os.path.join(self.opt.results_path, preds_filename))

            cate_hat = df_preds['cate_hat'].to_numpy()
            ate_hat = np.mean(cate_hat)

            # Scaled MSE (y_hat is scaled).
            y_hat = df_preds['y_hat'].to_numpy()
            test_mse = mse(y_hat, y_test_scaled)

            r2 = r2_score(y_test_scaled, y_hat)

            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            result = [iter_id, p_id, test_mse, test_ate, test_pehe, ate_hat, r2]

            if self.opt.scale_y:
                y_hat_inv = scaler.inverse_transform(y_hat)
                # Unscaled MSE.
                test_mse_inv = mse(y_hat_inv, y_test)
                result.insert(3, test_mse_inv)

            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)