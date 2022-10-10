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
        self.model = get_regressor(self.opt.base_model, n_jobs=self.opt.n_jobs)
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

        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params_grid):
            model1 = clone(self.model)
            model1.set_params(**params)

            model1.fit(Xt_tr, y_tr)

            y_hat = model1.predict(Xt_test)

            y0_hat = model1.predict(xt0).reshape(-1, 1)
            y1_hat = model1.predict(xt1).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            
            cate_hat = y1_hat - y0_hat

            y_hats.append(y_hat)
            y0_hats.append(y0_hat)
            y1_hats.append(y1_hat)
            cate_hats.append(cate_hat)
        
        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
            
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_grid)

        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class SEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.model_name = get_model_name(opt)
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.model_name}_params.csv'))

    def _run_valid(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
        else:
            y_test_scaled = y_test

        if y_test_scaled.ndim == 1:
            y_test_scaled = y_test_scaled.reshape(-1, 1)

        preds_filename_base = f'{self.model_name}_iter{iter_id}_fold{fold_id}'
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)

        results_cols = ['iter_id', 'fold_id', 'param_id', 'mse', 'r2_score'] + eval.metrics
        
        y_hats = preds['y_hat'].astype(float)
        cate_hats = preds['cate_hat'].astype(float)

        test_results = []
        for p_id in self.df_params['id']:
            # Scaled MSE (y_hat is scaled).
            y_hat = y_hats[p_id-1].reshape(-1, 1)
            test_mse = mse(y_hat, y_test_scaled)
            r2 = r2_score(y_test_scaled, y_hat)

            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, fold_id, p_id, test_mse, r2] + test_metrics
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def _run_test(self, iter_id, y_tr, t_test, y_test, eval):
        preds_filename_base = f'{self.model_name}_iter{iter_id}'
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)

        results_cols = ['iter_id', 'param_id'] + eval.metrics + ['ate_hat']

        cate_hats = preds['cate_hat'].astype(float)

        test_results = []
        for p_id in self.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            ate_hat = np.mean(cate_hat)
            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id] + test_metrics + [ate_hat]
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if fold_id > 0:
            return self._run_valid(iter_id, fold_id, y_tr, t_test, y_test, eval)
        else:
            return self._run_test(iter_id, y_tr, t_test, y_test, eval)

class SConverter():
    def __init__(self, opt):
        self.opt = opt
        self.model_name = get_model_name(opt)
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.model_name}_params.csv'))
    
    def convert(self, iter_id, fold_id):
        preds_filename_base = f'{self.model_name}_iter{iter_id}'
        if fold_id > 0:
            preds_filename_base += f'_fold{fold_id}'
        
        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for p_id in self.df_params['id']:
            preds_filename = f'{preds_filename_base}_param{p_id}.csv'
            df_preds = pd.read_csv(os.path.join(self.opt.results_path, preds_filename))

            y_hats.append(df_preds['y_hat'].to_numpy().reshape(-1, 1))
            y0_hats.append(df_preds['y0_hat'].to_numpy().reshape(-1, 1))
            y1_hats.append(df_preds['y1_hat'].to_numpy().reshape(-1, 1))
            cate_hats.append(df_preds['cate_hat'].to_numpy().reshape(-1, 1))
        
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, preds_filename_base), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

class SDebug():
    def __init__(self, opt):
        self.opt = opt
        self.model = get_regressor(self.opt.base_model)
        self.params_grid = get_params(self.opt.base_model)
    
    def run(self, train, test, iter_id, cate_test):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        y_test = test[2]
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        results = []
        for param_id, params in enumerate(ParameterGrid(self.params_grid)):
            model1 = clone(self.model)
            model1.set_params(**params)

            model1.fit(Xt_tr, y_tr)

            y_hat = model1.predict(Xt_test).reshape(-1, 1)
            y0_hat = model1.predict(xt0).reshape(-1, 1)
            y1_hat = model1.predict(xt1).reshape(-1, 1)
            
            cate_hat = y1_hat - y0_hat

            test_mse = mse(y_hat, y_test)
            r2 = r2_score(y_test, y_hat)
            test_pehe = pehe(cate_test, cate_hat)
            test_ate = abs_ate(cate_test, cate_hat)

            results.append([iter_id, param_id+1, test_mse, r2, test_ate, test_pehe])

        cols = ['fold_id', 'param_id', 'mse', 'r2_score', 'ate', 'pehe']
        return pd.DataFrame(results, columns=cols)