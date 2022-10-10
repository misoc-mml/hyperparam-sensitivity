import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

from ._common import get_params, get_regressor
from helpers.data import get_scaler
from helpers.metrics import mse
from helpers.utils import get_params_df

class TSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model0 = get_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.model1 = get_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
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
        y0_hats = []
        y0_hat_cates = []
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            m0 = clone(self.model0)
            m0.set_params(**p0)

            m0.fit(X0_tr, y0_tr)

            # Factual predictions for model selection purposes (predict X[t==0]).
            y0_hat = m0.predict(X0_test)
            y0_hats.append(y0_hat)

            # For CATE prediction purposes (predict ALL X).
            y0_hat_cate = m0.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat_cate = scaler.inverse_transform(y0_hat_cate)

            y0_hat_cates.append(y0_hat_cate)
        # ***

        # *** Model y1 ***
        y1_hats = []
        y1_hat_cates = []
        for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
            m1 = clone(self.model1)
            m1.set_params(**p1)

            m1.fit(X1_tr, y1_tr)
            # Factual predictions for model selection purposes (predict X[t==1]).
            y1_hat = m1.predict(X1_test)
            y1_hats.append(y1_hat)

            # For CATE prediction purposes (predict ALL X).
            y1_hat_cate = m1.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y1_hat_cate = scaler.inverse_transform(y1_hat_cate)

            y1_hat_cates.append(y1_hat_cate)
        # ***

        # *** CATE estimator ***
        cate_hats = []
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
                cate_hat = y1_hat_cates[p1_id] - y0_hat_cates[p0_id]
                cate_hats.append(cate_hat)
        # ***

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

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

class TEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_m0_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m0_params.csv'))
        self.df_m1_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m1_params.csv'))
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))

    def _run_valid(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
        else:
            y_test_scaled = y_test
        
        # Split y_test into y0 and y1
        y0_test = y_test_scaled[t_test < 1].reshape(-1, 1)
        y1_test = y_test_scaled[t_test > 0].reshape(-1, 1)

        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)

        results_cols = ['iter_id', 'fold_id', 'param_id', 'mse_m0', 'mse_m1', 'r2_score_m0', 'r2_score_m1'] + eval.metrics
        m0_mse = {}
        m0_r2 = {}
        m1_mse = {}
        m1_r2 = {}
        y0_hats = preds['y0_hat'].astype(float)
        y1_hats = preds['y1_hat'].astype(float)
        cate_hats = preds['cate_hat'].astype(float)
        # m0 and m1 share the same hyperparam search space (same models), so can do 1 loop instead of 2.
        for p0_id in self.df_m0_params['id']:
            m0_mse[p0_id] = mse(y0_hats[p0_id-1].reshape(-1, 1), y0_test)
            m0_r2[p0_id] = r2_score(y0_test, y0_hats[p0_id-1].reshape(-1, 1))
        
            m1_mse[p0_id] = mse(y1_hats[p0_id-1].reshape(-1, 1), y1_test)
            m1_r2[p0_id] = r2_score(y1_test, y1_hats[p0_id-1].reshape(-1, 1))
        
        test_results = []
        for p_id, p0_id, p1_id in zip(self.df_params['id'], self.df_params['m0'], self.df_params['m1']):
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            test_metrics = eval.get_metrics(cate_hat)
            result = [iter_id, fold_id, p_id, m0_mse[p0_id], m1_mse[p1_id], m0_r2[p0_id], m1_r2[p1_id]] + test_metrics
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)
    
    def _run_test(self, iter_id, y_tr, t_test, y_test, eval):
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)

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

class TSSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model0 = get_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.params0 = get_params(opt.base_model)
    
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

        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params0):
            m0 = clone(self.model0)
            m1 = clone(self.model0)

            m0.set_params(**params)
            m1.set_params(**params)

            m0.fit(X0_tr, y0_tr)
            m1.fit(X1_tr, y1_tr)

            # Factual predictions for model selection purposes (predict X[t==0]).
            y0_hat = m0.predict(X0_test)
            y0_hats.append(y0_hat)

            # Factual predictions for model selection purposes (predict X[t==1]).
            y1_hat = m1.predict(X1_test)
            y1_hats.append(y1_hat)

            # For CATE prediction purposes (predict ALL X).
            y0_hat_cate = m0.predict(X_test).reshape(-1, 1)
            y1_hat_cate = m1.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat_cate = scaler.inverse_transform(y0_hat_cate)
                y1_hat_cate = scaler.inverse_transform(y1_hat_cate)

            cate_hat = y1_hat_cate - y0_hat_cate
            cate_hats.append(cate_hat)

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

    def save_params_info(self):
        df_p0 = get_params_df(self.params0)
        df_p0.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class TSEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'))

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
        else:
            y_test_scaled = y_test
        
        # Split y_test into y0 and y1
        y0_test = y_test_scaled[t_test < 1].reshape(-1, 1)
        y1_test = y_test_scaled[t_test > 0].reshape(-1, 1)

        results_cols = ['iter_id', 'param_id', 'mse_m0', 'mse_m1'] + eval.metrics + ['ate_hat', 'r2_score_m0', 'r2_score_m1']
        
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        if fold_id > 0:
            preds_cate_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)

        test_results = []
        for p_id in self.df_params['id']:
            m0_mse = mse(preds['y0_hat'][p_id-1].reshape(-1, 1).astype(float), y0_test)
            m0_r2 = r2_score(y0_test, preds['y0_hat'][p_id-1].reshape(-1, 1).astype(float))

            m1_mse = mse(preds['y1_hat'][p_id-1].reshape(-1, 1).astype(float), y1_test)
            m1_r2 = r2_score(y1_test, preds['y1_hat'][p_id-1].reshape(-1, 1).astype(float))

            cate_hat = preds['cate_hat'][p_id-1].reshape(-1, 1).astype(float)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id, m0_mse, m1_mse] + test_metrics + [ate_hat, m0_r2, m1_r2]
            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

class TConverter():
    def __init__(self, opt):
        self.opt = opt
        self.df_m0_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m0_params.csv'))
        self.df_m1_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m1_params.csv'))
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))
    
    def convert(self, iter_id, fold_id):
        preds_m0_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_m0_iter{iter_id}'
        preds_m1_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_m1_iter{iter_id}'
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}'

        if fold_id > 0:
            preds_m0_filename_base += f'_fold{fold_id}'
            preds_m1_filename_base += f'_fold{fold_id}'
            preds_cate_filename_base += f'_fold{fold_id}'
        
        y0_hats = []
        for p0_id in self.df_m0_params['id']:
            preds_m0_filename = f'{preds_m0_filename_base}_param{p0_id}.csv'
            df_m0 = pd.read_csv(os.path.join(self.opt.results_path, preds_m0_filename))
            y0_hats.append(df_m0['y_hat'].to_numpy().reshape(-1, 1))
        
        y1_hats = []
        for p1_id in self.df_m1_params['id']:
            preds_m1_filename = f'{preds_m1_filename_base}_param{p1_id}.csv'
            df_m1 = pd.read_csv(os.path.join(self.opt.results_path, preds_m1_filename))
            y1_hats.append(df_m1['y_hat'].to_numpy().reshape(-1, 1))
        
        cate_hats = []
        for p_id in self.df_params['id']:
            preds_cate_filename = f'{preds_cate_filename_base}_param{p_id}.csv'
            df_cate = pd.read_csv(os.path.join(self.opt.results_path, preds_cate_filename))
            cate_hats.append(df_cate['cate_hat'].to_numpy().reshape(-1, 1))
        
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, preds_cate_filename_base), y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)