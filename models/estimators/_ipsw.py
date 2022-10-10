import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

from ._common import get_params, get_weighted_regressor, get_classifier
from helpers.utils import get_params_df
from helpers.data import xt_from_x, get_scaler
from helpers.metrics import mse

def _get_ps_weights(model, x, t, eps=0.0001):
    e = model.predict_proba(x).T[1].T + eps
    return (t / e) + ((1.0 - t) / (1.0 - e))

class IPSWSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_weighted_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.m_prop = get_classifier(opt.base_model, n_jobs=self.opt.n_jobs)
        self.params_reg = get_params(opt.base_model)
        self.params_prop = get_params(opt.base_model)

    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        t_hats = []
        t_prob_hats = []
        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for ps_params in ParameterGrid(self.params_prop):
            ps_model = clone(self.m_prop)
            ps_model.set_params(**ps_params)

            ps_model.fit(X_tr, t_tr)
            weights = _get_ps_weights(ps_model, X_tr, t_tr)

            ps_hat = ps_model.predict(X_test).reshape(-1, 1)
            ps_prob_hat = ps_model.predict_proba(X_test).reshape(-1, 1)

            t_hats.append(ps_hat)
            t_prob_hats.append(ps_prob_hat)
    
            for reg_params in ParameterGrid(self.params_reg):
                reg_model = clone(self.m_reg)
                reg_model.set_params(**reg_params)

                reg_model.fit(Xt_tr, y_tr, sample_weight=weights)

                y_hat = reg_model.predict(Xt_test)

                y0_hat = reg_model.predict(xt0).reshape(-1, 1)
                y1_hat = reg_model.predict(xt1).reshape(-1, 1)

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
        
        t_hats_arr = np.array(t_hats, dtype=object)
        t_prob_hats_arr = np.array(t_prob_hats, dtype=object)
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr, t_hat=t_hats_arr, t_prob_hat=t_prob_hats_arr)

    def save_params_info(self):
        df_prop = get_params_df(self.params_prop)
        df_reg = get_params_df(self.params_reg)

        p_global_id = 1
        params_mapping = []
        for prop_id, _ in enumerate(ParameterGrid(self.params_prop)):
            for reg_id, _ in enumerate(ParameterGrid(self.params_reg)):
                params_mapping.append([p_global_id, prop_id+1, reg_id+1])
                p_global_id += 1
        df_all = pd.DataFrame(params_mapping, columns=['id', 'prop', 'reg'])

        df_prop.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_prop_params.csv'), index=False)
        df_reg.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_reg_params.csv'), index=False)
        df_all.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'), index=False)

class IPSWEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_prop_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_prop_params.csv'))
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))
    
    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        if self.opt.scale_y:
            # Replicate the scaler
            scaler = get_scaler(self.opt.scaler)
            scaler.fit(y_tr)
            y_test_scaled = scaler.transform(y_test)
        else:
            y_test_scaled = y_test
        
        if y_test_scaled.ndim == 1:
            y_test_scaled = y_test_scaled.reshape(-1, 1)
        
        results_cols = ['iter_id', 'param_id', 'mse_prop', 'mse_reg'] + eval.metrics + ['ate_hat', 'r2_score_prop', 'r2_score_reg']
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        if fold_id > 0:
            preds_cate_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')

        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)

        prop_mse = {}
        prop_r2 = {}
        for prop_id in self.df_prop_params['id']:
            prop_mse[prop_id] = mse(preds['t_prob_hat'][prop_id-1].reshape(-1, 1).astype(float), t_test)
            prop_r2[prop_id] = r2_score(t_test, preds['t_prob_hat'][prop_id-1].reshape(-1, 1).astype(float))
        
        test_results = []
        for p_id, prop_id in zip(self.df_params['id'], self.df_params['prop']):
            cate_hat = preds['cate_hat'][p_id-1].reshape(-1, 1).astype(float)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            # Scaled MSE (y_hat is scaled).
            y_hat = preds['y_hat'][p_id-1].reshape(-1, 1).astype(float)
            reg_mse = mse(y_hat, y_test_scaled)
            reg_r2 = r2_score(y_test_scaled, y_hat)

            result = [iter_id, p_id, prop_mse[prop_id], reg_mse] + test_metrics + [ate_hat, prop_r2[prop_id], reg_r2]
            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)

class IPSWSSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_weighted_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.m_prop = get_classifier(opt.base_model, n_jobs=self.opt.n_jobs)
        self.params_base = get_params(opt.base_model)

    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        t_hats = []
        t_prob_hats = []
        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params_base):
            # *** Propensity model ***
            ps_model = clone(self.m_prop)
            ps_model.set_params(**params)

            ps_model.fit(X_tr, t_tr)
            weights = _get_ps_weights(ps_model, X_tr, t_tr)

            ps_hat = ps_model.predict(X_test).reshape(-1, 1)
            # TODO: take only P(t==1) -- second column
            ps_prob_hat = ps_model.predict_proba(X_test).reshape(-1, 1)

            t_hats.append(ps_hat)
            t_prob_hats.append(ps_prob_hat)
            # ***

            # *** Regression model ***
            reg_model = clone(self.m_reg)
            reg_model.set_params(**params)
    
            reg_model.fit(Xt_tr, y_tr, sample_weight=weights)

            y_hat = reg_model.predict(Xt_test)

            y0_hat = reg_model.predict(xt0).reshape(-1, 1)
            y1_hat = reg_model.predict(xt1).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            # ***
                
            # *** CATE ***
            cate_hat = y1_hat - y0_hat

            y_hats.append(y_hat)
            y0_hats.append(y0_hat)
            y1_hats.append(y1_hat)
            cate_hats.append(cate_hat)
            # ***
        
        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        t_hats_arr = np.array(t_hats, dtype=object)
        t_prob_hats_arr = np.array(t_prob_hats, dtype=object)
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr, t_hat=t_hats_arr, t_prob_hat=t_prob_hats_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_base)
        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class IPSWSEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'))
    
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
        
        results_cols = ['iter_id', 'fold_id', 'param_id', 'mse_prop', 'mse_reg', 'r2_score_prop', 'r2_score_reg'] + eval.metrics
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'

        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)
        cate_hats = preds['cate_hat'].astype(float)
        
        t_prob_hats = preds['t_prob_hat'].astype(float)
        y_hats = preds['y_hat'].astype(float)

        test_results = []
        for p_id in self.df_params['id']:
            #prop_mse = mse(t_prob_hats[p_id-1].reshape(-1, 1), t_test)
            #prop_r2 = r2_score(t_test, t_prob_hats[p_id-1].reshape(-1, 1))
            # Temporary hack to counter the bug in the search part.
            prop_mse = mse(t_prob_hats[p_id-1].reshape(-1, 2)[:, 1:], t_test)
            prop_r2 = r2_score(t_test, t_prob_hats[p_id-1].reshape(-1, 2)[:, 1:])

            # Scaled MSE (y_hat is scaled).
            y_hat = y_hats[p_id-1].reshape(-1, 1)
            reg_mse = mse(y_hat, y_test_scaled)
            reg_r2 = r2_score(y_test_scaled, y_hat)

            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, fold_id, p_id, prop_mse, reg_mse, prop_r2, reg_r2] + test_metrics
            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)
    
    def _run_test(self, iter_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'param_id'] + eval.metrics + ['ate_hat']
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        preds = np.load(os.path.join(self.opt.results_path, f'{preds_cate_filename_base}.npz'), allow_pickle=True)
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