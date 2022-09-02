import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score

from ._common import get_params, get_weighted_regressor, get_classifier, plugin_score, r_score
from helpers.utils import get_params_df
from helpers.data import xt_from_x, get_scaler
from helpers.metrics import mse

def _get_ps_weights(model, x, t, eps=0.0001):
    e = model.predict_proba(x).T[1].T + eps
    return (t / e) + ((1.0 - t) / (1.0 - e))

class IPSWSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_weighted_regressor(opt.base_model)
        self.m_prop = get_classifier(opt.base_model)
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

        global_id = 1
        for ps_id, ps_params in enumerate(ParameterGrid(self.params_prop)):
            ps_model = clone(self.m_prop)
            ps_model.set_params(**ps_params)

            ps_model.fit(X_tr, t_tr)
            weights = _get_ps_weights(ps_model, X_tr, t_tr)

            ps_hat = ps_model.predict(X_test).reshape(-1, 1)
            ps_prob_hat = ps_model.predict_proba(X_test).reshape(-1, 1)

            self._save_predictions(np.hstack([ps_hat, ps_prob_hat]), ['y_hat', 'y_prob_hat'], iter_id, fold_id, ps_id+1, 'prop')

            for _, reg_params in enumerate(ParameterGrid(self.params_reg)):
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

                cols = ['y_hat', 'y0_hat', 'y1_hat', 'cate_hat']
                results = np.concatenate([y_hat.reshape(-1, 1), y0_hat.reshape(-1, 1), y1_hat.reshape(-1, 1), cate_hat.reshape(-1, 1)], axis=1)

                self._save_predictions(results, cols, iter_id, fold_id, global_id, 'cate')

                global_id += 1

    def save_params_info(self):
        df_prop = get_params_df(self.params_prop)
        df_reg = get_params_df(self.params_reg)

        p_global_id = 1
        params_mapping = []
        for prop_id, _ in enumerate(ParameterGrid(self.params_prop)):
            for reg_id, _ in enumerate(ParameterGrid(self.params_reg)):
                params_mapping.append([p_global_id, prop_id, reg_id])
                p_global_id += 1
        df_all = pd.DataFrame(params_mapping, columns=['id', 'prop', 'reg'])

        df_prop.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_prop_params.csv'), index=False)
        df_reg.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_reg_params.csv'), index=False)
        df_all.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'), index=False)

    def _save_predictions(self, preds, cols, iter_id, fold_id, param_id, model):
        filename = f'{self.opt.estimation_model}_{self.opt.base_model}_{model}_iter{iter_id}'

        if fold_id > 0:
            filename += f'_fold{fold_id}'

        filename += f'_param{param_id}.csv'

        pd.DataFrame(preds, columns=cols).to_csv(os.path.join(self.opt.output_path, filename), index=False)

class IPSWEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_prop_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_prop_params.csv'))
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))
    
    def rscore(self, iter_id, fold_id, scorer):
        filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}_fold{fold_id}'
        return r_score(self, iter_id, fold_id, scorer, filename_base)

    def score_cate(self, iter_id, fold_id, plugin):
        filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}_fold{fold_id}'
        return plugin_score(self, iter_id, fold_id, plugin, filename_base)
    
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
        preds_prop_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_prop_iter{iter_id}'
        preds_cate_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_cate_iter{iter_id}'

        if fold_id > 0:
            preds_prop_filename_base += f'_fold{fold_id}'
            preds_cate_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')
        
        prop_mse = {}
        prop_r2 = {}
        for prop_id in self.df_prop_params['id']:
            preds_prop_filename = f'{preds_prop_filename_base}_param{prop_id}.csv'
            df_prop = pd.read_csv(os.path.join(self.opt.results_path, preds_prop_filename))
            prop_mse[prop_id] = mse(df_prop['y_prob_hat'].to_numpy().reshape(-1, 1), t_test)
            prop_r2[prop_id] = r2_score(t_test, df_prop['y_prob_hat'].to_numpy().reshape(-1, 1))
        
        test_results = []
        for p_id, prop_id in zip(self.df_params['id'], self.df_params['prop']):
            preds_cate_filename = f'{preds_cate_filename_base}_param{p_id}.csv'
            df_cate = pd.read_csv(os.path.join(self.opt.results_path, preds_cate_filename))

            cate_hat = df_cate['cate_hat'].to_numpy().reshape(-1, 1)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            # Scaled MSE (y_hat is scaled).
            y_hat = df_cate['y_hat'].to_numpy().reshape(-1, 1)
            reg_mse = mse(y_hat, y_test_scaled)
            reg_r2 = r2_score(y_test_scaled, y_hat)

            result = [iter_id, p_id, prop_mse[prop_id], reg_mse] + test_metrics + [ate_hat, prop_r2[prop_id], reg_r2]
            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)