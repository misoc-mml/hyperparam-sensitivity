import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, GridSearchCV
from econml.dr import DRLearner

from ._common import get_params, get_regressor, get_classifier
from helpers.utils import get_params_df
from helpers.data import xt_from_x

class DRSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_regressor(opt.base_model, n_jobs=self.opt.n_jobs)
        self.m_prop = get_classifier(opt.base_model, n_jobs=self.opt.n_jobs)
        self.params_reg = get_params(opt.base_model)
        self.params_prop = get_params(opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        X_test = test[0]
        t_test = test[1].flatten()
        y_test = test[2].flatten()

        if fold_id > 0:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        scores = []
        cate_hats = []
        for p_reg in ParameterGrid(self.params_reg):
            model_reg = clone(self.m_reg)
            model_reg.set_params(**p_reg)

            for p_prop in ParameterGrid(self.params_prop):
                model_prop = clone(self.m_prop)
                model_prop.set_params(**p_prop)

                # Tune 'model_final' but not part of full exploration.
                model_final = GridSearchCV(get_regressor(self.opt.base_model), get_params(self.opt.base_model), cv=self.opt.inner_cv, n_jobs=-1)

                dr = DRLearner(model_propensity=model_prop, model_regression=model_reg, model_final=model_final, cv=self.opt.inner_cv, random_state=self.opt.seed)
                dr.fit(y_tr, t_tr, X=X_tr)

                score_reg = np.mean(dr.nuisance_scores_regression)
                score_prop = np.mean(dr.nuisance_scores_propensity)
                score_final = dr.score(y_test, t_test, X=X_test)
                scores.append([score_reg, score_prop, score_final])

                cate_hat = dr.effect(X_test)
                cate_hats.append(cate_hat)
        
        scores_arr = np.array(scores)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, base_filename), cate_hat=cate_hats_arr, scores=scores_arr)

    def save_params_info(self):
        df_reg = get_params_df(self.params_reg)
        df_prop = get_params_df(self.params_prop)

        global_id = 1
        mapping = []
        for reg_id, _ in enumerate(ParameterGrid(self.params_reg)):
            for prop_id, _ in enumerate(ParameterGrid(self.params_prop)):
                mapping.append([global_id, reg_id+1, prop_id+1])
                global_id += 1
        df_all = pd.DataFrame(mapping, columns=['id', 'reg', 'prop'])

        df_reg.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_reg_params.csv'), index=False)
        df_prop.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_prop_params.csv'), index=False)
        df_all.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'), index=False)

class DREvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'))

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'param_id', 'ate_hat'] + eval.metrics + ['reg_score', 'prop_score', 'final_score']
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        if fold_id > 0:
            preds_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)

        test_results = []
        for p_id in self.df_params['id']:
            cate_hat = preds['cate_hat'][p_id-1].reshape(-1, 1).astype(float)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id, ate_hat] + test_metrics

            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        test_results_arr = np.array(test_results)
        all_results = np.hstack((test_results_arr, preds['scores']).astype(float))

        return pd.DataFrame(all_results, columns=results_cols)

class DRSSearch():
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
        t_test = test[1].flatten()
        y_test = test[2].flatten()

        if fold_id > 0:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        scores = []
        cate_hats = []
        for params in ParameterGrid(self.params_base):
            model_reg = clone(self.m_reg)
            model_reg.set_params(**params)

            model_prop = clone(self.m_prop)
            model_prop.set_params(**params)

            model_final = clone(self.m_reg)
            model_final.set_params(**params)

            dr = DRLearner(model_propensity=model_prop, model_regression=model_reg, model_final=model_final, cv=self.opt.inner_cv, random_state=self.opt.seed, min_propensity=0.1)
            dr.fit(y_tr, t_tr, X=X_tr)

            cate_hat = dr.effect(X_test).reshape(-1, 1)
            cate_hats.append(cate_hat)

            score_reg = np.mean(dr.nuisance_scores_regression)
            score_prop = np.mean(dr.nuisance_scores_propensity)
            score_final = dr.score(y_test, t_test, X=X_test)
            scores.append([score_reg, score_prop, score_final])
        
        scores_arr = np.array(scores)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, base_filename), cate_hat=cate_hats_arr, scores=scores_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_base)
        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class DRSEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'))

    def _run_valid(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'fold_id', 'param_id'] + eval.metrics + ['reg_score', 'prop_score', 'final_score']
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)
        cate_hats = preds['cate_hat'].astype(float)
        
        test_results = []
        for p_id in self.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            test_metrics = eval.get_metrics(cate_hat)
            result = [iter_id, fold_id, p_id] + test_metrics
            test_results.append(result)
        
        test_results_arr = np.array(test_results)
        all_results = np.hstack((test_results_arr, preds['scores'].astype(float)))

        return pd.DataFrame(all_results, columns=results_cols)

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

class DRCustomSearch():
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
        t_test = test[1].reshape(-1, 1)
        y_test = test[2].reshape(-1, 1)

        X0_tr = X_tr[t_tr < 1]
        X1_tr = X_tr[t_tr > 0]
        y0_tr = y_tr[t_tr < 1]
        y1_tr = y_tr[t_tr > 0]
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)

        if fold_id > 0:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            base_filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        #model_prop = clone(self.m_prop)
        #model_prop.set_params(**params)
        model_prop = GridSearchCV(self.m_prop, self.params_base, cv=5, n_jobs=-1)
        model_prop.fit(X_tr, t_tr)
        ps = model_prop.predict_proba(X_test)[:, 1].reshape(-1, 1) + 1e-6
        ps = np.clip(ps, 0.01, 0.99).reshape(-1, 1)

        scores = []
        cate_hats = []

        xt0, xt1 = xt_from_x(X_test)

        for params in ParameterGrid(self.params_base):
            model_y0 = clone(self.m_reg)
            model_y0.set_params(**params)
            model_y0.fit(Xt_tr, y_tr)
            mu0 = model_y0.predict(xt0).reshape(-1, 1)
            mu1 = model_y0.predict(xt1).reshape(-1, 1)
            Y0 = mu0 + ((1.0 - t_test) * (y_test - mu0) / (1.0 - ps))
            Y1 = mu1 + (t_test * (y_test - mu1) / ps)
            cate_hat = Y1 - Y0
            cate_hats.append(cate_hat)
            scores.append([0, 0, 0])

        """
        y0s = []
        for params in ParameterGrid(self.params_base):
            model_y0 = clone(self.m_reg)
            model_y0.set_params(**params)
            model_y0.fit(X0_tr, y0_tr)
            mu0 = model_y0.predict(X_test).reshape(-1, 1)
            Y0 = mu0 + ((1.0 - t_test) * (y_test - mu0) / (1.0 - ps))
            y0s.append(Y0)
        
        y1s = []
        for params in ParameterGrid(self.params_base):
            model_y1 = clone(self.m_reg)
            model_y1.set_params(**params)
            model_y1.fit(X1_tr, y1_tr)
            mu1 = model_y1.predict(X_test).reshape(-1, 1)
            Y1 = mu1 + (t_test * (y_test - mu1) / ps)
            y1s.append(Y1)
        
        for p0, params0 in enumerate(ParameterGrid(self.params_base)):
            for p1, params1 in enumerate(ParameterGrid(self.params_base)):
                cate_hat = y1s[p1] - y0s[p0]
                cate_hats.append(cate_hat)
                scores.append([0, 0, 0])
        """

        """
        for params in ParameterGrid(self.params_base):
            model_y0 = clone(self.m_reg)
            model_y0.set_params(**params)
            model_y0.fit(X0_tr, y0_tr)
            mu0 = model_y0.predict(X_test).reshape(-1, 1)

            model_y1 = clone(self.m_reg)
            model_y1.set_params(**params)
            model_y1.fit(X1_tr, y1_tr)
            mu1 = model_y1.predict(X_test).reshape(-1, 1)

            #model_prop = clone(self.m_prop)
            #model_prop.set_params(**params)
            #model_prop.fit(X_tr, t_tr)
            #ps = model_prop.predict_proba(X_test)[:, 1].reshape(-1, 1) + 1e-6
            #ps = np.clip(ps, 0.01, 0.99).reshape(-1, 1)

            #Y0 = ( (1 - t_test) * (y_test - mu0) / (1 - ps) + mu0 )
            #Y1 = ( (t_test) * (y_test - mu1) / (ps) + mu1 )
            Y0 = mu0 + ((1.0 - t_test) * (y_test - mu0) / (1.0 - ps))
            Y1 = mu1 + (t_test * (y_test - mu1) / ps)
            cate_hat = Y1 - Y0
            
            scores.append([0, 0, 0])

            #score_reg = np.mean(dr.nuisance_scores_regression)
            #score_prop = np.mean(dr.nuisance_scores_propensity)
            #score_final = dr.score(y_test, t_test, X=X_test)
            #scores.append([score_reg, score_prop, score_final])

            #cate_hat = dr.effect(X_test)
            cate_hats.append(cate_hat)
            """
        
        scores_arr = np.array(scores)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, base_filename), cate_hat=cate_hats_arr, scores=scores_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_base)

        """
        p_global_id = 1
        params_mapping = []
        for p0_id, _ in enumerate(ParameterGrid(self.params_base)):
            for p1_id, _ in enumerate(ParameterGrid(self.params_base)):
                params_mapping.append([p_global_id, p0_id+1, p1_id+1])
                p_global_id += 1
        df_params = pd.DataFrame(params_mapping, columns=['id', 'm0', 'm1'])
        """

        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)

class DRCustomEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'))

    def _run_valid(self, iter_id, fold_id, y_tr, t_test, y_test):
        results_cols = ['iter_id', 'fold_id', 'param_id', 'reg_score', 'prop_score', 'final_score']
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        
        preds = np.load(os.path.join(self.opt.results_path, f'{preds_filename_base}.npz'), allow_pickle=True)
        
        test_results = []
        for p_id in self.df_params['id']:
            result = [iter_id, fold_id, p_id]
            test_results.append(result)
        
        test_results_arr = np.array(test_results)
        all_results = np.hstack((test_results_arr, preds['scores'].astype(float)))

        return pd.DataFrame(all_results, columns=results_cols)

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
            return self._run_valid(iter_id, fold_id, y_tr, t_test, y_test)
        else:
            return self._run_test(iter_id, y_tr, t_test, y_test, eval)