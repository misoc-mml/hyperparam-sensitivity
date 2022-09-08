import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, GridSearchCV
from econml.dr import DRLearner

from ._common import get_params, get_regressor, get_classifier
from helpers.utils import get_params_df

class DRSearch():
    def __init__(self, opt):
        self.opt = opt
        self.m_reg = get_regressor(opt.base_model)
        self.m_prop = get_classifier(opt.base_model)
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