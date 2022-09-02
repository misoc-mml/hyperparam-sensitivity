import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, GridSearchCV
from econml.dr import DRLearner

from ._common import get_params, get_regressor, get_classifier, plugin_score, r_score
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

        global_id = 1
        scores = []
        for p_reg in ParameterGrid(self.params_reg):
            model_reg = clone(self.m_reg)
            model_reg.set_params(**p_reg)

            for p_prop in ParameterGrid(self.params_prop):
                model_prop = clone(self.m_prop)
                model_prop.set_params(**p_prop)

                # Tune 'model_final' but not part of full exploration.
                model_final = GridSearchCV(get_regressor(self.opt.base_model), get_params(self.opt.base_model), cv=self.opt.inner_cv)

                dr = DRLearner(model_propensity=model_prop, model_regression=model_reg, model_final=model_final, cv=self.opt.inner_cv, random_state=self.opt.seed)
                dr.fit(y_tr, t_tr, X=X_tr)

                score_reg = np.mean(dr.nuisance_scores_regression)
                score_prop = np.mean(dr.nuisance_scores_propensity)
                score_final = dr.score(y_test, t_test, X=X_test)
                scores.append([global_id, score_reg, score_prop, score_final])

                cate_hat = dr.effect(X_test)
                    
                pd.DataFrame(cate_hat, columns=['cate_hat']).to_csv(os.path.join(self.opt.output_path, f'{base_filename}_param{global_id}.csv'), index=False)
                    
                global_id += 1
        
        pd.DataFrame(scores, columns=['id', 'score_reg', 'score_prop', 'score_final']).to_csv(os.path.join(self.opt.output_path, f'{base_filename}_scores.csv'), index=False)

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
    
    def rscore(self, iter_id, fold_id, scorer):
        filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        return r_score(self, iter_id, fold_id, scorer, filename_base)

    def score_cate(self, iter_id, fold_id, plugin):
        filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        return plugin_score(self, iter_id, fold_id, plugin, filename_base)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'param_id', 'reg_score', 'prop_score', 'final_score'] + eval.metrics + ['ate_hat']
        preds_filename_base = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'

        if fold_id > 0:
            preds_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')
        
        df_scores = pd.read_csv(os.path.join(self.opt.results_path, f'{preds_filename_base}_scores.csv'))

        test_results = []
        for p_id, s_reg, s_prop, s_final in zip(df_scores['id'], df_scores['score_reg'], df_scores['score_prop'], df_scores['score_final']):
            preds_filename = f'{preds_filename_base}_param{p_id}.csv'
            df_preds = pd.read_csv(os.path.join(self.opt.results_path, preds_filename))

            cate_hat = df_preds['cate_hat'].to_numpy().reshape(-1, 1)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id, s_reg, s_prop, s_final] + test_metrics + [ate_hat]

            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)