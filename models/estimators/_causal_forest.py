import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from econml.grf import CausalForest

from ._common import get_params, plugin_score, r_score
from helpers.utils import get_params_df

class CausalForestSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model = CausalForest(n_estimators=1000, random_state=opt.seed)
        self.params_grid = get_params(opt.estimation_model)

    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        X_test = test[0]

        if fold_id > 0:
            filename_base = f'{self.opt.estimation_model}_iter{iter_id}_fold{fold_id}_param'
        else:
            filename_base = f'{self.opt.estimation_model}_iter{iter_id}_param'

        for param_id, params in enumerate(ParameterGrid(self.params_grid)):
            model1 = clone(self.model)
            model1.set_params(**params)

            model1.fit(X=X_tr, T=t_tr, y=y_tr)
            cate_hat = model1.predict(X_test)

            filename = f'{filename_base}{param_id+1}.csv'
            pd.DataFrame(cate_hat, columns=['cate_hat']).to_csv(os.path.join(self.opt.output_path, filename), index=False)

    def save_params_info(self):
        df_params = get_params_df(self.params_grid)
        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_params.csv'), index=False)

class CausalForestEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_params = pd.read_csv(os.path.join(self.opt.results_path, f'{self.opt.estimation_model}_params.csv'))

    def rscore(self, iter_id, fold_id, scorer):
        filename_base = f'{self.opt.estimation_model}_iter{iter_id}_fold{fold_id}'
        return r_score(self, iter_id, fold_id, scorer, filename_base)

    def score_cate(self, iter_id, fold_id, plugin):
        filename_base = f'{self.opt.estimation_model}_iter{iter_id}_fold{fold_id}'
        return plugin_score(self, iter_id, fold_id, plugin, filename_base)

    def run(self, iter_id, fold_id, y_tr, t_test, y_test, eval):
        results_cols = ['iter_id', 'param_id'] + eval.metrics + ['ate_hat']
        preds_filename_base = f'{self.opt.estimation_model}_iter{iter_id}'

        if fold_id > 0:
            preds_filename_base += f'_fold{fold_id}'
            results_cols.insert(1, 'fold_id')
        
        test_results = []
        for p_id in self.df_params['id']:
            preds_filename = f'{preds_filename_base}_param{p_id}.csv'
            df_preds = pd.read_csv(os.path.join(self.opt.results_path, preds_filename))

            cate_hat = df_preds['cate_hat'].to_numpy().reshape(-1, 1)
            ate_hat = np.mean(cate_hat)

            test_metrics = eval.get_metrics(cate_hat)

            result = [iter_id, p_id] + test_metrics + [ate_hat]

            if fold_id > 0: result.insert(1, fold_id)

            test_results.append(result)
        
        return pd.DataFrame(test_results, columns=results_cols)