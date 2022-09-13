import os
import numpy as np
import pandas as pd
# This is just to fix the import error related to RScorer.
from econml.dml import CausalForestDML
from econml.score import RScorer
from sklearn.model_selection import GridSearchCV

from models.estimators._common import get_params, get_regressor, get_classifier
from helpers.metrics import rscore
from helpers.utils import get_model_name

class RScorerWrapper():
    def __init__(self, opt):
        self.opt = opt
    
    def run(self, X, t, y):
        model_y_cv = GridSearchCV(get_regressor(self.opt.base_model, self.opt.seed, 1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)
        model_t_cv = GridSearchCV(get_classifier(self.opt.base_model, self.opt.seed, 1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)

        model_y_cv.fit(X, y.flatten())
        model_t_cv.fit(X, t.flatten())

        rscorer = RScorer(model_y=model_y_cv.best_estimator_, model_t=model_t_cv.best_estimator_, discrete_treatment=True, cv=self.opt.cv, random_state=self.opt.seed)
        rscorer.fit(y.flatten(), t.flatten(), X=X)

        # ((Y_res, T_res), base_score)
        return rscorer.lineardml_._cached_values.nuisances, rscorer.base_score_

class RScorerEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_base_scores = pd.read_csv(os.path.join(self.opt.scorer_path, f'{self.opt.scorer_name}_base_scores.csv'))

    def score(self, est, iter_id, fold_id):
        filename = f'{get_model_name(self.opt)}_iter{iter_id}_fold{fold_id}.npz'

        preds = np.load(os.path.join(self.opt.results_path, filename), allow_pickle=True)
        cate_hats = preds['cate_hat'].astype(float)

        base_score = float(self.df_base_scores.loc[(self.df_base_scores['iter_id'] == iter_id) & (self.df_base_scores['fold_id'] == fold_id), 'base_score'])
        res = np.load(os.path.join(self.opt.scorer_path, f'{self.opt.scorer_name}_iter{iter_id}_fold{fold_id}.npz'), allow_pickle=True)

        test_results = []
        for p_id in est.df_params['id']:
            cate_hat = cate_hats[p_id-1].reshape(-1, 1)
            
            _score = rscore(cate_hat, res['y_res'], res['t_res'], base_score)

            result = [iter_id, fold_id, p_id, _score]
            test_results.append(result)
        
        results_cols = ['iter_id', 'fold_id', 'param_id', 'rscore']
        return pd.DataFrame(test_results, columns=results_cols)

class RScorerConverter():
    def __init__(self, opt):
        self.opt = opt
    
    def convert(self, iter_id, fold_id):
        df = pd.read_csv(os.path.join(self.opt.results_path, f'rs_{self.opt.base_model}_iter{iter_id}_fold{fold_id}.csv'))

        y_res = df['y_res'].to_numpy().reshape(-1, 1)
        t_res = df['t_res'].to_numpy().reshape(-1, 1)

        np.savez_compressed(os.path.join(self.opt.output_path, f'rs_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'), y_res=y_res, t_res=t_res)