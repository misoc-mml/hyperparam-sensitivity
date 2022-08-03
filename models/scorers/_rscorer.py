import os
import pandas as pd
from econml.score import RScorer
from sklearn.model_selection import GridSearchCV

from models.estimators._common import get_params, get_regressor, get_classifier
from helpers.metrics import rscore

class RScorerWrapper():
    def __init__(self, opt):
        self.opt = opt
    
    def run(self, X, t, y):
        model_y_cv = GridSearchCV(get_regressor(self.opt.base_model, self.opt.seed, 1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)
        model_t_cv = GridSearchCV(get_classifier(self.opt.base_model, self.opt.seed, 1), get_params(self.opt.base_model), n_jobs=self.opt.n_jobs, cv=self.opt.cv)

        model_y_cv.fit(X, y.flatten())
        model_t_cv.fit(X, t.flatten())

        rscorer = RScorer(model_y=model_y_cv.best_estimator_, model_t=model_t_cv.best_estimator_, discrete_treatment=True, cv=self.opt.cv, random_state=self.opt.seed)
        rscorer.fit(y, t, X=X)

        # ((Y_res, T_res), base_score)
        return rscorer.lineardml_._cached_values.nuisances, rscorer.base_score_

class RScorerEvaluator():
    def __init__(self, opt):
        self.opt = opt
        self.df_base_scores = pd.read_csv(os.path.join(self.opt.scorer_path, f'rs_{self.opt.base_model}_base_scores.csv'))
    
    def score(self, iter_id, fold_id, cate_hat):
        base_score = float(self.df_base_scores.loc[(self.df_base_scores['iter_id'] == iter_id) & (self.df_base_scores['fold_id'] == fold_id), 'base_score'])

        df_res = pd.read_csv(os.path.join(self.opt.scorer_path, f'rs_{self.opt.base_model}_iter{iter_id}_fold{fold_id}.csv'))

        return rscore(cate_hat, df_res['y_res'].to_numpy(), df_res['t_res'].to_numpy(), base_score)