import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, LassoLars, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_params(name):
    if name == 'l1':
        return {'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 20], 'max_iter': [1000, 10000]}
    elif name == 'l2':
        return {'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 20], 'max_iter': [1000, 10000]}
    elif name == 'tr':
        return {'feat__degree': [1, 2], 'clf__power': [0, 2, 3] + list(np.arange(1, 2, 0.1)),
                'clf__alpha': [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 20], 'clf__max_iter':[1000, 10000]}
    elif name == 'dt':
        return {'max_depth': list(np.arange(2, 10)) + [10, 15, 20],
                'min_samples_leaf': list(np.arange(1, 10)) + list(np.arange(0.01, 0.06, 0.01))}
    elif name == 'rf':
        return {'max_depth': list(np.arange(2, 10)) + [10, 15, 20],
                'min_samples_leaf': list(np.arange(1, 10)) + list(np.arange(0.01, 0.06, 0.01))}
    elif name == 'et':
        return {'max_depth': list(np.arange(2, 10)) + [10, 15, 20],
                'min_samples_leaf': list(np.arange(1, 10)) + list(np.arange(0.01, 0.06, 0.01))}
    elif name == 'kr':
        return {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5), "kernel": ["rbf", "poly"], "degree": [2, 3, 4]}
    elif name == 'cb':
        return {"depth": list(np.arange(5, 11)), "l2_leaf_reg": [1, 3, 10, 100]}
    elif name == 'lgbm':
        return {"max_depth": list(np.arange(5, 11)), "reg_lambda": [0.1, 0, 1, 5, 10]}
    else:
        raise ValueError("Unrecognised 'get_params' key.")

def get_regressor(name, seed=1):
    if name == 'l1':
        return LassoLars(fit_intercept=False, random_state=seed)
    elif name == 'l2':
        return Ridge(fit_intercept=False, random_state=seed)
    elif name == 'tr':
        return Pipeline([('feat', PolynomialFeatures()), ('clf', TweedieRegressor(fit_intercept=False))])
    elif name == 'dt':
        return DecisionTreeRegressor()
    elif name == 'rf':
        return RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=seed)
    elif name == 'et':
        return ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, random_state=seed)
    elif name == 'kr':
        return KernelRidge()
    elif name == 'cb':
        return CatBoostRegressor(iterations=1000, random_state=seed, verbose=False, thread_count=-1)
    elif name == 'lgbm':
        return LGBMRegressor(n_estimators=1000, random_state=seed, n_jobs=-1)
    else:
        raise ValueError("Unrecognised 'get_regressor' key.")

def get_params_df(params):
        # ParameterGrid is deterministic, so we can safely do this.
        # First, determine if all combinations have the same number of params.
        param_list = list(ParameterGrid(params))
        equal_len = True
        row_len = len(param_list[0])
        for params in ParameterGrid(params):
            if len(params) != row_len:
                equal_len = False
                break

        if equal_len:
            df_params = pd.DataFrame.from_records(param_list)
        else:
            df_params = pd.DataFrame(param_list, columns=['params'])

        df_params.insert(0, 'id', range(1, len(df_params) + 1))

        return df_params