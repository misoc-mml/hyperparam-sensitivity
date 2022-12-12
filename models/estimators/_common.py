import numpy as np

from sklearn.linear_model import Ridge, LassoLars, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from econml.sklearn_extensions.linear_model import WeightedLasso

class RidgeClassifier(Ridge):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)
        return np.concatenate([1 - p, p], axis=1)

class LassoLarsClassifier(LassoLars):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)
        return np.concatenate([1 - p, p], axis=1)

class KernelRidgeClassifier(KernelRidge):
    def predict_proba(self, X):
        p = self.predict(X).reshape(-1, 1)
        p = np.clip(p, 0, 1)
        return np.concatenate([1 - p, p], axis=1)

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
    elif name == 'cf':
        return {'max_depth': list(np.arange(2, 10)) + [10, 15, 20],
                'min_samples_leaf': list(np.arange(1, 10)) + list(np.arange(0.01, 0.06, 0.01))}
    else:
        raise ValueError("Unrecognised 'get_params' key.")

def get_default_params(name, ds):
    if name == 'l1':
        return {'alpha': 1, 'max_iter': 1000}
    elif name == 'l2':
        return {'alpha': 1, 'max_iter': 1000}
    elif name == 'dt':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'rf':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'et':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    elif name == 'kr':
        return {"alpha": 1, "gamma": 1, "kernel": "poly", "degree": 3}
    elif name == 'cb':
        return {"depth": 10, "l2_leaf_reg": 1}
    elif name == 'lgbm':
        return {"max_depth": 10, "reg_lambda": 0.1}
    elif name == 'cf':
        if ds == 'news':
            return {'max_depth': 10, 'min_samples_leaf': 1}
        else:
            return {'max_depth': 20, 'min_samples_leaf': 1}
    else:
        raise ValueError("Unrecognised 'get_default_params' key.")

def get_weighted_regressor(name, seed=1, n_jobs=-1):
    if name == 'l1':
        # LassoLars doesn't support sample weights, so use WeightedLasso instead.
        return WeightedLasso(random_state=seed)
    else:
        return get_regressor(name, seed, n_jobs)

def get_regressor(name, seed=1, n_jobs=1):
    if name == 'l1':
        return LassoLars(random_state=seed)
    elif name == 'l2':
        return Ridge(random_state=seed)
    elif name == 'tr':
        return Pipeline([('feat', PolynomialFeatures()), ('clf', TweedieRegressor())])
    elif name == 'dt':
        return DecisionTreeRegressor()
    elif name == 'rf':
        return RandomForestRegressor(n_estimators=1000, n_jobs=n_jobs, random_state=seed)
    elif name == 'et':
        return ExtraTreesRegressor(n_estimators=1000, n_jobs=n_jobs, random_state=seed)
    elif name == 'kr':
        return KernelRidge()
    elif name == 'cb':
        return CatBoostRegressor(iterations=1000, random_state=seed, verbose=False, thread_count=n_jobs)
    elif name == 'lgbm':
        return LGBMRegressor(n_estimators=1000, random_state=seed, n_jobs=n_jobs)
    else:
        raise ValueError("Unrecognised 'get_regressor' key.")

def get_classifier(name, seed=1, n_jobs=1):
    if name == 'l1':
        return LassoLarsClassifier(random_state=seed)
    elif name == 'l2':
        return RidgeClassifier(random_state=seed)
    elif name == 'tr':
        return None
    elif name == 'dt':
        return DecisionTreeClassifier()
    elif name == 'rf':
        return RandomForestClassifier(n_estimators=1000, n_jobs=n_jobs, random_state=seed)
    elif name == 'et':
        return ExtraTreesClassifier(n_estimators=1000, n_jobs=n_jobs, random_state=seed)
    elif name == 'kr':
        return KernelRidgeClassifier()
    elif name == 'cb':
        return CatBoostClassifier(iterations=1000, random_state=seed, verbose=False, thread_count=n_jobs)
    elif name == 'lgbm':
        return LGBMClassifier(n_estimators=1000, random_state=seed, n_jobs=n_jobs)
    else:
        raise ValueError("Unrecognised 'get_classifier' key.")
