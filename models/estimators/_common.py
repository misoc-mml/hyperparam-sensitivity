import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

def get_params(name):
    if name == 'dt':
        return {'max_depth': range(2, 10)}
    elif name == 'rf':
        return {'estimators': [100, 1000], 'max_depth': range(2, 10)}
    else:
        raise ValueError("Unrecognised 'get_params' key.")

def get_regressor(name):
    if name == 'dt':
        return DecisionTreeRegressor()
    elif name == 'rf':
        return RandomForestRegressor()
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