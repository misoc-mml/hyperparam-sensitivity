from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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