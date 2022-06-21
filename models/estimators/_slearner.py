import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from ._common import get_params, get_regressor
from helpers.data import xt_from_x

class SSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model = get_regressor(self.opt.base_model)
        self.params_grid = get_params(self.opt.base_model)
    
    def run(self, train, test, scaler, opt, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        for param_id, params in enumerate(ParameterGrid(self.params_grid)):
            model1 = clone(self.model)
            model1.set_params(**params)

            model1.fit(Xt_tr, y_tr)

            y_hat = model1.predict(Xt_test)

            y0_hat = model1.predict(xt0)
            y1_hat = model1.predict(xt1)

            if opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            
            cate_hat = y1_hat - y0_hat

            cols = ['y_hat', 'y0_hat', 'y1_hat', 'cate_hat']
            results = np.concatenate([y_hat.reshape(-1, 1), y0_hat.reshape(-1, 1), y1_hat.reshape(-1, 1), cate_hat.reshape(-1, 1)], axis=1)

            if fold_id > 0:
                filename = f'{opt.estimation_model}_{opt.base_model}_iter{iter_id}_fold{fold_id}_param{param_id+1}.csv'
            else:
                filename = f'{opt.estimation_model}_{opt.base_model}_iter{iter_id}_param{param_id+1}.csv'

            pd.DataFrame(results, columns=cols).to_csv(os.path.join(opt.output_path, filename), index=False)

    
    def get_params_info(self):
        # ParameterGrid is deterministic, so we can safely do this.
        # First, determine if all combinations have the same number of params.
        equal_len = True
        row_len = -1
        for params in ParameterGrid(self.params_grid):
            if row_len < 0:
                row_len = len(params)
            
            if len(params) != row_len:
                equal_len = False
                break

        param_list = list(ParameterGrid(self.params_grid))

        if equal_len:
            df_params = pd.DataFrame.from_records(param_list)
        else:
            df_params = pd.DataFrame(param_list, columns=['params'])

        df_params.insert(0, 'id', range(1, len(df_params) + 1))

        return df_params