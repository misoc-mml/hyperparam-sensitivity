import os
import numpy as np
from sklearn.model_selection import ParameterGrid

from ._common import get_params, get_mlp_reg
from helpers.data import xt_from_x
from helpers.utils import get_params_df

class SSearch():
    def __init__(self, opt):
        self.opt = opt
        self.params_grid = get_params(self.opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()
        Xt_tr = np.concatenate([X_tr, t_tr.reshape(-1, 1)], axis=1)
        X_test = test[0]
        t_test = test[1].flatten()
        Xt_test = np.concatenate([X_test, t_test.reshape(-1, 1)], axis=1)

        xt0, xt1 = xt_from_x(X_test)

        input_size = Xt_tr.shape[1]

        y_hats = []
        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params_grid):
            model1 = get_mlp_reg(input_size, params)

            model1.fit(Xt_tr, y_tr)

            y_hat = model1.predict(Xt_test)

            y0_hat = model1.predict(xt0).reshape(-1, 1)
            y1_hat = model1.predict(xt1).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat = scaler.inverse_transform(y0_hat)
                y1_hat = scaler.inverse_transform(y1_hat)
            
            cate_hat = y1_hat - y0_hat

            y_hats.append(y_hat)
            y0_hats.append(y0_hat)
            y1_hats.append(y1_hat)
            cate_hats.append(cate_hat)

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        y_hats_arr = np.array(y_hats, dtype=object)
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y_hat=y_hats_arr, y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_grid)

        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)