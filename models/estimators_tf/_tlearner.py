import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ._common import get_params, get_mlp_reg
from helpers.utils import get_params_df

class TSearch():
    def __init__(self, opt):
        self.opt = opt
        self.params0 = get_params(opt.base_model)
        self.params1 = get_params(opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()

        X0_tr = X_tr[t_tr < 1]
        X1_tr = X_tr[t_tr > 0]
        y0_tr = y_tr[t_tr < 1]
        y1_tr = y_tr[t_tr > 0]

        X_test = test[0]
        t_test = test[1].flatten()
        X0_test = X_test[t_test < 1]
        X1_test = X_test[t_test > 0]

        input_size = X0_tr.shape[1]

        # *** Model y0 ***
        y0_hats = []
        y0_hat_cates = []
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            m0 = get_mlp_reg(input_size, p0)

            m0.fit(X0_tr, y0_tr)

            # Factual predictions for model selection purposes (predict X[t==0]).
            y0_hat = m0.predict(X0_test)
            y0_hats.append(y0_hat)

            # For CATE prediction purposes (predict ALL X).
            y0_hat_cate = m0.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat_cate = scaler.inverse_transform(y0_hat_cate)

            y0_hat_cates.append(y0_hat_cate)
        # ***

        # *** Model y1 ***
        y1_hats = []
        y1_hat_cates = []
        for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
            m1 = get_mlp_reg(input_size, p1)

            m1.fit(X1_tr, y1_tr)
            # Factual predictions for model selection purposes (predict X[t==1]).
            y1_hat = m1.predict(X1_test)
            y1_hats.append(y1_hat)

            # For CATE prediction purposes (predict ALL X).
            y1_hat_cate = m1.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y1_hat_cate = scaler.inverse_transform(y1_hat_cate)

            y1_hat_cates.append(y1_hat_cate)
        # ***

        # *** CATE estimator ***
        cate_hats = []
        for p0_id, p0 in enumerate(ParameterGrid(self.params0)):
            for p1_id, p1 in enumerate(ParameterGrid(self.params1)):
                cate_hat = y1_hat_cates[p1_id] - y0_hat_cates[p0_id]
                cate_hats.append(cate_hat)
        # ***

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

    def save_params_info(self):
        # Individual (id, params) pairs per model.
        df_p0 = get_params_df(self.params0)
        df_p1 = get_params_df(self.params1)

        # Keep the same order as in 'run()'.
        p_global_id = 1
        params_mapping = []
        for p0_id, _ in enumerate(ParameterGrid(self.params0)):
            for p1_id, _ in enumerate(ParameterGrid(self.params1)):
                params_mapping.append([p_global_id, p0_id+1, p1_id+1])
                p_global_id += 1
        df_all = pd.DataFrame(params_mapping, columns=['id', 'm0', 'm1'])

        df_p0.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m0_params.csv'), index=False)
        df_p1.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_m1_params.csv'), index=False)
        df_all.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_cate_params.csv'), index=False)

class TSSearch():
    def __init__(self, opt):
        self.opt = opt
        self.params_base = get_params(opt.base_model)
    
    def run(self, train, test, scaler, iter_id, fold_id):
        X_tr = train[0]
        t_tr = train[1].flatten()
        y_tr = train[2].flatten()

        X0_tr = X_tr[t_tr < 1]
        X1_tr = X_tr[t_tr > 0]
        y0_tr = y_tr[t_tr < 1]
        y1_tr = y_tr[t_tr > 0]

        X_test = test[0]
        t_test = test[1].flatten()
        X0_test = X_test[t_test < 1]
        X1_test = X_test[t_test > 0]

        input_size = X0_tr.shape[1]

        y0_hats = []
        y1_hats = []
        cate_hats = []
        for params in ParameterGrid(self.params_base):
            m0 = get_mlp_reg(input_size, params)
            m1 = get_mlp_reg(input_size, params)

            m0.fit(X0_tr, y0_tr)
            m1.fit(X1_tr, y1_tr)

            # Factual predictions for model selection purposes (predict X[t==0]).
            y0_hat = m0.predict(X0_test)
            y0_hats.append(y0_hat)

            # Factual predictions for model selection purposes (predict X[t==1]).
            y1_hat = m1.predict(X1_test)
            y1_hats.append(y1_hat)

            # For CATE prediction purposes (predict ALL X).
            y0_hat_cate = m0.predict(X_test).reshape(-1, 1)
            y1_hat_cate = m1.predict(X_test).reshape(-1, 1)

            if self.opt.scale_y:
                y0_hat_cate = scaler.inverse_transform(y0_hat_cate)
                y1_hat_cate = scaler.inverse_transform(y1_hat_cate)

            cate_hat = y1_hat_cate - y0_hat_cate
            cate_hats.append(cate_hat)

        if fold_id > 0:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}_fold{fold_id}'
        else:
            filename = f'{self.opt.estimation_model}_{self.opt.base_model}_iter{iter_id}'
        
        y0_hats_arr = np.array(y0_hats, dtype=object)
        y1_hats_arr = np.array(y1_hats, dtype=object)
        cate_hats_arr = np.array(cate_hats, dtype=object)

        np.savez_compressed(os.path.join(self.opt.output_path, filename), y0_hat=y0_hats_arr, y1_hat=y1_hats_arr, cate_hat=cate_hats_arr)

    def save_params_info(self):
        df_params = get_params_df(self.params_base)
        df_params.to_csv(os.path.join(self.opt.output_path, f'{self.opt.estimation_model}_{self.opt.base_model}_params.csv'), index=False)