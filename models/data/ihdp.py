import logging
import numpy as np
from .eval import Evaluator

class IHDP(object):
    def __init__(self, path_data="datasets/IHDP", replications=1000):
        self.path_data = path_data
        self.path_data_train = path_data + '/ihdp_npci_1-1000.train.npz'
        self.path_data_test = path_data + '/ihdp_npci_1-1000.test.npz'
        self.arr_train = np.load(self.path_data_train)
        self.arr_test = np.load(self.path_data_test)
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        self.logger = logging.getLogger('models.data.ihdp')

    def split_ihdp_dataset(self, arr, i_rep):
        t, y, y_cf = arr['t'], arr['yf'], arr['ycf']
        mu_0, mu_1, Xs = arr['mu0'], arr['mu1'], arr['x']

        t, y, y_cf = t[:, i_rep][:, np.newaxis], y[:, i_rep][:, np.newaxis], y_cf[:, i_rep][:, np.newaxis]
        mu_0, mu_1, Xs = mu_0[:, i_rep][:, np.newaxis], mu_1[:, i_rep][:, np.newaxis], Xs[:, :, i_rep]
        Xs[:, 13] -= 1  # this binary feature is in {1, 2}
        return (Xs, t, y), (y_cf, mu_0, mu_1)

    def _get_train_test(self, i):
        train = self.split_ihdp_dataset(self.arr_train, i)
        test = self.split_ihdp_dataset(self.arr_test, i)
        return train, test

    def get_train_xt(self, i):
        (x, t, y), _ = self.split_ihdp_dataset(self.arr_train, i)
        return x, t

    def get_xty(self, data):
        (x, t, y), _ = data
        return x, t, y
    
    def get_eval(self, data):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0, mu_1)

    def get_eval_idx(self, data, idx):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0[idx], mu_1[idx])