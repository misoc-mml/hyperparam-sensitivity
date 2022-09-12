import os
import logging
import numpy as np
from .eval import Evaluator
from scipy.sparse import coo_matrix

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname, "rb"), delimiter=",")
    H = E[0, :]
    n = int(H[0])
    d = int(H[1])
    E = E[1:, :]
    S = coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
    K = S.tocsr()
    return K

class NEWS(object):
    """
    Class for the NEWS dataset.

    Parameters
    ----------
    path_data    :  str, default="datasets/NEWS/csv"
                    Path to data
    n_iterations       :  int, default=10
                    Number of simulations/runs
    """
    def __init__(self, path_data="datasets/NEWS", n_iterations=10, static_splits=False):
        self.path_data = path_data
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.news')
        self.binfeats = []
        self.contfeats = []
        self.static_splits = static_splits
        if static_splits:
            self.splits_file = np.load(os.path.join(self.path_data, 'news_splits_10iters.npz'), allow_pickle=True)

    def _load_batch(self, i):
        x = load_sparse(self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv.x')  # features
        f = open(self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv.y', 'rb')
        data2 = np.loadtxt(f, delimiter=",")
        f.close()
        t, y, y_cf, mu_0, mu_1 = data2[:, 0], data2[:, 1], data2[:, 2], data2[:, 3], data2[:, 4]
        # t is the treatment
        # y is the observed outcome
        # y_cf is the counterfactual outcome
        # mu_0 is the correct outcome if no treatment (in absence of noise)
        # mu_1 is the correct outcome if treatment is administered (in absence of noise)

        return x.A, t.reshape(-1, 1), y.reshape(-1, 1), y_cf, mu_0, mu_1

    def get_rows_count(self, i):
        x = load_sparse(self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv.x')
        return x.shape[0]
    
    def _get_train_test(self, i):
        x, t, y, y_cf, mu_0, mu_1 = self._load_batch(i)
        itr, ite = self.splits_file['train'][i].astype(int), self.splits_file['test'][i].astype(int)
        train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
        test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])

        return train, test

    def get_train_xt(self, i):
        x, t, y, y_cf, mu_0, mu_1 = self._load_batch(i)
        itr = self.splits_file['train'][i].astype(int)
        return x[itr], t[itr]

    def get_xty(self, data):
        (x, t, y), _ = data
        return x, t, y
    
    def get_eval(self, data):
        _, (y_cf, mu0, mu1) = data
        return Evaluator(mu0, mu1)
    
    def get_eval_idx(self, data, idx):
        _, (y_cf, mu0, mu1) = data
        return Evaluator(mu0[idx], mu1[idx])