import os
import logging
import numpy as np
from .eval import EvaluatorTwins as Evaluator

class TWINS(object):
    """
        Class for the TWINS dataset.

        Parameters
        ----------
        path_data    :  str, default="datasets/TWINS/csv"
                        Path to data
        n_iterations :  int, default=10
                        Number of simulations/runs
        """

    def __init__(self, path_data="datasets/TWINS/csv", n_iterations=10, static_splits=False):
        self.path_data = path_data
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.twins')
        # which features are binary
        # 50 and 51 are (unique values):
        # 50: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 17.]
        # 51: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 15.]
        self.contfeats = [192, 193]
        self.binfeats = [i for i in range(194) if i not in self.contfeats]
        self.static_splits = static_splits
        if static_splits:
            self.splits_file = np.load(os.path.join(self.path_data, 'twins_splits_10iters.npz'), allow_pickle=True)

    def _load_batch(self, i):
        data = np.loadtxt(self.path_data + '/twins_' + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        x = data[:, 3:]
        return x, t, y, y_cf

    def get_rows_count(self, i):
        data = np.loadtxt(self.path_data + '/twins_' + str(i + 1) + '.csv', delimiter=',')
        x = data[:, 3:]
        return x.shape[0]

    def _get_train_test(self, i):
        x, t, y, y_cf = self._load_batch(i)
        itr, ite = self.splits_file['train'][i].astype(int), self.splits_file['test'][i].astype(int)
        train = (x[itr], t[itr], y[itr], y_cf[itr])
        test = (x[ite], t[ite], y[ite], y_cf[ite])
        return train, test

    def get_train_xt(self, i):
        x, t, y, y_cf = self._load_batch(i)
        itr = self.splits_file['train'][i].astype(int)
        return x[itr], t[itr]

    def get_xty(self, data):
        (x, t, y, y_cf) = data
        return x, t, y
    
    def get_eval(self, data):
        (x, t, y, y_cf) = data
        return Evaluator(y, t, y_cf)
    
    def get_eval_idx(self, data, idx):
        (x, t, y, y_cf) = data
        return Evaluator(y[idx], t[idx], y_cf[idx])