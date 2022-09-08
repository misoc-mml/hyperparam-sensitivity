import logging
import numpy as np
from .eval import EvaluatorJobs as Evaluator

class JOBS(object):
    """
    Class to load the JOBS dataset.

    Parameters
    ----------
    path_data :         str, default="datasets/JOBS"
                        Path to data

    n_iterations :      int, default=10
                        Number of simulations/runs
    """
    def __init__(self, path_data="datasets/JOBS", n_iterations=10):
        self.path_data = path_data
        # Load train and test data from disk
        self.train_data = np.load(self.path_data + "/jobs_DW_bin.new.10.train.npz")
        self.test_data = np.load(self.path_data +"/jobs_DW_bin.new.10.test.npz")
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.jobs')
        self.binfeats = [2, 3, 4, 5, 13, 14, 16]
        self.contfeats = [i for i in range(17) if i not in self.binfeats]
    
    def _split_data(self, arr, i):
        x = arr['x'][:, :, i]
        t = arr['t'][:, i].reshape(-1, 1)
        yf = arr['yf'][:, i].reshape(-1, 1)
        e = arr['e'][:, i].reshape(-1, 1)
        return x, t, yf, e

    def _get_train_test(self, i):
        train = self._split_data(self.train_data, i)
        test = self._split_data(self.test_data, i)
        return train, test

    def get_train_xt(self, i):
        x, t, y, e = self._split_data(self.train_data, i)
        return x, t

    def get_xty(self, data):
        x, t, y, e = data
        return x, t, y

    def get_eval(self, data):
        x, t, y, e = data
        return Evaluator(y, t, e)

    def get_eval_idx(self, data, idx):
        x, t, y, e = data
        return Evaluator(y[idx], t[idx], e[idx])