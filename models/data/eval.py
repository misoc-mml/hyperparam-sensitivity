import numpy as np

def rmse(a, b):
    """"
    Root mean squared error between two arrays
    """
    return np.sqrt(((a - b)**2).mean())

class Evaluator(object):
    """
    Class that provides some functionality to evaluate the results.

    Param:
    ------

    y :     array-like, shape=(num_samples)
            Observed outcome

    t :     array-like, shape=(num_samples)
            Binary array representing the presence (t[i] == 1) or absence (t[i]==0) of treatment

    y_cf :  array-like, shape=(num_samples) or None, optional
            Counterfactual outcome (i.e., what would the outcome be if !t

    mu0 :   array-like, shape=(num_samples) or None, optional
            Outcome if no treatment and in absence of noise

    mu1 :   array-like, shape=(num_samples) or None, optional
            Outcome if treatment and in absence of noise

    """
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        """"
        Root mean squared error of the Individual Treatment Effect (ITE)

        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case

        :return: the RMSE of the ITE

        """
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return rmse(self.true_ite, pred_ite)

    def abs_ate(self, ypred1, ypred0):
        """
        Absolute error for the Average Treatment Effect (ATE)
        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case
        :return: absolute ATE
        """
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        """
        Precision in Estimating the Heterogeneous Treatment Effect (PEHE)

        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case

        :return: PEHE
        """
        return rmse(self.mu1 - self.mu0, ypred1 - ypred0)

    def calc_stats(self, ypred1, ypred0):
        """
        Calculate some metrics

        :param ypred1: predicted outcome if treated
        :param ypred0: predicted outcome if not treated

        :return ite: RMSE of ITE
        :return ate: absolute error for the ATE
        :return pehe: PEHE
        """
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe

    def calc_stats_effect(self, pred_ite):
        ite = pehe = rmse(self.true_ite, pred_ite)
        ate = np.abs(np.mean(pred_ite) - np.mean(self.true_ite))
        return ite, ate, pehe
    
    def train_val_split(self, train_idx, val_idx):
        train_eval = Evaluator(self.y[train_idx], self.t[train_idx], y_cf=self.y_cf[train_idx], mu0=self.mu0[train_idx], mu1=self.mu1[train_idx])
        val_eval = Evaluator(self.y[val_idx], self.t[val_idx], y_cf=self.y_cf[val_idx], mu0=self.mu0[val_idx], mu1=self.mu1[val_idx])
        return train_eval, val_eval

