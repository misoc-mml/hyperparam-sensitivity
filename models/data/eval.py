import numpy as np
from sklearn.metrics import roc_auc_score

from helpers.metrics import abs_ate, pehe

class Evaluator(object):
    def __init__(self, mu0, mu1):
        self.mu0 = mu0.reshape(-1, 1) if mu0.ndim == 1 else mu0
        self.mu1 = mu1.reshape(-1, 1) if mu1.ndim == 1 else mu1
        self.cate_true = self.mu1 - self.mu0
        self.metrics = ['ate', 'pehe']

    def get_metrics(self, cate_hat):
        ate_val = abs_ate(self.cate_true, cate_hat)
        pehe_val = pehe(self.cate_true, cate_hat)
        return [ate_val, pehe_val]

class EvaluatorJobs(object):
    def __init__(self, yf, t, e):
        self.yf = yf
        self.t = t
        self.e = e
        self.metrics = ['att', 'policy']
        self.att = np.mean(self.yf[self.t > 0]) - np.mean(self.yf[(1 - self.t + self.e) > 1])
        self.t_e = self.t[self.e > 0]
        self.yf_e = self.yf[self.e > 0]

    def get_metrics(self, cate_hat):
        att_pred = np.mean(cate_hat[(self.t + self.e) > 1])
        bias_att = att_pred - self.att

        policy_value = self.policy_val(cate_hat[self.e > 0])

        # ATT, Policy
        return [np.abs(bias_att), 1 - policy_value]

    def policy_val(self, cate_hat):
        """
        Computes the value of the policy defined by predicted effect

        :param cate_hat: predicted effect (for the experimental data only)

        :return: policy value

        """
        # Consider only the cases for which we have experimental data (i.e., e > 0)

        if np.any(np.isnan(cate_hat)):
            return np.nan

        policy = cate_hat > 0.0
        policy_t = policy == self.t_e
        treat_overlap = policy_t * (self.t_e > 0)
        control_overlap = policy_t * (self.t_e < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(self.yf_e[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(self.yf_e[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value

class EvaluatorTwins(object):
    def __init__(self, y, t, y_cf):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = self.y * (1 - self.t) + self.y_cf * self.t
        self.mu1 = self.y * self.t + self.y_cf * (1 - self.t)
        self.cate_true = self.mu1 - self.mu0
        if self.cate_true.ndim == 1:
            self.cate_true = self.cate_true.reshape(-1, 1)
        self.metrics = ['ate', 'pehe']
        self.metrics_y = ['ate', 'pehe', 'auc', 'cf_auc']

    def get_metrics(self, cate_hat):
        ate_val = abs_ate(self.cate_true, cate_hat)
        pehe_val = pehe(self.cate_true, cate_hat)
        return [ate_val, pehe_val]

    def get_metrics_y(self, y0_hat, y1_hat):
        if y0_hat.ndim == 1:
            y0_hat = y0_hat.reshape(-1, 1)
        if y1_hat.ndim == 1:
            y1_hat = y1_hat.reshape(-1, 1)
        
        ate_val, pehe_val = self.get_metrics(y1_hat - y0_hat)
        auc, cf_auc = self._get_auc(y0_hat, y1_hat)
        return [ate_val, pehe_val, auc, cf_auc]

    def _get_auc(self, ypred0, ypred1):
        # Combined AUC (as in Yao et al.)
        # https://github.com/Osier-Yi/SITE/blob/master/simi_ite/evaluation.py
        y_label = np.concatenate((self.mu0, self.mu1), axis=0)
        y_label_pred = np.concatenate((ypred0, ypred1), axis=0)
        auc = roc_auc_score(y_label, y_label_pred)

        # Counterfactual AUC (as in Louizos et al.)
        y_cf = (1 - self.t) * ypred1 + self.t * ypred0
        cf_auc = roc_auc_score(self.y_cf, y_cf)  

        return auc, cf_auc