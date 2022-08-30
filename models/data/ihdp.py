import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .eval import Evaluator
from scipy.stats import sem

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

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:] # the Xs (features ar from row 5 onwards
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def _get_train_valid_test(self, i):
        '''
        train size = 57% (90*63/100)
        validation size = 24% of the training set  (val_size=90*27/100)
        According to Shalit et. al 2017. https://arxiv.org/pdf/1606.03976.pdf
        '''
        (x, t, y), (y_cf, mu_0, mu_1) = self.split_ihdp_dataset(self.arr_train, i)

        itr, iva = train_test_split(np.arange(x.shape[0]), test_size=0.24, random_state=1)
        train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
        valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
        test = self.split_ihdp_dataset(self.arr_test, i)
        return train, valid, test

    def _get_train_test(self, i):
        train = self.split_ihdp_dataset(self.arr_train, i)
        test = self.split_ihdp_dataset(self.arr_test, i)
        return train, test

    def get_train_xt(self, i):
        (x, t, y), _ = self.split_ihdp_dataset(self.arr_train, i)
        return x, t

    def _get_batch(self, i, merge=False):
        if merge:
            train, test = self._get_train_test(i)
            (x_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
            (x_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr, mu0=mu0_tr, mu1=mu1_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test, mu0=mu0_test, mu1=mu1_test)
            t_tr = t_tr.reshape(-1, 1)
            t_test = t_test.reshape(-1, 1)
            batch = (x_tr, t_tr, y_tr), (x_test, t_test, y_test), (evaluator_train, evaluator_test)
        else:
            train, valid, test = self._get_train_valid_test(i)
            (x_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
            (x_val, t_val, y_val), (y_cf_val, mu0_val, mu1_val) = valid
            (x_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr, mu0=mu0_tr, mu1=mu1_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test, mu0=mu0_test, mu1=mu1_test)
            t_tr = t_tr.reshape(-1, 1)
            t_val = t_val.reshape(-1, 1)
            t_test = t_test.reshape(-1, 1)
            batch = (x_tr, t_tr, y_tr), (x_val, t_val, y_val), (x_test, t_test, y_test), (evaluator_train, evaluator_test)

        return batch

    def split_ihdp_dataset(self, arr, i_rep):
        t, y, y_cf = arr['t'], arr['yf'], arr['ycf']
        mu_0, mu_1, Xs = arr['mu0'], arr['mu1'], arr['x']

        t, y, y_cf = t[:, i_rep][:, np.newaxis], y[:, i_rep][:, np.newaxis], y_cf[:, i_rep][:, np.newaxis]
        mu_0, mu_1, Xs = mu_0[:, i_rep][:, np.newaxis], mu_1[:, i_rep][:, np.newaxis], Xs[:, :, i_rep]
        Xs[:, 13] -= 1  # this binary feature is in {1, 2}
        return (Xs, t, y), (y_cf, mu_0, mu_1)
    
    def get_xty(self, data):
        (x, t, y), _ = data
        return x, t, y
    
    def get_eval(self, data):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0, mu_1)

    def get_eval_idx(self, data, idx):
        (x, t, y), (y_cf, mu_0, mu_1) = data
        return Evaluator(mu_0[idx], mu_1[idx])

    def get_processed_data(self, merge=False):
        for i in range(self.replications):
            yield self._get_batch(i, merge)

    def _scale_train_test(self, train, test, scalers, scale_bin=False, scale_y=False):
        x_tr, t_tr, y_tr = train
        x_test, t_test, y_test = test

        scaler_x, scaler_y = scalers

        if scale_bin:
            x_tr = scaler_x.fit_transform(x_tr)
            x_test = scaler_x.transform(x_test)
        else:
            x_tr[:, self.contfeats] = scaler_x.fit_transform(x_tr[:, self.contfeats])
            x_test[:, self.contfeats] = scaler_x.transform(x_test[:, self.contfeats])

        if scale_y:
            y_tr = scaler_y.fit_transform(y_tr)
            y_test = scaler_y.transform(y_test)

        return (x_tr, t_tr, y_tr), (x_test, t_test, y_test)

    def _scale_train_valid_test(self, train, valid, test, scalers, scale_bin=False, scale_y=False):
        x_tr, t_tr, y_tr = train
        x_val, t_val, y_val = valid
        x_test, t_test, y_test = test

        scaler_x, scaler_y = scalers

        if scale_bin:
            x_tr = scaler_x.fit_transform(x_tr)
            x_val = scaler_x.transform(x_val)
            x_test = scaler_x.transform(x_test)
        else:
            x_tr[:, self.contfeats] = scaler_x.fit_transform(x_tr[:, self.contfeats])
            x_val[:, self.contfeats] = scaler_x.transform(x_val[:, self.contfeats])
            x_test[:, self.contfeats] = scaler_x.transform(x_test[:, self.contfeats])

        if scale_y:
            y_tr = scaler_y.fit_transform(y_tr)
            y_val = scaler_y.transform(y_val)
            y_test = scaler_y.transform(y_test)

        return (x_tr, t_tr, y_tr), (x_val, t_val, y_val), (x_test, t_test, y_test)

    def scale_data(self, train, valid, test, scalers, scale_bin=False, scale_y=False):
        if valid is None:
            result = self._scale_train_test(train, test, scalers, scale_bin, scale_y)
        else:
            result = self._scale_train_valid_test(train, valid, test, scalers, scale_bin, scale_y)
        
        return result
    
    def evaluate_batch(self, estimates, scaler_y, evaluators):
        y0_tr, y1_tr, y0_test, y1_test = estimates
        evaluator_train, evaluator_test = evaluators
        score_tr = self._get_score(y0_tr, y1_tr, evaluator_train, scaler_y)
        score_test = self._get_score(y0_test, y1_test, evaluator_test, scaler_y)
        return score_tr, score_test
    
    def evaluate_batch_effect(self, estimates, evaluators):
        te_tr, te_test = estimates
        eval_tr, eval_test = evaluators
        score_tr = eval_tr.calc_stats_effect(te_tr)
        score_test = eval_test.calc_stats_effect(te_test)
        return score_tr, score_test

    def print_scores(self, i, iter_max, score, score_test):
        self.logger.info(f'Iteration: {i+1}/{iter_max}, train ITE: {score[0]:0.3f}, train ATE: {score[1]:0.3f}, train PEHE: {score[2]:0.3f}, test ITE: {score_test[0]:0.3f}, test ATE: {score_test[1]:0.3f}, test PEHE: {score_test[2]:0.3f}')

    def print_scores_agg(self, scores, scores_test):
        means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
        self.logger.info(f'train ITE: {means[0]:.3f}+-{stds[0]:.3f}, train ATE: {means[1]:.3f}+-{stds[1]:.3f}, train PEHE: {means[2]:.3f}+-{stds[2]:.3f}')

        means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
        self.logger.info(f'test ITE: {means[0]:.3f}+-{stds[0]:.3f}, test ATE: {means[1]:.3f}+-{stds[1]:.3f}, test PEHE: {means[2]:.3f}+-{stds[2]:.3f}')
    
    def get_score_headers(self):
        return ['train ITE', 'train ATE', 'train PEHE', 'test ITE', 'test ATE', 'test PEHE']

    def _get_score(self, y0, y1, evaluator, scaler):
        if len(y0.shape) == 1:
            y0 = y0[:, np.newaxis]
            y1 = y1[:, np.newaxis]
        if scaler:
            y0 = scaler.inverse_transform(y0)
            y1 = scaler.inverse_transform(y1)
        return evaluator.calc_stats(y1, y0)