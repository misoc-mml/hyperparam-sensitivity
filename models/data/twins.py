import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .eval import EvaluatorTwins as Evaluator
from scipy.stats import sem

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

    def __init__(self, path_data="datasets/TWINS/csv", n_iterations=10):
        self.path_data = path_data
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.twins')
        # which features are binary
        # 50 and 51 are (unique values):
        # 50: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 17.]
        # 51: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 15.]
        self.contfeats = [192, 193]
        self.binfeats = [i for i in range(194) if i not in self.contfeats]

    def _get_train_valid_test(self, i):
        data = np.loadtxt(self.path_data + '/twins_' + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        x = data[:, 3:]
        idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.20, random_state=1)
        itr, iva = train_test_split(idxtrain, test_size=0.24, random_state=1)
        train = (x[itr], t[itr], y[itr], y_cf[itr])
        valid = (x[iva], t[iva], y[iva], y_cf[iva])
        test = (x[ite], t[ite], y[ite], y_cf[ite])

        return train, valid, test

    def _get_train_test(self, i, test_ratio=0.2):
        data = np.loadtxt(self.path_data + '/twins_' + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        x = data[:, 3:]
        itr, ite = train_test_split(np.arange(x.shape[0]), test_size=test_ratio, random_state=1)
        train = (x[itr], t[itr], y[itr], y_cf[itr])
        test = (x[ite], t[ite], y[ite], y_cf[ite])

        return train, test

    def get_train_xt(self, i, test_ratio=0.2):
        data = np.loadtxt(self.path_data + '/twins_' + str(i + 1) + '.csv', delimiter=',')
        t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        x = data[:, 3:]
        itr, _ = train_test_split(np.arange(x.shape[0]), test_size=test_ratio, random_state=1)
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

    def _get_batch(self, i, merge=False):
        if merge:
            train, test = self._get_train_test(i)
            (x_tr, t_tr, y_tr, y_cf_tr) = train
            (x_test, t_test, y_test, y_cf_test) = test
            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test)
            batch = (x_tr, t_tr, y_tr), (x_test, t_test, y_test), (evaluator_train, evaluator_test)
        else:
            train, valid, test = self._get_train_valid_test(i)
            (x_tr, t_tr, y_tr, y_cf_tr) = train
            (x_val, t_val, y_val, y_cf_val) = valid
            (x_test, t_test, y_test, y_cf_test) = test
            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test)
            batch = (x_tr, t_tr, y_tr), (x_val, t_val, y_val), (x_test, t_test, y_test), (evaluator_train, evaluator_test)
        
        return batch

    def get_processed_data(self, merge=False):
        for i in range(self.n_iterations):
            # Get another batch of data.
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

        # No scaling for Y as it's binary.

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

        # No scaling for Y as it's binary.

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
        score = evaluator_train.calc_stats(y1_tr, y0_tr)
        score_test = evaluator_test.calc_stats(y1_test, y0_test)
        return score, score_test
    
    def evaluate_batch_effect(self, estimates, evaluators):
        te_tr, te_test = estimates
        eval_tr, eval_test = evaluators
        score_tr = eval_tr.calc_stats_effect(te_tr)
        score_test = eval_test.calc_stats_effect(te_test)
        return score_tr, score_test

    def print_scores(self, i, iter_max, score, score_test):
        self.logger.info(f'Iteration: {i+1}/{iter_max}, train ATE: {score[0]:0.3f}, train PEHE: {score[1]:0.3f}, train AUC: {score[2]:0.3f}, train CF AUC: {score[3]:0.3f}, test ATE: {score_test[0]:0.3f}, test PEHE: {score_test[1]:0.3f}, test AUC: {score_test[2]:0.3f}, test CF AUC: {score_test[3]:0.3f}')
    
    def print_scores_agg(self, scores, scores_test):
        means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
        self.logger.info(f'train ATE: {means[0]:.3f}+-{stds[0]:.3f}, train PEHE: {means[1]:.3f}+-{stds[1]:.3f}, train AUC: {means[2]:.3f}+-{stds[2]:.3f}, train CF AUC: {means[3]:.3f}+-{stds[3]:.3f}')

        means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
        self.logger.info(f'test ATE: {means[0]:.3f}+-{stds[0]:.3f}, test PEHE: {means[1]:.3f}+-{stds[1]:.3f}, test AUC: {means[2]:.3f}+-{stds[2]:.3f}, test CF AUC: {means[3]:.3f}+-{stds[3]:.3f}')
    
    def get_score_headers(self):
        return ['train ATE', 'train PEHE', 'train AUC', 'train CF AUC', 'test ATE', 'test PEHE', 'test AUC', 'test CF AUC']