import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .eval import Evaluator
from scipy.stats import sem
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
    def __init__(self, path_data="datasets/NEWS", n_iterations=10):
        self.path_data = path_data
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.news')
        self.binfeats = []
        self.contfeats = []

    def __iter__(self):
        for i in range(self.n_iterations):
            x = load_sparse(self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv.x')  # features
            f = open(self.path_data + '/topic_doc_mean_n5000_k3477_seed_' + str(i + 1) + '.csv.y','rb')
            data2 = np.loadtxt(f, delimiter=",")
            f.close()
            t, y, y_cf, mu_0, mu_1 = data2[:, 0], data2[:, 1], data2[:, 2], data2[:, 3], data2[:, 4]
            yield (x, t, y), (y_cf, mu_0, mu_1)

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

        return x, t, y, y_cf, mu_0, mu_1

    def _get_batch(self, i, merge=False, validation_ratio=0.3, test_ratio=0.1):
        if merge:
            train, test = self._get_train_test(i, test_ratio)
            (x_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
            (x_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test

            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr, mu0=mu0_tr, mu1=mu1_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test, mu0=mu0_test, mu1=mu1_test)

            # Convert matrices to ndarrays.
            x_tr = x_tr.A
            x_test = x_test.A

            # Y and T are vectors - convert them to (n, 1) ndarrays.
            y_tr = y_tr.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            t_tr = t_tr.reshape(-1, 1)
            t_test = t_test.reshape(-1, 1)

            batch = (x_tr, t_tr, y_tr), (x_test, t_test, y_test), (evaluator_train, evaluator_test)
        else:
            train, valid, test = self._get_train_valid_test(i, validation_ratio, test_ratio)
            (x_tr, t_tr, y_tr), (y_cf_tr, mu0_tr, mu1_tr) = train
            (x_val, t_val, y_val), (y_cf_val, mu0_val, mu1_val) = valid
            (x_test, t_test, y_test), (y_cf_test, mu0_test, mu1_test) = test
            evaluator_train = Evaluator(y_tr, t_tr, y_cf=y_cf_tr, mu0=mu0_tr, mu1=mu1_tr)
            evaluator_test = Evaluator(y_test, t_test, y_cf=y_cf_test, mu0=mu0_test, mu1=mu1_test)

            # Convert matrices to ndarrays.
            x_tr = x_tr.A
            x_val = x_val.A
            x_test = x_test.A

            # Y and T are vectors - convert them to (n, 1) ndarrays.
            y_tr = y_tr.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            t_tr = t_tr.reshape(-1, 1)
            t_val = t_val.reshape(-1, 1)
            t_test = t_test.reshape(-1, 1)

            batch = (x_tr, t_tr, y_tr), (x_val, t_val, y_val), (x_test, t_test, y_test), (evaluator_train, evaluator_test)

        return batch

    def _get_train_valid_test(self, i, validation_ratio=0.3, test_ratio=0.1):
        """
        Returns train, validation and test sets from the dataset
        :param merge: Whether to merge the training and test sets for a new train/validation/test split
        :param validation_ratio: Percentage of the training set to use for validation
        :param test_ratio: Percentage of the dataset to use for testing
        :return: train, validation and test sets
        """
        x, t, y, y_cf, mu_0, mu_1 = self._load_batch(i)
        idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=test_ratio, random_state=1)
        itr, iva = train_test_split(idxtrain, test_size=validation_ratio, random_state=1)
        train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
        valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
        test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])

        return train, valid, test
    
    def _get_train_test(self, i, test_ratio=0.1):
        x, t, y, y_cf, mu_0, mu_1 = self._load_batch(i)
        itr, ite = train_test_split(np.arange(x.shape[0]), test_size=test_ratio, random_state=1)
        train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
        test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])

        return train, test

    # TODO: generate static train/test splits.
    def get_train_xt(self, i, test_ratio=0.1):
        x, t, y, y_cf, mu_0, mu_1 = self._load_batch(i)
        itr, _ = train_test_split(np.arange(x.shape[0]), test_size=test_ratio, random_state=1)
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

    def get_processed_data(self, merge=False):
        for i in range(self.n_iterations):
            # Get another batch of data.
            yield self._get_batch(i, merge)

    def _scale_train_test(self, train, test, scalers, scale_bin=False, scale_y=False):
        x_tr, t_tr, y_tr = train
        x_test, t_test, y_test = test

        scaler_x, scaler_y = scalers

        # X in fact describes word counts, but we'll use this flag for conditional scaling anyway.
        if scale_bin:
            x_tr = scaler_x.fit_transform(x_tr)
            x_test = scaler_x.transform(x_test)

        if scale_y:
            y_tr = scaler_y.fit_transform(y_tr)
            # Not necessary as scaled y_test isn't used anywhere, but we're doing it for consistency with the rest.
            y_test = scaler_y.transform(y_test)
        
        return (x_tr, t_tr, y_tr), (x_test, t_test, y_test)

    def _scale_train_valid_test(self, train, valid, test, scalers, scale_bin=False, scale_y=False):
        x_tr, t_tr, y_tr = train
        x_val, t_val, y_val = valid
        x_test, t_test, y_test = test

        scaler_x, scaler_y = scalers

        # X in fact describes word counts, but we'll use this flag for conditional scaling anyway.
        if scale_bin:
            x_tr = scaler_x.fit_transform(x_tr)
            x_val = scaler_x.transform(x_val)
            x_test = scaler_x.transform(x_test)

        if scale_y:
            y_tr = scaler_y.fit_transform(y_tr)
            y_val = scaler_y.transform(y_val)
            # Not necessary as scaled y_test isn't used anywhere, but we're doing it for consistency with the rest.
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
        score = self._get_score(y0_tr, y1_tr, evaluator_train, scaler_y)
        score_test = self._get_score(y0_test, y1_test, evaluator_test, scaler_y)
        return score, score_test

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
        if scaler:
            y0 = scaler.inverse_transform(y0.reshape(-1, 1))
            y1 = scaler.inverse_transform(y1.reshape(-1, 1))
        return evaluator.calc_stats(y1.reshape(-1), y0.reshape(-1))