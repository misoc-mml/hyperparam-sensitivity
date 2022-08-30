import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .eval import EvaluatorJobs as Evaluator
from scipy.stats import sem

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
        self.train_data = np.load(self.path_data + "/jobs_DW_bin.train.npz")
        self.test_data = np.load(self.path_data +"/jobs_DW_bin.test.npz")
        self.n_iterations = n_iterations
        self.logger = logging.getLogger('models.data.jobs')
        self.binfeats = [2, 3, 4, 5, 13, 14, 16]
        self.contfeats = [i for i in range(17) if i not in self.binfeats]

    def _get_train_valid_test(self, merge=False, train_ratio=0.56, validation_ratio=0.24, test_ratio=0.2):
        """
        Returns train, validation and test sets from the dataset
        :param merge: Whether to merge the training and test sets for a new train/validation/test split
        :param train_ratio: Percentage of the dataset to use of training
        :param validation_ratio: Percentage of the dataset to use for validation
        :param test_ratio: Percentage of the dataset to use for testing (must be 1 - train_ratio - validation_ratio)
        :return: train, validation and test sets
        """
        if train_ratio + validation_ratio + test_ratio != 1:
            print("The sum of the train, validation and test ratios must be 1. Using default values:")
            train_ratio, validation_ratio, test_ratio = 0.56, 0.24, 0.2
            print("Percentage of dataset for training: %d%%" % int(train_ratio*100))
            print("Percentage of dataset for validation: %d%%" % int(validation_ratio * 100))
            print("Percentage of dataset for testing: %d%%" % int(test_ratio * 100))

        if merge:
            x = np.concatenate([np.squeeze(self.train_data["x"]), np.squeeze(self.test_data["x"])])  # features [nsamp, nfeat]
            e = np.concatenate([self.train_data["e"], self.test_data["e"]]) # [nsamp, 1]
            # 'e' indicates whether the instance belongs to the experimental data (1) or not (0).
            t = np.concatenate([self.train_data["t"], self.test_data["t"]])  # treatment (0 == no treatment) [ nsamp, 1]
            y = np.concatenate([self.train_data["yf"], self.test_data["yf"]])  # outcome [ nsamp, 1]

            # Train/test split
            idx_tr, idx_test = train_test_split(np.arange(x.shape[0]), test_size=1-train_ratio)
            # Divide the test split into validation and test
            idx_val, idx_test = train_test_split(idx_test, test_size=test_ratio / (test_ratio + validation_ratio))
            train = (x[idx_tr], t[idx_tr], y[idx_tr], e[idx_tr])  # Train set
            valid = (x[idx_val], t[idx_val], y[idx_val], e[idx_val])  # Validation set
            test = (x[idx_test], t[idx_test], y[idx_test], e[idx_test])  # Test set

        else:
            x = np.squeeze(self.train_data["x"])  # features
            e = self.train_data["e"]  # whether the instance belongs to the experimental data (1) or not (0).
            t = self.train_data["t"]  # treatment (0 == no treatment)
            y = self.train_data["yf"]  # outcome (binary classification); 1 == hired

            # Divide training set into training and validation sets
            idx_tr, idx_val = train_test_split(np.arange(x.shape[0]), test_size=1-train_ratio)
            train = (x[idx_tr], t[idx_tr], y[idx_tr], e[idx_tr])
            valid = (x[idx_val], t[idx_val], y[idx_val], e[idx_val])
            # Test set
            test = (np.squeeze(self.test_data["x"]), self.test_data["t"], self.test_data["yf"], self.test_data["e"])

        return train, valid, test
    
    def _get_train_test(self, i=0, merge=False, test_ratio=0.2):
        if merge:
            x = np.concatenate([np.squeeze(self.train_data["x"]), np.squeeze(self.test_data["x"])])
            e = np.concatenate([self.train_data["e"], self.test_data["e"]])
            t = np.concatenate([self.train_data["t"], self.test_data["t"]])
            y = np.concatenate([self.train_data["yf"], self.test_data["yf"]])

            # fix seed?
            idx_tr, idx_test = train_test_split(np.arange(x.shape[0]), test_size=test_ratio)
            train = (x[idx_tr], t[idx_tr], y[idx_tr], e[idx_tr])  # Train set
            test = (x[idx_test], t[idx_test], y[idx_test], e[idx_test])  # Test set
        else:
            train = (np.squeeze(self.train_data["x"]), self.train_data["t"], self.train_data["yf"], self.train_data["e"])
            test = (np.squeeze(self.test_data["x"]), self.test_data["t"], self.test_data["yf"], self.test_data["e"])

        return train, test

    # 'i' param just for compatibility with other datasets.
    def get_train_xt(self, i=0):
        return np.squeeze(self.train_data["x"]), self.train_data["t"]

    def get_xty(self, data):
        x, t, y, e = data
        return x, t, y

    def get_eval(self, data):
        x, t, y, e = data
        return Evaluator(y, t, e)

    def get_eval_idx(self, data, idx):
        x, t, y, e = data
        return Evaluator(y[idx], t[idx], e[idx])

    def _get_batch(self, merge=False):
        if merge:
            train, test = self._get_train_test(merge=False)
            (x_tr, t_tr, y_tr, e_tr) = train  # training set
            (x_test, t_test, y_test, e_test) = test  # test set
            evaluator_train = Evaluator(y_tr.flatten(), t_tr.flatten(), e_tr.flatten())
            evaluator_test = Evaluator(y_test.flatten(), t_test.flatten(), e_test.flatten())
            batch = (x_tr, t_tr, y_tr), (x_test, t_test, y_test), (evaluator_train, evaluator_test)
        else:
            train, valid, test = self._get_train_valid_test()
            (x_tr, t_tr, y_tr, e_tr) = train  # training set
            (x_val, t_val, y_val, e_val) = valid  # validation set
            (x_test, t_test, y_test, e_test) = test  # test set
            evaluator_train = Evaluator(y_tr.flatten(), t_tr.flatten(), e_tr.flatten())
            evaluator_test = Evaluator(y_test.flatten(), t_test.flatten(), e_test.flatten())
            batch = (x_tr, t_tr, y_tr), (x_val, t_val, y_val), (x_test, t_test, y_test), (evaluator_train, evaluator_test)

        return batch

    def get_processed_data(self, merge=False):
        for _ in range(self.n_iterations):
            # Get another batch of data.
            yield self._get_batch(merge)

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
        yf_tr, ycf_tr, yf_test, ycf_test = estimates
        evaluator_train, evaluator_test = evaluators
        score = evaluator_train.calc_stats(yf_tr.flatten(), ycf_tr.flatten())
        score_test = evaluator_test.calc_stats(yf_test.flatten(), ycf_test.flatten())
        return score, score_test
    
    def evaluate_batch_effect(self, estimates, evaluators):
        te_tr, te_test = estimates
        eval_tr, eval_test = evaluators
        score_tr = eval_tr.calc_stats_effect(te_tr.flatten())
        score_test = eval_test.calc_stats_effect(te_test.flatten())
        return score_tr, score_test

    def evaluate_batch_pol(self, estimates, evaluators):
        pol_tr, pol_test = estimates
        evaluator_train, evaluator_test = evaluators
        r_pol = evaluator_train.calc_r_pol(pol_tr.flatten())
        r_pol_test = evaluator_test.calc_r_pol(pol_test.flatten())
        return (0.0, r_pol), (0.0, r_pol_test)

    def evaluate_batch_pol_alt(self, estimates, evaluators):
        pol_tr, pol_test = estimates
        evaluator_train, evaluator_test = evaluators
        r_pol = evaluator_train.calc_r_pol(pol_tr.flatten())
        r_pol_test = evaluator_test.calc_r_pol(pol_test.flatten())

        r_pol_all = evaluator_train.calc_r_pol_all(pol_tr.flatten())
        r_pol_all_test = evaluator_test.calc_r_pol_all(pol_test.flatten())

        return (r_pol_all, r_pol), (r_pol_all_test, r_pol_test)

    def print_scores(self, i, iter_max, score, score_test):
        self.logger.info(f'Iteration: {i+1}/{iter_max}, train ATT: {score[0]:0.3f}, train policy: {score[1]:0.3f}, test ATT: {score_test[0]:0.3f}, test policy: {score_test[1]:0.3f}')
    
    def print_scores_agg(self, scores, scores_test):
        means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
        self.logger.info(f'train ATT: {means[0]:.3f}+-{stds[0]:.3f}, train policy: {means[1]:.3f}+-{stds[1]:.3f}')

        means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
        self.logger.info(f'test ATT: {means[0]:.3f}+-{stds[0]:.3f}, test policy: {means[1]:.3f}+-{stds[1]:.3f}')
    
    def get_score_headers(self):
        return ['train ATT', 'train policy', 'test ATT', 'test policy']
    
    def get_score_headers_alt(self):
        return ['train policy_all', 'train policy', 'test policy_all', 'test policy']