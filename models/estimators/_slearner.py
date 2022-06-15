from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from ._common import get_params, get_regressor

class SSearch():
    def __init__(self, opt):
        self.opt = opt
        self.model = get_regressor(self.base_model)
        self.params_grid = get_params(self.base_model)
    
    def run(self, train, test, splits, iter_id):
        for params in ParameterGrid(self.params_grid):
            self.model.set_params(params)

            # cross_val_pred(self.model, train, splits, opt)

            # scale train and test
            cate_model = clone(self.model)
            # fit the model on train data
            # make cate predictions on test data
            # save cate predictions