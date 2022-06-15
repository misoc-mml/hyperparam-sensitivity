# Helper functions for the running scripts.

import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def init_logger(options):
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(options.output_path, 'info.log'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def xt_from_x(x):
    xt0 = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    xt1 = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xt0, xt1

def get_scaler(name):
    result = None
    if name == 'minmax':
        result = MinMaxScaler(feature_range=(-1, 1))
    elif name == 'std':
        result = StandardScaler()
    else:
        raise ValueError('Unknown scaler type.')
    return result