# General purpose utility functions.

import os
import logging
import pandas as pd

from sklearn.model_selection import ParameterGrid

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

def get_params_df(params):
    # ParameterGrid is deterministic, so we can safely do this.
    # First, determine if all combinations have the same number of params.
    param_list = list(ParameterGrid(params))
    equal_len = True
    row_len = len(param_list[0])
    for params in ParameterGrid(params):
        if len(params) != row_len:
            equal_len = False
            break

    if equal_len:
        df_params = pd.DataFrame.from_records(param_list)
    else:
        df_params = pd.DataFrame(param_list, columns=['params'])

    df_params.insert(0, 'id', range(1, len(df_params) + 1))

    return df_params

def get_model_name(opt):
    if opt.estimation_model in ('two-head', 'cf'):
        return opt.estimation_model
    else:
        return f'{opt.estimation_model}_{opt.base_model}'