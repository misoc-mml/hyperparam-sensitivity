# Main running script

fixed_seed = 1
import random
import numpy as np
random.seed(fixed_seed)
np.random.seed(fixed_seed)

import os
import time
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path', type=str)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--seed', type=int, default=1)

    return parser

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

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    # Check if output folder exists and create if necessary.
    if not os.path.isdir(options.output_path):
        os.mkdir(options.output_path)

    # Initialise the logger (writes simultaneously to a file and the console).
    init_logger(options)
    logging.debug(options)