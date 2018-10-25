import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import logging
from sklearn import metrics
from utils import utilities, data_generator
import core
import inference_rt

# import keras
# from keras.models import Model
# from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
#                           Activation, Concatenate, LSTM, GRU, Reshape)
# import keras.backend as K
# from keras.optimizers import Adam

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

import vim_params as vp

def train(args):
    # model_type = args.model_type
    # time_steps = 10
    # freq_bins = 128
    # classes_num = 527

    # # Hyper parameters
    # hidden_units = 1024
    # drop_rate = 0.5

    args.batch_size = vp.BATCH_SIZE    #128, 192

    # Train
    core.train(args)


# Main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--workspace', type=str, required=True)

    parser.add_argument('--mini_data', action='store_true',
                        default=False)

    parser.add_argument('--quantize', action='store_true',
                        default=False)  # quantize the model

    parser.add_argument('--balance_type', type=str,
                        default='balance_in_batch',
                        choices=['no_balance', 'balance_in_batch'])

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['decision_level_max_pooling', 
                                 'decision_level_average_pooling', 
                                 'decision_level_single_attention',
                                 'decision_level_multi_attention',
                                 'feature_level_attention'])

    parser.add_argument('--learning_rate', type=float, default=1e-3)

    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')
    parser_get_avg_stats = subparsers.add_parser('get_avg_stats')
    parser_inference = subparsers.add_parser('inference_rt')

    args = parser.parse_args()

    args.filename = utilities.get_filename(__file__)

    # Logs
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    utilities.create_folder(logs_dir)
    logging = utilities.create_logging(logs_dir, filemode='w')

    logging.info(os.path.abspath(__file__))
    logging.info(args)

    if args.mode == "train":
        train(args)

    elif args.mode == 'get_avg_stats':
        args.bgn_iteration = 10000
        args.fin_iteration = 50001
        args.interval_iteration = 5000
        utilities.get_avg_stats(args)

    elif args.mode == 'inference_rt':
        inference_rt.core(args)

    else:
        raise Exception("Error!")
