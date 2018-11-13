"""
Summary:  DCASE 2017 task 4 Large-scale weakly supervised 
          sound event detection for smart cars. Ranked 1 in DCASE 2017 Challenge.
Author:   Yong Xu, Qiuqiang Kong
Created:  03/04/2017
Modified: 31/10/2017
"""
from __future__ import print_function 
import sys
try:
    import cPickle
except BaseException:
    print('cPickle not found, use _pickle instead ...')
    import _pickle as cPickle

import numpy as np
import argparse
import glob
import time
import os

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Input, merge   #, Merge not found
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Activation
from keras.layers.normalization import BatchNormalization
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random
from keras.optimizers import Adam

# import config as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator, QueueDataGenerator
# from evaluation import io_task4, evaluate

print(sys.path[0])
sys.path.insert(1, os.path.join(sys.path[0], '../'))
sys.path.append(os.path.join(sys.path[0], '../vim/'))
from vim import vim_params as vp
from vim import tasksmq
import pandas as pd
from utils import utilities
import tensorflow as tf
from sklearn import metrics
import logging

def evaluate(model, input, target, stats_dir, probs_dir, iteration, labels_map):
    """Evaluate a model.

    Args:
      model: object
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      stats_dir: str, directory to write out statistics.
      probs_dir: str, directory to write out output (samples_num, classes_num)
      iteration: int

    Returns:
      None
    """

    utilities.create_folder(stats_dir)
    utilities.create_folder(probs_dir)

    # Predict presence probabilittarget
    callback_time = time.time()
    # (clips_num, time_steps, freq_bins) = input.shape
    # (input, target) = utilities.transform_data(input, target)

    output = model.predict(input)
    output = output.astype(np.float32)  # (clips_num, classes_num)

    # Write out presence probabilities
    prob_path = os.path.join(probs_dir, "prob_{}_iters.p".format(iteration))
    cPickle.dump(output, open(prob_path, 'wb'))

    # Calculate statistics
    stats = utilities.calculate_stats(output, target, labels_map)

    # Write out statistics
    stat_path = os.path.join(stats_dir, "stat_{}_iters.p".format(iteration))
    cPickle.dump(stats, open(stat_path, 'wb'))

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    logging.info(
        "mAP: {:.6f}, AUC: {:.6f}, Callback time: {:.3f} s".format(
            mAP, mAUC, time.time() - callback_time))

    if False:
        logging.info("Saveing prob to {}".format(prob_path))
        logging.info("Saveing stat to {}".format(stat_path))

    print('mAP: ', mAP)
    return mAP, mAUC, [stat['AP'] for stat in stats]

# def f1(y_true, y_pred):
#     def recall(y_true, y_pred):
#         """Recall metric.

#         Only computes a batch-wise average of recall.

#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         """Precision metric.

#         Only computes a batch-wise average of precision.

#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

# Train model
def train(args):
    data_dir = args.data_dir
    workspace = args.workspace
    balance_type = args.balance_type
    filename = args.filename
    model_type = args.model_type

    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type),
                           'sr={}'.format(vp.feature_sr))

    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)

    # 
    num_classes = vp.TOTAL_NUM_CLASS
    batch_size = vp.BATCH_SIZE

    df = pd.read_csv(vp.FILE_CLASS_LABELS)
    labels_dict = {}
    labels_dict['name'] = np.array(df[df['transfer'] == 1]['display_name'])
    labels_dict['id'] = np.array(df[df['transfer'] == 1]['index'])
    labels_dict['count'] = []

    # Load training & testing data
    # (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    # (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    # print("tr_x.shape: %s" % (tr_x.shape,))

    # # Scale data
    # tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    # te_x = do_scale(te_x, args.scaler_path, verbose=1)
    # (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)

    # Load testing data
    (test_x, test_y) = load_test_data(data_dir, labels_dict)

    # Build model

    # Data generator
    gen = QueueDataGenerator(batch_size=batch_size, type='train')
    batch_x, _ = next(gen.generate())
    (_, n_time, n_freq) = batch_x.shape        # model change dynamically
    print('n_time ', n_time, 'n_freq ', n_freq)

    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    print(input_logmel)

    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((n_time, 256))(a1) # (N, 240, 256) bugfix: replace n_time for 240
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])

    model = Model(input_logmel, out)

    # del model
    # # load json and create model

    # save_out_path = os.path.join(
    #     models_dir, "md_{}_iters.h5".format(18000))
    # model = load_model(save_out_path)

    # # json_file = open(save_out_path + '.json', 'r')
    # # loaded_model_json = json_file.read()
    # # json_file.close()
    # # model = model_from_json(loaded_model_json)
    # # # load weights into new model
    # # model.load_weights(save_out_path + '.weights.h5')
    # print("Loaded model from disk")

    model.summary()

    # Compile model
    # optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    iteration = 0
    call_freq = 500
    train_time = time.time()

    for (batch_x, batch_y) in gen.generate():

        # Compute stats every several interations
        if (iteration % call_freq == 0) and (iteration != 0):

            logging.info("------------------")

            logging.info(
                "Iteration: {}, train time: {:.3f} s".format(
                    iteration, time.time() - train_time))

            # logging.info("Balance train statistics:")
            # evaluate(
            #     model=model,
            #     input=bal_train_x,
            #     target=bal_train_y,
            #     stats_dir=os.path.join(stats_dir, 'bal_train'),
            #     probs_dir=os.path.join(probs_dir, 'bal_train'),
            #     iteration=iteration)

            logging.info("Test statistics:")
            mAP, _, AP = evaluate(
                model=model,
                input=test_x,
                target=test_y,
                stats_dir=os.path.join(stats_dir, "test"),
                probs_dir=os.path.join(probs_dir, "test"),
                iteration=iteration,
                labels_map=labels_dict['id'])

            labels_dict['AP'] = AP
            for (name, ap) in zip(labels_dict['name'], labels_dict['AP']):
                print(name, '\t', ap)            
            train_time = time.time()

        # Update params
        # (batch_x, batch_y) = utilities.transform_data(batch_x, batch_y) !!!!
        cost = model.train_on_batch(x=batch_x, y=batch_y)
        if (iteration % 10) == 0:
            print(iteration,':', cost)

        iteration += 1
        
        # Save model
        if (iteration % 500) == 0:
            # save_out_path = os.path.join(
            #     models_dir, "md_{}_iters.h5".format(iteration))
            # model.save(save_out_path)

            # serialize model to JSON
            save_out_path = os.path.join(
                models_dir, "model.{}".format(iteration))
            model_json = model.to_json()
            with open(save_out_path + '.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(save_out_path + '.weights.h5')
            print("Saved model to disk")

        # Stop training when maximum iteration achieves
        if iteration == 50001:
            break

    # def as_keras_metric(method):
    #     import functools
    #     from keras import backend as K
    #     import tensorflow as tf
    #     @functools.wraps(method)
    #     def wrapper(self, args, **kwargs):
    #         """ Wrapper for turning tensorflow metrics into keras metrics """
    #         value, update_op = method(self, args, **kwargs)
    #         K.get_session().run(tf.local_variables_initializer())
    #         with tf.control_dependencies([update_op]):
    #             value = tf.identity(value)
    #         return value
    #     return wrapper

    # precision = as_keras_metric(tf.metrics.precision)
    # recall = as_keras_metric(tf.metrics.recall)
    
    # @as_keras_metric
    # def auc_pr(y_true, y_pred, curve='PR'):
    #     return tf.metrics.auc(y_true, y_pred, curve=curve)

    # @as_keras_metric
    # def mAP(y_true, y_pred):
    #     _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, 5)
    #     return m_ap

    #     return metrics.average_precision_score(y_true, y_pred, average=None)
        # _, m_ap = tf.metrics.average_precision_at_k(y_true, y_pred, k)         k= 5           # return m_ap

    #   metrics=['accuracy', precision, recall, auc_pr])  # 'accuracy', mAP
    #   metrics=['accuracy', mAP])  # 'accuracy', mAP

    # Save model callback
    # filepath = os.path.join('/work/audio/audioset_classification/work/models', 
    #     "gatedAct_rationBal44_lr0.001_normalization_at_cnnRNN_64newMel_240fr.{epoch:02d}-{val_acc:.4f}.hdf5")
    # create_folder(os.path.dirname(filepath))
    # save_model = ModelCheckpoint(filepath=filepath,
    #                              monitor='val_acc', 
    #                              verbose=0,
    #                              save_best_only=False,
    #                              save_weights_only=False,
    #                              mode='auto',
    #                              period=1)  # each epoch??? 

    # Data generator
    # gen = RatioDataGenerator(batch_size=44, type='train')

    # Train
    # model.fit_generator(generator=gen.generate(), 
    #                     steps_per_epoch=500,    # iters is called an 'epoch', old val: 100
    #                     epochs=101,              # Maximum 'epoch' to train, old val: 31
    #                     verbose=1, 
    #                     callbacks=[save_model], 
    #                     validation_data=(te_x, te_y))

def load_test_data(data_dir, labels_dict):
    labels_map_mask = [False] * vp.TOTAL_NUM_CLASS
    for x in labels_dict['id']:
        labels_map_mask[x] = True

    # Load data
    load_time = time.time()

    # train_x = []
    # train_y = []
    # train_id_list = []
    test_x = []
    test_y = []
    test_id_list = []

    # early_stop = 0
    for aclass in labels_dict['name']:
        print(aclass)

    #     local_train_x = []
    #     local_train_y = []
    #     local_train_id_list = []
        local_test_x = []
        local_test_y = []
        local_test_id_list = []

    #     # Path of hdf5 data
    #     bal_train_hdf5_path = os.path.join(data_dir, aclass, "balanced_train_segments.hdf5")
    #     unbal_train_hdf5_path = os.path.join(data_dir, aclass, "unbalanced_train_segments.hdf5")
        test_hdf5_path = os.path.join(data_dir, aclass, "eval_segments.hdf5")

    #     if mini_data:
    #         # Only load balanced data
    #         (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
    #             bal_train_hdf5_path)

    #         local_train_x = bal_train_x
    #         local_train_y = bal_train_y
    #         local_train_id_list = bal_train_id_list

    #     else:
    #         # Load both balanced and unbalanced data
    #         (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
    #             bal_train_hdf5_path)

    #         (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
    #             unbal_train_hdf5_path)

    #         local_train_x = np.concatenate((bal_train_x, unbal_train_x))
    #         local_train_y = np.concatenate((bal_train_y, unbal_train_y))
    #         local_train_id_list = bal_train_id_list + unbal_train_id_list

    #     labels_dict['count'].append(len(local_train_id_list))
        # Test data
        (local_test_x, local_test_y, local_test_id_list) = utilities.load_data(test_hdf5_path)
        print('local_test: ', local_test_x.shape, local_test_y.shape)

    #     train_x = ( local_train_x if (train_x == []) else np.concatenate((train_x, local_train_x)) )
    #     train_y = ( local_train_y if (train_y == []) else np.concatenate((train_y, local_train_y)) )
    #     train_id_list = train_id_list + local_train_id_list
        test_x = ( local_test_x if (test_x == []) else np.concatenate((test_x, local_test_x)) )
        test_y = ( local_test_y if (test_y == []) else np.concatenate((test_y, local_test_y)) )
        test_id_list = test_id_list + local_test_id_list

    # # Mask other classes.
    # for ii, item in  enumerate(train_y):
    #     train_y[ii] = np.logical_and(item, labels_map_mask)
    for ii, item in  enumerate(test_y):
        test_y[ii] = np.logical_and(item, labels_map_mask)

    for ii, item in  enumerate(test_y):
        if not any(item):
            print(test_id_list[ii])
            print(ii, item)
            raise Exception('False item, no positive label.')

    test_x_mfcc, test_y_mfcc, test_seq_len = tasksmq.batch_wav_to_mfcc_parallel(test_x, test_y, agumentation=False)           # test_seq_len = np.ones(len(test_x_mfcc)) * 240     # length array of the batch
    print("Loading data time: {:.3f} s".format(time.time() - load_time))

    return test_x_mfcc, test_y_mfcc

# Run function in mini-batch to save memory. 
# def run_func(func, x, batch_size):
#     pred_all = []
#     batch_num = int(np.ceil(len(x) / float(batch_size)))
#     for i1 in xrange(batch_num):
#         batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
#         [preds] = func([batch_x, 0.])
#         pred_all.append(preds)
#     pred_all = np.concatenate(pred_all, axis=0)
#     return pred_all

# # Recognize and write probabilites. 
# def recognize(args, at_bool, sed_bool):
#     (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
#     x = te_x
#     y = te_y
#     na_list = te_na_list
    
#     x = do_scale(x, args.scaler_path, verbose=1)
    
#     fusion_at_list = []
#     fusion_sed_list = []
#     for epoch in range(20, 30, 1):
#         t1 = time.time()
#         [model_path] = glob.glob(os.path.join(args.model_dir, 
#             "*.%02d-0.*.hdf5" % epoch))
#         model = load_model(model_path)
        
#         # Audio tagging
#         if at_bool:
#             pred = model.predict(x)
#             fusion_at_list.append(pred)
        
#         # Sound event detection
#         if sed_bool:
#             in_layer = model.get_layer('in_layer')
#             loc_layer = model.get_layer('localization_layer')
#             func = K.function([in_layer.input, K.learning_phase()], 
#                               [loc_layer.output])
#             pred3d = run_func(func, x, batch_size=20)
#             fusion_sed_list.append(pred3d)
        
#         print("Prediction time: %s" % (time.time() - t1,))
    
#     # Write out AT probabilities
#     if at_bool:
#         fusion_at = np.mean(np.array(fusion_at_list), axis=0)
#         print("AT shape: %s" % (fusion_at.shape,))
#         io_task4.at_write_prob_mat_to_csv(
#             na_list=na_list, 
#             prob_mat=fusion_at, 
#             out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
#     # Write out SED probabilites
#     if sed_bool:
#         fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
#         print("SED shape:%s" % (fusion_sed.shape,))
#         io_task4.sed_write_prob_mat_list_to_csv(
#             na_list=na_list, 
#             prob_mat_list=fusion_sed, 
#             out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
#     print("Prediction finished!")

# # Get stats from probabilites. 
# def get_stat(args, at_bool, sed_bool):
#     lbs = cfg.lbs
#     step_time_in_sec = cfg.step_time_in_sec
#     max_len = cfg.max_len
#     thres_ary = [0.3] * len(lbs)

#     # Calculate AT stat
#     if at_bool:
#         pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
#         at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
#         at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
#         at_evaluator = evaluate.AudioTaggingEvaluate(
#             weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv", 
#             lbs=lbs)
        
#         at_stat = at_evaluator.get_stats_from_prob_mat_csv(
#                         pd_prob_mat_csv=pd_prob_mat_csv_path, 
#                         thres_ary=thres_ary)
                        
#         # Write out & print AT stat
#         at_evaluator.write_stat_to_csv(stat=at_stat, 
#                                        stat_path=at_stat_path)
#         at_evaluator.print_stat(stat_path=at_stat_path)
        
#         # Write AT to submission format
#         io_task4.at_write_prob_mat_csv_to_submission_csv(
#             at_prob_mat_path=pd_prob_mat_csv_path, 
#             lbs=lbs, 
#             thres_ary=at_stat['thres_ary'], 
#             out_path=at_submission_path)
               
#     # Calculate SED stat
#     if sed_bool:
#         sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
#         sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
#         sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
#         sed_evaluator = evaluate.SoundEventDetectionEvaluate(
#             strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv", 
#             lbs=lbs, 
#             step_sec=step_time_in_sec, 
#             max_len=max_len)
                            
#         # Write out & print SED stat
#         sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
#                     pd_prob_mat_list_csv=sed_prob_mat_list_path, 
#                     thres_ary=thres_ary)
                    
#         # Write SED to submission format
#         sed_evaluator.write_stat_to_csv(stat=sed_stat, 
#                                         stat_path=sed_stat_path)                     
#         sed_evaluator.print_stat(stat_path=sed_stat_path)
        
#         # Write SED to submission format
#         io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
#             sed_prob_mat_list_path=sed_prob_mat_list_path, 
#             lbs=lbs, 
#             thres_ary=thres_ary, 
#             step_sec=step_time_in_sec, 
#             out_path=sed_submission_path)
                                                        
#     print("Calculating stat finished!")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="")
    # subparsers = parser.add_subparsers(dest='mode')
    
    # parser_train = subparsers.add_parser('train')
    # parser_train.add_argument('--tr_hdf5_path', type=str)
    # parser_train.add_argument('--te_hdf5_path', type=str)
    # parser_train.add_argument('--scaler_path', type=str)
    # parser_train.add_argument('--out_model_dir', type=str)
    
    # parser_recognize = subparsers.add_parser('recognize')
    # parser_recognize.add_argument('--te_hdf5_path', type=str)
    # parser_recognize.add_argument('--scaler_path', type=str)
    # parser_recognize.add_argument('--model_dir', type=str)
    # parser_recognize.add_argument('--out_dir', type=str)
    
    # parser_get_stat = subparsers.add_parser('get_stat')
    # parser_get_stat.add_argument('--pred_dir', type=str)
    # parser_get_stat.add_argument('--stat_dir', type=str)
    # parser_get_stat.add_argument('--submission_dir', type=str)
    
    # args = parser.parse_args()
    
    # if args.mode == 'train':
    #     train(args)
    # elif args.mode == 'recognize':
    #     recognize(args, at_bool=True, sed_bool=True)
    # elif args.mode == 'get_stat':
    #     get_stat(args, at_bool=True, sed_bool=True)
    # else:
    #     raise Exception("Incorrect argument!")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--workspace', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--balance_type', type=str,
                        default='balance_in_batch',
                        choices=['no_balance', 'balance_in_batch'])

    parser.add_argument('--model_type', type=str, required=True,
                        choices=['crnn_sed', 
                                 'decision_level_average_pooling', 
                                 'decision_level_single_attention',
                                 'decision_level_multi_attention',
                                 'feature_level_attention'])

    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')

    args = parser.parse_args()
    args.filename = utilities.get_filename(__file__)
    args.data_dir = vp.DATA_DIR
    if args.mode == 'train':
        train(args)
