import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import h5py
import argparse
import time
import logging
from sklearn import metrics
from utils import utilities, data_generator, agument

import tensorflow as tf
slim = tf.contrib.slim
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# from mfcc import vggish_input as mfcc
from mfcc import vggish_params
import crnn

import multiprocessing as mp
import horovod.tensorflow as hvd

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

import pika
import tasksmq_transfer as tasksmq
import vim_params as vp
import numpy as np

# less than 1.0, improve precision
# larger than 1.0. improve recall.
def pos_weight_init():
    weight_dict = {}
    weight_dict = {
        23: 1.0,        # !!! Baby_cryAND_infant_cry
        75: 1.0,        # Bark
        470: 1.0,       # !!! Breaking
        322: 1.0,       # Emergency_vehicle
        343: 1.0,       # Engine
        428: 1.0,       # Machine_gun
        14: 1.0,        # !!! Screaming
        399: 1.0,       # Smoke_detectorAND_smoke_alarm
        288: 1.0,       # Water
    }
    w = np.ones(vp.TOTAL_NUM_CLASS)
    for key in weight_dict:
        w[key] = weight_dict[key]
    return w


def tf_evaluate(target, output, stats_dir, probs_dir, iteration, labels_map):
    """Evaluate a model.

    Args:
      model: object
      output: 2d array, (samples_num, _NUM_CLASS)
      target: 2d array, (samples_num, _NUM_CLASS)
      stats_dir: str, directory to write out statistics.
      probs_dir: str, directory to write out output (samples_num, _NUM_CLASS)
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
    output = output.astype(np.float32)  # (clips_num, _NUM_CLASS)

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

    return mAP, mAUC, [stat['AP'] for stat in stats]


def train(args):

    data_dir = args.data_dir
    workspace = args.workspace
    mini_data = args.mini_data
    quantize = args.quantize
    balance_type = args.balance_type
    init_learning_rate = args.learning_rate           # transfer learning rate
    filename = args.filename
    model_type = args.model_type
    # model = args.model
    batch_size = args.batch_size

    hvd.init()

    # # Load test data.
    df = pd.read_csv(vp.FILE_CLASS_LABELS)
    labels_dict = {}
    labels_dict['name'] = np.array(df[df['transfer'] == 1]['display_name'])
    labels_dict['id'] = np.array(df[df['transfer'] == 1]['index'])
    labels_dict['count'] = []

    labels_map_mask = [False] * vp.TOTAL_NUM_CLASS
    for x in labels_dict['id']:
        labels_map_mask[x] = True

    if (hvd.rank() == 0):
        # Load data
        load_time = time.time()

        # train_x = []
        # train_y = []
        # train_id_list = []
        test_x = []
        test_y = []
        test_id_list = []

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

        logging.info("Loading data time: {:.3f} s".format(time.time() - load_time))

    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)

    # weighted class
    pos_weight = pos_weight_init()

    # Data generator
    # if balance_type == 'no_balance':
    #     DataGenerator = data_generator.VanillaDataGenerator

    # elif balance_type == 'balance_in_batch':
    #     DataGenerator = data_generator.BalancedDataGenerator

    # else:
    #     raise Exception("Incorrect balance_type!")

    # train_gen = DataGenerator(
    #     x=train_x,
    #     y=train_y,
    #     batch_size=batch_size,
    #     labels_map=labels_dict['id'],
    #     shuffle=True,
    #     seed=1234)

    # create work thread for DataGenerator
    # if IS_DISTRIBUTE and hvd.rank() == 0:
    #     # q_batch = mp.Queue (maxsize=10)
    #     task_generate_batch = mp.Process (target = tasksmq.generate_batch, args = (train_gen,))
    #     task_generate_batch.start()

    # use tf.get_default_graph()
    logits_tensor = crnn.build_crnn_model(is_training=tf.contrib.learn.ModeKeys.TRAIN)     # training=False，模型参数不可被修改
    with tf.variable_scope('mix'):
        output_tensor = tf.sigmoid(logits_tensor, name='prediction')

    # Add training ops.
    with tf.variable_scope('train'):
        global_step = tf.train.get_or_create_global_step()    # global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

        labels = tf.placeholder(
            tf.float32, shape=(None, vp.TOTAL_NUM_CLASS), name='labels')

        # xent = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=logits_tensor, labels=labels, name='xent')
        xent = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits_tensor, targets=labels, pos_weight=pos_weight, name='xent')
        loss_tensor = tf.reduce_mean(xent, name='loss_op')
        
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step=global_step, decay_steps=50000, decay_rate=1e-6 * hvd.size())

        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            epsilon=vggish_params.ADAM_EPSILON)
        opt = hvd.DistributedOptimizer(opt)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_list = [v for v in trainable_vars if 'rnn' in v.name or 'gru' in v.name]

        opt.minimize(loss_tensor, global_step=global_step, name='train_op', var_list=var_list)

    #    ----- tensorboard-------
    tf.summary.scalar('loss', loss_tensor)    # do not needed. summary_op = tf.summary.merge_all()

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),
        
        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=40008 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss_tensor},
                                every_n_iter=10),

        # tf.train.SummarySaverHook(save_secs=5, output_dir='./tblogs',summary_op=summary_op),
        # tf.train.StepCounterHook(every_n_steps=10),
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    checkpoint_dir = './checkpoints_transfer' if hvd.rank() == 0 else None
    result_queue = 'result_queue'
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=20))

    restart = 1
    reinit_global_step = global_step.assign(3000)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession( checkpoint_dir=checkpoint_dir,
                                            save_checkpoint_steps=500,
                                            summary_dir='./tblogs',
                                            save_summaries_steps=5,
                                            hooks=hooks,
                                            config=config,
                                            scaffold=scaffold ) as sess:
        
        # Locate all the tensors and ops we need for the training loop.
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        sequence_length = sess.graph.get_tensor_by_name('vggish/input_sequence_length:0')
        output_tensor = sess.graph.get_tensor_by_name('mix/prediction:0')
        labels_tensor = sess.graph.get_tensor_by_name('train/labels:0')
        global_step_tensor = tf.train.get_global_step()
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')
        
        train_op = sess.graph.get_operation_by_name('train/train_op')

        # connect batch msg queue
        credentials = pika.PlainCredentials('myuser', 'mypassword')
        with pika.BlockingConnection(
                    pika.ConnectionParameters('ice-P910',5672,'myvhost',
                    credentials)) as connection:
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)   # 消息未处理完前不要发送新的消息
            
            while not sess.should_stop():
                method_frame, header_frame, body = channel.basic_get(queue=result_queue)      # consumer
                if method_frame:
                    # print(method_frame)
                    batch_x_mfcc, batch_y_mfcc, batch_seq_len = cPickle.loads(body)   # print(batch_x_mfcc.shape)      # print(batch_y_mfcc.shape)
                    channel.basic_ack(method_frame.delivery_tag)

                    if len(batch_x_mfcc) != batch_size:
                        continue
                    
                    if method_frame.message_count > 400:         # consumer can't process too much messages.
                        print('result_queue has too much msgs, clip them.')
                        for _ in range(100):
                            method_frame, header_frame, body = channel.basic_get(queue=result_queue)
                            channel.basic_ack(method_frame.delivery_tag)

                # train
                if restart:
                    restart = 0
                    [num_steps, loss, lr, _] = sess.run([reinit_global_step, loss_tensor, learning_rate, train_op],
                            feed_dict={ features_tensor: batch_x_mfcc, 
                                        labels_tensor: batch_y_mfcc, 
                                        sequence_length: batch_seq_len})
                else:
                    [num_steps, loss, lr, _] = sess.run([global_step_tensor, loss_tensor, learning_rate, train_op],
                            feed_dict={ features_tensor: batch_x_mfcc, 
                                        labels_tensor: batch_y_mfcc, 
                                        sequence_length: batch_seq_len})

                if num_steps % 10 == 0:
                    print('steps: ', num_steps, 'loss: ', loss, 'lr: ', lr)

                # evaluate
                if (num_steps != 0) and (num_steps % 1000 == 0) and (hvd.rank() == 0):
                    logging.info("------------------")
                    # logging.info(
                    #     "Iteration: {}, train time: {:.3f} s".format(
                    #         num_steps, time.time() - train_time))
                    logging.info("Test statistics:")

                    # tensorflow/core/framework/allocator.cc:101] Allocation of 561807360 exceeds 10% of system memory.
                    output = []
                    test_len = len(test_x_mfcc)
                    start_pos = 0
                    max_iter_len = batch_size   # 300 overflow GPU
                    while True:
                        print(start_pos)
                        iter_test_len = ((test_len - start_pos) if ((test_len - start_pos) < max_iter_len) else max_iter_len)
                        if (iter_test_len <= 0):
                            break
                        local_output = sess.run(output_tensor, 
                            feed_dict={
                                features_tensor: test_x_mfcc[start_pos:start_pos+iter_test_len], 
                                labels_tensor: test_y_mfcc[start_pos:start_pos+iter_test_len],
                                sequence_length: test_seq_len[start_pos:start_pos+iter_test_len],
                            })
                        # print('local_output.shape', local_output.shape)
                        output = ( local_output if (output == []) else np.concatenate((output, local_output)) )
                        start_pos += iter_test_len
                    
                    # output = sess.run(output_tensor, feed_dict={features_tensor: test_x_mfcc})       #output = model.predict(input)
                    print('output', output.shape)
                    print('test_y_mfcc', test_y_mfcc.shape)

                    mAP, _, AP = tf_evaluate(
                        target=test_y_mfcc,
                        output=output,
                        stats_dir=os.path.join(stats_dir, "test"),
                        probs_dir=os.path.join(probs_dir, "test"),
                        iteration=num_steps,
                        labels_map=labels_dict['id'])
                    labels_dict['AP'] = AP
                    for (name, ap) in zip(labels_dict['name'], labels_dict['AP']):
                        print(name, '\t', ap)
                    # infer = np.argsort (-output[0])
                    # print(output[0][infer][:40])
                
    # if IS_DISTRIBUTE:
    #     task_generate_batch.terminate()     # parallel related.
    # train_writer.close()      # summary related.
