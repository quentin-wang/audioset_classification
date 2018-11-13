import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np

import tensorflow as tf
import crnn
import vim_params as vp

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

if __name__ == '__main__':
    logits_tensor = crnn.build_crnn_model(is_training=tf.contrib.learn.ModeKeys.TRAIN)     # training=False，模型参数不可被修改
    with tf.variable_scope('mix'):
        output_tensor = tf.sigmoid(logits_tensor, name='prediction')

    # Add training ops.
    with tf.variable_scope('train'):
        global_step = tf.train.get_or_create_global_step()
        # global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

        labels = tf.placeholder(
            tf.float32, shape=(None, vp.TOTAL_NUM_CLASS), name='labels')

        # xent = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=logits_tensor, labels=labels, name='xent')
        xent = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits_tensor, targets=labels, pos_weight=pos_weight_init(), name='xent')
        loss_tensor = tf.reduce_mean(xent, name='loss_op')
        
        learning_rate = tf.train.exponential_decay(
            0.001, global_step=global_step, decay_steps=50000, decay_rate=1e-6)

    # vggish_var_names = [v.name for v in tf.global_variables()]
    # print('\n all variables:')
    # print(vggish_var_names)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(trainable_vars)

    vggish_var = [v for v in trainable_vars
        if 'rnn' in v.name or 'gru' in v.name]
    print(vggish_var)

#  all variables:
# ['vggish/conv1/conv1_1/weights:0', 'vggish/conv1/conv1_1/biases:0', 
# 'vggish/conv1/conv1_2/weights:0', 'vggish/conv1/conv1_2/biases:0', 
# 'vggish/conv2/conv2_1/weights:0', 'vggish/conv2/conv2_1/biases:0', 
# 'vggish/conv2/conv2_2/weights:0', 'vggish/conv2/conv2_2/biases:0', 
# 'vggish/conv3/conv3_1/weights:0', 'vggish/conv3/conv3_1/biases:0', 
# 'vggish/conv3/conv3_2/weights:0', 'vggish/conv3/conv3_2/biases:0', 
# 'vggish/conv4/conv4_1/weights:0', 'vggish/conv4/conv4_1/biases:0', 
# 'vggish/conv4/conv4_2/weights:0', 'vggish/conv4/conv4_2/biases:0', 
# 'vggish/conv5/weights:0', 'vggish/conv5/biases:0', 
# 'vggish/rnnweights:0', 'vggish/rnnbiases:0', 
# 'vggish/rnn/gru_cell/gates/kernel:0', 'vggish/rnn/gru_cell/gates/bias:0', 
# 'vggish/rnn/gru_cell/candidate/kernel:0', 'vggish/rnn/gru_cell/candidate/bias:0', 'train/global_step:0']
