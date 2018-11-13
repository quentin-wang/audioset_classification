# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the 'VGGish' model used to generate AudioSet embedding features.

The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.

Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally.

For comparison, here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""

import tensorflow as tf
from mfcc import vggish_params as params
import numpy as np

slim = tf.contrib.slim

# def length(x):
#     used = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
#     length = tf.reduce_sum(used, reduction_indices=1)
#     length = tf.cast(length, tf.int32)
#     return length

# def _last_relevant_dynamic(output, length):
#     batch_size = tf.shape(output)[0]
#     max_length = int(output.get_shape()[1])
#     output_size = int(output.get_shape()[2])
#     index = tf.range(0, batch_size) * max_length + (length - 1)
#     flat = tf.reshape(output, [-1, output_size])
#     relevant = tf.gather(flat, index)
#     return relevant

# dynamic_rnn
# def RNN(x, w, b, n_hidden, seq=None):
#     lstm_cell =tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis)
#     outputs, states =tf.nn.dynamic_rnn(lstm_cell, x, sequence_length=seq, dtype=tf.float32)  # sequence_length=
#     outputs = _last_relevant_dynamic(outputs, seq)
#     out = tf.matmul(outputs, w['out']) + b['out']     # [N, 527]        # print(aa.shape)
#     return out

# def _last_relevant_static(output, length):
#     batch_size = tf.shape(output[0])[0]
#     max_length = len(output)
#     output_size = int(output[0].get_shape()[1])
#     index = tf.range(0, batch_size) * max_length + (length - 1)
#     flat = tf.reshape(output, [-1, output_size])
#     relevant = tf.gather(flat, index)
#     return relevant

# def rnn_placeholders(state):
#     """Convert RNN state tensors to placeholders with the zero state as default."""
#     if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
#         c, h = state
#         c = tf.placeholder_with_default(c, c.shape, c.op.name)
#         h = tf.placeholder_with_default(h, h.shape, h.op.name)
#         return tf.contrib.rnn.LSTMStateTuple(c, h)
#     elif isinstance(state, tf.Tensor):
#         h = state
#         h = tf.placeholder_with_default(h, h.shape, h.op.name)
#         return h
#     else:
#         structure = [rnn_placeholders(x) for x in state]
#         return tuple(structure)

# dynamic_rnn
def RNN(x, w, b, n_hidden, seq=None):
    cell =tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    _, states =tf.nn.dynamic_rnn(cell, x, sequence_length=seq, dtype=tf.float32)
    out = tf.matmul(states.h, w['out']) + b['out']     # [N, 527]
    return out

# static_rnn
# def RNN(x, w, b, n_hidden, seq=None):

#     cell =tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
#     # initial_state = rnn_placeholders(cell.zero_state(tf.shape(x)[0], tf.float32))
#     # c_state = orthogonal_init(shape=[batch_size, cell.state_size.c])
#     # h_state = orthogonal_init(shape=[batch_size, cell.state_size.h])
#     # initial_state = tf.contrib.rnn.LSTMStateTuple(c_state, h_state)
#     # print('initial_state {}'.format(initial_state))
    
#     cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)    # 0.8
#     x=tf.unstack(x, axis=1)
#     outputs, states = tf.contrib.rnn.static_rnn(cell, x, # initial_state=initial_state,
#         dtype=tf.float32, sequence_length=seq)
#     # print(states)
#     out = tf.matmul(states.h, w['out']) + b['out']     # [N, 527]
#     return out

def build_crnn_model(is_training=tf.contrib.learn.ModeKeys.TRAIN):

    # Defaults:
    # - All weights are initialized to N(0, INIT_STDDEV).
    # - All biases are initialized to 0.
    # - All activations are ReLU.
    # - All convolutions are 3x3 with stride 1 and SAME padding.
    # - All max-pools are 2x2 with stride 2 and SAME padding.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            # normalizer_fn = tf.contrib.layers.layer_norm,
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': training},
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=tf.nn.relu,  # is_training=is_training
            ), \
            slim.arg_scope([slim.conv2d],
                kernel_size=[3, 3], stride=1, padding='SAME'), \
             slim.arg_scope([slim.max_pool2d],
                kernel_size=[1, 2], stride=[1, 2], padding='SAME'), \
            tf.variable_scope('vggish'):
        # Input: a batch of 2-D log-mel-spectrogram patches.
        features = tf.placeholder(tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
                name='input_features')      # (N, SEQ_LEN, 64)
        sequence_length = tf.placeholder(tf.int32, shape=[None], name='input_sequence_length') # (N)

        net = tf.reshape(features, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])    # [N, SEQ_LEN, 64, 1]

        # The VGG stack of alternating convolutions and max-pools.
        net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv1')                  # (N, SEQ_LEN, 64, 128)
        net = slim.max_pool2d(net, scope='pool1')                                   # (N, SEQ_LEN, 32, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv2')                  # (N, SEQ_LEN, 32, 128)
        net = slim.max_pool2d(net, scope='pool2')                                   # (N, SEQ_LEN, 16, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv3')                  # (N, SEQ_LEN, 16, 128)
        net = slim.max_pool2d(net, scope='pool3')                                   # (N, SEQ_LEN, 8, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv4')                  # (N, SEQ_LEN, 8, 128)
        net = slim.max_pool2d(net, scope='pool4')                                   # (N, SEQ_LEN, 4, 128)

        net = slim.conv2d(net, 256, scope='conv5')
        net = slim.max_pool2d(net, kernel_size=[1, 4], stride=[1, 4], scope='pool5')     # (N, SEQ_LEN, 1, 256)

        net = tf.squeeze(net, axis=2)           # net = tf.reshape(net, [-1, SEQ_LEN, 256])   # TODO: 使用 收缩的接口
        print('net before RNN:', net.shape)

        n_hidden = 128
        vocab_size = 527
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), name='rnnweights')
        }
        biases = {
            'out': tf.Variable(tf.random_normal([vocab_size]), name='rnnbiases')
        }
        net = RNN(net, weights, biases, n_hidden, sequence_length)      # (N, 527)
        print('net.shape', net.shape)

        return tf.identity(net, name='embeddings')

def load_checkpoint(session, checkpoint_path):
    """Loads a pre-trained VGGish-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the VGGish model
    definition. Only variables defined by VGGish will be loaded.

    Args:
        session: an active TensorFlow session.
        checkpoint_path: path to a file containing a checkpoint that is
            compatible with the VGGish model definition.
    """
    # Get the list of names of all VGGish variables that exist in
    # the checkpoint (i.e., all inference-mode VGGish variables).
    with tf.Graph().as_default():
        build_crnn_model(is_training=tf.contrib.learn.ModeKeys.TEST)

        vggish_var_names = [v.name for v in tf.global_variables()]
        print('\n all variables:')
        print(vggish_var_names)

        vggish_var_names = [v.name for v in tf.global_variables() 
            if 'conv2' in v.name or 'conv3' in v.name or 'conv4' in v.name]
        print('\n loaded variables:')
        print(vggish_var_names)
        print('++++++++++++++++')

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]

    # Use a Saver to restore just the variables selected above.
    saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained',
                                                 write_version=1)
    saver.restore(session, checkpoint_path)
