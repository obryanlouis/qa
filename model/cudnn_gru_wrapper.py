"""Wrapper for CudnnGruCell.
NOTE: GRU's don't seem to perform as well as LSTM's for this task, and they
also don't seem to be that much faster on CUDA.

NOTE: cuDNN dropout does not apply to the input or output layers, only between
layers. A single layer RNN will not have any dropout applied by cuDNN itself,
although this implementation adds it to the input.
"""

import tensorflow as tf

from model.dropout_util import *


class CudnnGruWrapper:
    def __init__(self, train_gru, eval_gru, train_gru_buffer, eval_gru_buffer,
        num_layers, layer_size, input_dim, bidirectional):
        self.train_gru = train_gru
        self.eval_gru = eval_gru
        self.train_gru_buffer = train_gru_buffer
        self.eval_gru_buffer = eval_gru_buffer
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.input_dim = input_dim
        self.bidirectional = bidirectional

def create_cudnn_gru(input_dim, sess, options, scope, keep_prob, num_layers=None,
        layer_size=None, bidirectional=True):
    """
        Inputs:
            input_dim: The input dimension of the data, which is not the
                same as the sequence length of the data.
            layer_size: The dimension of the rnn.
    """
    raise Exception("testing")
    num_layers = options.num_rnn_layers if num_layers is None \
        else num_layers
    layer_size = options.rnn_size if layer_size is None else layer_size
    with tf.variable_scope(scope):
        train_gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, layer_size, input_dim,
            direction="bidirectional" if bidirectional else "unidirectional",
            dropout=options.dropout)
        eval_gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, layer_size, input_dim,
            direction="bidirectional" if bidirectional else "unidirectional",
            dropout=0.0)
        params_size = sess.run(train_gru.params_size())
        train_gru_buffer = tf.get_variable("train_buffer", shape=params_size,
            dtype=tf.float32)
        eval_gru_buffer = tf.get_variable("eval_buffer", shape=params_size,
            dtype=tf.float32)
        return CudnnGruWrapper(train_gru, eval_gru, train_gru_buffer,
            eval_gru_buffer, num_layers, layer_size, input_dim, bidirectional)

def run_cudnn_gru(inputs, keep_prob, options, gru_wrapper, batch_size,
    use_dropout, initial_state=None):
    """
        Inputs:
            inputs: A tensor of size [batch_size, max_time, input_size]
            use_dropout: A tensor that indicates whether to use dropout
        Outputs:
            (output, output_h)
            output: A tensor of size [batch_size, max_time,
                rnn_size or 2 * rnn_size (if bidirectional)]
            output_h: A tensor of size [batch_size,
                1 (if unidirectional) or 2 (if bidirectional), rnn_size]
    """
    assert(gru_wrapper.num_layers > 0)
    initial_state_shape = [gru_wrapper.num_layers * \
            (2 if gru_wrapper.bidirectional else 1), batch_size,
            gru_wrapper.layer_size]
    if initial_state is None:
        initial_state = tf.zeros(initial_state_shape)
    sh = initial_state.get_shape()
    initial_state_shape_expected = (len(sh) == 3 \
        and sh[0] == initial_state_shape[0] \
        and sh[2] == initial_state_shape[2])
    if not initial_state_shape_expected:
        print("initial state shape:", initial_state.get_shape(),
              "expected shape:", initial_state_shape)
    assert initial_state_shape_expected
    transposed_inputs = tf.transpose(inputs, perm=[1, 0, 2])
    dropout_inputs = sequence_dropout(transposed_inputs, keep_prob=keep_prob)
    train_output, train_output_h = gru_wrapper.train_gru(dropout_inputs,
        initial_state, gru_wrapper.train_gru_buffer, is_training=True)
    assign_eval_buffer = tf.assign(gru_wrapper.eval_gru_buffer,
        gru_wrapper.train_gru_buffer)
    with tf.control_dependencies([assign_eval_buffer]):
        # CudnnGRU doesn't support training/eval dropout switching,
        # so this must be is_training=True.
        eval_output, eval_output_h = gru_wrapper.eval_gru(transposed_inputs,
            initial_state, gru_wrapper.eval_gru_buffer, is_training=True)
    output = tf.cond(use_dropout, true_fn=lambda: train_output,
        false_fn=lambda: eval_output)
    output_h = tf.cond(use_dropout, true_fn=lambda: train_output_h,
        false_fn=lambda: eval_output_h)
    # size(output) = [max_time, batch_size, (1 + bidirectional) * rnn_size]
    # size(output_h) = [(1 + bidirectional), batch_size, rnn_size]
    output = tf.transpose(output, perm=[1, 0, 2])
    output_h = tf.transpose(output_h, perm=[1, 0, 2])
    return output, output_h

def run_cudnn_gru_and_return_outputs(inputs, keep_prob, options,
    gru_wrapper, batch_size, use_dropout, initial_state=None):
    output, output_h = run_cudnn_gru(inputs, keep_prob, options, gru_wrapper,
        batch_size, use_dropout, initial_state=initial_state)
    return output

def run_cudnn_gru_and_return_hidden_outputs(inputs, keep_prob, options,
    gru_wrapper, batch_size, use_dropout, initial_state=None):
    output, output_h = run_cudnn_gru(inputs, keep_prob, options, gru_wrapper,
        batch_size, use_dropout, initial_state=initial_state)
    return output_h
