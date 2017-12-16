"""Wrapper for CudnnLSTM.

NOTE: cuDNN dropout does not apply to the input or output layers, only between
layers. A single layer RNN will not have any dropout applied by cuDNN itself,
although this implementation adds it to the input.
"""

import tensorflow as tf

from model.dropout_util import *

class CudnnLstmWrapper:
    def __init__(self, train_lstm, eval_lstm, train_lstm_buffer, eval_lstm_buffer,
        num_layers, layer_size, input_dim, bidirectional):
        self.train_lstm = train_lstm
        self.eval_lstm = eval_lstm
        self.train_lstm_buffer = train_lstm_buffer
        self.eval_lstm_buffer = eval_lstm_buffer
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.input_dim = input_dim
        self.bidirectional = bidirectional

def create_cudnn_lstm(input_dim, sess, options, scope, keep_prob, num_layers=None,
        layer_size=None, bidirectional=True):
    """
        Inputs:
            input_dim: The input dimension of the data, which is not the
                same as the sequence length of the data.
            layer_size: The dimension of the rnn.
    """
    num_layers = options.num_rnn_layers if num_layers is None \
        else num_layers
    layer_size = options.rnn_size if layer_size is None else layer_size
    with tf.variable_scope(scope):
        train_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, layer_size, input_dim,
            direction="bidirectional" if bidirectional else "unidirectional",
            dropout=options.dropout)
        eval_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, layer_size, input_dim,
            direction="bidirectional" if bidirectional else "unidirectional",
            dropout=0.0)
        params_size = sess.run(train_lstm.params_size())
        train_lstm_buffer = tf.get_variable("train_buffer", shape=params_size,
            dtype=tf.float32)
        eval_lstm_buffer = tf.get_variable("eval_buffer", shape=params_size,
            dtype=tf.float32)
        return CudnnLstmWrapper(train_lstm, eval_lstm, train_lstm_buffer,
            eval_lstm_buffer, num_layers, layer_size, input_dim, bidirectional)

def run_cudnn_lstm(inputs, keep_prob, options, lstm_wrapper, batch_size,
    use_dropout, initial_state_h=None, initial_state_c=None):
    """
        Inputs:
            inputs: A tensor of size [batch_size, max_time, input_size]
            use_dropout: A tensor that indicates whether to use dropout
        Outputs:
            (output, output_h, output_c)
            output: A tensor of size [batch_size, max_time,
                rnn_size or 2 * rnn_size (if bidirectional)]
            output_h: A tensor of size [batch_size,
                1 (if unidirectional) or 2 (if bidirectional), rnn_size]
            output_c: A tensor of size [batch_size,
                1 (if unidirectional) or 2 (if bidirectional), rnn_size]
    """
    assert(lstm_wrapper.num_layers > 0)
    initial_state_shape = [lstm_wrapper.num_layers * \
            (2 if lstm_wrapper.bidirectional else 1), batch_size,
            lstm_wrapper.layer_size]
    if initial_state_h is None or initial_state_c is None:
        initial_state_h = tf.zeros(initial_state_shape)
        initial_state_c = tf.zeros(initial_state_shape)
    transposed_inputs = tf.transpose(inputs, perm=[1, 0, 2])
    dropout_inputs = sequence_dropout(transposed_inputs, keep_prob=keep_prob)
    train_output, train_output_h, train_output_c = \
        lstm_wrapper.train_lstm(dropout_inputs,
            initial_state_h, initial_state_c, lstm_wrapper.train_lstm_buffer,
            is_training=True)
    assign_eval_buffer = tf.assign(lstm_wrapper.eval_lstm_buffer,
        lstm_wrapper.train_lstm_buffer)
    with tf.control_dependencies([assign_eval_buffer]):
        # CudnnLSTM doesn't support training/eval dropout switching,
        # so this must be is_training=True.
        eval_output, eval_output_h, eval_output_c = lstm_wrapper.eval_lstm(
            transposed_inputs, initial_state_h, initial_state_c,
            lstm_wrapper.eval_lstm_buffer, is_training=True)
    output = tf.cond(use_dropout, true_fn=lambda: train_output,
        false_fn=lambda: eval_output)
    output_h = tf.cond(use_dropout, true_fn=lambda: train_output_h,
        false_fn=lambda: eval_output_h)
    output_c = tf.cond(use_dropout, true_fn=lambda: train_output_c,
        false_fn=lambda: eval_output_c)
    # size(output) = [max_time, batch_size, (1 + bidirectional) * rnn_size]
    # size(output_h) = [(1 + bidirectional), batch_size, rnn_size]
    # size(output_c) = size(output_h)
    output = tf.transpose(output, perm=[1, 0, 2])
    output_h = tf.transpose(output_h, perm=[1, 0, 2])
    output_c = tf.transpose(output_c, perm=[1, 0, 2])
    return output, output_h, output_c

def run_cudnn_lstm_and_return_outputs(inputs, keep_prob, options,
    lstm_wrapper, batch_size, use_dropout, initial_state_h=None,
    initial_state_c=None):
    output, _, _ = run_cudnn_lstm(inputs, keep_prob, options, lstm_wrapper,
        batch_size, use_dropout, initial_state_h=initial_state_h,
        initial_state_c=initial_state_c)
    return output

def run_cudnn_lstm_and_return_hidden_outputs(inputs, keep_prob, options,
    lstm_wrapper, batch_size, use_dropout, initial_state_h=None,
    initial_state_c=None):
    _, output_h, output_c = run_cudnn_lstm(inputs, keep_prob, options, lstm_wrapper,
        batch_size, use_dropout, initial_state_h=initial_state_h,
        initial_state_c=initial_state_c)
    return output_h, output_c
