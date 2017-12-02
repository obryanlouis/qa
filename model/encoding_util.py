"""Functions for encoding passage and question inputs.
"""

import tensorflow as tf

from model.cudnn_gru_wrapper import *
from model.rnn_util import *

def _run_bidirectional_preprocessing_lstm(inputs, cell_fw, cell_bw):
    '''Returns (output_states_fw_final, output_states_bw_final, outputs)
    '''
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
        inputs, dtype=tf.float32)
    outputs_fw, outputs_bw = outputs
    return tf.concat([outputs_fw, outputs_bw], axis=2)

def encode_low_level_and_high_level_representations(sess, scope, options, inputs,
    keep_prob, batch_size, is_train):
    """Runs a bi-directional GRU over the inputs to get low level and
       high level outputs.

       Inputs:
            inputs: A tensor shaped [batch_size, M, d]

       Outputs:
            A tensor shaped [batch_size, M, 2 * rnn_size]
    """
    with tf.variable_scope(scope):
        gru = create_cudnn_gru(inputs.get_shape()[2], sess, options, "low",
            keep_prob)
        low_level_outputs = run_cudnn_rnn_and_return_outputs(inputs,
            keep_prob, options, gru, batch_size, is_train)
        gru = create_cudnn_gru(inputs.get_shape()[2], sess, options, "high",
            keep_prob)
        high_level_outputs = run_cudnn_rnn_and_return_outputs(low_level_outputs,
            keep_prob, options, gru, batch_size, is_train)

        return low_level_outputs, high_level_outputs

def encode_passage_and_question(options, passage, question, keep_prob,
    sess, batch_size, is_train):
    """Returns (passage_outputs, question_outputs), which are both of size
       [batch_size, max ctx length | max qst length, 2 * rnn_size].
    """
    with tf.variable_scope("preprocessing_lstm"):
        with tf.variable_scope("passage"):
            gru = create_cudnn_gru(passage.get_shape()[2], sess, options,
                "passage_gru", keep_prob)
            passage_outputs = run_cudnn_rnn_and_return_outputs(passage,
                keep_prob, options, gru, batch_size, is_train) # size = [batch_size, max_ctx_length, 2 * rnn_size]
        with tf.variable_scope("question"):
            gru = create_cudnn_gru(question.get_shape()[2], sess, options,
                "question_gru", keep_prob)
            question_outputs = run_cudnn_rnn_and_return_outputs(question,
                keep_prob, options, gru, batch_size, is_train) # size = [batch_size, max_qst_length, 2 * rnn_size]
        return passage_outputs, question_outputs
