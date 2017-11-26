"""Functions for encoding passage and question inputs.
"""

import tensorflow as tf

from model.rnn_util import *

def _run_bidirectional_preprocessing_lstm(inputs, cell_fw, cell_bw):
    '''Returns (output_states_fw_final, output_states_bw_final, outputs)
    '''
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
        inputs, dtype=tf.float32)
    outputs_fw, outputs_bw = outputs
    return tf.concat([outputs_fw, outputs_bw], axis=2)

def _create_rnn_cells(scope, options, keep_prob):
    with tf.variable_scope(scope):
        rnn_cell_fw = create_multi_rnn_cell(options,
            "preprocessing_rnn_forward_cell", keep_prob,
            num_rnn_layers=options.num_rnn_layers)
        rnn_cell_bw = create_multi_rnn_cell(options,
            "preprocessing_rnn_backward_cell", keep_prob,
            num_rnn_layers=options.num_rnn_layers)
        return rnn_cell_fw, rnn_cell_bw

def encode_low_level_and_high_level_representations(scope, options, inputs,
    keep_prob):
    """Runs a bi-directional GRU over the inputs to get low level and
       high level outputs.

       Inputs:
            inputs: A tensor shaped [batch_size, M, d]

       Outputs:
            A tensor shaped [batch_size, M, 2 * rnn_size]
    """
    with tf.variable_scope(scope):
        with tf.variable_scope("low"):
            low_cell_fw, low_cell_bw = _create_rnn_cells("low_rnn_cells", options,
                keep_prob)
            low_level_outputs = _run_bidirectional_preprocessing_lstm(inputs,
                low_cell_fw, low_cell_bw) # size = [batch_size, M, 2 * rnn_size]
        with tf.variable_scope("high"):
            high_cell_fw, high_cell_bw = _create_rnn_cells("high_rnn_cells",
                options, keep_prob)
            high_level_outputs = _run_bidirectional_preprocessing_lstm(
                low_level_outputs, high_cell_fw, high_cell_bw) # size = [batch_size, M, 2 * rnn_size]
        return low_level_outputs, high_level_outputs

def encode_passage_and_question(options, passage, question, keep_prob):
    """Returns (passage_outputs, question_outputs), which are both of size
       [batch_size, max ctx length | max qst length, 2 * rnn_size].
    """
    with tf.variable_scope("preprocessing_lstm"):
        with tf.variable_scope("passage"):
            rnn_cell_fw, rnn_cell_bw = _create_rnn_cells("rnn_cells", options,
                keep_prob)
            passage_outputs = _run_bidirectional_preprocessing_lstm(passage,
                rnn_cell_fw, rnn_cell_bw) # size = [batch_size, max_ctx_length, 2 * rnn_size]
        with tf.variable_scope("question"):
            rnn_cell_fw, rnn_cell_bw = _create_rnn_cells("rnn_cells", options,
                keep_prob)
            question_outputs = _run_bidirectional_preprocessing_lstm(question,
                rnn_cell_fw, rnn_cell_bw) # size = [batch_size, max_qst_length, 2 * rnn_size]
        return passage_outputs, question_outputs
