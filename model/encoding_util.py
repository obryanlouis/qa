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

def encode_passage_and_question(options, passage, question, keep_prob):
    """Returns (passage_outputs, question_outputs), which are both of size
       [batch_size, max ctx length | max qst length, 2 * rnn_size].
    """
    with tf.variable_scope("preprocessing_lstm"):
        rnn_cell_fw = create_multi_rnn_cell(options,
            "preprocessing_rnn_forward_cell", keep_prob, num_rnn_layers=options.num_rnn_layers)
        rnn_cell_bw = create_multi_rnn_cell(options,
            "preprocessing_rnn_backward_cell", keep_prob, num_rnn_layers=options.num_rnn_layers)
        passage_outputs = _run_bidirectional_preprocessing_lstm(passage,
            rnn_cell_fw, rnn_cell_bw)
        # size(passage_outputs) = [batch_size, max_ctx_length, 2 * rnn_size]
        question_outputs = _run_bidirectional_preprocessing_lstm(question,
            rnn_cell_fw, rnn_cell_bw)
        # size(question_outputs) = [batch_size, max_qst_length, 2 * rnn_size]
        return passage_outputs, question_outputs
