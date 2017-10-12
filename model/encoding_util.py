"""Functions for encoding passage and question inputs.
"""

import tensorflow as tf

from model.rnn_util import *

def _run_bidirectional_preprocessing_lstm(inputs, cell_fw, cell_bw,
        initial_state_fw=None, initial_state_bw=None):
    '''Returns (output_states_fw_final, output_states_bw_final, outputs)
    '''
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, inputs,
            dtype=tf.float32, initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw)
    outputs_fw, outputs_bw = outputs
    return output_states[0], output_states[1], tf.concat([outputs_fw, outputs_bw], axis=2)

def encode_passage_and_question(options, passage, question, keep_prob):
    """Returns (passage_outputs, question_outputs), which are both of size
       [batch_size, max ctx length | max qst length, 2 * rnn_size].
    """
    with tf.variable_scope("preprocessing_lstm"):
        rnn_cell_fw = create_multi_rnn_cell(options, "preprocessing_rnn_forward_cell", keep_prob)
        rnn_cell_bw = create_multi_rnn_cell(options, "preprocessing_rnn_backward_cell", keep_prob)
        state_fw, state_bw, passage_outputs = _run_bidirectional_preprocessing_lstm(passage,
                rnn_cell_fw, rnn_cell_bw)
        # size(passage_outputs) = [batch_size, max_ctx_length, 2 * rnn_size]
        _, _, question_outputs = _run_bidirectional_preprocessing_lstm(question,
                rnn_cell_fw, rnn_cell_bw, initial_state_fw=state_fw, initial_state_bw=state_bw)
        # size(question_outputs) = [batch_size, max_qst_length, 2 * rnn_size]
        return passage_outputs, question_outputs
