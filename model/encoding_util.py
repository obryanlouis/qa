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

def create_word_similarity(primary_iterator, secondary_iterator, v_wiq_or_wic):
    """Creates a word similarity tensor, which is used as an input feature.
       Inputs:
         primary_iterator: Either the contexts or questions shaped [batch_size, N, W]
         secondary_iterator: Vice versa of the primary iterator shaped [batch_size, M, W]
         v_wiq_or_wic: 1-Dimensional vector shaped [W]
       Output:
         A word-similarity vector shaped [batch_size, N, 1]
    """
    sh_prim = tf.shape(primary_iterator)
    sh_sec = tf.shape(secondary_iterator)
    batch_size, N, W = sh_prim[0], sh_prim[1], sh_prim[2]
    M = sh_sec[1]
    prim = tf.reshape(primary_iterator, shape=[batch_size * W, N, 1])
    sec = tf.reshape(secondary_iterator, shape=[batch_size * W, 1, M])
    mult = tf.reshape(tf.matmul(prim, sec), shape=[batch_size * N * M, W]) # size = [batch_size * W, N, M]
    similarity =  tf.reshape(
       tf.matmul(mult, tf.reshape(v_wiq_or_wic, shape=[W, 1]))
       , shape=[batch_size, N, M]) # size = [batch_size, N, M]
    sm = tf.nn.softmax(similarity, dim=1) # size = [batch_size, N, M]
    return tf.reshape(
            tf.reduce_sum(sm, axis=2) # size = [batch_size, N]
            , [batch_size, N, 1])

def create_model_inputs(words_placeholder, ctx, qst, options, wiq, wic, sq_dataset):
    with tf.device("/cpu:0"):
        with tf.variable_scope("model_inputs"):
            v_wiq = tf.get_variable("v_wiq", shape=[sq_dataset.word_vec_size])
            v_wic = tf.get_variable("v_wic", shape=[sq_dataset.word_vec_size])
            ctx_embedded = tf.nn.embedding_lookup(words_placeholder, ctx)
            qst_embedded = tf.nn.embedding_lookup(words_placeholder, qst)
            ctx_inputs_list = [ctx_embedded]
            qst_inputs_list = [qst_embedded]
            wiq_sh = tf.shape(wiq)
            wiq_feature_shape = [wiq_sh[0], wiq_sh[1]] + [1]
            wic_sh = tf.shape(wic)
            wic_feature_shape = [wic_sh[0], wic_sh[1]] + [1]
            if options.use_word_in_question_feature:
                ctx_inputs_list.append(tf.reshape(tf.cast(wiq, dtype=tf.float32), shape=wiq_feature_shape))
                qst_inputs_list.append(tf.reshape(tf.cast(wic, dtype=tf.float32), shape=wic_feature_shape))
            if options.use_word_similarity_feature:
                ctx_inputs_list.append(create_word_similarity(ctx_embedded, qst_embedded, v_wiq))
                qst_inputs_list.append(create_word_similarity(qst_embedded, ctx_embedded, v_wic))
            if len(ctx_inputs_list) == 1:
                return ctx_inputs_list[0], qst_inputs_list[0]
            else:
                return tf.concat(ctx_inputs_list, axis=-1), tf.concat(qst_inputs_list, axis=-1)
