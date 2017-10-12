"""Functions for decoding model components to get loss and span probabilities.
"""

import tensorflow as tf

from model.tf_util import *
from model.rnn_util import *

def _get_question_attention(options, question_rnn_outputs):
    '''Gets an attention pooling vector of the question to be used as the
       initial input to the answer recurrent network.

       Inputs:
            question_rnn_outputs: Tensor of the question rnn outputs of
                size [batch_size, Q, 2 * rnn_size].
       Output:
            A tensor of size [batch_size, rnn_size].
    '''
    with tf.variable_scope("decode_question_attention"):
        sh = tf.shape(question_rnn_outputs)
        batch_size = sh[0]
        Q = sh[1]

        W = 2 * options.rnn_size
        W_question = tf.get_variable("W", dtype=tf.float32, shape=[W, options.rnn_size])
        W_param = tf.get_variable("W_param", dtype=tf.float32, shape=[options.rnn_size, options.rnn_size])
        V_param = tf.get_variable("V_param", dtype=tf.float32, shape=[options.rnn_size, 1])
        v = tf.get_variable("v", dtype=tf.float32, shape=[options.rnn_size, 1])
        W_reduce_size = tf.get_variable("W_reduce_size", dtype=tf.float32, shape=[W, options.rnn_size])

        s = multiply_3d_and_2d_tensor(
                    multiply_3d_and_2d_tensor(question_rnn_outputs,
                              W_question) # size = [batch_size, Q, rnn_size]
                    + tf.squeeze(tf.matmul(W_param, V_param)) # size = [rnn_size]
                    , v) # size = [batch_size, Q, 1]
        a = tf.nn.softmax(s, dim=1) # size = [batch_size, Q, 1]
        reduced_sum = tf.reduce_sum(
                  a * question_rnn_outputs # size = [batch_size, Q, W]
                  , axis=1) # size = [batch_size, W]
        return tf.matmul(reduced_sum, W_reduce_size)

def decode_answer_pointer_boundary(options, batch_size, keep_prob, spans,
        attention_outputs, sq_dataset, question_outputs):
    with tf.variable_scope("answer_pointer"):
        V = tf.get_variable("V", shape=[2 * options.rnn_size,
                options.rnn_size])
        Wa_h = tf.get_variable("Wa_h", shape=[options.rnn_size,
                options.rnn_size])
        v = tf.get_variable("v", shape=[options.rnn_size])
        ba = tf.get_variable("ba", shape=[1, options.rnn_size])
        c = tf.get_variable("c", shape=[1])
        answer_lstm_cell = create_multi_rnn_cell(options, "answer_lstm",
                keep_prob)
    initial_state = _get_question_attention(options, question_outputs)
    answer_pointer_state = (initial_state,) * options.num_rnn_layers
    loss = tf.constant(0.0, dtype=tf.float32)
    start_span_probs = None
    end_span_probs = None
    HrV = multiply_3d_and_2d_tensor(attention_outputs, V) # size = [batch_size, max_ctx_length, rnn_size]
    v_reshaped = tf.reshape(v, [1, -1, 1]) # size = [1, rnn_size, 1]
    v_tiled = tf.tile(v_reshaped, [batch_size, 1, 1]) # size = [batch_size, rnn_size, 1]
    for z in range(2):
        Wa_h_ha = tf.constant(0.0, dtype=tf.float32)
        for s in answer_pointer_state:
            Wa_h_ha += tf.matmul(s, Wa_h) # size = [batch_size, rnn_size]
        inner_sum = Wa_h_ha + ba
        inner_sum = HrV + tf.reshape(inner_sum, [batch_size, 1, options.rnn_size]) # size = [batch_size, max_ctx_length, rnn_size]
        F = tf.tanh(inner_sum) # size = [batch_size, max_ctx_length, rnn_size]
        vF = tf.reshape(tf.matmul(F, v_tiled), [batch_size, sq_dataset.get_max_ctx_len()]) # size = [batch_size, max_ctx_length]
        logits = vF + c # size = [batch_size, max_ctx_length]
        beta = tf.nn.softmax(logits)

        if z == 0:
            start_span_probs = beta
        else:
            end_span_probs = beta
        labels = tf.minimum(spans[:,z], sq_dataset.get_max_ctx_len() - 1)
        loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)) \
               / tf.cast(batch_size, tf.float32)

        HrBeta = tf.matmul(tf.reshape(beta, [-1, 1, sq_dataset.get_max_ctx_len()]), attention_outputs) # size = [batch_size, 1, 2 * rnn_size]
        HrBeta = tf.reshape(HrBeta, [batch_size, 2 * options.rnn_size]) # size = [batch_size, 2 * rnn_size]
        with tf.variable_scope("answer_pointer_lstm", reuse=z > 0):
            _, answer_pointer_state = answer_lstm_cell(HrBeta, answer_pointer_state)
    return loss, start_span_probs, end_span_probs
