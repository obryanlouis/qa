"""The stochastic answer pointer from https://arxiv.org/pdf/1712.03556.pdf.
"""


import tensorflow as tf

from model.cudnn_lstm_wrapper import *
from model.dropout_util import *
from model.rnn_util import *
from model.tf_util import *

_NUMERICAL_STABILITY_EPSILON = 1e-8


def _get_probs_from_state_and_memory(state, memory, matrix,
    batch_size, ctx_dim, max_ctx_len):
    """Performs softmax(state . matrix . memory)
        Inputs:
            state: The state of size [batch_size, 1, d]
            memory: The memory of size [batch_size, max_ctx_len, 2 * rnn_size]
            matrix: The matrix of size [d, 2 * rnn_size]

        Outputs:
            A tensor of size [batch_size, max_ctx_len]
    """
    sW = multiply_tensors(state, matrix) # size = [batch_size, 1, 2 * rnn_size]
    sWM = tf.matmul(memory, tf.transpose(sW, perm=[0, 2, 1])) # size = [batch_size, max_ctx_length, 1]
    sWM = tf.reshape(sWM, [batch_size, max_ctx_len])
    return tf.nn.softmax(sWM, dim=1) # size = [batch_size, max_ctx_len]

def _compute_avg_probs(probs_list, keep_prob, batch_size, options):
    """Computes the average probability from the list after dropping out some
       of the lists. See the paper.
    """
    probs = tf.stack(probs_list, axis=2) # size = [batch_size, max_ctx_len, num_steps]
    probs += _NUMERICAL_STABILITY_EPSILON
    probs_dropout = tf.nn.dropout(probs,
        noise_shape=[batch_size, 1, options.num_stochastic_answer_pointer_steps],
        keep_prob=keep_prob)
    dropout_nonzero = tf.cast(probs_dropout > 0, dtype=tf.float32)
    dropped_out_probs = dropout_nonzero * probs
    sum_values = tf.reduce_sum(dropped_out_probs, axis=2) # size = [batch_size, max_ctx_len]
    num_values = tf.maximum(tf.reduce_sum(dropout_nonzero, axis=2), 1) # size = [batch_size, max_ctx_len]
    avg_probs = sum_values / num_values
    return avg_probs


def _compute_loss(probs, spans, max_ctx_len):
    labels = tf.minimum(spans, max_ctx_len)
    one_hot_labels = tf.one_hot(spans, depth=max_ctx_len,
        dtype=tf.float32) # size = [batch_size, max_ctx_len]
    return tf.reduce_mean(
        -tf.reduce_sum(one_hot_labels
            * tf.log(probs + _NUMERICAL_STABILITY_EPSILON), axis=1) # size = [batch_size]
    )

def stochastic_answer_pointer(options, ctx, qst, spans, sq_dataset, keep_prob,
    sess, batch_size, use_dropout):
    """Runs a stochastic answer pointer to get start/end span predictions

       Input:
         ctx: The passage representation of shape [batch_size, M, d]
         qst: The question representation of shape [batch_size, N, d]
         spans: The target spans of shape [batch_size, 2]. spans[:,0] are the
            start spans while spans[:,1] are the end spans.
         sq_dataset: A SquadDataBase object
         keep_prob: Probability used for dropout.

       Output:
         (loss, start_span_probs, end_span_probs)
         loss - a single scalar
         start_span_probs - the probabilities of the start spans of shape
            [batch_size, M]
         end_span_probs - the probabilities of the end spans of shape
            [batch_size, M]
    """
    with tf.variable_scope("stochastic_answer_pointer"):
        max_qst_len = sq_dataset.get_max_qst_len()
        max_ctx_len = sq_dataset.get_max_ctx_len()
        ctx_dim = ctx.get_shape()[-1].value # 2 * rnn_size
        assert qst.get_shape()[-1].value == ctx.get_shape()[-1].value
        w = tf.get_variable("w", shape=[ctx_dim], dtype=tf.float32)
        Qw = multiply_tensors(qst, w) # size = [batch_size, max_qst_length]
        sm = tf.nn.softmax(Qw, dim=1) # size = [batch_size, max_qst_length]
        s = tf.matmul(tf.reshape(sm, [batch_size, 1, max_qst_len])
            , qst) # size = [batch_size, 1, 2 * rnn_size]
        ctx_transpose = tf.transpose(ctx, perm=[0, 2, 1]) # size = [batch_size, 2 * rnn_size, max_ctx_len]

        lstm = create_cudnn_lstm(ctx_dim,
            sess, options, "lstm", keep_prob,
            bidirectional=False, layer_size=ctx_dim, num_layers=1)
        state_h = s
        state_c = s

        W = tf.get_variable("W", shape=[ctx_dim, ctx_dim], dtype=tf.float32)
        W_start = tf.get_variable("W_start", [ctx_dim, ctx_dim], dtype=tf.float32)
        W_end = tf.get_variable("W_end", [2 * ctx_dim, ctx_dim], dtype=tf.float32)

        start_prob_list = []
        end_prob_list = []
        for z in range(options.num_stochastic_answer_pointer_steps):
            with tf.variable_scope("step_" + str(z)):
                beta = _get_probs_from_state_and_memory(s, ctx, W,
                    batch_size, ctx_dim, max_ctx_len) # size = [batch_size, max_ctx_len]
                x = tf.reshape(
                        tf.matmul(ctx_transpose,
                            tf.reshape(beta, [batch_size, max_ctx_len, 1])) # size = [batch_size, 2 * rnn_size, 1]
                    , [batch_size, 1, ctx_dim]) # size = [batch_size, 1, 2 * rnn_size]

                s, state_h, state_c = run_cudnn_lstm(x, keep_prob, options,
                    lstm, batch_size, use_dropout,
                    initial_state_h=state_h, initial_state_c=state_c) # size(s) = [batch_size, 1, 2 * rnn_size]

                probs_start = _get_probs_from_state_and_memory(s, ctx, W_start,
                    batch_size, ctx_dim, max_ctx_len) # size = [batch_size, max_ctx_len]
                start_prob_list.append(probs_start)
                additional_end_inputs = tf.matmul(
                    tf.reshape(probs_start, [batch_size, 1, max_ctx_len]), ctx) # size = [batch_size, 1, 2 * rnn_size]
                end_inputs = tf.concat([s, additional_end_inputs], axis=2) # size = [batch_size, 1, 4 * rnn_size]

                probs_end = _get_probs_from_state_and_memory(end_inputs,
                    ctx, W_end, batch_size, ctx_dim, max_ctx_len) # size = [batch_size, max_ctx_len]
                end_prob_list.append(probs_end)
        avg_start_probs = _compute_avg_probs(start_prob_list, keep_prob,
            batch_size, options)
        avg_end_probs = _compute_avg_probs(end_prob_list, keep_prob,
            batch_size, options)
        start_loss = _compute_loss(avg_start_probs, spans[:,0], max_ctx_len)
        end_loss = _compute_loss(avg_end_probs, spans[:,1], max_ctx_len)
        loss = start_loss + end_loss
        return loss, avg_start_probs, avg_end_probs
