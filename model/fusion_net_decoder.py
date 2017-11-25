"""Provides functionality to get the start & end spans for FusionNet.
"""

import tensorflow as tf

from model.tf_util import *
from model.rnn_util import *


def _attend_to_prob(scope, d, final_ctx, qst_repr, batch_size, max_ctx_length,
    spans):
    with tf.variable_scope(scope):
        Ws = tf.get_variable("Ws", shape=[d, d], dtype=tf.float32)
        logits = tf.matmul(
            final_ctx,
            tf.reshape(multiply_tensors(qst_repr, Ws), # size = [batch_size, d]
                shape=[batch_size, d, 1]) # size = [batch_size, d, 1]
            ) # size = [batch_size, max_ctx_length, 1]
        logits = tf.reshape(logits, shape=[batch_size, max_ctx_length])
        span_probs = tf.nn.softmax(logits, dim=1) # size = [batch_size, max_ctx_length]
        labels = tf.minimum(spans, max_ctx_length - 1)
        loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)) \
               / tf.cast(batch_size, tf.float32)
        return span_probs, loss

def decode_fusion_net(options, sq_dataset, keep_prob, final_ctx,
    qst_understanding, batch_size, spans):
    with tf.variable_scope("fusion_net_decoder"):
        max_ctx_length = sq_dataset.get_max_ctx_len()
        max_qst_length = sq_dataset.get_max_qst_len()
        d = 2 * options.rnn_size
        w = tf.get_variable("w", shape=[d], dtype=tf.float32)
        w_times_qst = multiply_tensors(qst_understanding, w) # size = [batch_size, max_qst_length]
        softmax_w_times_qst = tf.reshape(tf.nn.softmax(w_times_qst, dim=1),
            shape=[batch_size, max_qst_length, 1]) # size = [batch_size, max_qst_length, 1]
        qst_summary = tf.reshape(tf.matmul(
            tf.transpose(qst_understanding, perm=[0, 2, 1]),
            softmax_w_times_qst), shape=[batch_size, d]) # size = [batch_size, d]
        start_probs, start_loss = _attend_to_prob("start_probs", d, final_ctx,
            qst_summary, batch_size, max_ctx_length, spans[:,0])
        weighted_ctx = tf.reshape(
                tf.matmul(tf.reshape(start_probs,
                shape=[batch_size, 1, max_ctx_length]), final_ctx)
            , shape=[batch_size, d]) # size = [batch_size, d]
        gru_cell = create_multi_rnn_cell(options, "answer_lstm",
            keep_prob, layer_size=d)
        vq, _ = gru_cell(weighted_ctx, (qst_summary,))
        end_probs, end_loss = _attend_to_prob("end_loss", d, final_ctx,
            vq, batch_size, max_ctx_length, spans[:,1])
        loss = start_loss + end_loss
        return loss, start_probs, end_probs
