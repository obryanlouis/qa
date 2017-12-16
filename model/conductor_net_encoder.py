"""Provides functionality to run conductor net encoding.
"""

import tensorflow as tf

from model.rnn_util import *
from model.encoding_util import *


def encode_conductor_net(ctx, qst, keep_prob, use_dropout, batch_size,
    options, sess):
    """Performs the 'encoding' step of the conductor net model.

       Inputs:
            ctx: The passage of size [batch_size, max_ctx_length, *]
            qst: The question of size [batch_size, max_qst_length, *]

       Outputs:
            (encoded_ctx, encoded_qst)
            encoded_ctx: The encoded passage of size [batch_size, max_ctx_length, 2 * rnn_size * num_conductor_net_encoder_layers]
            encoded_qst: The encoded question of size [batch_size, max_qst_length, 2 * rnn_size]
    """
    with tf.variable_scope("conductor_net_encoder"):
        # Create an independently-encoded question representation.
        vq = run_bidirectional_cudnn_lstm("independent_question", qst, keep_prob,
            options, batch_size, sess, use_dropout) # size = [batch_size, max_qst_length, 2 * rnn_size]
        # Create representations for the question and passage using a shared LSTM.
        hp, uq = encode_passage_and_question(options, ctx, qst, keep_prob,
            sess, batch_size, use_dropout) # size(hp) = [batch_size, max_ctx_length, 2 * rnn_size], size(uq) = size(vq)
        ht = hp
        uq_transpose = tf.transpose(uq, perm=[0, 2, 1])
        question_passage_layers = []
        for z in range(options.num_conductor_net_encoder_layers):
            M = tf.matmul(ht, uq_transpose) # size = [batch_size, max_ctx_length, max_qst_length]
            A = tf.nn.softmax(M, dim=2) # size(A) = size(M)
            ht = tf.matmul(A, vq) # size = [batch_size, max_ctx_length, 2 * rnn_size]
            question_passage_layers.append(ht)
        encoded_ctx = tf.concat(question_passage_layers, axis=-1) # size = [batch_size, max_ctx_length, 2 * rnn_size * num_conductor_net_encoder_layers]
        # TODO: Verify that the question encoded with the shared LSTM, uq, is
        # the right one.
        return encoded_ctx, uq
