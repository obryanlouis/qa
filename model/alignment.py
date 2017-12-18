"""Provides methods for interactive and self alignment as defined in
https://arxiv.org/pdf/1705.02798.pdf
"""

import tensorflow as tf

from model.rnn_util import *
from model.semantic_fusion import *

def run_alignment(options, ctx, qst, ctx_dim, keep_prob, batch_size, sess,
    is_train):
    with tf.variable_scope("alignment"):
        assert options.num_interactive_alignment_hops > 0
        c = ctx
        for z in range(options.num_interactive_alignment_hops):
            with tf.variable_scope("hop_" + str(z)):
                c = align_tensors("question_passage_alignment",
                    options, c, qst, qst, batch_size)
                c = align_tensors("self_alignment", options, c, c, c, batch_size)
                c = run_bidirectional_cudnn_lstm("bidirectional_ctx",
                    c, keep_prob, options, batch_size, sess, is_train)
        return c

def align_tensors(scope, options, primary_tensor, secondary_tensor, apply_tensor,
    batch_size):
    """Runs alignment. Softmax probabilities are computed using the
       primary and secondary tensors, and they are applied to the third
       tensor.

       Input:
         primary_tensor: A tensor of shape [batch_size, M, d]
         secondary_tensor: A tensor of shape [batch_size, N, d]
         apply_tensor: A tensor of shape [batch_size, N, d]

       Output:
         An output tensor of shape [batch_size, M, d]
    """
    with tf.variable_scope(scope):
        coattention_matrix = tf.matmul(primary_tensor, tf.transpose(
            secondary_tensor, perm=[0, 2, 1])) # size = [batch_size, M, N]
        if primary_tensor == secondary_tensor:
            sh = tf.shape(primary_tensor)
            M = sh[1]
            eye = tf.eye(M, batch_shape=[batch_size]) # size = [batch_size, M, M]
            ones = tf.fill([batch_size, M, M], 1.0) # size = [batch_size, M, M]
            mult = ones - eye # size = [batch_size, M, M]
            coattention_matrix *= mult # size = [batch_size, M, M]
        softmax = tf.nn.softmax(coattention_matrix, dim=-1) # size = [batch_size, M, N]
        attention = tf.matmul(softmax, apply_tensor) # size = [batch_size, M, d]
        apply_tensor_times_attention = primary_tensor * attention # size = [batch_size, M, d]
        apply_tensor_minus_attention = primary_tensor - attention # size = [batch_size, M, d]
        return semantic_fusion(primary_tensor, primary_tensor.get_shape()[-1].value,
            [attention, apply_tensor_times_attention, apply_tensor_minus_attention],
            "semantic_fusion") # size = [batch_size, M, d]
