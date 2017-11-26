"""Provides methods for interactive and self alignment as defined in
https://arxiv.org/pdf/1705.02798.pdf
"""

import tensorflow as tf

from model.rnn_util import *
from model.semantic_fusion import *

def run_alignment(options, ctx, qst, ctx_dim, sq_dataset, keep_prob):
    with tf.variable_scope("alignment"):
        assert options.num_interactive_alignment_hops > 0
        c = ctx
        for z in range(options.num_interactive_alignment_hops):
            with tf.variable_scope("hop_" + str(z)):
                c = _interactive_alignment(options, c, qst, ctx_dim, sq_dataset,
                        keep_prob)
                c = _self_alignment(options, c, ctx_dim, sq_dataset)
                c = run_bidirectional_lstm("bidirectional_ctx_" + str(z),
                    c, keep_prob, options)
        return c

def _interactive_alignment(options, ctx, qst, ctx_dim, sq_dataset, keep_prob):
    """Runs interactive alignment on the passage and question.

       Input:
         ctx: The passage of shape [batch_size, M, d]
         qst: The question of shape [batch_size, N, d]
         ctx_dim: d, from above

       Output:
         A question-aware representation of shape [batch_size, M, d]
    """
    with tf.variable_scope("interactive_alignment"):
        coattention_matrix = tf.matmul(qst, tf.transpose(ctx, perm=[0, 2, 1])) # size = [batch_size, N, M]
        softmax = tf.nn.softmax(coattention_matrix, dim=1) # size = [batch_size, N, M]
        q_attention = tf.matmul(
            tf.transpose(softmax, perm=[0, 2, 1]) # size = [batch_size, M, N]
            , qst) # size = [batch_size, M, d]
        c_times_q_attention = ctx * q_attention # size = [batch_size, M, d]
        c_minus_q_attention = ctx - q_attention # size = [batch_size, M, d]
        return semantic_fusion(ctx, ctx_dim,
            [q_attention, c_times_q_attention, c_minus_q_attention], "interactive_alignment_semantic_fusion") # size = [batch_size, M, d]

def _self_alignment(options, ctx, ctx_dim, sq_dataset):
    """Runs self alignment on the passage.

       Input:
         ctx: The (question-aware) passage of shape [batch_size, M, d]
         ctx_dim: d, from above

       Output:
         A self-aligned representation of shape [batch_size, M, d]
    """
    with tf.variable_scope("self_alignment"):
        sh = tf.shape(ctx)
        batch_size, M = sh[0], sh[1]
        coattention_matrix = tf.matmul(ctx, tf.transpose(ctx, perm=[0, 2, 1])) # size = [batch_size, M, M]
        eye = tf.eye(M, batch_shape=[batch_size]) # size = [batch_size, M, M]
        ones = tf.fill([batch_size, M, M], 1.0) # size = [batch_size, M, M]
        mult = ones - eye # size = [batch_size, M, M]
        B = coattention_matrix * mult # size = [batch_size, M, M]
        softmax = tf.nn.softmax(B, dim=2) # size = [batch_size, M, M]
        attention = tf.matmul(softmax, ctx) # size = [batch_size, M, d]
        ctx_times_attention = ctx * attention # size = [batch_size, M, d]
        ctx_minus_attention = ctx - attention # size = [batch_size, M, d]
        return semantic_fusion(ctx, ctx_dim, [
            attention, ctx_times_attention, ctx_minus_attention], "self_aligned_semantic_fusion") # size = [batch_size, M, d]
