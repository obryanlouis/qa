"""Provides functions for using Semantic Fusion Units.
https://arxiv.org/pdf/1705.02798.pdf
"""

import tensorflow as tf

from model.tf_util import *

def semantic_fusion(input_vector, input_dim, fusion_vectors, scope):
    """Runs a semantic fusion unit on the input vector and the fusion vectors
       to produce an output.

       Input:
         input_vector: Vector of size [batch_size, ..., input_dim].
            This vector must have the same size as each of the vectors in
            the fusion_vectors list.
         fusion_vectors: List of vectors of size [batch_size, ..., input_dim].
            The vectors in this list must all have the same size as the
            input_vector.
         input_dim: The last dimension of the vectors (python scalar)

       Output:
         Vector of the same size as the input_vector.
    """
    with tf.variable_scope(scope):
        assert len(fusion_vectors) > 0
        stacked_vectors = tf.concat(fusion_vectors + [input_vector], axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
        num_total_vectors = len(fusion_vectors) + 1
        Wr = tf.get_variable("Wr", dtype=tf.float32, shape=[num_total_vectors * input_dim, input_dim])
        Wg = tf.get_variable("Wg", dtype=tf.float32, shape=[num_total_vectors * input_dim, input_dim])
        br = tf.get_variable("br", dtype=tf.float32, shape=[input_dim])
        bg = tf.get_variable("bg", dtype=tf.float32, shape=[input_dim])
        r = tf.tanh(multiply_tensors(stacked_vectors, Wr) + br) # size = [batch_size, ..., input_dim]
        g = tf.sigmoid(multiply_tensors(stacked_vectors, Wg) + bg) # size = [batch_size, ..., input_dim]
        return g * r + (1 - g) * input_vector # size = [batch_size, ..., input_dim]
