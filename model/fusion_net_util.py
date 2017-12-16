"""Functions that can be used for building FusionNet models.
"""

import tensorflow as tf

from model.semantic_fusion import *
from model.tf_util import *

def vector_fusion(scope, options, vec_a, vec_b, vec_last_dim_size, vec_apply,
    keep_prob):
    """Takes two input vectors and applies a fusion to them to get a weighted
       sum output.
       See the FusionNet paper https://arxiv.org/pdf/1711.07341.pdf.

       Inputs:
            vec_a: First input vector of size [batch_size, A, d]
            vec_b: Second input vector of size [batch_size, B, d]
            vec_last_dim_size: Last dimension, d, from above. A constant, not
                a tensor.
            vec_apply: Vector to apply the fusion of vectors A & B to, of size
                [batch_size, B, e]

       Output:
            Tensor of shape [batch_size, A, e]
    """
    with tf.variable_scope(scope):
        D = tf.get_variable("D", shape=[options.fusion_matrix_dimension],
            dtype=tf.float32) # size = [k]
        D = tf.diag(D) # size = [k, k]
        U = tf.get_variable("U", shape=[vec_last_dim_size,
            options.fusion_matrix_dimension], dtype=tf.float32) # size = [d, k]

        a_times_U = tf.nn.relu(multiply_tensors(tf.nn.dropout(vec_a, keep_prob)
            , U)) # size = [batch_size, A, k]
        b_times_U = tf.nn.relu(multiply_tensors(tf.nn.dropout(vec_b, keep_prob)
            , U)) # size = [batch_size, B, k]

        intermediate = multiply_tensors(b_times_U, D) # size = [batch_size, B, k]
        scores = tf.matmul(a_times_U,
            tf.transpose(intermediate, perm=[0, 2, 1])) # size = [batch_size, A, B]
        softmax = tf.nn.softmax(scores, dim=2) # size = [batch_size, A, B]
        return tf.matmul(softmax, vec_apply) # size = [batch_size, A, e]
