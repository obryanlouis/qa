"""Provides functionality to run conductor self attention and inner fusion.
"""

import tensorflow as tf

from model.tf_util import *


def conductor_net_self_attention(options, outer_fusion, keep_prob):
    with tf.variable_scope("conductor_net_self_attention"):
        h = outer_fusion # size = [batch_size, max_ctx_length, *]
        h_dim = h.get_shape()[-1]
        B_last = None
        for z in range(options.num_conductor_net_self_attention_layers):
            with tf.variable_scope("layer_" + str(z)):
                M = tf.matmul(h, tf.transpose(h, perm=[0, 2, 1])) # size = [batch_size, max_ctx_length, max_ctx_length]
                S = tf.nn.softmax(M, dim=2)
                B = tf.matmul(S, h) # size(B) = [batch_size, max_ctx_length, *]

                B_inputs = [B]
                if B_last is not None: # I assume.
                    B_inputs.extend([B_last, B_last * B])
                B_in = tf.concat(B_in)

                B_in_dim = B_in.get_shape()[-1]
                Wb = tf.get_variable("Wb", shape=[B_in_dim, h_dim], dtype=tf.float32)
                bb = tf.get_variable("bb", shape=[h_dim], dtype=tf.float32)
                Wf = tf.get_variable("Wf", shape=[B_in_dim, h_dim], dtype=tf.float32)
                bf = tf.get_variable("bf", shape=[h_dim], dtype=tf.float32)

                B_tilde = tf.tanh(multiply_tensors(B_in, Wb) + bb)
                f = tf.nn.sigmoid(multiply_tensors(B_in, Wf) + bf)
                h = (1 - f) * h + f * B_tilde
                B_last = B
        return h
