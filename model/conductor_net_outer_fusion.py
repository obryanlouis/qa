"""Provides functionality to run conductor outer fusion.
"""

import tensorflow as tf

from model.tf_util import *


def conductor_net_outer_fusion(options, encoded_passsage, keep_prob):
    with tf.variable_scope("conductor_net_outer_fusion"):
        C = encoded_passsage
        C_dim = C.get_shape()[-1]
        for z in range(options.num_conductor_net_outer_fusion_layers):
            with tf.variable_scope("layer_" + str(z)):
                Wc = tf.get_variable("Wc", shape=[C_dim, C_dim], dtype=tf.float32)
                bc = tf.get_variable("bc", shape=[C_dim], dtype=tf.float32)
                Wz = tf.get_variable("Wz", shape=[C_dim, C_dim], dtype=tf.float32)
                bz = tf.get_variable("bz", shape=[C_dim], dtype=tf.float32)

                C = tf.nn.dropout(C, keep_prob) # TODO: use sequence dropout.
                C_tilde = tf.nn.relu(multiply_tensors(C, Wc) + bc) # size = size(C)
                z = tf.nn.sigmoid(multiply_tensors(C, Wz) + bz) # size = size(C)
                C = (1 - z) * C + z * C_tilde
        return C
