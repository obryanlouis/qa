"""Provides functionality to run conductor self attention and inner fusion.
"""

import tensorflow as tf

from model.alignment import *
from model.dropout_util import *
from model.rnn_util import *
from model.semantic_fusion import *
from model.tf_util import *


def conductor_net_self_attention(options, outer_fusion, keep_prob,
    batch_size, sess, use_dropout):
    with tf.variable_scope("conductor_net_self_attention"):
        h = outer_fusion # size = [batch_size, max_ctx_length, *]
        for z in range(options.num_conductor_net_self_attention_layers):
            h = align_tensors("self_alignment_" + str(z), options, h, h, h,
                batch_size)
        return h
