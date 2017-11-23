"""Experimental function to run an RNN over sentence fragments.
"""

import tensorflow as tf

from model.rnn_util import *


def get_sentence_fragments(sq_dataset, scope, options, ctx, ctx_dim, keep_prob):
    """Returns a tensor representing a list of sentence fragments produced by
       running an RNN over segments of the context.

       Inputs:
            ctx: A tensor of shape [batch_size, max_ctx_length, d]
            ctx_dim: The dimension 'd' from above

       Output:
            A tensor of shape [batch_size, *, 2 * rnn_size]
    """
    with tf.variable_scope(scope):
        frag_outputs = []
        word_idx = 0
        cell_fw = create_multi_rnn_cell(options, "cell_fw", keep_prob)
        cell_bw = create_multi_rnn_cell(options, "cell_bw", keep_prob)
        data_set_len = sq_dataset.get_max_ctx_len()
        while word_idx < data_set_len:
            next_word_idx = min(data_set_len,
                word_idx + options.sent_frag_length)
            print("word_idx", word_idx, "next_word_idx", next_word_idx)
            sent_frag = ctx[:, word_idx:next_word_idx, :]
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                sent_frag,
                dtype=tf.float32)
            fw_outputs, bw_outputs = rnn_outputs
            print("Creating last output fw")
            last_output_fw = fw_outputs[:, next_word_idx - word_idx - 1, :] # size = [batch_size, rnn_size]
            print("Creating first output bw")
            last_output_bw = bw_outputs[:, 0, :] # size = [batch_size, rnn_size]
            frag_outputs.append(last_output_fw)
            frag_outputs.append(last_output_bw)
            word_idx = next_word_idx
        sent_frags_len = len(frag_outputs)
        print("Num sent frags:", sent_frags_len)
        sent_frags = tf.stack(frag_outputs, axis=1) # size = [batch_size, sent_frags_len, 2 * rnn_size]
        sent_cell_fw = create_multi_rnn_cell(options, "sent_fw", keep_prob)
        sent_cell_bw = create_multi_rnn_cell(options, "sent_bw", keep_prob)
        with tf.variable_scope("sent_frag_rnn"):
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                sent_cell_fw,
                sent_cell_bw,
                sent_frags,
                dtype=tf.float32)
        sent_fw_output, sent_bw_output = rnn_outputs # size(each) = [batch_size, sent_frags_len, rnn_size]
        sent_frags = tf.concat([sent_fw_output, sent_bw_output], axis=2) # size = [batch_size, sent_frags_len, 2 * rnn_size]
        print("sent_frags shape", sent_frags.get_shape())
        return sent_frags, sent_frags_len
