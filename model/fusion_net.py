"""Implements the "FusionNet" model.
https://arxiv.org/pdf/1711.07341.pdf
"""

import tensorflow as tf

from model.alignment import *
from model.base_model import BaseModel
from model.encoding_util import *
from model.fusion_net_decoder import *
from model.fusion_net_util import *
from model.memory_answer_pointer import *
from model.rnn_util import *

class FusionNet(BaseModel):
    def setup(self):
        super(FusionNet, self).setup()
        # Step 1. Form the "low-level" and "high-level" representations of
        # the context and question.
        ctx_dropout = tf.nn.dropout(self.ctx_inputs, self.keep_prob)
        qst_dropout = tf.nn.dropout(self.qst_inputs, self.keep_prob)
        ctx_low_level, ctx_high_level = \
            encode_low_level_and_high_level_representations(
                "ctx_preprocessing", self.options, ctx_dropout, self.keep_prob)
        qst_low_level, qst_high_level = \
            encode_low_level_and_high_level_representations(
                "qst_preprocessing", self.options, qst_dropout, self.keep_prob)

        # Step 2. Get the "question understanding" representation.
        qst_understanding = run_bidirectional_lstm("qst_understanding",
            tf.concat([qst_low_level, qst_high_level], axis=-1),
            self.keep_prob, self.options)

        # Step 3. Fuse the "history-of-word" question vectors into the
        # "history-of-word" context vectors.
        ctx_how = tf.concat([self.ctx_glove, ctx_low_level, ctx_high_level], axis=-1)
        qst_how = tf.concat([self.qst_glove, qst_low_level, qst_high_level], axis=-1)
        how_dim = ctx_how.get_shape()[-1]
        ctx_low_fusion = vector_fusion("ctx_qst_low_fusion", self.options,
            ctx_how, qst_how, how_dim, qst_low_level)
        ctx_high_fusion = vector_fusion("ctx_qst_high_fusion", self.options,
            ctx_how, qst_how, how_dim, qst_high_level)
        ctx_understanding_fusion = vector_fusion("ctx_qst_understanding_fusion",
            self.options, ctx_how, qst_how, how_dim, qst_understanding)
        ctx_fusion_input = tf.concat([ctx_how,
            ctx_low_fusion, ctx_high_fusion, ctx_understanding_fusion], axis=-1)
        ctx_full_qst_fusion = run_bidirectional_lstm("ctx_qst_fusion",
            ctx_fusion_input, self.keep_prob, self.options) # size = [batch_size, max_ctx_length, 2 * rnn_size]

        # Step 4. Use the "history-of-word" context vectors to perform
        # self matching and then get the final context "understanding" vectors.
        self_matching_ctx_how = tf.concat([ctx_fusion_input,
            ctx_full_qst_fusion], axis=-1)
        how_dim = self_matching_ctx_how.get_shape()[-1]
        self_matching_fusion = vector_fusion("self_matching_fusion",
             self.options, self_matching_ctx_how, self_matching_ctx_how,
             how_dim, ctx_full_qst_fusion)
        final_ctx = run_bidirectional_lstm("final_ctx",
            tf.concat([ctx_full_qst_fusion, self_matching_fusion], axis=-1),
            self.keep_prob, self.options) # size = [batch_size, max_ctx_length, 2 * rnn_size]

        # Step 5. Decode the answer start & end.
        self.loss, self.start_span_probs, self.end_span_probs = \
            decode_fusion_net(self.options, self.sq_dataset, self.keep_prob,
                final_ctx, qst_understanding, self.batch_size, self.spn_iterator)
