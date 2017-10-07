"""Implements an r-net model.
"""

import tensorflow as tf

from model.base_model import BaseModel
from model.tf_util import *
from model.rnn_util import *
from model.encoding_util import *
from model.decoding_util import *

class Rnet(BaseModel):
    def __init__(self, options, embeddings, tf_iterators):
        super().__init__(options, embeddings, tf_iterators)
        self.loss = None
        self.start_span_probs = None
        self.end_span_probs = None

    def setup(self):
        super(Rnet, self).setup()
        # Step 1. Encode the passage and question.
        ctx_dropout = tf.nn.dropout(self.ctx_embedded, self.keep_prob)
        qst_dropout = tf.nn.dropout(self.qst_embedded, self.keep_prob)
        passage_outputs, question_outputs = encode_passage_and_question(
                self.options, ctx_dropout, qst_dropout, self.keep_prob)
        # Step 2. Run a bi-lstm over the passage with the question as the
        # attention.
        ctx_attention = run_attention(self.options, passage_outputs,
                2 * self.options.rnn_size, question_outputs,
                2 * self.options.rnn_size, "attention_birnn", self.batch_size,
                self.options.max_qst_length, self.keep_prob, num_rnn_layers=1)
        # Step 3. Run self-matching attention of the previous result over
        # itself.
        ctx_attention = run_attention(self.options, ctx_attention,
                2 * self.options.rnn_size, ctx_attention,
                2 * self.options.rnn_size, "self_matching_attention", self.batch_size,
                self.options.max_ctx_length, self.keep_prob, num_rnn_layers=1)
        # Step 3. Create the answer output layer using answer-pointer boundary
        # decoding.
        self.loss, self.start_span_probs, self.end_span_probs = \
            decode_answer_pointer_boundary(self.options, self.batch_size,
                self.keep_prob, self.spn_iterator, ctx_attention)

    def get_loss_op(self):
        return self.loss

    def _get_start_span_probs(self):
        return self.start_span_probs

    def _get_end_span_probs(self):
        return self.end_span_probs


