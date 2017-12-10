"""Implements an r-net model.
"""

import tensorflow as tf

from model.base_model import BaseModel
from model.tf_util import *
from model.rnn_util import *
from model.encoding_util import *
from model.decoding_util import *

class Rnet(BaseModel):
    def setup(self):
        super(Rnet, self).setup()
        # Step 1. Encode the passage and question.
        ctx_dropout = tf.nn.dropout(self.ctx_inputs, self.keep_prob)
        qst_dropout = tf.nn.dropout(self.qst_inputs, self.keep_prob)
        passage_outputs, question_outputs = encode_passage_and_question(
                self.options, ctx_dropout, qst_dropout, self.keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder)
        # Step 2. Run a bi-lstm over the passage with the question as the
        # attention.
        ctx_attention = run_attention(self.options,
                passage_outputs, 2 * self.options.rnn_size, question_outputs,
                2 * self.options.rnn_size, "attention_birnn", self.batch_size,
                self.sq_dataset.get_max_qst_len(), self.keep_prob,
                self.sq_dataset.get_max_ctx_len(), num_rnn_layers=1)
        # Step 3. Run self-matching attention of the previous result over
        # itself.
        ctx_attention = run_attention(self.options,
                ctx_attention, 2 * self.options.rnn_size, ctx_attention,
                2 * self.options.rnn_size, "self_matching_attention", self.batch_size,
                self.sq_dataset.get_max_ctx_len(), self.keep_prob,
                self.sq_dataset.get_max_ctx_len(), num_rnn_layers=1)
        # Step 4. Use a bi-lstm over the context again.
        ctx_attention = run_bidirectional_cudnn_lstm("ctx_bidirectional",
            ctx_attention, self.keep_prob, self.options,
            self.batch_size, self.sess, self.use_dropout_placeholder)
        # Step 5. Create the answer output layer using answer-pointer boundary
        # decoding.
        self.loss, self.start_span_probs, self.end_span_probs = \
            decode_answer_pointer_boundary(self.options, self.batch_size,
                self.keep_prob, self.spn_iterator, ctx_attention,
                self.sq_dataset, question_outputs)
