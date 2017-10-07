"""Implements a logistic regression model.
"""

import tensorflow as tf

from model.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, options, embeddings, tf_iterators):
        super().__init__(options, embeddings, tf_iterators)
        self.loss = None
        self.start_span_probs = None
        self.end_span_probs = None

    def setup(self):
        super(LogisticRegression, self).setup()
        concat_inputs = tf.concat([self.ctx_embedded, self.qst_embedded], axis=1) # size = [batch_size, max_ctx_length + max_qst_length, word_dim]
        input_dim = (self.options.max_ctx_length + self.options.max_qst_length) * self.word_dim
        inputs = tf.reshape(concat_inputs, [self.batch_size, input_dim])
        max_len = tf.reshape(
                    tf.tile(tf.constant([self.options.max_ctx_length - 1])
                        , [self.batch_size])
                , [self.batch_size])
        start_loss, self.start_span_probs = self._add_logistic_regression("start_span_probs", inputs, input_dim, max_len, self.spn_iterator[:,0])
        end_loss, self.end_span_probs = self._add_logistic_regression("end_span_probs", inputs, input_dim, max_len, self.spn_iterator[:,1])
        self.loss = (start_loss + end_loss) / 2

    def _add_logistic_regression(self, scope, inputs, input_dim, max_len, expected_spans):
        with tf.variable_scope(scope):
            output_dim = self.options.max_ctx_length
            W = tf.get_variable("W", shape=[input_dim, output_dim], dtype=tf.float32)
            b = tf.get_variable("b", shape=[output_dim], dtype=tf.float32)
            logits = tf.matmul(inputs, W) + b # size = [batch_size, max_ctx_length]
            probs = tf.nn.softmax(logits)
            labels = tf.reshape(tf.minimum(expected_spans, max_len), [self.batch_size])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)) / tf.cast(self.batch_size, tf.float32)
            return loss, probs

    def get_loss_op(self):
        return self.loss

    def _get_start_span_probs(self):
        return self.start_span_probs

    def _get_end_span_probs(self):
        return self.end_span_probs
