"""Defines a model for a stochastic answer network.
https://arxiv.org/abs/1712.03556v1
"""

import tensorflow as tf

from model.alignment import *
from model.base_model import BaseModel
from model.dropout_util import *
from model.encoding_util import *
from model.memory_answer_pointer import *
from model.rnn_util import *
from model.stochastic_answer_pointer import *

class StochasticAnswerNetwork(BaseModel):
    def setup(self):
        super(StochasticAnswerNetwork, self).setup()

        # Step 1. Map all the inputs to the same dimension and run encoders.
        ffn_ctx = _feed_forward_network("feed_forward_ctx",
            sequence_dropout(self.ctx_inputs_without_cove, self.input_keep_prob),
            self.options, self.options.rnn_size, self.keep_prob)
        ffn_qst = _feed_forward_network("feed_forward_qst",
            sequence_dropout(self.qst_inputs_without_cove, self.input_keep_prob),
            self.options, self.options.rnn_size, self.keep_prob)
        # Concatenate the word-level inputs with Cove and then run encoders
        # that will use the same LSTMs for both the passage and question.
        ctx = tf.concat([ffn_ctx, sequence_dropout(self.ctx_cove,
            self.input_keep_prob)], axis=-1)
        qst = tf.concat([ffn_qst, sequence_dropout(self.qst_cove,
            self.input_keep_prob)], axis=-1)
        with tf.variable_scope("low_level_encoder"):
            ctx_low, qst_low = encode_passage_and_question(
                self.options, ctx, qst, self.rnn_keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder) # size(each) = [batch_size, max_ctx_len | max_qst_len, 2 * rnn_size]
        with tf.variable_scope("high_level_encoder"):
            ctx_high, qst_high = encode_passage_and_question(
                self.options, ctx_low, qst_low, self.rnn_keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder) # size(each) = [batch_size, max_ctx_len | max_qst_len, 2 * rnn_size]
        ctx_low = _maxout("ctx_low_maxout",
            ctx_low) # size = [batch_size, max_ctx_len, rnn_size]
        qst_low = _maxout("qst_low_maxout",
            qst_low) # size = [batch_size, max_qst_len, rnn_size]
        ctx_high = _maxout("ctx_high_maxout",
            ctx_high) # size = [batch_size, max_ctx_len, rnn_size]
        qst_high = _maxout("qst_high_maxout",
            qst_high) # size = [batch_size, max_qst_len, rnn_size]

        Hp = tf.concat([ctx_low, ctx_high], axis=-1) # size = [batch_size, max_ctx_len, 2 * rnn_size]
        Hq = tf.concat([qst_low, qst_high], axis=-1) # size = [batch_size, max_qst_len, 2 * rnn_size]

        # Step 2. Generate the "memory" for the network.
        Hp_tilde = _feed_forward_network("Hp_tilde",
            Hp, self.options,
            2 * self.options.rnn_size, self.keep_prob)
        Hq_tilde = _feed_forward_network("Hq_tilde",
            Hq, self.options,
            2 * self.options.rnn_size, self.keep_prob)
        C = tf.nn.dropout(_attention(Hp_tilde, Hq_tilde),
            self.keep_prob) # size = [batch_size, max_ctx_len, max_qst_len]
        Up = tf.concat([Hp, tf.matmul(C, Hq)], axis=-1) # size = [batch_size, max_ctx_len, 4 * rnn_size]
        self_attention = _attention(Up, Up) # size = [batch_size, max_ctx_len, max_ctx_len]

        # Kill the diagonal so that words don't match themselves.
        max_ctx_len = self.sq_dataset.get_max_ctx_len()
        eye = tf.eye(max_ctx_len, batch_shape=[self.batch_size]) # size = [batch_size, max_ctx_len, max_ctx_len]
        ones = tf.fill([self.batch_size, max_ctx_len, max_ctx_len], 1.0) # size = [batch_size, max_ctx_len, max_ctx_len]
        mult = ones - eye # size = [batch_size, max_ctx_len, max_ctx_len]
        self_attention *= mult # size = [batch_size, max_ctx_len, max_ctx_len]

        Up_tilde = tf.matmul(self_attention, Up) # size = [batch_size, max_ctx_len, 4 * rnn_size]
        memory = run_bidirectional_cudnn_lstm("memory",
            tf.concat([Up, Up_tilde], axis=-1), self.rnn_keep_prob, self.options,
            self.batch_size, self.sess, self.use_dropout_placeholder) # size = [batch_size, max_ctx_len, 2 * rnn_size]

        # Step 3. Use an answer pointer mechanism to get the loss,
        # and start & end span probabilities
        self.loss, self.start_span_probs, self.end_span_probs = \
            stochastic_answer_pointer(self.options, memory, Hq,
                self.spn_iterator, self.sq_dataset, self.keep_prob,
                self.sess, self.batch_size, self.use_dropout_placeholder)

def _attention(primary_sequences, secondary_sequences):
    """Gets the attention between the two sequences using the standard mechanism.

       Inputs:
         primary_sequences: A tensor of shape [batch_size, M, d]
         secondary_sequences: A tensor of shape [batch_size, N, d]

       Output:
         A tensor of shape [batch_size, M, N]
    """
    mult = tf.matmul(primary_sequences,
        tf.transpose(secondary_sequences, perm=[0, 2, 1]))
    return tf.nn.softmax(mult, dim=-1)

def _maxout(scope, inputs):
    with tf.variable_scope(scope):
        half_input_dim = int(inputs.get_shape()[-1].value) // 2
        maxes = tf.maximum(inputs[:, :, half_input_dim:],
                inputs[:, :, :half_input_dim])
        return maxes

def _feed_forward_network(scope, inputs, options, output_dim, keep_prob):
    with tf.variable_scope(scope):
        inputs = tf.nn.dropout(inputs, keep_prob)
        input_dim = inputs.get_shape()[-1].value
        w1 = tf.get_variable("w1", dtype=tf.float32, shape=[input_dim, output_dim])
        b1 = tf.get_variable("b1", dtype=tf.float32, shape=[output_dim])
        w2 = tf.get_variable("w2", dtype=tf.float32, shape=[output_dim, output_dim])
        b2 = tf.get_variable("b2", dtype=tf.float32, shape=[output_dim])
        y = tf.nn.relu(multiply_tensors(inputs, w1) + b1)
        return multiply_tensors(y, w2) + b2
