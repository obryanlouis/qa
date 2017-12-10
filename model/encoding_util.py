"""Functions for encoding passage and question inputs.
"""

import tensorflow as tf

from model.cudnn_lstm_wrapper import *
from model.rnn_util import *

def encode_low_level_and_high_level_representations(sess, scope, options, inputs,
    keep_prob, batch_size, is_train):
    """Runs a bi-directional rnn over the inputs to get low level and
       high level outputs.

       Inputs:
            inputs: A tensor shaped [batch_size, M, d]

       Outputs:
            A tensor shaped [batch_size, M, 2 * rnn_size]
    """
    with tf.variable_scope(scope):
        lstm = create_cudnn_lstm(inputs.get_shape()[2], sess, options, "low",
            keep_prob)
        low_level_outputs = run_cudnn_lstm_and_return_outputs(inputs,
            keep_prob, options, lstm, batch_size, is_train)
        lstm = create_cudnn_lstm(inputs.get_shape()[2], sess, options, "high",
            keep_prob)
        high_level_outputs = run_cudnn_lstm_and_return_outputs(low_level_outputs,
            keep_prob, options, lstm, batch_size, is_train)

        return low_level_outputs, high_level_outputs

def encode_passage_and_question(options, passage, question, keep_prob,
    sess, batch_size, is_train):
    """Returns (passage_outputs, question_outputs), which are both of size
       [batch_size, max ctx length | max qst length, 2 * rnn_size].
    """
    with tf.variable_scope("preprocessing_lstm"):
        with tf.variable_scope("passage"):
            lstm = create_cudnn_lstm(passage.get_shape()[2], sess, options,
                "passage_lstm", keep_prob)
            passage_outputs = run_cudnn_lstm_and_return_outputs(passage,
                keep_prob, options, lstm, batch_size, is_train) # size = [batch_size, max_ctx_length, 2 * rnn_size]
        with tf.variable_scope("question"):
            lstm = create_cudnn_lstm(question.get_shape()[2], sess, options,
                "question_lstm", keep_prob)
            question_outputs = run_cudnn_lstm_and_return_outputs(question,
                keep_prob, options, lstm, batch_size, is_train) # size = [batch_size, max_qst_length, 2 * rnn_size]
        return passage_outputs, question_outputs
