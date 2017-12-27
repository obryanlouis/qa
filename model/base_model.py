"""Defines a base model to hold a common model interface.
"""

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from model.input_util import *

class BaseModel:
    def __init__(self, options, tf_iterators, sq_dataset, embeddings,
            word_chars, cove_cells, sess):
        self.sq_dataset = sq_dataset
        self.options = options
        self.num_words = self.sq_dataset.embeddings.shape[0]
        self.word_dim = self.sq_dataset.embeddings.shape[1]
        self.ctx_iterator = tf_iterators.ctx
        self.qst_iterator = tf_iterators.qst
        self.spn_iterator = tf_iterators.spn
        self.data_index_iterator = tf_iterators.question_ids
        self.wiq_iterator = tf_iterators.word_in_question
        self.wic_iterator = tf_iterators.word_in_context
        self.ctx_pos_iterator = tf_iterators.context_pos
        self.qst_pos_iterator = tf_iterators.question_pos
        self.ctx_ner_iterator = tf_iterators.context_ner
        self.qst_ner_iterator = tf_iterators.question_ner
        self.embeddings = embeddings
        self.word_chars = word_chars
        self.cove_cells = cove_cells
        self.sess = sess

    def get_use_dropout_placeholder(self):
        return self.use_dropout_placeholder

    def get_data_index_iterator(self):
        return self.data_index_iterator

    def get_keep_prob_placeholder(self):
        return self.keep_prob

    def get_input_keep_prob_placeholder(self):
        return self.input_keep_prob

    def get_rnn_keep_prob_placeholder(self):
        return self.rnn_keep_prob

    def setup(self):
        self.use_dropout_placeholder = tf.placeholder(tf.bool, name="use_dropout")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.rnn_keep_prob = tf.placeholder(tf.float32, name="rnn_keep_prob")
        self.batch_size = tf.shape(self.ctx_iterator)[0]
        model_inputs = create_model_inputs(self.sess,
                self.embeddings, self.ctx_iterator,
                self.qst_iterator,
                self.options, self.wiq_iterator,
                self.wic_iterator, self.sq_dataset,
                self.ctx_pos_iterator, self.qst_pos_iterator,
                self.ctx_ner_iterator, self.qst_ner_iterator,
                self.word_chars, self.cove_cells,
                self.use_dropout_placeholder,
                self.batch_size)
        self.ctx_inputs = model_inputs.ctx_concat
        self.qst_inputs = model_inputs.qst_concat
        self.ctx_glove = model_inputs.ctx_glove
        self.qst_glove = model_inputs.qst_glove
        self.ctx_cove = model_inputs.ctx_cove
        self.qst_cove = model_inputs.qst_cove
        self.ctx_inputs_without_cove = model_inputs.ctx_concat_without_cove 
        self.qst_inputs_without_cove = model_inputs.qst_concat_without_cove 

    def get_qst(self):
        return self.qst_iterator

    def get_start_spans(self):
        return tf.argmax(self.get_start_span_probs(), axis=1)

    def get_end_spans(self):
        return tf.argmax(self.get_end_span_probs(), axis=1)

    def get_loss_op(self):
        return self.loss

    def get_start_span_probs(self):
        return self.start_span_probs

    def get_end_span_probs(self):
        return self.end_span_probs
