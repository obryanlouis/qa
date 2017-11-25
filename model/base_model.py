"""Defines a base model to hold a common model interface.
"""

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from model.input_util import *

class BaseModel:
    def __init__(self, options, tf_iterators, sq_dataset, embeddings):
        self.sq_dataset = sq_dataset
        self.options = options
        self.batch_size = None
        self.num_words = self.sq_dataset.embeddings.shape[0]
        self.word_dim = self.sq_dataset.embeddings.shape[1]
        self.char_embedding_placeholder = None
        self.keep_prob = None
        self.ctx_iterator = tf_iterators.ctx
        self.qst_iterator = tf_iterators.qst
        self.ctx_chars_iterator = tf_iterators.ctx_chars
        self.qst_chars_iterator = tf_iterators.qst_chars
        self.spn_iterator = tf_iterators.spn
        self.data_index_iterator = tf_iterators.data_index
        self.wiq_iterator = tf_iterators.wiq
        self.wic_iterator = tf_iterators.wic
        self.ctx_pos_iterator = tf_iterators.ctx_pos
        self.qst_pos_iterator = tf_iterators.qst_pos
        self.ctx_ner_iterator = tf_iterators.ctx_ner
        self.qst_ner_iterator = tf_iterators.qst_ner
        self.ctx_inputs = None
        self.qst_inputs = None
        self.embeddings = embeddings

    def get_data_index_iterator(self):
        return self.data_index_iterator

    def get_keep_prob_placeholder(self):
        return self.keep_prob

    def setup(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.shape(self.ctx_iterator)[0]
        model_inputs = create_model_inputs(
                self.embeddings, self.ctx_iterator,
                self.qst_iterator, self.ctx_chars_iterator,
                self.qst_chars_iterator,
                self.options, self.wiq_iterator,
                self.wic_iterator, self.sq_dataset,
                self.ctx_pos_iterator, self.qst_pos_iterator,
                self.ctx_ner_iterator, self.qst_ner_iterator)
        self.ctx_inputs = model_inputs.ctx_concat
        self.qst_inputs = model_inputs.qst_concat
        self.ctx_glove = model_inputs.ctx_glove
        self.qst_glove = model_inputs.qst_glove

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
