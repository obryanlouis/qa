"""Defines a base model to hold a common model interface.
"""

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from model.input_util import *

class BaseModel:
    def __init__(self, options, tf_iterators, sq_dataset):
        self.sq_dataset = sq_dataset
        self.options = options
        self.embeddings = sq_dataset.embeddings
        self.batch_size = None
        self.num_words = self.embeddings.shape[0]
        self.word_dim = self.embeddings.shape[1]
        self.embedding_placeholder = None
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
        self.ctx_inputs = None
        self.qst_inputs = None

    def get_data_index_iterator(self):
        return self.data_index_iterator

    def get_embedding_placeholder(self):
        return self.embedding_placeholder

    def get_keep_prob_placeholder(self):
        return self.keep_prob

    def _create_word_embedding_placeholder(self):
        # Need to add a vector for padding words and a vector for unique words.
        # The result, which should be used in further calculations, is stored
        # in self.words_placeholder rather than self.embedding_placeholder.
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.num_words, self.word_dim])
        self.padding_vector = tf.get_variable("padding_vector", shape=[1, self.word_dim])
        self.unique_word_vector = tf.get_variable("unique_word_vector", shape=[1, self.word_dim])
        with tf.device("/cpu:0"):
            # Order of normal vocab, pad, unk must match vocab_util.py
            self.words_placeholder = tf.concat([self.embedding_placeholder,
                self.padding_vector, self.unique_word_vector], axis=0)

    def setup(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self._create_word_embedding_placeholder()
        self.batch_size = tf.shape(self.ctx_iterator)[0]
        self.ctx_inputs, self.qst_inputs = create_model_inputs(
                self.words_placeholder, self.ctx_iterator,
                self.qst_iterator, self.ctx_chars_iterator,
                self.qst_chars_iterator,
                self.options, self.wiq_iterator,
                self.wic_iterator, self.sq_dataset)

    def get_start_spans(self):
        return tf.argmax(self.get_start_span_probs(), axis=1)

    def get_end_spans(self):
        return tf.argmax(self.get_end_span_probs(), axis=1)

    @abstractmethod
    def get_loss_op(self):
        pass

    @abstractmethod
    def get_start_span_probs(self):
        pass

    @abstractmethod
    def get_end_span_probs(self):
        pass

