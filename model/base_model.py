
import tensorflow as tf

from abc import ABCMeta, abstractmethod

class BaseModel:
    def __init__(self, options, embeddings):
        self.options = options
        self.ctx_placeholder = None
        self.qst_placeholder = None
        self.spn_placeholder = None
        self.embeddings = embeddings
        self.batch_size = None
        self.num_words = embeddings.shape[0]
        self.word_dim = embeddings.shape[1]
        self.embedding_placeholder = None
        self.keep_prob = None

    def get_embedding_placeholder(self):
        return self.embedding_placeholder

    def get_contexts_placeholder(self):
        return self.ctx_placeholder

    def get_questions_placeholder(self):
        return self.qst_placeholder

    def get_spans_placeholder(self):
        return self.spn_placeholder

    def get_keep_prob_placeholder(self):
        return self.keep_prob

    def _create_embedding_placeholder(self):
        # Need to add a vector for padding words and a vector for unique words.
        # The result, which should be used in further calculations, is stored
        # in self.words_placeholder rather than self.embedding_placeholder.
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.num_words, self.word_dim])
        self.padding_vector = tf.get_variable("padding_vector", shape=[1, self.word_dim])
        self.unique_word_vector = tf.get_variable("unique_word_vector", shape=[1, self.word_dim])
        with tf.device("/cpu:0"):
            self.words_placeholder = tf.concat([self.embedding_placeholder,
                self.padding_vector, self.unique_word_vector], axis=0)

    def setup(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self._create_embedding_placeholder()
        self.ctx_placeholder = tf.placeholder(tf.int32, shape=[None, self.options.max_ctx_length])
        self.qst_placeholder = tf.placeholder(tf.int32, shape=[None, self.options.max_qst_length])
        self.spn_placeholder = tf.placeholder(tf.int32, shape=[None, 2])

        self.batch_size = tf.shape(self.ctx_placeholder)[0]
        with tf.device("/cpu:0"):
            self.ctx_embedded = tf.nn.embedding_lookup(self.words_placeholder, self.ctx_placeholder) # size = [batch_size, max_ctx_length, word_dim]
            self.qst_embedded = tf.nn.embedding_lookup(self.words_placeholder, self.qst_placeholder) # size = [batch_size, max_qst_length, word_dim]

    def get_start_spans(self):
        return tf.argmax(self._get_start_span_probs(), axis=1)

    def get_end_spans(self):
        return tf.argmax(self._get_end_span_probs(), axis=1)

    @abstractmethod
    def get_loss_op(self):
        pass

    @abstractmethod
    def _get_start_span_probs(self):
        pass

    @abstractmethod
    def _get_end_span_probs(self):
        pass

