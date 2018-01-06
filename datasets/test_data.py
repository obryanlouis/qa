"""Defines a small debug data set. Models should get nearly 100% on it.
"""

import glob
import math
import numpy as np
import os
import preprocessing.constants as constants
import tensorflow as tf

from datasets.iterator_wrapper import *
from preprocessing.vocab_util import get_vocab


_WORD_DIM = 300
_WORD_LEN = 25

_NUM_SAMPLES_PER_FILE = 100
_NUM_FILES = 3

_CTX_LEN = 15
_QST_LEN = 17

_CONTEXT_KEY = "context"
_QUESTION_KEY = "question"
_SPAN_KEY = "span"
_WORD_IN_QUESTION_KEY = "word_in_question"
_WORD_IN_CONTEXT_KEY = "word_in_context"
_QUESTION_IDS_KEY = "question_ids"
_CONTEXT_POS_KEY = "context.pos"
_CONTEXT_NER_KEY = "context.ner"
_QUESTION_POS_KEY = "question.pos"
_QUESTION_NER_KEY = "question.ner"

class _TestDataset:
    def __init__(self, test_data):
        self.test_data = test_data
        self.question_ids_to_squad_ids = None
        self.question_ids_to_passage_context = None

    def get_sentences_for_all_gnd_truths(self, ctx_id):
        sentences = []
        start_idx, end_idx = self.test_data.spn[ctx_id, 0], self.test_data.spn[ctx_id, 1]
        for _ in range(3):
            sentences.append(self.get_sentence(ctx_id, start_idx, end_idx))
        return sentences

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.test_data.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])


class TestData:
    def __init__(self, options):
        self.vocab = get_vocab(options.data_dir)
        vocab_size = self.vocab.get_vocab_size_without_pad_or_unk()
        self.embeddings = np.random.uniform(-1.0, 1.0,
            size=(vocab_size + 2, _WORD_DIM))
        self.word_vec_size = _WORD_DIM
        self.max_word_len = _WORD_LEN
        self.text_tokens = [
            [self.vocab.get_word_for_id(np.random.randint(0, vocab_size)) \
                for _ in range(_CTX_LEN)] \
            for _ in range(_NUM_SAMPLES_PER_FILE) ]


        self.ctx = np.random.randint(0, vocab_size, size=(_NUM_SAMPLES_PER_FILE, _CTX_LEN))
        self.qst = np.random.randint(0, vocab_size, size=(_NUM_SAMPLES_PER_FILE, _QST_LEN))
        self.spn = np.zeros((_NUM_SAMPLES_PER_FILE, 2), dtype=np.int32)
        for z in range(_NUM_SAMPLES_PER_FILE):
            spns = sorted([np.random.randint(0, _CTX_LEN),
                           np.random.randint(0, _CTX_LEN)])
            self.spn[z, 0] = spns[0]
            self.spn[z, 1] = spns[1]
        self.data_index = np.arange(self.ctx.shape[0])
        self.word_in_question = np.random.randint(0, 2, size=(_NUM_SAMPLES_PER_FILE, _CTX_LEN))
        self.word_in_context = np.random.randint(0, 2, size=(_NUM_SAMPLES_PER_FILE, _QST_LEN))
        self.question_ids = self.data_index
        self.context_pos  = np.random.randint(0, 2**7, size=(_NUM_SAMPLES_PER_FILE, _CTX_LEN), dtype=np.int8)
        self.question_pos = np.random.randint(0, 2**7, size=(_NUM_SAMPLES_PER_FILE, _QST_LEN), dtype=np.int8)
        self.context_ner  = np.random.randint(0, 2**7, size=(_NUM_SAMPLES_PER_FILE, _CTX_LEN), dtype=np.int8)
        self.question_ner = np.random.randint(0, 2**7, size=(_NUM_SAMPLES_PER_FILE, _QST_LEN), dtype=np.int8)
        self.word_chars = np.random.randint(0, 2**8 - 2, size=(vocab_size + 2, _WORD_LEN), dtype=np.uint8)


        self.ctx_ds = tf.contrib.data.Dataset.from_tensor_slices(self.ctx)
        self.qst_ds = tf.contrib.data.Dataset.from_tensor_slices(self.qst)
        self.spn_ds = tf.contrib.data.Dataset.from_tensor_slices(self.spn)
        self.data_index_ds = tf.contrib.data.Dataset.from_tensor_slices(self.data_index)
        self.word_in_question_ds = tf.contrib.data.Dataset.from_tensor_slices(self.word_in_question)
        self.word_in_context_ds = tf.contrib.data.Dataset.from_tensor_slices(self.word_in_context)
        self.question_ids_ds = tf.contrib.data.Dataset.from_tensor_slices(self.question_ids)
        self.context_pos_ds = tf.contrib.data.Dataset.from_tensor_slices(self.context_pos)
        self.question_pos_ds = tf.contrib.data.Dataset.from_tensor_slices(self.question_pos)
        self.context_ner_ds = tf.contrib.data.Dataset.from_tensor_slices(self.context_ner)
        self.question_ner_ds = tf.contrib.data.Dataset.from_tensor_slices(self.question_ner)

        self.zip_ds = tf.contrib.data.Dataset.zip({
            _CONTEXT_KEY: self.ctx_ds,
            _QUESTION_KEY: self.qst_ds,
            _SPAN_KEY: self.spn_ds,
            _WORD_IN_QUESTION_KEY: self.word_in_question_ds,
            _WORD_IN_CONTEXT_KEY: self.word_in_context_ds,
            _QUESTION_IDS_KEY: self.question_ids_ds,
            _CONTEXT_POS_KEY: self.context_pos_ds,
            _CONTEXT_NER_KEY: self.context_ner_ds,
            _QUESTION_POS_KEY: self.question_pos_ds,
            _QUESTION_NER_KEY: self.question_ner_ds,
        }) \
        .batch(options.batch_size) \
        .repeat() \
        .shuffle(buffer_size=10)
        self.zip_iterator = self.zip_ds.make_initializable_iterator()

        self.train_handle = None
        self.val_handle = None
        self.handle = tf.placeholder(tf.string, shape=[], name="data_handle")
        self.iterator = tf.contrib.data.Iterator.from_string_handle(
            self.handle, self.zip_ds.output_types,
            self.zip_ds.output_shapes)
        self.total_samples_processed = 0

        self.train_ds = _TestDataset(self)
        self.dev_ds = self.train_ds

    def get_max_ctx_len(self):
        return _CTX_LEN

    def get_max_qst_len(self):
        return _QST_LEN

    def get_word_vec_size(self):
        return self.word_vec_size

    def get_current_dev_file_number(self):
        return math.floor(
            (float(self.total_samples_processed) / float(_NUM_SAMPLES_PER_FILE))) \
            % self.estimate_total_dev_ds_size()

    def get_num_dev_files(self):
        return _NUM_FILES

    def estimate_total_dev_ds_size(self):
        return _NUM_FILES * _NUM_SAMPLES_PER_FILE

    def estimate_total_train_ds_size(self):
        return _NUM_FILES * _NUM_SAMPLES_PER_FILE

    def get_num_samples_in_current_dev_file(self):
        return _NUM_SAMPLES_PER_FILE

    def increment_train_samples_processed(self, num_samples):
        pass

    def increment_val_samples_processed(self, num_samples):
        self.total_samples_processed += num_samples

    def setup_with_tf_session(self, sess):
        self.sess = sess
        self.sess.run(self.zip_iterator.initializer, feed_dict={})
        self.train_handle = sess.run(self.zip_iterator.string_handle())
        self.val_handle = self.train_handle

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_handle

    def get_dev_handle(self):
        return self.val_handle

    def create_iterators(self):
        with tf.device("/cpu:0"):
            next_elem = self.iterator.get_next()
            return IteratorWrapper(
                next_elem[_CONTEXT_KEY],
                next_elem[_QUESTION_KEY],
                next_elem[_SPAN_KEY],
                next_elem[_WORD_IN_QUESTION_KEY],
                next_elem[_WORD_IN_CONTEXT_KEY],
                next_elem[_QUESTION_IDS_KEY],
                next_elem[_CONTEXT_POS_KEY],
                next_elem[_CONTEXT_NER_KEY],
                next_elem[_QUESTION_POS_KEY],
                next_elem[_QUESTION_NER_KEY])
