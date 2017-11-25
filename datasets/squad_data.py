"""Provides SQuAD data for training and dev.
"""

import glob
import numpy as np
import os
import pickle
import preprocessing.constants as constants
import tensorflow as tf

from datasets.iterator_wrapper import *
from datasets.file_util import *
from datasets.squad_data_base import SquadDataBase
from preprocessing.vocab_util import get_vocab


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

# Class used to hold data and state for a single dataset: the training
# dataset or the validation dataset.
class _SquadDataset:
    def __init__(self, files_dir, options):
        self.options = options

        self.context_placeholder = tf.placeholder(tf.int32)
        self.question_placeholder = tf.placeholder(tf.int32)
        self.span_placeholder = tf.placeholder(tf.int32)
        self.word_in_question_placeholder = tf.placeholder(tf.float32)
        self.word_in_context_placeholder = tf.placeholder(tf.float32)
        self.question_ids_placeholder = tf.placeholder(tf.int32)
        self.question_ids_to_gnd_truths = None
        self.context_pos_placeholder = tf.placeholder(tf.uint8)
        self.context_ner_placeholder = tf.placeholder(tf.uint8)
        self.question_pos_placeholder = tf.placeholder(tf.uint8)
        self.question_ner_placeholder = tf.placeholder(tf.uint8)
        self.text_tokens = None

        self.current_file_number = 0
        self.files_dir = files_dir

        self.context_files = \
            self._get_files_list(constants.CONTEXT_FILE_PATTERN)
        self.question_files = \
            self._get_files_list(constants.QUESTION_FILE_PATTERN)
        self.span_files = \
            self._get_files_list(constants.SPAN_FILE_PATTERN)
        self.word_in_question_files = \
            self._get_files_list(constants.WORD_IN_QUESTION_FILE_PATTERN)
        self.word_in_context_files = \
            self._get_files_list(constants.WORD_IN_CONTEXT_FILE_PATTERN)
        self.question_ids_files = \
            self._get_files_list(constants.QUESTION_IDS_FILE_PATTERN)
        self.question_ids_to_gnd_truths_files = \
            self._get_files_list(constants.QUESTION_IDS_TO_GND_TRUTHS_FILE_PATTERN)
        self.context_pos_files = \
            self._get_files_list(constants.CONTEXT_POS_FILE_PATTERN)
        self.context_ner_files = \
            self._get_files_list(constants.CONTEXT_NER_FILE_PATTERN)
        self.question_pos_files = \
            self._get_files_list(constants.QUESTION_POS_FILE_PATTERN)
        self.question_ner_files = \
            self._get_files_list(constants.QUESTION_NER_FILE_PATTERN)
        self.text_tokens_files = \
            self._get_files_list(constants.TEXT_TOKENS_FILE_PATTERN)

        assert len(self.context_files) == len(self.question_files)
        assert len(self.context_files) == len(self.span_files)
        assert len(self.context_files) == len(self.word_in_question_files)
        assert len(self.context_files) == len(self.word_in_context_files)
        assert len(self.context_files) == len(self.question_ids_files)
        assert len(self.context_files) == len(self.question_ids_to_gnd_truths_files)
        assert len(self.context_files) == len(self.context_pos_files)
        assert len(self.context_files) == len(self.context_ner_files)
        assert len(self.context_files) == len(self.question_pos_files)
        assert len(self.context_files) == len(self.question_ner_files)
        assert len(self.context_files) == len(self.text_tokens_files)

        self.zip_ds = tf.data.Dataset.zip({
            _CONTEXT_KEY: self._create_ds(self.context_placeholder),
            _QUESTION_KEY: self._create_ds(self.question_placeholder),
            _SPAN_KEY: self._create_ds(self.span_placeholder),
            _WORD_IN_QUESTION_KEY: self._create_ds(self.word_in_question_placeholder),
            _WORD_IN_CONTEXT_KEY: self._create_ds(self.word_in_context_placeholder),
            _QUESTION_IDS_KEY: self._create_ds(self.question_ids_placeholder),
            _CONTEXT_POS_KEY: self._create_ds(self.context_pos_placeholder),
            _CONTEXT_NER_KEY: self._create_ds(self.context_ner_placeholder),
            _QUESTION_POS_KEY: self._create_ds(self.question_pos_placeholder),
            _QUESTION_NER_KEY: self._create_ds(self.question_ner_placeholder),
        })
        self.iterator = self.zip_ds.make_initializable_iterator()
        self.handle = None
        self.samples_processed_in_current_files = 0
        self.num_samples_in_current_files = 0
        self.finished_processing_files = False

    def _get_files_list(self, file_name_pattern):
        return sorted(glob.glob(os.path.join(self.files_dir,
            file_name_prefix.replace("%d", "[0-9]+"))))

    def _create_ds(self, placeholder):
        return tf.data.Dataset.from_tensor_slices(placeholder) \
            .batch(self.options.batch_size) \
            .repeat()

    def setup_with_tf_session(self, sess):
        self.sess = sess
        self.handle = sess.run(self.iterator.string_handle())
        self.load_next_file()

    def load_next_file(self):
        ctx = np.load(self.context_files[self.current_file_number])
        qst = np.load(self.question_files[self.current_file_number])
        spn = np.load(self.span_files[self.current_file_number])
        wiq = np.load(self.word_in_question_files[self.current_file_number])
        wic = np.load(self.word_in_context_files[self.current_file_number])
        qid = np.load(self.question_ids_files[self.current_file_number])
        qid_gnd = load_text_file(self.question_ids_to_gnd_truths_files[self.current_file_number])
        ctx_pos = np.load(self.context_pos_files[self.current_file_number])
        ctx_ner = np.load(self.context_ner_files[self.current_file_number])
        qst_pos = np.load(self.question_pos_files[self.current_file_number])
        qst_ner = np.load(self.question_ner_files[self.current_file_number])
        txt_tokens = load_text_file(self.text_tokens_files[self.current_file_number])

        self.question_ids_to_gnd_truths = qid_gnd
        self.text_tokens = txt_tokens

        if self.current_file_number >= len(self.context_files) - 1:
            self.finished_processing_files = True
        self.current_file_number = (self.current_file_number + 1) % len(self.context_files)
        self.sess.run(self.iterator.initializer, feed_dict={
            self.context_placeholder: ctx,
            self.question_placeholder: qst,
            self.span_placeholder: spn,
            self.word_in_question_placeholder: wiq,
            self.word_in_context_placeholder: wic,
            self.question_ids_placeholder: qid,
            self.context_pos_placeholder: ctx_pos,
            self.context_ner_placeholder: ctx_ner,
            self.question_pos_placeholder: qst_pos,
            self.question_ner_placeholder: qst_ner,
        })
        self.samples_processed_in_current_files = 0
        self.num_samples_in_current_files = ctx.shape[0]

    def increment_samples_processed(self, num_samples):
        self.samples_processed_in_current_files += num_samples
        if self.samples_processed_in_current_files >= self.num_samples_in_current_files:
            self.load_next_file()


# Class that provides Squad data through Tensorflow iterators by cycling through
# a set of Numpy & pickle files. There doesn't seem to be a native way to do this
# easily through the Dataset API.
class SquadData(SquadDataBase):
    def __init__(self, options):
        self.options = options
        training_dir = os.path.join(options.data_dir,
            constants.TRAINING_FOLDER_NAME)
        validation_dir = os.path.join(options.data_dir,
            constants.DEV_FOLDER_NAME)

        self.train_ds = _SquadDataset(training_dir, options)
        self.val_ds = _SquadDataset(validation_dir, options)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.contrib.data.Iterator.from_string_handle(
            self.handle, self.train_ds.zip_ds.output_types,
            self.train_ds.zip_ds.output_shapes)

        self.vocab = get_vocab(data_dir)
        self.embeddings = np.load(os.path.join(data_dir,
            constants.EMBEDDING_FILE))
        # Add in all 0 embeddings for the padding and unk vectors
        self.embeddings = np.concatenate((self.embeddings,
            np.zeros((2, self.embeddings.shape[1]))))
        self.word_chars = np.load(os.path.join(options.data_dir,
            constants.VOCAB_CHARS_FILE))
        self.word_vec_size = constants.WORD_VEC_DIM
        self.max_word_len = constants.MAX_WORD_LEN

    def get_word_vec_size(self):
        return self.word_vec_size

    def get_current_dev_file_number(self):
        return self.val_ds.current_file_number

    def get_num_dev_files(self):
        return len(self.val_ds.context_files)

    def estimate_total_dev_ds_size(self):
        """Must not call this before setup_with_tf_session is called."""
        return self.val_ds.num_samples_in_current_files \
            * len(self.val_ds.context_files)

    def estimate_total_train_ds_size(self):
        """Must not call this before setup_with_tf_session is called."""
        return self.train_ds.num_samples_in_current_files \
            * len(self.train_ds.context_files)

    def is_done_processing_dev_files(self):
        return self.train_ds.finished_processing_files

    def get_num_samples_in_current_dev_file(self):
        return self.train_ds.num_samples_in_current_files

    def increment_train_samples_processed(self, num_samples):
        self.train_ds.increment_samples_processed(num_samples)

    def increment_val_samples_processed(self, num_samples):
        self.val_ds.increment_samples_processed(num_samples)

    def setup_with_tf_session(self, sess):
        print("Setting up tensorflow data iterator handles")
        self.train_ds.setup_with_tf_session(sess)
        self.val_ds.setup_with_tf_session(sess)

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_ds.handle

    def get_dev_handle(self):
        return self.val_ds.handle

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
