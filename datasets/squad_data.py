"""Provides SQuAD data for training and dev.
"""

import glob
import numpy as np
import os
import pickle
import preprocessing.constants as constants
import preprocessing.embedding_util as embedding_util
import re
import tensorflow as tf

from datasets.iterator_wrapper import *
from datasets.file_util import *
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
    def __init__(self, files_dir, options, vocab):
        self.options = options
        self.vocab = vocab

        self.context_placeholder = tf.placeholder(tf.int32, shape=[None, options.max_ctx_length])
        self.question_placeholder = tf.placeholder(tf.int32, shape=[None, options.max_qst_length])
        self.span_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.word_in_question_placeholder = tf.placeholder(tf.float32, shape=[None, options.max_ctx_length])
        self.word_in_context_placeholder = tf.placeholder(tf.float32, shape=[None, options.max_qst_length])
        self.question_ids_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.context_pos_placeholder = tf.placeholder(tf.uint8, shape=[None, options.max_ctx_length])
        self.context_ner_placeholder = tf.placeholder(tf.uint8, shape=[None, options.max_ctx_length])
        self.question_pos_placeholder = tf.placeholder(tf.uint8, shape=[None, options.max_qst_length])
        self.question_ner_placeholder = tf.placeholder(tf.uint8, shape=[None, options.max_qst_length])

        self.current_file_number = 0
        self.files_dir = files_dir

        self.all_files_in_files_dir = glob.glob(os.path.join(files_dir, "*"))
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

        assert len(self.context_files) > 0
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

        self.zip_ds = tf.contrib.data.Dataset.zip({
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

    def get_sentences_for_all_gnd_truths(self, question_id):
        gnd_truths_list = self.question_ids_to_ground_truths[question_id]
        sentences = []
        for start_idx, end_idx in gnd_truths_list:
            sentences.append(self.get_sentence(question_id, start_idx, end_idx))
        return sentences

    def get_sentence(self, example_idx, start_idx, end_idx):
        list_text_tokens = self.text_tokens_dict[example_idx]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def _get_files_list(self, file_name_pattern):
        search_pattern = os.path.join(self.files_dir,
            file_name_pattern.replace("%d", "[0-9]+"))
        return [f for f in self.all_files_in_files_dir if \
            re.match(search_pattern, f)]

    def _create_ds(self, placeholder):
        return tf.contrib.data.Dataset.from_tensor_slices(placeholder) \
            .batch(self.options.batch_size) \
            .repeat()

    def setup_with_tf_session(self, sess):
        self.sess = sess
        self.handle = sess.run(self.iterator.string_handle())
        self.load_next_file()

    def _load_2d_np_arr_with_possible_padding(self, full_file_name,
            max_second_dim, pad_value):
        np_arr = np.load(full_file_name)[:,:max_second_dim]
        return np.pad(np_arr,
            pad_width=((0, 0), (0, max_second_dim - np_arr.shape[1])),
            mode="constant",
            constant_values=(pad_value,))

    def load_next_file(self):
        max_ctx_len = self.options.max_ctx_length
        max_qst_len = self.options.max_qst_length
        WORD_PAD_ID = self.vocab.PAD_ID
        self.ctx = self._load_2d_np_arr_with_possible_padding(self.context_files[self.current_file_number], max_ctx_len, pad_value=WORD_PAD_ID)
        self.qst = self._load_2d_np_arr_with_possible_padding(self.question_files[self.current_file_number], max_qst_len, pad_value=WORD_PAD_ID)
        self.spn = np.load(self.span_files[self.current_file_number])
        self.wiq = self._load_2d_np_arr_with_possible_padding(self.word_in_question_files[self.current_file_number], max_ctx_len, pad_value=0)
        self.wic = self._load_2d_np_arr_with_possible_padding(self.word_in_context_files[self.current_file_number], max_qst_len, pad_value=0)
        self.qid = np.load(self.question_ids_files[self.current_file_number])
        self.question_ids_to_ground_truths = load_text_file(
            self.question_ids_to_gnd_truths_files[self.current_file_number])
        self.ctx_pos = self._load_2d_np_arr_with_possible_padding(self.context_pos_files[self.current_file_number], max_ctx_len, pad_value=0)
        self.ctx_ner = self._load_2d_np_arr_with_possible_padding(self.context_ner_files[self.current_file_number], max_ctx_len, pad_value=0)
        self.qst_pos = self._load_2d_np_arr_with_possible_padding(self.question_pos_files[self.current_file_number], max_qst_len, pad_value=0)
        self.qst_ner = self._load_2d_np_arr_with_possible_padding(self.question_ner_files[self.current_file_number], max_qst_len, pad_value=0)
        self.text_tokens_dict = load_text_file(
            self.text_tokens_files[self.current_file_number])

        self.current_file_number = (self.current_file_number + 1) % len(self.context_files)
        self.sess.run(self.iterator.initializer, feed_dict={
            self.context_placeholder: self.ctx,
            self.question_placeholder: self.qst,
            self.span_placeholder: self.spn,
            self.word_in_question_placeholder: self.wiq,
            self.word_in_context_placeholder: self.wic,
            self.question_ids_placeholder: self.qid,
            self.context_pos_placeholder: self.ctx_pos,
            self.context_ner_placeholder: self.ctx_ner,
            self.question_pos_placeholder: self.qst_pos,
            self.question_ner_placeholder: self.qst_ner,
        })
        self.samples_processed_in_current_files = 0
        self.num_samples_in_current_files = self.ctx.shape[0]

    def increment_samples_processed(self, num_samples):
        self.samples_processed_in_current_files += num_samples
        if self.samples_processed_in_current_files >= self.num_samples_in_current_files:
            self.load_next_file()


# Class that provides Squad data through Tensorflow iterators by cycling through
# a set of Numpy & pickle files. There doesn't seem to be a native way to do this
# easily through the Dataset API.
class SquadData:
    def __init__(self, options):
        self.options = options
        training_dir = os.path.join(options.data_dir,
            constants.TRAIN_FOLDER_NAME)
        validation_dir = os.path.join(options.data_dir,
            constants.DEV_FOLDER_NAME)
        self.vocab = get_vocab(options.data_dir)

        self.train_ds = _SquadDataset(training_dir, options, self.vocab)
        self.dev_ds = _SquadDataset(validation_dir, options, self.vocab)
        print("%d train files batches, %d dev" % (
            len(self.train_ds.context_files),
            len(self.dev_ds.context_files)))

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.contrib.data.Iterator.from_string_handle(
            self.handle, self.train_ds.zip_ds.output_types,
            self.train_ds.zip_ds.output_shapes)

        self.embeddings = np.load(os.path.join(options.data_dir,
            constants.EMBEDDING_FILE))
        self.word_chars = np.load(os.path.join(options.data_dir,
            constants.VOCAB_CHARS_FILE))
        # Add in all 0 embeddings for the padding and unk vectors
        self.embeddings = np.concatenate((self.embeddings,
            np.zeros((2, self.embeddings.shape[1]))))
        self.word_chars = np.concatenate((self.word_chars,
            np.full((2, self.word_chars.shape[1]),
                fill_value=embedding_util.CHAR_UNK_ID)))
        self.word_vec_size = constants.WORD_VEC_DIM
        self.max_word_len = constants.MAX_WORD_LEN

    def get_max_ctx_len(self):
        return self.options.max_ctx_length

    def get_max_qst_len(self):
        return self.options.max_qst_length

    def get_word_vec_size(self):
        return self.word_vec_size

    def get_current_dev_file_number(self):
        return self.dev_ds.current_file_number

    def get_num_dev_files(self):
        return len(self.dev_ds.context_files)

    def estimate_total_dev_ds_size(self):
        """Must not call this before setup_with_tf_session is called."""
        return self.dev_ds.num_samples_in_current_files \
            * len(self.dev_ds.context_files)

    def estimate_total_train_ds_size(self):
        """Must not call this before setup_with_tf_session is called."""
        return self.train_ds.num_samples_in_current_files \
            * len(self.train_ds.context_files)

    def increment_train_samples_processed(self, num_samples):
        self.train_ds.increment_samples_processed(num_samples)

    def increment_val_samples_processed(self, num_samples):
        self.dev_ds.increment_samples_processed(num_samples)

    def setup_with_tf_session(self, sess):
        print("Setting up tensorflow data iterator handles")
        self.train_ds.setup_with_tf_session(sess)
        self.dev_ds.setup_with_tf_session(sess)

    def get_iterator_handle(self):
        return self.handle

    def get_train_handle(self):
        return self.train_ds.handle

    def get_dev_handle(self):
        return self.dev_ds.handle

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
