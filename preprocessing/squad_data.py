"""Provides a way to load data for training or evaluation.
"""

import numpy as np
import os
import pickle
import preprocessing.constants as constants

from preprocessing.vocab_util import get_vocab

class Dataset:
    def __init__(self, text_tokens, contexts, questions, spans, options):
        self.text_tokens = text_tokens
        self.ctx = contexts[:, :options.max_ctx_length]
        self.qst = questions[:, :options.max_qst_length]
        self.spn = spans
        self.data_index = np.arange(self.ctx.shape[0])

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def get_size(self):
        return self.ctx.shape[0]

class SquadData:
    def __init__(self, options):
        data_dir = options.data_dir
        self.data_dir = data_dir
        self.vocab = get_vocab(data_dir)
        self.train_ds = Dataset(
            self._load_text_file(constants.TRAIN_FULL_TEXT_TOKENS_FILE),
            self._load_file(constants.TRAIN_CONTEXT_FILE),
            self._load_file(constants.TRAIN_QUESTION_FILE),
            self._load_file(constants.TRAIN_SPAN_FILE),
            options)
        self.dev_ds = Dataset(
            self._load_text_file(constants.DEV_FULL_TEXT_TOKENS_FILE),
            self._load_file(constants.DEV_CONTEXT_FILE),
            self._load_file(constants.DEV_QUESTION_FILE),
            self._load_file(constants.DEV_SPAN_FILE),
            options)
        self.embeddings = np.load(os.path.join(data_dir,
                    constants.EMBEDDING_FILE))

    def _load_text_file(self, file_name):
        f = open(os.path.join(self.data_dir, file_name), "rb")
        text_tokens = pickle.load(f)
        f.close()
        return text_tokens

    def _load_file(self, file_name):
        return np.load(os.path.join(self.data_dir, file_name))
