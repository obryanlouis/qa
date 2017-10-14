"""Defines a small debug data set. Models should get nearly 100% on it.
"""

import numpy as np
import os
import pickle
import preprocessing.constants as constants

from datasets.test_dataset import TestDataset
from datasets.squad_data_base import SquadDataBase
from datasets.file_util import *
from preprocessing.vocab_util import get_vocab

WORD_DIM = 7
WORD_LEN = 5

class TestData(SquadDataBase):
    def __init__(self, options):
        data_dir = options.data_dir
        self.data_dir = data_dir
        self.vocab = get_vocab(data_dir)
        self.train_ds = TestDataset(
            load_text_file(data_dir, constants.TRAIN_FULL_TEXT_TOKENS_FILE),
            self.vocab, WORD_LEN)
        self.dev_ds = self.train_ds
        vocab_size = self.vocab.get_vocab_size_without_pad_or_unk()
        self.embeddings = np.random.uniform(-1.0, 1.0, size=(vocab_size, WORD_DIM))
        self.word_vec_size = WORD_DIM
        self.max_word_len = WORD_LEN
