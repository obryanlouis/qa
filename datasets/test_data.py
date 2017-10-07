"""Defines a small debug data set. Models should get nearly 100% on it.
"""

import numpy as np
import os
import pickle
import preprocessing.constants as constants

from datasets.test_dataset import TestDataset
from datasets.file_util import *
from preprocessing.vocab_util import get_vocab

class TestData:
    def __init__(self, options):
        data_dir = options.data_dir
        self.data_dir = data_dir
        self.vocab = get_vocab(data_dir)
        self.train_ds = TestDataset(
            load_text_file(data_dir, constants.TRAIN_FULL_TEXT_TOKENS_FILE),
            load_file(data_dir, constants.TRAIN_CONTEXT_FILE),
            load_file(data_dir, constants.TRAIN_QUESTION_FILE),
            load_file(data_dir, constants.TRAIN_SPAN_FILE),
            options)
        self.dev_ds = self.train_ds
        self.embeddings = np.load(os.path.join(data_dir,
                    constants.EMBEDDING_FILE))
