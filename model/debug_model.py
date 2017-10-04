"""Implements a debug model that should get 100% exact match and f1 scores.
"""

import tensorflow as tf

from model.logistic_regression import LogisticRegression

class DebugModel(LogisticRegression):
    def __init__(self, options, embeddings, tf_iterators):
        super().__init__(options, embeddings, tf_iterators)

    def get_start_spans(self):
        return self.spn_iterator[:,0]

    def get_end_spans(self):
        return self.spn_iterator[:,1]
