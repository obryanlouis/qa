"""Implements a debug model that should get 100% exact match and f1 scores.
"""

import tensorflow as tf

from model.logistic_regression import LogisticRegression

class DebugModel(LogisticRegression):
    def __init__(self, options, embeddings):
        super().__init__(options, embeddings)

    def get_start_spans(self):
        return self.spn_placeholder[:,0]

    def get_end_spans(self):
        return self.spn_placeholder[:,1]
