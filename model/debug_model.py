"""Implements a debug model that should get 100% exact match and f1 scores.
"""

import tensorflow as tf

from model.base_model import BaseModel

class DebugModel(BaseModel):
    def __init__(self, options, embeddings, tf_iterators):
        super().__init__(options, embeddings, tf_iterators)
        self.loss = tf.get_variable("debug_loss", shape=[],
                dtype=tf.float32, initializer=tf.zeros_initializer())

    def get_loss_op(self):
        return self.loss

    def get_start_spans(self):
        return self.spn_iterator[:,0]

    def get_end_spans(self):
        return self.spn_iterator[:,1]
