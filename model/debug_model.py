"""Implements a debug model that should get 100% exact match and f1 scores.
"""

import tensorflow as tf

from model.base_model import BaseModel

class DebugModel(BaseModel):
    def __init__(self, options, tf_iterators, sq_dataset, embeddings):
        super().__init__(options, tf_iterators, sq_dataset, embeddings)
        self.loss = tf.get_variable("debug_loss", shape=[],
                dtype=tf.float32, initializer=tf.zeros_initializer())

    def get_loss_op(self):
        return self.loss

    def get_start_span_probs(self):
        return tf.one_hot(self.spn_iterator[:,0],
                depth=self.sq_dataset.get_max_ctx_len(),
                dtype=tf.float32)

    def get_end_span_probs(self):
        return tf.one_hot(self.spn_iterator[:,1],
                depth=self.sq_dataset.get_max_ctx_len(),
                dtype=tf.float32)

    def get_start_spans(self):
        return self.spn_iterator[:,0]

    def get_end_spans(self):
        return self.spn_iterator[:,1]
