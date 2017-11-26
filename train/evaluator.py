"""Evaluates a set of models.
"""

import boto3
import os
import shutil
import time

from model.model_types import MODEL_TYPES
from train.evaluation_util import *
from train.model_builder import ModelBuilder
from train.model_util import *
from train.print_utils import *
from train.s3_util import *
from train.train_util import *

class Evaluator:
    def __init__(self, options):
        self.options = options
        self.s3_save_key = create_s3_save_key(options)
        self.checkpoint_file_name = create_checkpoint_file_name(options)

    def evaluate(self):
        self.s3 = boto3.resource('s3') if self.options.use_s3 else None
        os.makedirs(self.options.checkpoint_dir, exist_ok=True)

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            self.session = create_session()
            self.sq_dataset = create_sq_dataset(self.options)
            embedding_placeholder = tf.placeholder(tf.float32,
                shape=self.sq_dataset.embeddings.shape)
            embedding_var = \
                tf.Variable(embedding_placeholder, trainable=False)
            word_chars_placeholder = tf.placeholder(tf.float32,
                shape=self.sq_dataset.word_chars.shape)
            word_chars_var = \
                tf.Variable(word_chars_placeholder, trainable=False)
            self.model_builder = ModelBuilder(None, self.options,
                self.sq_dataset, embedding_var, word_chars_var,
                compute_gradients=False)
            self.saver = create_saver()
            maybe_restore_model(self.s3, self.s3_save_key, self.options,
                self.session, self.checkpoint_file_name, self.saver,
                embedding_placeholder, self.sq_dataset.embeddings,
                word_chars_placeholder, self.sq_dataset.word_chars)
            maybe_print_model_parameters(self.options)
            self.sq_dataset.setup_with_tf_session(self.session)

            eval_fn = evaluate_dev_and_visualize if \
                self.options.visualize_evaluated_results else evaluate_dev
            dev_em, dev_f1 = eval_fn(self.session,
                self.model_builder.get_towers(), self.sq_dataset, self.options)
            print("Dev Em:", dev_em, "Dev F1:", dev_f1)
