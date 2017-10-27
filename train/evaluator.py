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
        self.model_builder = None
        self.options = options
        self.session = None
        self.sq_dataset = None
        self.tf_dataset = None
        self.saver = None
        self.train_writer = None
        self.val_writer = None
        self.em = None
        self.f1 = None
        self.summary_assignments = {}
        self.s3 = None
        self.s3_save_key = create_s3_save_key(options)
        self.checkpoint_file_name = create_checkpoint_file_name(options)
        self.optimizer = None

    def evaluate(self):
        if self.options.use_s3:
            self.s3 = boto3.resource('s3')
        if not os.path.exists(self.options.checkpoint_dir):
            os.makedirs(self.options.checkpoint_dir)

        self.sq_dataset = create_sq_dataset(self.options)
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            self.tf_dataset = create_tf_dataset(self.options, self.sq_dataset)
            self.session = create_session()
            self.model_builder = ModelBuilder(None, self.options,
                self.tf_dataset, self.sq_dataset, compute_gradients=False)
            self.saver = create_saver()
            maybe_restore_model(self.s3, self.s3_save_key, self.options,
                self.session, self.checkpoint_file_name, self.saver)
            maybe_print_model_parameters(self.options)
            self.tf_dataset.setup_with_tf_session(self.session)

            eval_fn = evaluate_dev_and_visualize if self.options.visualize_evaluated_results else evaluate_dev
            dev_em, dev_f1 = eval_fn(self.session,
                self.model_builder.get_towers(), self.sq_dataset, self.options,
                self.tf_dataset)
            print("Dev Em:", dev_em, "Dev F1:", dev_f1)
