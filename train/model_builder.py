"""Builds a TensorFlow model to run on SQuAD data.
"""

import tensorflow as tf
import time

from model.cove_lstm import *
from model.model_types import MODEL_TYPES

class ModelBuilder:
    def __init__(self, optimizer, options, sq_dataset, embeddings, word_chars,
            compute_gradients, sess):
        self.sq_dataset = sq_dataset
        self.optimizer = optimizer
        self.options = options
        self.towers = []
        self.compute_gradients = compute_gradients
        self.tower_grads = []
        self.loss = None
        self.embeddings = embeddings
        self.word_chars = word_chars
        self.sess = sess
        self._setup()

    def get_num_towers(self):
        return len(self.towers)

    def get_towers(self):
        return self.towers

    def get_tower_grads(self):
        return self.tower_grads

    def get_loss(self):
        return self.loss

    def _validate_parameters(self):
        if self.options.model_type not in MODEL_TYPES:
            raise Exception("Model type %s not recognized. Must be in set %s."
                % (self.options.model_type, MODEL_TYPES.keys()))

    def _add_tower_and_compute_loss(self, scope, iterators):
        print("Creating tower in model")
        tower = MODEL_TYPES[self.options.model_type](self.options,
                iterators, self.sq_dataset, self.embeddings,
                self.word_chars, self.cove_cells, self.sess)
        tower.setup()
        print("Tower created")
        self.towers.append(tower)
        return tower.get_loss_op()

    def _setup(self):
        self._validate_parameters()

        with tf.device('/cpu:0'):
            self.loss = None
            create_model_start_time = time.time()
            self.cove_cells = None
            if self.options.use_cove_vectors:
                self.cove_cells = load_cove_lstm(self.options)
            tower_creation_time = 0
            gradient_computation_time = 0
            print("Creating towers")
            with tf.variable_scope(tf.get_variable_scope()):
                if self.options.num_gpus == 0:
                    iterators = self.sq_dataset.create_iterators()
                    tower_start_time = time.time()
                    self.loss = self._add_tower_and_compute_loss("single_tower_scope",
                            iterators)
                    tower_creation_time += (time.time() - tower_start_time)
                    if self.compute_gradients:
                        gradient_start_time = time.time()
                        self.tower_grads.append(
                            self.optimizer.compute_gradients(self.loss))
                        gradient_computation_time += (time.time() - gradient_start_time)
                else:
                    for i in range(self.options.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_%d' % i) as scope:
                                iterators = self.sq_dataset.create_iterators()
                                tower_start_time = time.time()
                                self.loss = self._add_tower_and_compute_loss(scope,
                                        iterators)
                                tower_creation_time += (time.time() - tower_start_time)
                                # This should make each tower share variables.
                                tf.get_variable_scope().reuse_variables()
                                if self.compute_gradients:
                                    gradient_start_time = time.time()
                                    grads = self.optimizer.compute_gradients(
                                        self.loss)
                                    gradient_computation_time += (time.time() - gradient_start_time)
                                    self.tower_grads.append(grads)
            print("Time to create towers: %s" % tower_creation_time)
            print("Time to compute gradients: %s" % gradient_computation_time)
            print("Time to create towers and calculate gradients: %s"
                  % (time.time() - create_model_start_time))
