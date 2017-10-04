"""Trains a model on SQuAD data.
"""

import boto3
import os
import shutil
import time

from model.model_types import MODEL_TYPES
from preprocessing.squad_data import SquadData
from train.evaluation_util import *
from train.print_utils import *
from train.s3_util import *
from train.tf_dataset import *
from train.train_util import *

class Trainer:
    def __init__(self, options):
        self.options = options
        self.session = None
        self.towers = []
        self.sq_dataset = None
        self.tf_dataset = None
        self.saver = None
        self.train_writer = None
        self.val_writer = None
        self.em = None
        self.f1 = None
        self.summary_assignments = {}
        self.s3 = None
        self.s3_save_key = self.options.model_type + "." + self.options.experiment_name
        self.checkpoint_file_name = os.path.join(self.options.checkpoint_dir,
            self.options.model_type + "." + self.options.experiment_name)

    def _validate_parameters(self):
        if self.options.model_type not in MODEL_TYPES:
            raise Exception("Model type %s not recognized. Must be in set %s." % (self.options.model_type, MODEL_TYPES.keys()))

    def _add_tower_and_compute_loss(self, scope, iterators):
        # NOTE: This is so slow. Is there a way in tensorflow to just copy the
        # graph instead of recreating and recompiling the whole thing?
        tower = MODEL_TYPES[self.options.model_type](self.options,
                self.sq_dataset.embeddings, iterators)
        tower.setup()
        self.towers.append(tower)
        return tower.get_loss_op()

    def _perform_summary_assignment(self, summary, value):
        assignment_dict = self.summary_assignments[summary]
        self.session.run(assignment_dict["assign_op"],
            feed_dict={assignment_dict["placeholder"]:value})

    def _maybe_restore_model(self):
        print("Restoring or creating new model...")
        start = time.time()
        maybe_download_files_from_s3(self.s3, self.s3_save_key,
                self.options.checkpoint_dir, self.options)
        if os.path.exists(self.checkpoint_file_name + ".index"):
            print("Restoring model from checkpoint %s" % self.checkpoint_file_name)
            self.saver.restore(self.session, self.checkpoint_file_name)
        else:
            print("Creating model with new parameters")
            self.session.run(tf.global_variables_initializer())
        print("Model initialization time %s" % (time.time() - start))

    def train(self):
        self._validate_parameters()
        if self.options.use_s3:
            self.s3 = boto3.resource('s3')
        if not os.path.exists(self.options.checkpoint_dir):
            os.makedirs(self.options.checkpoint_dir)

        self.sq_dataset = SquadData(self.options)
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            self.tf_dataset = TfDataset(self.options, self.sq_dataset)
            self.session = tf.Session(config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False))
            tower_grads = []
            learning_rate = tf.Variable(initial_value=
                self.options.learning_rate, trainable=False, dtype=tf.float32)
            learning_rate_placeholder = tf.placeholder(tf.float32)
            assign_learning_rate = tf.assign(learning_rate,
                    tf.maximum(self.options.min_learning_rate, learning_rate_placeholder))
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.options.learning_rate)
            loss = None
            create_model_start_time = time.time()
            with tf.variable_scope(tf.get_variable_scope()):
                if self.options.num_gpus == 0:
                    iterators = self.tf_dataset.create_tf_iterators()
                    loss = self._add_tower_and_compute_loss("single_tower_scope",
                            iterators)
                    tower_grads.append(optimizer.compute_gradients(loss))
                else:
                    for i in range(self.options.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_%d' % i) as scope:
                                iterators = self.tf_dataset.create_tf_iterators()
                                loss = self._add_tower_and_compute_loss(scope,
                                        iterators)
                                # This should make each tower share variables.
                                tf.get_variable_scope().reuse_variables()
                                grads = optimizer.compute_gradients(loss)
                                tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads)
            iteration_num = tf.Variable(initial_value=1, trainable=False,
                dtype=tf.int32)
            incr_iter = tf.assign(iteration_num, iteration_num + 1)

            self.saver = tf.train.Saver(var_list=
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            if self.options.clear_logs_before_training:
                shutil.rmtree(self.options.log_dir, ignore_errors=True)
            if not os.path.exists(self.options.log_dir):
                os.makedirs(self.options.log_dir)
            self.train_writer = tf.summary.FileWriter(os.path.join(
                self.options.log_dir, "train"), graph=tf.get_default_graph())
            self.val_writer = tf.summary.FileWriter(os.path.join(
                self.options.log_dir, "val"), graph=tf.get_default_graph())
            self.em = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            self.f1 = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
            em_summary = tf.summary.scalar("exact_match", self.em)
            f1_summary = tf.summary.scalar("f1_score", self.f1)
            for summary in [self.em, self.f1]:
                assignment_dict = {}
                self.summary_assignments[summary] = assignment_dict
                placeholder = tf.placeholder(tf.float32)
                assignment_dict["placeholder"] = placeholder
                assignment_dict["assign_op"] = tf.assign(summary, placeholder)
            loss_summary = tf.summary.scalar("loss", loss)
            global_norm = tf.global_norm([grad for grad, var in grads])
            gradients_summary = tf.summary.scalar("gradients", global_norm)
            print("Time to create model and compute gradients: %s"
                    % (time.time() - create_model_start_time))


            self._maybe_restore_model()
            maybe_print_model_parameters(self.options)
            self.tf_dataset.setup_with_tf_session(self.session)

            current_iter = int(self.session.run(iteration_num))
            iterations_per_epoch = self.sq_dataset.train_ds.get_size() / self.options.batch_size
            total_iter = int(self.options.epochs * iterations_per_epoch)
            start_time = time.time()
            print("Current iteration: %d, Total iterations: %d" % (current_iter, total_iter))
            start_iter = current_iter
            for i in range(current_iter, total_iter):
                _, loss_value, _, loss_summary_value, gradients_summary_value = \
                    self.session.run([train_op, loss, incr_iter,
                        loss_summary, gradients_summary], feed_dict=
                        get_train_feed_dict(self.sq_dataset, self.tf_dataset,
                            self.options, self.towers))
                elapsed = time.time() - start_time
                time_per_iter = elapsed / (i - start_iter + 1)
                time_per_epoch = time_per_iter * iterations_per_epoch
                remaining_iters = total_iter - i - 1
                eta = remaining_iters * time_per_iter
                print("iteration:", str(i) + "/" + str(total_iter),
                        "percent done:", 100.0 * float(i) / float(total_iter), 
                        "loss:", loss_value,
                        "Seconds elapsed", elapsed,
                        "Sec/iter", time_per_iter, 
                        "time/epoch", readable_time(time_per_epoch), 
                        readable_eta(eta), end="\r")
                if i % self.options.log_every == 0:
                    if self.options.log_gradients:
                        self.train_writer.add_summary(gradients_summary_value, i)
                    if self.options.log_loss:
                        self.train_writer.add_summary(loss_summary_value, i)
                if i % self.options.log_valid_every == 0:
                    loss_summary_value, gradients_summary_value, loss_value = \
                        self.session.run([
                            loss_summary, gradients_summary, loss], 
                            feed_dict=get_dev_feed_dict(self.sq_dataset,
                                self.tf_dataset, self.options, self.towers))
                    if self.options.log_gradients:
                        self.val_writer.add_summary(gradients_summary_value, i)
                    if self.options.log_loss:
                        self.val_writer.add_summary(loss_summary_value, i)
                    print("")
                    print("[Validation] iteration:",
                          str(i) + "/" + str(total_iter),
                          "loss:", loss_value)
                if i % self.options.compute_accuracy_every == 0:
                    em, f1 = evaluate_train_partial(self.session, self.towers, self.sq_dataset, self.options, self.tf_dataset)
                    print("")
                    print("[Train] F1", f1, "Em", em)
                    val_em, val_f1 = evaluate_dev_partial(self.session, self.towers, self.sq_dataset, self.options, self.tf_dataset)
                    print("[Valid] F1", val_f1, "Em", val_em)
                    self._perform_summary_assignment(self.em, em)
                    self._perform_summary_assignment(self.f1, f1)
                    if self.options.log_exact_match:
                        self.train_writer.add_summary(self.session.run(em_summary), i)
                    if self.options.log_f1_score:
                        self.train_writer.add_summary(self.session.run(f1_summary), i)
                    self._perform_summary_assignment(self.em, val_em)
                    self._perform_summary_assignment(self.f1, val_f1)
                    if self.options.log_exact_match:
                        self.val_writer.add_summary(self.session.run(em_summary), i)
                    if self.options.log_f1_score:
                        self.val_writer.add_summary(self.session.run(f1_summary), i)
                    self.train_writer.flush()
                    self.val_writer.flush()
                if i % self.options.save_every == 0:
                    self.saver.save(self.session, self.checkpoint_file_name)
                    maybe_upload_files_to_s3(self.s3, self.s3_save_key, self.options.checkpoint_dir, self.options)
                    print("Saved model at iteration", i, "with checkpoint path", self.options.checkpoint_dir)

                self.session.run(assign_learning_rate, feed_dict={
                    learning_rate_placeholder: self.options.learning_rate * 
                    (self.options.learning_rate_decay ** (float(i) / iterations_per_epoch))})
