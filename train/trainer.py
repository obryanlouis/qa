"""Trains a model on SQuAD data.
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

class Trainer:
    def __init__(self, options):
        self.options = options
        self.summary_assignments = {}
        self.s3_save_key = create_s3_save_key(options)
        self.checkpoint_file_name = create_checkpoint_file_name(options)

    def _perform_summary_assignment(self, summary, value):
        assignment_dict = self.summary_assignments[summary]
        self.session.run(assignment_dict["assign_op"],
            feed_dict={assignment_dict["placeholder"]:value})

    def train(self):
        train_start_time = time.time()
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

            learning_rate = tf.Variable(initial_value=
                self.options.learning_rate, trainable=False, dtype=tf.float32)
            learning_rate_placeholder = tf.placeholder(tf.float32)
            assign_learning_rate = tf.assign(learning_rate,
                    tf.maximum(self.options.min_learning_rate, learning_rate_placeholder))
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.options.learning_rate)
            self.model_builder = ModelBuilder(self.optimizer, self.options,
                self.sq_dataset, embedding_var, word_chars_var,
                compute_gradients=True, sess=self.session)
            print("Applying gradients")
            apply_gradients_start_time = time.time()
            train_op = None
            global_norm = None
            if self.options.model_type == "debug":
                train_op = tf.no_op()
                global_norm = tf.constant(0.0, dtype=tf.float32)
            else:
                grads, variables = zip(*average_gradients(self.model_builder.get_tower_grads()))
                grads, global_norm = tf.clip_by_global_norm(grads, self.options.max_global_norm)
                train_op = self.optimizer.apply_gradients(zip(grads, variables))
            iteration_num = tf.Variable(initial_value=1, trainable=False,
                dtype=tf.int32)
            incr_iter = tf.assign(iteration_num, iteration_num + 1)

            self.saver = create_saver()
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
            loss = self.model_builder.get_loss()
            loss_summary = tf.summary.scalar("loss", loss)
            gradients_summary = tf.summary.scalar("gradients", global_norm)
            print("Time to apply gradients: %s"
                    % (time.time() - apply_gradients_start_time))

            maybe_restore_model(self.s3, self.s3_save_key, self.options,
                self.session, self.checkpoint_file_name, self.saver,
                embedding_placeholder, self.sq_dataset.embeddings,
                word_chars_placeholder, self.sq_dataset.word_chars)
            maybe_print_model_parameters(self.options)
            self.sq_dataset.setup_with_tf_session(self.session)
            num_towers = self.model_builder.get_num_towers()

            print("Total setup time before starting training: %s"
                  % (time.time() - train_start_time))
            current_iter = int(self.session.run(iteration_num))
            total_ds_size = self.sq_dataset.estimate_total_train_ds_size()
            iterations_per_epoch = total_ds_size / \
                (self.options.batch_size * max(1, self.options.num_gpus))
            total_iter = max(int(self.options.epochs * iterations_per_epoch), 1)
            start_time = time.time()
            print("Current iteration: %d, Total iterations: %d"
                  % (current_iter, total_iter))
            i = current_iter - 1

            while True:
                i += 1
                iter_start = time.time()
                _, loss_value, _, loss_summary_value, \
                    gradients_summary_value, norm_value = \
                    self.session.run([train_op, loss, incr_iter,
                        loss_summary, gradients_summary, global_norm], feed_dict=
                        get_train_feed_dict(self.sq_dataset,
                            self.options, self.model_builder.get_towers()))
                self.sq_dataset.increment_train_samples_processed(
                    self.options.batch_size * num_towers)
                iter_end = time.time()
                time_per_iter = iter_end - iter_start
                time_per_epoch = time_per_iter * iterations_per_epoch
                remaining_iters = total_iter - i - 1
                eta = remaining_iters * time_per_iter
                print("iteration:", str(i) + "/" + str(total_iter),
                      "percent done:", 100.0 * float(i) / float(total_iter), 
                      "loss:", loss_value,
                      "Sec/iter", time_per_iter, 
                      "time/epoch", readable_time(time_per_epoch), end="\r")
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
                                self.options, self.model_builder.get_towers()))
                    self.sq_dataset.increment_val_samples_processed(
                        self.options.batch_size * num_towers)
                    if self.options.log_gradients:
                        self.val_writer.add_summary(gradients_summary_value, i)
                    if self.options.log_loss:
                        self.val_writer.add_summary(loss_summary_value, i)
                    print("")
                    print("[Validation] iteration:",
                          str(i) + "/" + str(total_iter),
                          "loss:", loss_value)
                if i % self.options.compute_accuracy_every == 0:
                    em, f1 = evaluate_train_partial(self.session,
                        self.model_builder.get_towers(), self.sq_dataset,
                        self.options)
                    print("")
                    print("[Train] F1", f1, "Em", em)
                    val_em, val_f1 = evaluate_dev_partial(self.session,
                        self.model_builder.get_towers(), self.sq_dataset,
                        self.options)
                    self.sq_dataset.increment_val_samples_processed(
                        self.options.num_evaluation_samples)
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
