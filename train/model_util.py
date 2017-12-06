"""Functions for working with TensorFlow models.
"""

import os
import tensorflow as tf
import time

from datasets.test_data import TestData
from datasets.squad_data import SquadData
from train.s3_util import *

def create_checkpoint_file_name(options):
    return os.path.join(options.checkpoint_dir, create_s3_save_key(options))

def create_s3_save_key(options):
    return options.model_type + "." + options.experiment_name

def create_sq_dataset(options):
    return TestData(options) if options.use_fake_dataset else SquadData(options)

def create_session():
    return tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))

def create_saver(extra_save_vars=[]):
    return tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        + extra_save_vars)

def maybe_restore_model(s3, s3_save_key, options, session,
        checkpoint_file_name, saver, embeddings_placeholder, embeddings,
        word_chars_placeholder, word_chars):
    print("Restoring or creating new model...")
    start = time.time()
    maybe_download_files_from_s3(s3, s3_save_key, options.checkpoint_dir, options)
    if os.path.exists(checkpoint_file_name + ".index"):
        print("Restoring model from checkpoint %s" % checkpoint_file_name)
        saver.restore(session, checkpoint_file_name)
    else:
        print("Creating model with new parameters")
        session.run(tf.global_variables_initializer(), feed_dict={
            embeddings_placeholder: embeddings,
            word_chars_placeholder: word_chars,
        })
    print("Model initialization time %s" % (time.time() - start))
