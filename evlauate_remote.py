"""Evaluates the performance of a single, existing model on training and dev.
"""
import tensorflow as tf

from flags import get_options_from_flags
from preprocessing.s3_util import *
from remote_util import *
from train.evaluator import Evaluator

def main(_):
    options = get_options_from_flags()
    update_remote_options(options)
    maybe_download_data_files(options)
    Evaluator(options).evaluate()

if __name__ == "__main__":
    tf.app.run()
