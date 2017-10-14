"""Evaluates the performance of a single, existing model on training and dev.
"""
import tensorflow as tf

from preprocessing.s3_util import *
from train.evaluator import Evaluator
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    maybe_download_data_files(options)
    Evaluator(options).evaluate()

if __name__ == "__main__":
    tf.app.run()
