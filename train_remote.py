"""Use this version of the training file to train on AWS.
"""
import tensorflow as tf

from flags import get_options_from_flags
from preprocessing.s3_util import *
from remote_util import *
from train.trainer import Trainer

def main(_):
    options = get_options_from_flags()
    update_remote_options(options)
    maybe_download_data_files_from_s3(options)
    Trainer(options).train()

if __name__ == "__main__":
    tf.app.run()
