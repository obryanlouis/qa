"""Use this version of the training file to train on AWS.
"""
import tensorflow as tf

from preprocessing.s3_util import *
from train.trainer import Trainer
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    # Make sure you have used S3 to preprocess files, or else remove the
    # option here.
    options.use_s3 = True
    options.num_gpus = 1 # Update as needed
    options.batch_size = 120
    options.use_fake_dataset = False
    maybe_download_data_files(options)
    Trainer(options).train()

if __name__ == "__main__":
    tf.app.run()
