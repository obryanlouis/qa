import tensorflow as tf

from preprocessing.s3_util import *
from train.trainer import Trainer
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    maybe_download_data_files_from_s3(options)
    Trainer(options).train()

if __name__ == "__main__":
    tf.app.run()
