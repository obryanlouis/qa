import tensorflow as tf

from preprocessing.create_train_data import DataParser
from preprocessing.download_data import download_data
from preprocessing.download_stanford_corenlp import download_stanford_corenlp
from preprocessing.embedding_util import split_vocab_and_embedding
from preprocessing.s3_util import *
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    data_dir = options.data_dir
    download_data(data_dir)
    download_stanford_corenlp(data_dir)
    split_vocab_and_embedding(data_dir)
    DataParser(data_dir).create_train_data()
    maybe_upload_data_files(options)

if __name__ == "__main__":
    tf.app.run()
