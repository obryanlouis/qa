import tensorflow as tf

from preprocessing.create_train_data import DataParser
from preprocessing.download_data import download_data
from preprocessing.download_stanford_corenlp import download_stanford_corenlp
from preprocessing.embedding_util import split_vocab_and_embedding
from preprocessing.s3_util import maybe_upload_data_files_to_s3
from flags import get_options_from_flags

def main(_):
    options = get_options_from_flags()
    data_dir = options.data_dir
    download_dir = options.download_dir
    for d in [data_dir, download_dir]:
        if not os.path.exists(d):
            print("Making directory", d)
            os.makedirs(d)
    download_data(download_dir)
    download_stanford_corenlp(download_dir)
    split_vocab_and_embedding(data_dir, download_dir)
    DataParser(data_dir, download_dir).create_train_data()
    maybe_upload_data_files_to_s3(options)

if __name__ == "__main__":
    tf.app.run()
