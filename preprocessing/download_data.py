"""Downloads GloVe vectors from https://nlp.stanford.edu/projects/glove/.
"""

import os
import preprocessing.constants as constants
import sys
import urllib.request as request
import zipfile

from preprocessing.download_utils import download_file_with_progress, unzip_file_and_remove

def download_pretrained_vectors(download_dir):
    if os.path.isfile(os.path.join(download_dir, constants.VECTOR_FILE)):
        print("Already downloaded word vectors in directory " + download_dir + ".")
        return
    zip_file_name = os.path.join(download_dir, constants.VECTOR_ZIP_FILE)
    download_file_with_progress(constants.VECTORS_URL, zip_file_name)
    unzip_file_and_remove(zip_file_name, download_dir)

def download_squad_data(download_dir):
    squad_files = [constants.SQUAD_TRAIN_FILE, constants.SQUAD_DEV_FILE]
    if all(os.path.isfile(os.path.join(download_dir, squad_file)) for squad_file in squad_files):
        print("Already downloaded SQuAD files in directory " + download_dir + ".")
        return
    download_file_with_progress(constants.SQUAD_TRAIN_URL, os.path.join(download_dir, constants.SQUAD_TRAIN_FILE))
    download_file_with_progress(constants.SQUAD_DEV_URL, os.path.join(download_dir, constants.SQUAD_DEV_FILE))

def download_data(download_dir):
    download_pretrained_vectors(download_dir)
    download_squad_data(download_dir)
