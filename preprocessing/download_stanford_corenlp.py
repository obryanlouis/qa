"""Downloads the Stanford CoreNLP jar file to use for text tokenization.
"""

import os
import sys
import urllib.request as request
import zipfile
import preprocessing.constants as constants

from preprocessing.download_utils import download_file_with_progress, unzip_file_and_remove


def download_stanford_corenlp(download_dir):
    if os.path.isdir(os.path.join(download_dir, "stanford-corenlp-full-2017-06-09")):
        print("Already have Stanford CoreNLP downloaded and unzipped. Not redownloading.")
        return
    zip_file_name = os.path.join(download_dir, constants.CORENLP_ZIP_FILE_NAME)
    download_file_with_progress(constants.CORENLP_URL, zip_file_name)
    unzip_file_and_remove(zip_file_name, download_dir)
