"""Provides functions for saving and loading train/dev data and word vectors
   using AWS S3 storage.
"""

import boto3
import os
import preprocessing.constants as constants
import time

from sets import Set

SAVE_FILE_NAMES = Set([
    # Save the context/question/span numpy files
    constants.TRAIN_CONTEXT_FILE,
    constants.TRAIN_QUESTION_FILE,
    constants.TRAIN_SPAN_FILE,
    constants.DEV_CONTEXT_FILE,
    constants.DEV_QUESTION_FILE,
    constants.DEV_SPAN_FILE,
    # Save the word vectors
    constants.EMBEDDING_FILE,
    # Save the text tokens
    constants.TRAIN_FULL_TEXT_TOKENS_FILE,
    constants.DEV_FULL_TEXT_TOKENS_FILE,
    # Save the vocab
    constants.VOCAB_FILE,
])

def already_uploaded_s3_files(options, bucket):
    key_prefix = os.path.join(options.s3_data_folder_name)
    data_objs = {}
    for obj in bucket.objects.filter(Prefix=key_prefix):
        file_name = os.path.basename(obj.key)
        data_objs[file_name] = True
    return all([file_name in data_objs for file_name in SAVE_FILE_NAMES])

def maybe_upload_data_files(options):
    if not options.use_s3:
        print("S3 not enabled; not uploading to S3.")
        return
    start = time.time()
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(options.s3_bucket_name)
    if already_uploaded_s3_files(options, bucket):
        print("Already uploaded all data files to s3. Not reuploading.")
        return
    print("Uploading data files to S3")
    for file_name in SAVE_FILE_NAMES:
        key = os.path.join(options.s3_data_folder_name, file_name)
        full_file_name = os.path.join(options.data_dir, file_name)
        print("Uploading %s to %s" % (full_file_name, key))
        bucket.upload_file(full_file_name, key)
    print("Uploaded %d data files to AWS S3 in %f seconds" % (
                len(SAVE_FILE_NAMES), time.time() - start))

def maybe_download_data_files(options):
    """Downloads preprocessed training/dev data from S3 storage, if s3 is
       enabled and those files haven't already been downloaded.
    """
    if not options.use_s3:
        return
    if all([os.path.exists(os.path.join(options.data_dir, file_name)) for \
            file_name in SAVE_FILE_NAMES]):
        print("Already have all data files. Not downloading from S3.")
        return
    print("Downloading data files from S3")
    start = time.time()
    if not os.path.isdir(options.data_dir):
        os.makedirs(options.data_dir)
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(options.s3_bucket_name)
    key_prefix = os.path.join(options.s3_data_folder_name)
    num_files = 0
    for obj in bucket.objects.filter(Prefix=key_prefix):
        file_name = os.path.basename(obj.key)
        full_file_name = os.path.join(options.data_dir, file_name)
        if os.path.exists(full_file_name):
            print("Already downloaded file %s. Not downloading again."
                  % file_name)
            continue
        num_files += 1
        print("[S3] Downloading key", obj.key, "to file", full_file_name)
        bucket.download_file(obj.key, full_file_name)
    print("Time to download %d data files: %s" % (num_files,
                time.time() - start))
