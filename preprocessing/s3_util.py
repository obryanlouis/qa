"""Provides functions for saving and loading train/dev data and word vectors
   using AWS S3 storage.
"""

import boto3
import os
import preprocessing.constants as constants
import time

def _get_save_file_names(options):
    file_names = {
        # Save the context/question/span numpy files
        constants.TRAIN_CONTEXT_FILE,
        constants.TRAIN_QUESTION_FILE,
        constants.TRAIN_SPAN_FILE,
        constants.TRAIN_WORD_IN_QUESTION_FILE,
        constants.TRAIN_WORD_IN_CONTEXT_FILE,
        constants.DEV_CONTEXT_FILE,
        constants.DEV_QUESTION_FILE,
        constants.DEV_SPAN_FILE,
        constants.DEV_WORD_IN_QUESTION_FILE,
        constants.DEV_WORD_IN_CONTEXT_FILE,
        # Save the word vectors
        constants.EMBEDDING_FILE,
        # Save the text tokens
        constants.TRAIN_FULL_TEXT_TOKENS_FILE,
        constants.DEV_FULL_TEXT_TOKENS_FILE,
        # Save the vocab
        constants.VOCAB_FILE,
        # Save the character-level data
        constants.TRAIN_CONTEXT_CHAR_FILE,
        constants.TRAIN_QUESTION_CHAR_FILE,
        constants.DEV_CONTEXT_CHAR_FILE,
        constants.DEV_QUESTION_CHAR_FILE,
        # Save the question id files
        constants.TRAIN_QUESTION_IDS_FILE,
        constants.TRAIN_QUESTION_IDS_TO_GND_TRUTHS_FILE,
        constants.DEV_QUESTION_IDS_FILE,
        constants.DEV_QUESTION_IDS_TO_GND_TRUTHS_FILE,
        # Save the POS and NER tags
        constants.TRAIN_CONTEXT_POS_FILE,
        constants.TRAIN_QUESTION_POS_FILE,
        constants.TRAIN_CONTEXT_NER_FILE,
        constants.TRAIN_QUESTION_NER_FILE,
        constants.DEV_CONTEXT_POS_FILE,
        constants.DEV_QUESTION_POS_FILE,
        constants.DEV_CONTEXT_NER_FILE,
        constants.DEV_QUESTION_NER_FILE,
    }
    if options.use_cove_vectors:
        file_names.add(constants.DEV_COVE_QST_FILE)
        file_names.add(constants.DEV_COVE_CTX_FILE)
        file_names.add(constants.TRAIN_COVE_QST_FILE)
        file_names.add(constants.TRAIN_COVE_CTX_FILE)
    return file_names

def already_uploaded_s3_files(options, bucket, files):
    key_prefix = os.path.join(options.s3_data_folder_name)
    data_objs = {}
    for obj in bucket.objects.filter(Prefix=key_prefix):
        file_name = os.path.basename(obj.key)
        data_objs[file_name] = True
    return all([file_name in data_objs for file_name in files
        if os.path.isfile(os.path.join(options.data_dir, file_name))])

def maybe_upload_data_files(options):
    if not options.use_s3:
        print("S3 not enabled; not uploading to S3.")
        return
    start = time.time()
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(options.s3_bucket_name)
    files = _get_save_file_names(options)
    if already_uploaded_s3_files(options, bucket, files):
        print("Already uploaded all data files to s3. Not reuploading.")
        return
    print("Uploading data files to S3")
    files_uploaded = 0
    for file_name in files:
        full_file_name = os.path.join(options.data_dir, file_name)
        if not os.path.isfile(full_file_name):
            continue
        key = os.path.join(options.s3_data_folder_name, file_name)
        print("Uploading %s to %s" % (full_file_name, key))
        bucket.upload_file(full_file_name, key)
        files_uploaded += 1
    print("Uploaded %d data files to AWS S3 in %f seconds" % (
                files_uploaded, time.time() - start))

def maybe_download_data_files(options):
    """Downloads preprocessed training/dev data from S3 storage, if s3 is
       enabled and those files haven't already been downloaded.
    """
    if not options.use_s3:
        return
    if all([os.path.exists(os.path.join(options.data_dir, file_name)) for \
            file_name in _get_save_file_names(options)]):
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
