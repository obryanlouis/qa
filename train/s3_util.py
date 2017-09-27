"""Functions for dealing with Amazon S3. It is used for model storage.
"""

import os
import time

def maybe_upload_files_to_s3(s3, save_key, dir_path, options):
    """Uploads everything from the path to s3. Each file is saved with a key
       of 'save_key/filename'
    """
    if s3 is None:
        return
    start = time.time()
    bucket = s3.Bucket(options.s3_bucket_name)
    files = [f for f in os.listdir(dir_path) if 
        os.path.isfile(os.path.join(dir_path, f))]
    for f in files:
       full_file_name = os.path.join(dir_path, f)
       key = os.path.join(save_key, f)
       print("[S3] Uploading file", full_file_name, "to key", key)
       bucket.upload_file(full_file_name, key)
    print("Time to upload %d files: %s" % (len(files), time.time() - start))

def maybe_download_files_from_s3(s3, save_key, dir_path, options):
    """Downloads all files with the prefix 'save_key' to the path 'dir_path.'
    """
    if s3 is None:
        return
    start = time.time()
    bucket = s3.Bucket(options.s3_bucket_name)
    num_files = 0
    for obj in bucket.objects.filter(Prefix=save_key):
        num_files += 1
        file_name = os.path.basename(obj.key)
        full_file_name = os.path.join(dir_path, file_name)
        print("[S3] Downloading key", obj.key, "to file", full_file_name)
        bucket.download_file(obj.key, full_file_name)
    print("Time to download %d files: %s" % (num_files, time.time() - start))
