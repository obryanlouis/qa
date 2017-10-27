"""Functions for dealing with remote model training and evaluation.
"""

def update_remote_options(options):
    # Make sure you have used S3 to preprocess files, or else remove the
    # option here.
    options.use_s3 = True
    options.num_gpus = 1 # Update as needed
    options.batch_size = 20
    options.use_fake_dataset = False
    options.verbose_logging = False
