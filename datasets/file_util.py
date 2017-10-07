"""Functions for loading data from files.
"""

import numpy as np
import os
import pickle

def load_text_file(data_dir, file_name):
    f = open(os.path.join(data_dir, file_name), "rb")
    text_tokens = pickle.load(f)
    f.close()
    return text_tokens

def load_file(data_dir, file_name):
    return np.load(os.path.join(data_dir, file_name))
