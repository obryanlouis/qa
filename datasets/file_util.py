"""Functions for loading data from files.
"""

import numpy as np
import os
import pickle

def load_text_file(full_file_name):
    f = open(full_file_name, "rb")
    text_tokens = pickle.load(f)
    f.close()
    return text_tokens
