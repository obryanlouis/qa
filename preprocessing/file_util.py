"""Functions for dealing with saving/loading from files
"""

import pickle

def save_pickle_file(full_file_name, python_obj):
    p_file = open(full_file_name, "wb")
    pickle.dump(python_obj, p_file)
    p_file.close()
