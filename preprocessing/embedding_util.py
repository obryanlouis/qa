"""Creates way to make a word embedding file that is usable by numpy.
"""

import numpy as np
import os
import preprocessing.constants as constants

def _get_line_count(filename):
    num_lines = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _ in f:
            num_lines += 1
    return num_lines

def split_vocab_and_embedding(data_dir):
    input_file = os.path.join(data_dir, constants.VECTOR_FILE)
    embedding_output_file = os.path.join(data_dir, constants.EMBEDDING_FILE)
    vocab_output_file = os.path.join(data_dir, constants.VOCAB_FILE)
    if all([os.path.exists(f) for f in 
            [embedding_output_file, vocab_output_file]]):
        print("Python word embedding file %s and vocab file %s already exist. Not recreating them."
                % (embedding_output_file, vocab_output_file))
        return
    print("Creating NumPy word embedding file and vocab text file")
    num_lines = _get_line_count(input_file)
    print("Vocab size: %d" % num_lines)
    embedding = np.zeros((num_lines, constants.WORD_VEC_DIM), dtype=np.float32)
    vocab_o_file = open(vocab_output_file, "w", encoding="utf-8")
    i_file = open(input_file, "r", encoding="utf-8")
    i = 0
    for line in i_file:
        idx = line.index(" ") + 1
        vocab_o_file.write(line[:idx] + "\n")
        embedding[i] = np.fromstring(line[idx:], dtype=np.float32, sep=' ')
        i += 1
        if i % 10000 == 0 or i == num_lines:
            print("Processed %d of %d (%f percent done)" % (i, num_lines, 100 * float(i) / float(num_lines)), end="\r")
    np.save(embedding_output_file, embedding)
    vocab_o_file.close()
    i_file.close()
    print("")
    print("Finished creating vocabulary and embedding file")
