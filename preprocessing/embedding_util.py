"""Creates way to make a word embedding file that is usable by numpy.
"""

import numpy as np
import operator
import os
import preprocessing.constants as constants

# Reduce the maximum number of characters to prevent blowing up the data set
# size.
MAX_CHARS = (2**8) - 2
CHAR_PAD_ID = MAX_CHARS + 1
CHAR_UNK_ID = CHAR_PAD_ID + 1

def _get_line_count(filename):
    num_lines = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _ in f:
            num_lines += 1
    return num_lines

def split_vocab_and_embedding(data_dir, download_dir):
    input_file = os.path.join(download_dir, constants.VECTOR_FILE)
    embedding_output_file = os.path.join(data_dir, constants.EMBEDDING_FILE)
    vocab_output_file = os.path.join(data_dir, constants.VOCAB_FILE)
    vocab_chars_output_file = os.path.join(data_dir, constants.VOCAB_CHARS_FILE)
    if all([os.path.exists(f) for f in 
        [embedding_output_file, vocab_output_file, vocab_chars_output_file]]):
        print("Word embedding and vocab files already exist")
        return
    print("Creating NumPy word embedding file and vocab files")
    num_lines = _get_line_count(input_file)
    print("Vocab size: %d" % num_lines)
    embedding = np.zeros((num_lines, constants.WORD_VEC_DIM), dtype=np.float32)
    vocab_o_file = open(vocab_output_file, "w", encoding="utf-8")
    vocab_chars = np.zeros((num_lines, constants.MAX_WORD_LEN), dtype=np.uint8)
    i_file = open(input_file, "r", encoding="utf-8")
    i = 0
    char_counts = {}
    vocab_list = []
    for line in i_file:
        idx = line.index(" ")
        word = line[:idx]
        vocab_list.append(word)
        for c in word:
            if c in char_counts:
                char_counts[c] += 1
            else:
                char_counts[c] = 1
        vocab_o_file.write(word + "\n")
        embedding[i] = np.fromstring(line[idx + 1:], dtype=np.float32, sep=' ')
        i += 1
        if i % 10000 == 0 or i == num_lines:
            print("Processed %d of %d (%f percent done)" % (i, num_lines, 100 * float(i) / float(num_lines)), end="\r")
    sorted_chars = sorted(char_counts.items(), key=operator.itemgetter(1),
        reverse=True)
    frequent_chars = dict((x[0], i) for i, x in enumerate(sorted_chars[:MAX_CHARS]))
    print("Creating word character data")
    for z in range(len(vocab_list)):
        word = vocab_list[z]
        for zz in range(constants.MAX_WORD_LEN):
            if zz >= len(word):
                vocab_chars[z, zz] = CHAR_PAD_ID
            elif word[zz] not in frequent_chars:
                vocab_chars[z, zz] = CHAR_UNK_ID
            else:
                vocab_chars[z, zz] = frequent_chars[word[zz]]
    np.save(vocab_chars_output_file, vocab_chars)
    np.save(embedding_output_file, embedding)
    vocab_o_file.close()
    i_file.close()
    print("")
    print("Finished creating vocabulary and embedding file")
