"""Utilities for dealing with word vectors.
"""

import numpy as np
import operator
import os
import preprocessing.constants as constants

# Reduce the maximum number of characters to prevent blowing up the data set
# size.
MAX_CHARS = (2**8) - 2

class Vocab:
    def __init__(self, word_to_position, position_to_word, char_to_position,
            position_to_char):
        self._word_to_position = word_to_position
        self._position_to_word = position_to_word
        max_id = max(position_to_word.keys())
        self.PAD_ID = max_id + 1
        self.UNK_ID = self.PAD_ID + 1
        self._char_to_position = char_to_position
        self._position_to_char = position_to_char
        max_id = max(char_to_position.values())
        self.CHAR_PAD_ID = max_id + 1
        self.CHAR_UNK_ID = self.CHAR_PAD_ID + 1
    def get_vocab_size_without_pad_or_unk(self):
        return len(self._word_to_position)
    def get_vocab_size_including_pad_and_unk(self):
        return len(self._word_to_position) + 2
    def get_num_chars_including_padding(self):
        return self.CHAR_PAD_ID
    def get_id_for_char(self, char):
        if char in self._char_to_position:
            return self._char_to_position[char]
        return self.CHAR_UNK_ID
    def get_char_for_id(self, char_id):
        if char_id in self._position_to_char:
            return self._position_to_char[char_id]
        if char_id == self.CHAR_PAD_ID:
            return "<CHAR_PAD>"
        return "<UNK_CHAR>"
    def get_id_for_word(self, word):
        if word in self._word_to_position:
            return self._word_to_position[word]
        return self.UNK_ID
    def get_word_for_id(self, word_id, print_padding_and_unique=True):
        if word_id in self._position_to_word:
            return self._position_to_word[word_id]
        if word_id != self.UNK_ID and word_id != self.PAD_ID:
            raise Exception("Can't find word for id %d" % word_id)
        if not print_padding_and_unique:
            return ""
        if word_id == self.UNK_ID:
            return "<UNIQUE_WORD>"
        if word_id == self.PAD_ID:
            return "<PADDING>"
    def has_word(self, word):
        return word in self._word_to_position
    def get_sentences(self, numpy_array, print_padding_and_unique=True):
        """numpy_array is a 1 or 2 dimensional array of integers from the
           vocabulary.
           Returns a string or list of strings.
        """
        sh = numpy_array.shape
        if len(sh) != 1 and len(sh) != 2:
           raise Exception("Shape must be 1 or 2 dimensional")
        arr = numpy_array
        if len(sh) == 1:
            arr = np.reshape(numpy_array, (1, sh[0]))
        sentences = []
        for z in range(arr.shape[0]):
            sentence = ""
            for zz in range(arr.shape[1]):
                if zz == 0:
                    sentence = self.get_word_for_id(arr[z, zz],
                        print_padding_and_unique=print_padding_and_unique)
                else:
                    sentence += " " + self.get_word_for_id(arr[z, zz],
                        print_padding_and_unique=print_padding_and_unique)
            sentences.append(sentence)
        if len(sh) == 1:
            return sentences[0]
        return sentences

def get_vocab(data_dir="data"):
    position_to_word = {}
    word_to_position = {}
    char_counts = {}
    i = 0
    with open(os.path.join(data_dir, constants.VOCAB_FILE), encoding="utf-8") as f:
        for line in f:
            word = line[:-1]
            word_to_position[word] = i
            for char in word:
                if char in char_counts:
                    char_counts[char] = char_counts[char] + 1
                else:
                    char_counts[char] = 1
            i += 1
    position_to_word = {i:word for word, i in word_to_position.items()}
    sorted_chars = sorted(char_counts.items(), key=operator.itemgetter(1),
            reverse=True)
    chars_list = [x[0] for x in sorted_chars[:MAX_CHARS]]
    char_to_position = {}
    i = 0
    for char in chars_list:
        char_to_position[char] = i
        i += 1
    position_to_char = {i:char for char,i in char_to_position.items()}
    print("Vocab size: " + str(len(word_to_position)))
    print("Num chars in vocab: " + str(len(char_to_position)))
    return Vocab(word_to_position, position_to_word, char_to_position,
            position_to_char)
