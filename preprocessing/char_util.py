"""Provides function(s) for creating data to be used for character embeddings
"""

import numpy as np
import preprocessing.constants as constants

def get_char_np_array(list_sentences, max_sentence_length, vocab):
    """Input:
        list_sentences: List of lists of lists of character ids from the vocab
        max_sentence_length: max length of a sentence in this data set
        vocab: a Vocab object
       Output: A numpy array shaped [num_examples, max_sentence_length,
          max_word_length]
    """
    print("Creating character-level data")
    arr = np.full((len(list_sentences),
                   max_sentence_length,
                   constants.MAX_WORD_LEN),
            vocab.CHAR_PAD_ID, dtype=np.uint8)
    for i in range(len(list_sentences)):
        sentence = list_sentences[i]
        for j in range(len(sentence)):
            word = sentence[j]
            for k in range(min(len(word), constants.MAX_WORD_LEN)):
                arr[i, j, k] = word[k]
    print("Done creating character-level data")
    return arr
