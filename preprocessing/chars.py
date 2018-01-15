"""Defines constants for character representations.
"""

# Reduce the maximum number of characters to prevent blowing up the data set
# size.
MAX_CHARS = (2**8)
# Beginning of sentence. Only corresponds to the BOS word id.
CHAR_BOS_ID = (2**8)
# End of sentence. Only corresponds to the EOS word id.
CHAR_EOS_ID = (2**8) + 1
CHAR_BOW_ID = (2**8) + 2 # Beginning of word
CHAR_EOW_ID = (2**8) + 3 # End of word
CHAR_PAD_ID = (2**8) + 4 # Pad character
CHAR_UNK_ID = (2**8) + 5 # Unique character

MAX_ID = CHAR_UNK_ID
