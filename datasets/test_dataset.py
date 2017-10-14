"""The same as dataset.py except with small, debug test data.
"""

import numpy as np

NUM_SAMPLES = 13
CTX_LEN = 19
QST_LEN = 17

class TestDataset:
    def __init__(self, text_tokens, vocab, max_word_len):
        self.text_tokens = text_tokens
        vocab_size = vocab.get_vocab_size_including_pad_and_unk()
        self.ctx = np.random.randint(0, vocab_size, size=(NUM_SAMPLES, CTX_LEN))
        self.qst = np.random.randint(0, vocab_size, size=(NUM_SAMPLES, QST_LEN))
        self.spn = np.zeros((NUM_SAMPLES, 2), dtype=np.int32)
        for z in range(NUM_SAMPLES):
            spns = sorted([np.random.randint(0, CTX_LEN),
                           np.random.randint(0, CTX_LEN)])
            self.spn[z, 0] = spns[0]
            self.spn[z, 1] = spns[1]
        self.data_index = np.arange(self.ctx.shape[0])
        self.word_in_question = np.random.randint(0, 2, size=(NUM_SAMPLES, CTX_LEN))
        self.word_in_context = np.random.randint(0, 2, size=(NUM_SAMPLES, QST_LEN))
        self.ctx_chars = np.random.randint(0, vocab.CHAR_PAD_ID,
            size=(NUM_SAMPLES, CTX_LEN, max_word_len), dtype=np.uint8)
        self.qst_chars = np.random.randint(0, vocab.CHAR_PAD_ID,
            size=(NUM_SAMPLES, QST_LEN, max_word_len), dtype=np.uint8)

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def get_size(self):
        return self.ctx.shape[0]

