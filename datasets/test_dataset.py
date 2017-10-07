"""The same as dataset.py except with small, debug test data.
"""

import numpy as np

DEBUG_DATA_SIZE = 20

class TestDataset:
    def __init__(self, text_tokens, contexts, questions, spans, options):
        self.text_tokens = text_tokens
        self.ctx = contexts[:DEBUG_DATA_SIZE, :options.max_ctx_length]
        self.qst = questions[:DEBUG_DATA_SIZE, :options.max_qst_length]
        self.spn = spans[:DEBUG_DATA_SIZE,:]
        self.data_index = np.arange(self.ctx.shape[0])

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def get_size(self):
        return self.ctx.shape[0]

