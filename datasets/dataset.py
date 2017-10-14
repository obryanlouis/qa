"""Defines a class for a set of data with passages, questions, and spans.
"""

import numpy as np

class Dataset:
    def __init__(self, text_tokens, contexts, questions, spans, context_chars,
            question_chars, options, word_in_question, word_in_context):
        self.text_tokens = text_tokens
        self.ctx = contexts[:, :options.max_ctx_length]
        self.qst = questions[:, :options.max_qst_length]
        self.spn = spans
        self.data_index = np.arange(self.ctx.shape[0])
        self.word_in_question = word_in_question[:, :options.max_ctx_length]
        self.word_in_context = word_in_context[:, :options.max_qst_length]
        self.ctx_chars = context_chars[:, :options.max_ctx_length, :]
        self.qst_chars = question_chars[:, :options.max_qst_length, :]

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def get_size(self):
        return self.ctx.shape[0]
