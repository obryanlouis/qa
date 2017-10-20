"""Defines a class for a set of data with passages, questions, and spans.
"""

import numpy as np

class Dataset:
    def __init__(self, text_tokens, contexts, questions, spans, context_chars,
            question_chars, options, word_in_question, word_in_context,
            question_ids, question_ids_to_ground_truths):
        assert len(text_tokens) == contexts.shape[0]
        assert len(text_tokens) == questions.shape[0]
        assert len(text_tokens) == spans.shape[0]
        assert len(text_tokens) == context_chars.shape[0]
        assert len(text_tokens) == question_chars.shape[0]
        assert len(text_tokens) == word_in_question.shape[0]
        assert len(text_tokens) == word_in_context.shape[0]
        assert len(text_tokens) == question_ids.shape[0]
        assert len(text_tokens) == len(question_ids_to_ground_truths)
        num_dataset = len(text_tokens)
        if options.truncate_dataset_percent < 1.0:
            assert options.truncate_dataset_percent >= 0
            num_dataset = int(options.truncate_dataset_percent * len(text_tokens))
        self.text_tokens = text_tokens[:num_dataset]
        self.ctx = contexts[:num_dataset, :options.max_ctx_length]
        self.qst = questions[:num_dataset, :options.max_qst_length]
        self.spn = spans[:num_dataset]
        self.data_index = np.arange(num_dataset)
        self.word_in_question = word_in_question[:num_dataset, :options.max_ctx_length]
        self.word_in_context = word_in_context[:num_dataset, :options.max_qst_length]
        self.ctx_chars = context_chars[:num_dataset, :options.max_ctx_length, :]
        self.qst_chars = question_chars[:num_dataset, :options.max_qst_length, :]
        self.question_ids = question_ids[:num_dataset]
        self.question_ids_to_ground_truths = question_ids_to_ground_truths
        assert len(self.text_tokens) == self.ctx.shape[0]
        assert len(self.text_tokens) == self.qst.shape[0]
        assert len(self.text_tokens) == self.spn.shape[0]
        assert len(self.text_tokens) == self.ctx_chars.shape[0]
        assert len(self.text_tokens) == self.qst_chars.shape[0]
        assert len(self.text_tokens) == self.word_in_question.shape[0]
        assert len(self.text_tokens) == self.word_in_context.shape[0]
        assert len(self.text_tokens) == self.question_ids.shape[0]

    def get_sentences_for_all_gnd_truths(self, example_idx):
        question_id = self.question_ids[example_idx]
        gnd_truths_list = self.question_ids_to_ground_truths[question_id]
        sentences = []
        for start_idx, end_idx in gnd_truths_list:
            sentences.append(self.get_sentence(example_idx, start_idx, end_idx))
        return sentences

    def get_sentence(self, ctx_id, start_idx, end_idx):
        list_text_tokens = self.text_tokens[ctx_id]
        return " ".join(list_text_tokens[start_idx: end_idx + 1])

    def get_size(self):
        return self.ctx.shape[0]
