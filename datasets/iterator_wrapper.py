"""Defines a wrapper class for dataset iterators.
"""

class IteratorWrapper:
    def __init__(self, ctx_iterator, qst_iterator, spn_iterator,
            word_in_question_iterator, word_in_context_iterator,
            question_ids_iterator, context_pos_iterator, context_ner_iterator,
            question_pos_iterator, question_ner_iterator):
        self.ctx = ctx_iterator
        self.qst = qst_iterator
        self.spn = spn_iterator
        self.word_in_question = word_in_question_iterator
        self.word_in_context = word_in_context_iterator
        self.question_ids = question_ids_iterator
        self.context_pos = context_pos_iterator
        self.context_ner = context_ner_iterator
        self.question_pos = question_pos_iterator
        self.question_ner = question_ner_iterator
