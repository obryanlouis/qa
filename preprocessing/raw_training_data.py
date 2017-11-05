"""Wrapper for the result of creating raw training data that potentially needs
to be formatted, and then saved.
"""

class RawTrainingData():
    def __init__(self,
                list_contexts,
                list_word_in_question,
                list_questions,
                list_word_in_context,
                spans,
                text_tokens,
                context_chars,
                question_chars,
                question_ids,
                question_ids_to_ground_truths):
        self.list_contexts = list_contexts
        self.list_word_in_question = list_word_in_question
        self.list_questions = list_questions
        self.list_word_in_context = list_word_in_context
        self.spans = spans
        self.text_tokens = text_tokens
        self.context_chars = context_chars
        self.question_chars = question_chars
        self.question_ids = question_ids
        self.question_ids_to_ground_truths = question_ids_to_ground_truths
