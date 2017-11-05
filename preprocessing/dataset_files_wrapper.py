"""Wrapper for a set of files associated with creating a dataset.
"""

import os

import preprocessing.constants as constants

class DatasetFilesWrapper():
    def __init__(self,
                 data_dir,
                 text_tokens_file_name,
                 qst_file_name,
                 ctx_file_name,
                 spn_file_name,
                 word_in_question_file_name,
                 word_in_context_file_name,
                 context_chars_file_name,
                 question_chars_file_name,
                 question_ids_file_name,
                 question_ids_to_ground_truths_file_name,
                 context_pos_file_name,
                 question_pos_file_name,
                 context_ner_file_name,
                 question_ner_file_name):
        self._files = []
        self.data_dir = data_dir
        self.text_tokens_file_name = self._add_file(text_tokens_file_name)
        self.qst_file_name = self._add_file(qst_file_name)
        self.ctx_file_name = self._add_file(ctx_file_name)
        self.spn_file_name = self._add_file(spn_file_name)
        self.word_in_question_file_name = \
            self._add_file(word_in_question_file_name)
        self.word_in_context_file_name = \
            self._add_file(word_in_context_file_name)
        self.context_chars_file_name = self._add_file(context_chars_file_name)
        self.question_chars_file_name = \
            self._add_file(question_chars_file_name)
        self.question_ids_file_name = self._add_file(question_ids_file_name)
        self.question_ids_to_ground_truths_file_name = \
            self._add_file(question_ids_to_ground_truths_file_name)
        self.context_pos_file_name = self._add_file(context_pos_file_name)
        self.question_pos_file_name = self._add_file(question_pos_file_name)
        self.context_ner_file_name = self._add_file(context_ner_file_name)
        self.question_ner_file_name = self._add_file(question_ner_file_name)

    def _add_file(self, file_name):
        full_file_name = os.path.join(self.data_dir, file_name)
        self._files.append(full_file_name)
        return full_file_name

    def get_all_files(self):
        return list(self._files)

    @staticmethod
    def create_train_files_wrapper(data_dir):
        return DatasetFilesWrapper(
            data_dir=data_dir,
            text_tokens_file_name=constants.TRAIN_FULL_TEXT_TOKENS_FILE,
            qst_file_name=constants.TRAIN_QUESTION_FILE,
            ctx_file_name=constants.TRAIN_CONTEXT_FILE,
            spn_file_name=constants.TRAIN_SPAN_FILE,
            word_in_question_file_name=constants.TRAIN_WORD_IN_QUESTION_FILE,
            word_in_context_file_name=constants.TRAIN_WORD_IN_CONTEXT_FILE,
            context_chars_file_name=constants.TRAIN_CONTEXT_CHAR_FILE,
            question_chars_file_name=constants.TRAIN_QUESTION_CHAR_FILE,
            question_ids_file_name=constants.TRAIN_QUESTION_IDS_FILE,
            question_ids_to_ground_truths_file_name=constants.TRAIN_QUESTION_IDS_TO_GND_TRUTHS_FILE,
            context_pos_file_name=constants.TRAIN_CONTEXT_POS_FILE,
            question_pos_file_name=constants.TRAIN_QUESTION_POS_FILE,
            context_ner_file_name=constants.TRAIN_CONTEXT_NER_FILE,
            question_ner_file_name=constants.TRAIN_QUESTION_NER_FILE)

    @staticmethod
    def create_dev_files_wrapper(data_dir):
        return DatasetFilesWrapper(
            data_dir=data_dir,
            text_tokens_file_name=constants.DEV_FULL_TEXT_TOKENS_FILE,
            qst_file_name=constants.DEV_QUESTION_FILE,
            ctx_file_name=constants.DEV_CONTEXT_FILE,
            spn_file_name=constants.DEV_SPAN_FILE,
            word_in_question_file_name=constants.DEV_WORD_IN_QUESTION_FILE,
            word_in_context_file_name=constants.DEV_WORD_IN_CONTEXT_FILE,
            context_chars_file_name=constants.DEV_CONTEXT_CHAR_FILE,
            question_chars_file_name=constants.DEV_QUESTION_CHAR_FILE,
            question_ids_file_name=constants.DEV_QUESTION_IDS_FILE,
            question_ids_to_ground_truths_file_name=constants.DEV_QUESTION_IDS_TO_GND_TRUTHS_FILE,
            context_pos_file_name=constants.DEV_CONTEXT_POS_FILE,
            question_pos_file_name=constants.DEV_QUESTION_POS_FILE,
            context_ner_file_name=constants.DEV_CONTEXT_NER_FILE,
            question_ner_file_name=constants.DEV_QUESTION_NER_FILE)
