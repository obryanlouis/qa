"""Wrapper for a set of files associated with creating a dataset.
"""

import os

import preprocessing.constants as constants

class DatasetFilesWrapper():
    def __init__(self, full_data_folder):
        os.makedirs(full_data_folder, exist_ok=True)
        self.data_dir = full_data_folder
        self.next_batch_number = 0

    def create_new_file_names(self):
        file_names = FileNames(
            self._full_file_name(constants.QUESTION_FILE_PATTERN),
            self._full_file_name(constants.CONTEXT_FILE_PATTERN),
            self._full_file_name(constants.SPAN_FILE_PATTERN),
            self._full_file_name(constants.WORD_IN_QUESTION_FILE_PATTERN),
            self._full_file_name(constants.WORD_IN_CONTEXT_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_IDS_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_IDS_TO_GND_TRUTHS_FILE_PATTERN),
            self._full_file_name(constants.CONTEXT_POS_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_POS_FILE_PATTERN),
            self._full_file_name(constants.CONTEXT_NER_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_NER_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_IDS_TO_SQUAD_QUESTION_ID_FILE_PATTERN),
            self._full_file_name(constants.QUESTION_IDS_TO_PASSAGE_CONTEXT_FILE_PATTERN))
        self.next_batch_number += 1
        return file_names

    def _full_file_name(self, file_name_pattern):
        return os.path.join(self.data_dir,
            file_name_pattern % self.next_batch_number)

class FileNames():
    def __init__(self,
                 qst_file_name,
                 ctx_file_name,
                 spn_file_name,
                 word_in_question_file_name,
                 word_in_context_file_name,
                 question_ids_file_name,
                 question_ids_to_ground_truths_file_name,
                 context_pos_file_name,
                 question_pos_file_name,
                 context_ner_file_name,
                 question_ner_file_name,
                 question_ids_to_squad_question_id_file_name,
                 question_ids_to_passage_context_file_name):
        self.qst_file_name = qst_file_name
        self.ctx_file_name = ctx_file_name
        self.spn_file_name = spn_file_name
        self.word_in_question_file_name = word_in_question_file_name
        self.word_in_context_file_name = word_in_context_file_name
        self.question_ids_file_name = question_ids_file_name
        self.question_ids_to_ground_truths_file_name = \
            question_ids_to_ground_truths_file_name
        self.context_pos_file_name = context_pos_file_name
        self.question_pos_file_name = question_pos_file_name
        self.context_ner_file_name = context_ner_file_name
        self.question_ner_file_name = question_ner_file_name
        self.question_ids_to_squad_question_id_file_name = \
            question_ids_to_squad_question_id_file_name
        self.question_ids_to_passage_context_file_name = \
            question_ids_to_passage_context_file_name
