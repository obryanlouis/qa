"""Provides a way to save dataset files after the raw data is created.
"""

import numpy as np

from preprocessing.char_util import *
from preprocessing.file_util import *


class DatasetFilesSaver():
    def __init__(self,
            dataset_files_wrapper,
            max_ctx_length,
            max_qst_length,
            vocab,
            raw_training_data):
        self.files_wrapper = dataset_files_wrapper
        self.max_ctx_length = max_ctx_length
        self.max_qst_length = max_qst_length
        self.vocab = vocab
        self.data = raw_training_data

    def _create_padded_array(self, list_of_py_arrays, max_len, pad_value):
        return [py_arr + [pad_value] * (max_len - len(py_arr)) for py_arr in list_of_py_arrays]

    def save(self):
        print("Saving text tokens to binary pickle files")
        save_pickle_file(self.files_wrapper.text_tokens_file_name,
            self.data.text_tokens)

        print("Saving span numpy arrays")
        np.save(self.files_wrapper.spn_file_name, self.data.spans)

        print("Saving context numpy arrays")
        ctx_np_arr = np.array(self._create_padded_array(self.data.list_contexts,
            self.max_ctx_length, self.vocab.PAD_ID), dtype=np.int32)
        np.save(self.files_wrapper.ctx_file_name, ctx_np_arr)

        print("Saving context character-level numpy arrays")
        ctx_char_arr = get_char_np_array(self.data.context_chars,
            self.max_ctx_length, self.vocab)
        np.save(self.files_wrapper.context_chars_file_name, ctx_char_arr)

        print("Saving question numpy arrays")
        qst_np_arr = np.array(self._create_padded_array(self.data.list_questions,
            self.max_qst_length, self.vocab.PAD_ID), dtype=np.int32)
        np.save(self.files_wrapper.qst_file_name, qst_np_arr)

        print("Saving question character-level numpy arrays")
        qst_char_arr = get_char_np_array(self.data.question_chars,
            self.max_qst_length, self.vocab)
        np.save(self.files_wrapper.question_chars_file_name, qst_char_arr)

        print("Saving additional feature numpy arrays")
        word_in_question_np_arr = np.array(self._create_padded_array(
            self.data.list_word_in_question, self.max_ctx_length, 0),
                dtype=np.float32)
        np.save(self.files_wrapper.word_in_question_file_name, word_in_question_np_arr)
        word_in_context_np_arr = np.array(self._create_padded_array(
            self.data.list_word_in_context, self.max_ctx_length, 0),
                dtype=np.float32)
        np.save(self.files_wrapper.word_in_context_file_name, word_in_context_np_arr)

        print("Saving question ids")
        question_ids_np_arr = np.array(self.data.question_ids, dtype=np.int32)
        np.save(self.files_wrapper.question_ids_file_name, question_ids_np_arr)

        print("Saving question ids to ground truths dict")
        save_pickle_file(self.files_wrapper.question_ids_to_ground_truths_file_name,
            self.data.question_ids_to_ground_truths)

        print("Saving POS and NER tags")
        ctx_pos_np_arr = np.array(self._create_padded_array(self.data.context_ner,
            self.max_ctx_length, 0), dtype=np.int8)
        np.save(self.files_wrapper.context_pos_file_name, ctx_pos_np_arr)
        qst_pos_np_arr = np.array(self._create_padded_array(self.data.question_ner,
            self.max_qst_length, 0), dtype=np.int8)
        np.save(self.files_wrapper.question_pos_file_name, qst_pos_np_arr)

        ctx_ner_np_arr = np.array(self._create_padded_array(self.data.context_ner,
            self.max_ctx_length, 0), dtype=np.int8)
        np.save(self.files_wrapper.context_ner_file_name, ctx_ner_np_arr)
        qst_ner_np_arr = np.array(self._create_padded_array(self.data.question_ner,
            self.max_qst_length, 0), dtype=np.int8)
        np.save(self.files_wrapper.question_ner_file_name, qst_ner_np_arr)
