"""Provides a way to save dataset files after the raw data is created.
"""

import math
import numpy as np
import preprocessing.constants as constants

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
        batch_idx = 0
        num_samples = len(self.data.list_contexts)
        while batch_idx < num_samples:
            file_names = self.files_wrapper.create_new_file_names()
            print("Saving batch %d / %d" % (
                batch_idx / constants.MAX_SAMPLES_PER_SPLIT,
                math.ceil(float(num_samples) / 
                    float(constants.MAX_SAMPLES_PER_SPLIT))))
            next_batch_idx = batch_idx + constants.MAX_SAMPLES_PER_SPLIT

            print("Saving question ids")
            question_ids_np_arr = np.array(
                self.data.question_ids[batch_idx:next_batch_idx],
                dtype=np.int32)
            np.save(file_names.question_ids_file_name, question_ids_np_arr)

            filtered_squad_ids_dict = {}
            filtered_passage_contexts_dict = {}
            for z in range(question_ids_np_arr.shape[0]):
                question_id = question_ids_np_arr[z]
                filtered_squad_ids_dict[question_id] = \
                    self.data.question_ids_to_squad_question_id[question_id]
                filtered_passage_contexts_dict[question_id] = \
                    self.data.question_ids_to_passage_context[question_id]

            print("Saving question ids to SQuAD question ids dict")
            save_pickle_file(file_names.question_ids_to_squad_question_id_file_name,
                filtered_squad_ids_dict)

            print("Saving passage contexts to SQuAD question ids dict")
            save_pickle_file(file_names.question_ids_to_passage_context_file_name,
                filtered_passage_contexts_dict)

            print("Saving span numpy arrays")
            np.save(file_names.spn_file_name,
                self.data.spans[batch_idx:next_batch_idx])

            print("Saving context numpy arrays")
            ctx_np_arr = np.array(self._create_padded_array(
                self.data.list_contexts[batch_idx:next_batch_idx],
                self.max_ctx_length, self.vocab.PAD_ID), dtype=np.int32)
            np.save(file_names.ctx_file_name, ctx_np_arr)

            print("Saving question numpy arrays")
            qst_np_arr = np.array(self._create_padded_array(
                self.data.list_questions[batch_idx:next_batch_idx],
                self.max_qst_length, self.vocab.PAD_ID), dtype=np.int32)
            np.save(file_names.qst_file_name, qst_np_arr)

            print("Saving additional feature numpy arrays")
            word_in_question_np_arr = np.array(self._create_padded_array(
                self.data.list_word_in_question[batch_idx:next_batch_idx],
                self.max_ctx_length, 0), dtype=np.float32)
            np.save(file_names.word_in_question_file_name, word_in_question_np_arr)
            word_in_context_np_arr = np.array(self._create_padded_array(
                self.data.list_word_in_context[batch_idx:next_batch_idx],
                self.max_ctx_length, 0), dtype=np.float32)
            np.save(file_names.word_in_context_file_name, word_in_context_np_arr)

            print("Saving POS and NER tags")
            ctx_pos_np_arr = np.array(self._create_padded_array(
                self.data.context_ner[batch_idx:next_batch_idx],
                self.max_ctx_length, 0), dtype=np.int8)
            np.save(file_names.context_pos_file_name, ctx_pos_np_arr)
            qst_pos_np_arr = np.array(self._create_padded_array(
                self.data.question_ner[batch_idx:next_batch_idx],
                self.max_qst_length, 0), dtype=np.int8)
            np.save(file_names.question_pos_file_name, qst_pos_np_arr)

            ctx_ner_np_arr = np.array(self._create_padded_array(
                self.data.context_ner[batch_idx:next_batch_idx],
                self.max_ctx_length, 0), dtype=np.int8)
            np.save(file_names.context_ner_file_name, ctx_ner_np_arr)
            qst_ner_np_arr = np.array(self._create_padded_array(
                self.data.question_ner[batch_idx:next_batch_idx],
                self.max_qst_length, 0), dtype=np.int8)
            np.save(file_names.question_ner_file_name, qst_ner_np_arr)

            batch_idx = next_batch_idx
