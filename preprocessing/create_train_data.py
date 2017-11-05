"""Uses downloaded files to create training and dev data.
"""

import json
import numpy as np
import os
import preprocessing.constants as constants
import time

from preprocessing.char_util import *
from preprocessing.dataset_files_saver import *
from preprocessing.dataset_files_wrapper import *
from preprocessing.file_util import *
from preprocessing.raw_training_data import *
from preprocessing.stanford_corenlp_util import StanfordCoreNlpCommunication
from preprocessing.vocab_util import get_vocab

# Note: Some of the training/dev data seems to be inaccurate. This code
# tries to make sure that at least one of the "qa" options in the acceptable
# answers list is accurate and includes it in the data set.

class DataParser():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.value_idx = 0
        self.vocab = None
        self.nlp = None
        self.question_id = 0

    def _maybe_add_samples(self, tok_context, tok_question, qa,
            ctx_offset_dict, ctx_end_offset_dict, list_contexts, list_word_in_question,
            list_questions, list_word_in_context, spans, num_values, text_tokens,
            question_ids, question_ids_to_ground_truths, context_chars, question_chars,
            is_dev):
        first_answer = True
        for answer in qa["answers"]:
            answer_start = answer["answer_start"]
            text = answer["text"]
            answer_end = answer_start + len(text)
            tok_start = None
            tok_end = None
            exact_match = answer_start in ctx_offset_dict and answer_end in ctx_end_offset_dict
            if not exact_match:
                # Sometimes, the given answer isn't actually in the context.
                # If so, find the smallest surrounding text instead.
                for z in range(len(tok_context)):
                    tok = tok_context[z]
                    st = tok["characterOffsetBegin"]
                    end = tok["characterOffsetEnd"]
                    if st <= answer_start and answer_start <= end:
                        tok_start = tok
                    elif tok_start is not None:
                        tok_end = tok
                        if end >= answer_end:
                            break
            tok_start = tok_start if tok_start is not None else ctx_offset_dict[answer_start]
            tok_end = tok_end if tok_end is not None else ctx_end_offset_dict[answer_end]
            tok_start_idx = tok_context.index(tok_start)
            tok_end_idx = tok_context.index(tok_end)
            gnd_truths_list = []
            if self.question_id in question_ids_to_ground_truths:
                gnd_truths_list = question_ids_to_ground_truths[self.question_id]
            gnd_truths_list.append((tok_start_idx, tok_end_idx))
            question_ids_to_ground_truths[self.question_id] = gnd_truths_list
            # For dev, only keep one exmaple per question, and the set of all
            # acceptable answers. This reduces the required memory for storing
            # data.
            if is_dev and not first_answer:
                continue
            first_answer = False

            spans.append([tok_start_idx, tok_end_idx])
            ctx_list = []
            list_contexts.append(ctx_list)
            ctx_word_list = []
            text_tokens.append(ctx_word_list)
            question_ids.append(self.question_id)
            ctx_word_ids = set()
            qst_word_ids = set()
            ctx_char_list = []
            context_chars.append(ctx_char_list)
            for zz in range(len(tok_context)):
                ctx_word = tok_context[zz]["word"]
                vocab_id = self.vocab.get_id_for_word(ctx_word)
                ctx_list.append(vocab_id)
                ctx_word_list.append(ctx_word)
                ctx_word_ids.add(vocab_id)
                word_char_list = []
                ctx_char_list.append(word_char_list)
                for char in ctx_word:
                    word_char_list.append(self.vocab.get_id_for_char(char))
            qst_list = []
            list_questions.append(qst_list)
            qst_char_list = []
            question_chars.append(qst_char_list)
            for zz in range(len(tok_question)):
                question_word = tok_question[zz]["word"]
                word_id = self.vocab.get_id_for_word(question_word)
                qst_list.append(self.vocab.get_id_for_word(question_word))
                qst_word_ids.add(word_id)
                word_char_list = []
                qst_char_list.append(word_char_list)
                for char in question_word:
                    word_char_list.append(self.vocab.get_id_for_char(char))
            word_in_question_list = [1 if word_id in qst_word_ids else 0 for word_id in ctx_list]
            word_in_context_list = [1 if word_id in ctx_word_ids else 0 for word_id in qst_list]
            list_word_in_question.append(word_in_question_list)
            list_word_in_context.append(word_in_context_list)
            print("Value", self.value_idx, "of", num_values, "percent done", 100 * float(self.value_idx) / float(num_values), end="\r")
            self.value_idx += 1

    def _get_num_data_values(self, dataset):
        num_values = 0
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    num_values += 1
        return num_values

    def _create_train_data_internal(self, data_file, is_dev):
        """Returns (contexts, word_in_question, questions, word_in_context, spans)
            contexts: list of lists of integer word ids
            word_in_question: list of lists of booleans indicating whether each
                word in the context is present in the question
            questions: list of lists of integer word ids
            word_in_context: list of lists of booleans indicating whether each
                word in the question is present in the context
            spans: numpy array of shape (num_samples, 2)
            text_tokens: list of strings in the context
            context_chars: a list of lists of lists of characters the contexts
            question_chars: a list of lists of lists of characters in the
                questions
            question_ids: a list of ints that indicates which question the
                given sample is part of. this has the same length as
                |contexts| and |questions|. multiple samples may come from
                the same question because there are potentially multiple valid
                answers for the same question
            question_id_to_ground_truths: a map whose keys are question id's
                the same as in the above |question_ids| return value and whose
                values are sets of acceptable answer strings
        """
        filename = os.path.join(self.data_dir, data_file)
        print("Reading data from file", filename)
        with open(filename) as data_file: 
            data = json.load(data_file)
            dataset = data["data"]
            num_values = self._get_num_data_values(dataset)
            spans = []
            list_contexts = []
            list_word_in_question = []
            list_questions = []
            text_tokens = []
            list_word_in_context = []
            question_ids = []
            question_ids_to_ground_truths = {}
            context_chars = []
            question_chars = []
            self.value_idx = 0
            for article in dataset:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    tok_context = self.nlp.tokenize_text(context)
                    if tok_context is None:
                        continue
                    ctx_offset_dict = {}
                    for tok in tok_context:
                        ctx_offset_dict[tok["characterOffsetBegin"]] = tok
                    ctx_end_offset_dict = {}
                    for tok in tok_context:
                        ctx_end_offset_dict[tok["characterOffsetEnd"]] = tok
                    for qa in paragraph["qas"]:
                        self.question_id += 1
                        question = qa["question"]
                        tok_question = self.nlp.tokenize_text(question)
                        if tok_question is None:
                            continue
                        found_answer_in_context = False
                        found_answer_in_context = self._maybe_add_samples(tok_context, tok_question, qa, ctx_offset_dict,
                                ctx_end_offset_dict, list_contexts, list_word_in_question, list_questions, list_word_in_context,
                                spans, num_values, text_tokens, question_ids, question_ids_to_ground_truths, context_chars, question_chars,
                                is_dev)
            print("")
            spans = np.array(spans[:self.value_idx], dtype=np.int32)
            return RawTrainingData(
                list_contexts = list_contexts,
                list_word_in_question = list_word_in_question,
                list_questions = list_questions,
                list_word_in_context = list_word_in_context,
                spans = spans,
                text_tokens = text_tokens,
                context_chars = context_chars,
                question_chars = question_chars,
                question_ids = question_ids,
                question_ids_to_ground_truths = question_ids_to_ground_truths)

    def _create_padded_array(self, list_of_py_arrays, max_len, pad_value):
        return [py_arr + [pad_value] * (max_len - len(py_arr)) for py_arr in list_of_py_arrays]

    def create_train_data(self):
        train_files_wrapper = DatasetFilesWrapper.create_train_files_wrapper(self.data_dir)
        dev_files_wrapper = DatasetFilesWrapper.create_dev_files_wrapper(self.data_dir)

        file_names = train_files_wrapper.get_all_files() \
            + dev_files_wrapper.get_all_files()
        if all([os.path.exists(filename) for filename in file_names]):
            print("Context, question, and span files already exist. Not creating data again.")
            return

        print("Getting vocabulary")
        self.vocab = get_vocab(self.data_dir)
        print("Finished getting vocabulary")
        self.nlp = StanfordCoreNlpCommunication(self.data_dir)
        self.nlp.start_server()
        print("Waiting for Core NLP server to start")
        # TODO: improve this logic by actually pinging the server until it
        # responds.
        time.sleep(5)
        print("Getting DEV dataset")
        dev_raw_data = self._create_train_data_internal(
            constants.DEV_SQUAD_FILE, is_dev=True)
        print("Getting TRAIN dataset")
        train_raw_data = self._create_train_data_internal(
            constants.TRAIN_SQUAD_FILE, is_dev=False)
        self.nlp.stop_server()

        max_context_length = max(
                max([len(x) for x in train_raw_data.list_contexts]),
                max([len(x) for x in dev_raw_data.list_contexts]))

        max_question_length = max(
                max([len(x) for x in train_raw_data.list_questions]),
                max([len(x) for x in dev_raw_data.list_questions]))

        print("Saving TRAIN data")
        train_file_saver = DatasetFilesSaver(
                train_files_wrapper,
                max_context_length,
                max_question_length,
                self.vocab,
                train_raw_data)
        train_file_saver.save()

        print("Saving DEV data")
        dev_file_saver = DatasetFilesSaver(
                dev_files_wrapper,
                max_context_length,
                max_question_length,
                self.vocab,
                dev_raw_data)
        dev_file_saver.save()

        print("Finished creating training data!")
