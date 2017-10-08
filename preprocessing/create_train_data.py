"""Uses downloaded files to create training and dev data.
"""

import json
import numpy as np
import os
import pickle
import preprocessing.constants as constants
import time

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

    def _maybe_add_sample(self, tok_context, tok_question, qa,
            ctx_offset_dict, ctx_end_offset_dict, list_contexts,
            list_questions, spans, num_values, text_tokens,
            allow_approximation=False):
        for answer in qa["answers"]:
            answer_start = answer["answer_start"]
            text = answer["text"]
            answer_end = answer_start + len(text)
            tok_start = None
            tok_end = None
            exact_match = answer_start in ctx_offset_dict and answer_end in ctx_end_offset_dict
            if allow_approximation and not exact_match:
                # Find the smallest surrounding text.
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
            elif not exact_match:
                continue
            tok_start = tok_start if tok_start is not None else ctx_offset_dict[answer_start]
            tok_end = tok_end if tok_end is not None else ctx_end_offset_dict[answer_end]
            spans[self.value_idx, 0] = tok_context.index(tok_start)
            spans[self.value_idx, 1] = tok_context.index(tok_end)
            ctx_list = []
            list_contexts.append(ctx_list)
            ctx_word_list = []
            text_tokens.append(ctx_word_list)
            for zz in range(len(tok_context)):
                ctx_word = tok_context[zz]["word"]
                vocab_id = self.vocab.get_id_for_word(ctx_word)
                ctx_list.append(vocab_id)
                ctx_word_list.append(ctx_word)
            qst_list = []
            list_questions.append(qst_list)
            for zz in range(len(tok_question)):
                question_word = tok_question[zz]["word"]
                qst_list.append(self.vocab.get_id_for_word(question_word))
            self.value_idx += 1
            print("Value", self.value_idx, "of", num_values, "percent done", 100 * float(self.value_idx) / float(num_values), end="\r")
            return True
        return False

    def _get_num_data_values(self, dataset):
        num_values = 0
        for article in dataset:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    num_values += 1
        return num_values

    def _create_train_data_internal(self, data_file):
        """Returns (contexts, questions, spans)
            contexts: list of lists of integer word ids
            questions: list of lists of integer word ids
            spans: numpy array of shape (num_samples, 2)
        """
        filename = os.path.join(self.data_dir, data_file)
        print("Reading data from file", filename)
        with open(filename) as data_file: 
            data = json.load(data_file)
            dataset = data["data"]
            num_values = self._get_num_data_values(dataset)
            spans = np.zeros((num_values, 2), dtype=np.int32)
            list_contexts = []
            list_questions = []
            text_tokens = []
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
                        question = qa["question"]
                        tok_question = self.nlp.tokenize_text(question)
                        if tok_question is None:
                            continue
                        found_answer_in_context = False
                        # Use the first answer that is found exactly in the context.
                        found_answer_in_context = self._maybe_add_sample(tok_context, tok_question, qa, ctx_offset_dict,
                                ctx_end_offset_dict, list_contexts, list_questions,
                                spans, num_values, text_tokens)
                        if not found_answer_in_context:
                            # The data seems to have some glitches.
                            # In the case that none of the answers were found exactly
                            # in the context, use an approximation.
                            found_answer_in_context = self._maybe_add_sample(tok_context, tok_question, qa, ctx_offset_dict,
                                ctx_end_offset_dict, list_contexts, list_questions,
                                spans, num_values, text_tokens, allow_approximation=True)
                        if not found_answer_in_context:
                            raise Exception(
                                  "Couldn't find answer for question in context",
                                  "Answers", qa["answers"],
                                  "Question", question,
                                  "Context", context,
                                  "Tok context", json.dumps(tok_context, indent=4, sort_keys=True))
            print("")
            spans = spans[:self.value_idx]
            return list_contexts, list_questions, spans, text_tokens

    def create_train_data(self):
        full_train_text_tokens_file_name = os.path.join(self.data_dir, constants.TRAIN_FULL_TEXT_TOKENS_FILE)
        full_dev_text_tokens_file_name = os.path.join(self.data_dir, constants.DEV_FULL_TEXT_TOKENS_FILE)
        full_train_qst_file_name = os.path.join(self.data_dir, constants.TRAIN_QUESTION_FILE)
        full_dev_qst_file_name = os.path.join(self.data_dir, constants.DEV_QUESTION_FILE)
        full_dev_ctx_file_name = os.path.join(self.data_dir, constants.DEV_CONTEXT_FILE)
        full_train_ctx_file_name = os.path.join(self.data_dir, constants.TRAIN_CONTEXT_FILE)
        full_dev_span_file_name = os.path.join(self.data_dir, constants.DEV_SPAN_FILE)
        full_train_span_file_name = os.path.join(self.data_dir, constants.TRAIN_SPAN_FILE)
        file_names = [
            full_train_text_tokens_file_name,
            full_dev_text_tokens_file_name,
            full_train_qst_file_name,
            full_dev_qst_file_name,
            full_dev_ctx_file_name,
            full_train_ctx_file_name,
            full_dev_span_file_name,
            full_train_span_file_name ]
        if all([os.path.exists(filename) for filename in file_names]):
            print("Context, question, and span files already exist. Not creating data again.")
            return

        print("Getting vocabulary")
        self.vocab = get_vocab(self.data_dir)
        print("Finished getting vocabulary")
        self.nlp = StanfordCoreNlpCommunication(self.data_dir)
        self.nlp.start_server()
        print("Getting TRAIN dataset")
        train_ctx_list, train_qst_list, train_spans_np_arr, train_text_tokens = \
            self._create_train_data_internal(constants.TRAIN_SQUAD_FILE)
        print("Getting DEV dataset")
        dev_ctx_list, dev_qst_list, dev_spans_np_arr, dev_text_tokens = \
            self._create_train_data_internal(constants.DEV_SQUAD_FILE)
        self.nlp.stop_server()

        print("Saving text tokens to binary pickle files")
        p_file = open(full_train_text_tokens_file_name, "wb")
        pickle.dump(train_text_tokens, p_file)
        p_file.close()
        p_file = open(full_dev_text_tokens_file_name, "wb")
        pickle.dump(dev_text_tokens, p_file)
        p_file.close()

        print("Saving span numpy arrays")
        np.save(full_train_span_file_name, train_spans_np_arr)
        np.save(full_dev_span_file_name, dev_spans_np_arr)

        print("Saving context numpy arrays")
        max_context_length = max(
                max([len(x) for x in train_ctx_list]),
                max([len(x) for x in dev_ctx_list]))
        train_ctx_np_arr = np.array([x + [self.vocab.PAD_ID] * (max_context_length - len(x)) for x in train_ctx_list], dtype=np.int32)
        np.save(full_train_ctx_file_name, train_ctx_np_arr)
        dev_ctx_np_arr = np.array([x + [self.vocab.PAD_ID] * (max_context_length - len(x)) for x in dev_ctx_list], dtype=np.int32)
        np.save(full_dev_ctx_file_name, dev_ctx_np_arr)

        print("Saving question numpy arrays")
        max_question_length = max(
                max([len(x) for x in train_qst_list]),
                max([len(x) for x in dev_qst_list]))
        train_qst_np_arr = np.array([x + [self.vocab.PAD_ID] * (max_question_length - len(x)) for x in train_qst_list], dtype=np.int32)
        np.save(full_train_qst_file_name, train_qst_np_arr)
        dev_qst_np_arr = np.array([x + [self.vocab.PAD_ID] * (max_question_length - len(x)) for x in dev_qst_list], dtype=np.int32)
        np.save(full_dev_qst_file_name, dev_qst_np_arr)

        print("Finished creating training data!")
