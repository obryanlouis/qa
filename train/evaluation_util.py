"""Utility methods for evaluating a model.
"""

import json
import math
import os
import time

from train.evaluation_functions import *
from train.train_util import *
from train.print_utils import *
from train.sentence_util import *
from util.string_util import *

class _EvalResult:
    def __init__(self, em, f1, passages, questions, text_predictions, ground_truths):
        self.em = em
        self.f1 = f1
        self.passages = passages
        self.questions = questions
        self.text_predictions = text_predictions
        self.ground_truths = ground_truths

def evaluate_train(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, is_train=True, sample_limit=None)
    return result.em, result.f1

def evaluate_train_partial(session, towers, squad_dataset, options,
    sample_limit):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, is_train=True, sample_limit=sample_limit)
    return result.em, result.f1

def evaluate_dev_partial(session, towers, squad_dataset, options,
    sample_limit):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, is_train=False, sample_limit=sample_limit)
    return result.em, result.f1

def evaluate_dev(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, is_train=False, sample_limit=None)
    return result.em, result.f1

def evaluate_dev_and_visualize(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1) and also prints contexts, questions,
       ground truths, and predictions to files.
    """
    if not os.path.exists(options.evaluation_dir):
        os.makedirs(options.evaluation_dir)
    result = _eval(session, towers, squad_dataset,
            options, is_train=False, sample_limit=None)
    ctx_file = open(os.path.join(options.evaluation_dir, "context.visualization.txt"), mode="w")
    qst_file = open(os.path.join(options.evaluation_dir, "question.visualization.txt"), mode="w")
    gnd_span_file = open(os.path.join(options.evaluation_dir, "ground_truth_spans.visualization.txt"), mode="w")
    spn_file = open(os.path.join(options.evaluation_dir, "predicted_spans.visualization.txt"), mode="w")
    print("Writing context, question, ground truth, and predictions to files in evaluation dir [" + options.evaluation_dir + "]")
    for z in range(len(result.passages)):
        ctx_file.write(utf8_str(result.passages[z]))
        ctx_file.write("\n")
        qst_file.write(utf8_str(result.questions[z]))
        qst_file.write("\n")
        gnd_span_file.write(utf8_str(result.ground_truths[z]))
        gnd_span_file.write("\n")
        spn_file.write(utf8_str(result.text_predictions[z]))
        spn_file.write("\n")
    for f in [ctx_file, qst_file, gnd_span_file, spn_file]:
        f.close()
    return result.em, result.f1

def _eval(session, towers, squad_dataset, options, is_train, sample_limit):
    passages = []
    questions = []
    text_predictions = []
    ground_truths = []
    run_ops = []
    for tower in towers:
        run_ops.append(tower.get_start_span_probs())
        run_ops.append(tower.get_end_span_probs())
        run_ops.append(tower.get_data_index_iterator())
        run_ops.append(tower.get_qst())
    dataset = squad_dataset.train_ds if is_train else squad_dataset.dev_ds

    num_dev_files = squad_dataset.get_num_dev_files()
    num_files_processed = 0
    estimated_total_dev_samples = squad_dataset.estimate_total_dev_ds_size()
    total_samples_processed = 0
    start_time = time.time()
    squad_prediction_format = {} # key=squad question id, value=prediction (string)
    while True:
        if total_samples_processed >= estimated_total_dev_samples \
            and num_dev_files == 1:
            break
        if sample_limit is None and num_files_processed >= num_dev_files:
            break
        if sample_limit is not None and total_samples_processed >= sample_limit:
            break
        feed_dict = get_eval_feed_dict(squad_dataset, options, towers, is_train=is_train)
        iter_start = time.time()
        towers_spans_values = session.run(run_ops, feed_dict=feed_dict)
        batch_increment = options.batch_size * max(1, options.num_gpus)
        total_samples_processed += batch_increment

        num_towers = len(towers)
        items_per_tower = int(len(run_ops) / num_towers)
        for z in range(num_towers):
            start_span_probs, end_span_probs, data_indices, qst_values = \
                towers_spans_values[items_per_tower * z], \
                towers_spans_values[items_per_tower * z + 1], \
                towers_spans_values[items_per_tower * z + 2], \
                towers_spans_values[items_per_tower * z + 3]
            if start_span_probs.shape != end_span_probs.shape:
                print("start_span_probs shape", start_span_probs.shape,
                      "end_span_probs shape", end_span_probs.shape,
                      "data_indices shape", data_indices.shape)
                print("start_span_probs", start_span_probs)
                print("end_span_probs", end_span_probs)
                print("data_indices", data_indices)
            assert start_span_probs.shape == end_span_probs.shape
            assert start_span_probs.shape[0] == data_indices.shape[0]
            assert start_span_probs.shape[0] == qst_values.shape[0]
            for zz in range(start_span_probs.shape[0]):
                start, end = get_best_start_and_end(start_span_probs[zz],
                    end_span_probs[zz], options)
                example_index = data_indices[zz]
                question_word_ids = qst_values[zz]
                question = find_question_sentence(question_word_ids, squad_dataset.vocab)
                prediction_str = dataset.get_sentence(example_index, start, end)
                if dataset.question_ids_to_squad_ids is not None:
                    squad_question_id = \
                        dataset.question_ids_to_squad_ids[example_index]
                    if squad_question_id in squad_prediction_format:
                        continue
                    squad_prediction_format[squad_question_id] = prediction_str
                text_predictions.append(prediction_str)
                acceptable_gnd_truths = dataset.get_sentences_for_all_gnd_truths(example_index)
                ground_truths.append(acceptable_gnd_truths)
                passages.append(dataset.get_sentence(example_index, 0, squad_dataset.get_max_ctx_len() - 1))
                questions.append(question)
        if not is_train:
            squad_dataset.increment_val_samples_processed(batch_increment)
        else:
            squad_dataset.increment_train_samples_processed(batch_increment)
        if squad_dataset.get_current_dev_file_number() != num_files_processed:
            num_files_processed += 1

            samples = sample_limit if sample_limit is not None \
                else estimated_total_dev_samples
            est_percent_done = min((100 * float(total_samples_processed) / float(samples)), 100)
            est_processing_rate = est_percent_done / (time.time() - start_time)
            iter_time = time.time() - iter_start
            est_time_left = (float(samples
                - total_samples_processed) / batch_increment) * iter_time
            clear_printed_line()
            print("Estimated percent evaluated: %f (processing files: %d of %d). %s"
                % (est_percent_done, min(num_files_processed + 1, num_dev_files),
                    num_dev_files, readable_eta(est_time_left)),
                end="\r", flush=True)
    print("")
    if not is_train:
        if not os.path.exists(options.evaluation_dir):
            os.makedirs(options.evaluation_dir)
        with open(os.path.join(options.evaluation_dir,
            "predictions.json"), mode="w") as predictions_file:
            json.dump(squad_prediction_format, predictions_file)
    if options.verbose_logging:
        print("text_predictions", utf8_str(text_predictions),
              "ground_truths", utf8_str(ground_truths))
    exact_match = avg_over_list(exact_match_score, text_predictions,
            ground_truths)
    f1 = avg_over_list(f1_score, text_predictions, ground_truths)
    return _EvalResult(exact_match, f1, passages, questions, text_predictions, ground_truths)
