"""Utility methods for evaluating a model.
"""

import math
import os

from train.train_util import *
from train.evaluation_functions import *

class _EvalResult:
    def __init__(self, em, f1, passages, questions, text_predictions, ground_truths):
        self.em = em
        self.f1 = f1
        self.passages = passages
        self.questions = questions
        self.text_predictions = text_predictions
        self.ground_truths = ground_truths

def _utf8_str(obj):
    return str(str(obj).encode("utf-8"))

def evaluate_train(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=True, limit_samples=False)
    return result.em, result.f1

def evaluate_train_partial(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=True, limit_samples=True)
    return result.em, result.f1

def evaluate_dev_partial(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=True)
    return result.em, result.f1

def evaluate_dev(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    result = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=False)
    return result.em, result.f1

def evaluate_dev_and_visualize(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1) and also prints contexts, questions,
       ground truths, and predictions to files.
    """
    if not os.path.exists(options.evaluation_dir):
        os.makedirs(options.evaluation_dir)
    result = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=False)
    ctx_file = open(os.path.join(options.evaluation_dir, "context.visualization.txt"), mode="w")
    qst_file = open(os.path.join(options.evaluation_dir, "question.visualization.txt"), mode="w")
    gnd_span_file = open(os.path.join(options.evaluation_dir, "ground_truth_spans.visualization.txt"), mode="w")
    spn_file = open(os.path.join(options.evaluation_dir, "predicted_spans.visualization.txt"), mode="w")
    print("Writing context, question, ground truth, and predictions to files in evaluation dir [" + options.evaluation_dir + "]")
    for z in range(len(result.passages)):
        ctx_file.write(_utf8_str(result.passages[z]))
        ctx_file.write("\n")
        qst_file.write(_utf8_str(result.questions[z]))
        qst_file.write("\n")
        gnd_span_file.write(_utf8_str(result.ground_truths[z]))
        gnd_span_file.write("\n")
        spn_file.write(_utf8_str(result.text_predictions[z]))
        spn_file.write("\n")
    for f in [ctx_file, qst_file, gnd_span_file, spn_file]:
        f.close()
    return result.em, result.f1

def _eval(session, towers, squad_dataset, options, tf_dataset, is_train, limit_samples):
    passages = []
    questions = []
    text_predictions = []
    ground_truths = []
    run_ops = []
    for tower in towers:
        run_ops.append(tower.get_start_span_probs())
        run_ops.append(tower.get_end_span_probs())
        run_ops.append(tower.get_data_index_iterator())
    dataset = squad_dataset.train_ds if is_train else squad_dataset.dev_ds

    num_samples = dataset.get_size() if not limit_samples else options.num_evaluation_samples
    num_batches = max(1, math.ceil(num_samples / options.batch_size)) # close enough
    for batch_number in range(num_batches):
        feed_dict = get_eval_feed_dict(squad_dataset, tf_dataset, options, towers, is_train=is_train)
        towers_spans_values = session.run(run_ops, feed_dict=feed_dict)

        num_towers = len(towers)
        items_per_tower = int(len(run_ops) / num_towers)
        for z in range(num_towers):
            start_span_probs, end_span_probs, data_indices = \
                towers_spans_values[items_per_tower * z], \
                towers_spans_values[items_per_tower * z + 1], \
                towers_spans_values[items_per_tower * z + 2]
            if start_span_probs.shape != end_span_probs.shape:
                print("start_span_probs shape", start_span_probs.shape,
                      "end_span_probs shape", end_span_probs.shape,
                      "data_indices shape", data_indices.shape)
                print("start_span_probs", start_span_probs)
                print("end_span_probs", end_span_probs)
                print("data_indices", data_indices)
            assert start_span_probs.shape == end_span_probs.shape
            assert start_span_probs.shape[0] == data_indices.shape[0]
            for zz in range(start_span_probs.shape[0]):
                start, end = get_best_start_and_end(start_span_probs[zz],
                        end_span_probs[zz], options)
                example_index = data_indices[zz]
                passages.append(dataset.get_sentence(example_index, 0, squad_dataset.get_max_ctx_len() - 1))
                questions.append(dataset.get_question_sentence(example_index))
                # These need to be the original sentences from the training/dev
                # sets, without any padding/unique word replacements.
                text_predictions.append(dataset.get_sentence(example_index, start, end))
                acceptable_gnd_truths = dataset.get_sentences_for_all_gnd_truths(example_index)
                ground_truths.append(acceptable_gnd_truths)
        if not limit_samples:
            print("Percent evaluated: %f (%d / %d)"
                  % ((100 * float(batch_number + 1) / float(num_batches)),
                     batch_number + 1, num_batches), end="\r")
    print("")
    if options.verbose_logging:
        print("text_predictions", _utf8_str(text_predictions),
              "ground_truths", _utf8_str(ground_truths))
    exact_match = avg_over_list(exact_match_score, text_predictions,
            ground_truths)
    f1 = avg_over_list(f1_score, text_predictions, ground_truths)
    return _EvalResult(exact_match, f1, passages, questions, text_predictions, ground_truths)
