"""Utility methods for evaluating a model.
"""

import os
import re
import string

from collections import Counter
from train.train_util import *

def evaluate_train(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset, squad_dataset.train_ds,
            options, limit_samples=True)
    return em, f1

def evaluate_train_partial(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset, squad_dataset.train_ds,
            options, limit_samples=True)
    return em, f1

def evaluate_dev_partial(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset, squad_dataset.dev_ds,
            options, limit_samples=True)
    return em, f1

def evaluate_dev(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset, squad_dataset.dev_ds,
            options, limit_samples=False)
    return em, f1

def evaluate_dev_and_visualize(session, towers, squad_dataset, options):
    """Returns dev (exact match, f1) and also prints contexts, questions,
       ground truths, and predictions to files.
    """
    em, f1, text_predictions, ground_truths = _eval(session, towers, dataset,
            dataset.dev_ds, options, limit_samples=False)
    ctx_file = open(os.path.join(options.data_dir, "context.visualization.txt"))
    qst_file = open(os.path.join(options.data_dir, "question.visualization.txt"))
    gnd_span_file = open(os.path.join(options.data_dir, "ground_truth_spans.visualization.txt"))
    spn_file = open(os.path.join(options.data_dir, "predicted_spans.visualization.txt"))
    print("Writing context, question, ground truth, and predictions to files")
    for z in range(dataset.dev_ds.get_size()):
        ctx_file.write(dataset.vocab.get_sentences(dataset.dev_ds.ctx[z]) + "\n")
        qst_file.write(dataset.vocab.get_sentences(dataset.dev_dx.qst[z]) + "\n")
        gnd_span_file.write(ground_truths[z] + "\n")
        span_file.write(text_predictions[z] + "\n")
    return em, f1

def _eval(session, towers, squad_dataset, dataset, options, limit_samples):
    text_predictions = []
    ground_truths = []
    data_index = 0
    effective_batch_size = len(towers) * options.batch_size
    run_ops = []
    for tower in towers:
        run_ops.append(tower.get_start_spans())
        run_ops.append(tower.get_end_spans())

    data_indices = np.arange(dataset.get_size())
    if limit_samples:
        data_indices = np.random.choice(np.arange(dataset.get_size()),
                options.num_evaluation_samples)
    data_size = data_indices.shape[0]

    while data_index < data_size:
        rng_start = data_index
        rng_end = min(data_index + effective_batch_size, data_size)
        batch_indices = data_indices[rng_start: rng_end]
        indices_per_tower = np.array_split(batch_indices, len(towers))
        feed_dict = get_feed_dict(squad_dataset, dataset, options, towers,
            is_train=False, indices_per_tower=indices_per_tower)

        towers_spans_values = session.run(run_ops, feed_dict=feed_dict)
        sub_batch_index = 0
        for z in range(len(towers)):
            tower_indices = indices_per_tower[z]
            start_spans, end_spans = \
                towers_spans_values[2 * z], towers_spans_values[2 * z + 1]
            for zz in range(tower_indices.shape[0]):
                start = start_spans[zz]
                end = end_spans[zz]
                example_index = batch_indices[sub_batch_index]
                # These need to be the original sentences from the training/dev
                # sets, without any padding/unique word replacements.
                text_predictions.append(dataset.get_sentence(example_index, start, end))
                gnd_start = dataset.spn[example_index, 0]
                gnd_end = dataset.spn[example_index, 1]
                ground_truths.append(dataset.get_sentence(example_index, gnd_start, gnd_end))
                sub_batch_index += 1
        data_index += effective_batch_size
    if options.debug:
        print("text_predictions", str(text_predictions).encode("utf-8"),
              "ground_truths", str(ground_truths).encode("utf-8"))
    exact_match = _avg_over_list(_exact_match_score, text_predictions,
            ground_truths)
    f1 = _avg_over_list(_f1_score, text_predictions, ground_truths)
    return exact_match, f1, text_predictions, ground_truths

def _avg_over_list(metric_fn, predictions, ground_truths):
    avg_value = 0.0
    for i in range(len(predictions)):
        avg_value += metric_fn(predictions[i], ground_truths[i])
    avg_value /= len(predictions)
    return avg_value

def _f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _exact_match_score(prediction, ground_truth):
    return (_normalize_answer(prediction) == _normalize_answer(ground_truth))

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
