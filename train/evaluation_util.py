"""Utility methods for evaluating a model.
"""

import os

from train.train_util import *
from train.evaluation_functions import *

def evaluate_train(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=True, limit_samples=True)
    return em, f1

def evaluate_train_partial(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=True, limit_samples=True)
    return em, f1

def evaluate_dev_partial(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=True)
    return em, f1

def evaluate_dev(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1)"""
    em, f1, _, _ = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=False)
    return em, f1

def evaluate_dev_and_visualize(session, towers, squad_dataset, options, tf_dataset):
    """Returns dev (exact match, f1) and also prints contexts, questions,
       ground truths, and predictions to files.
    """
    em, f1, text_predictions, ground_truths = _eval(session, towers, squad_dataset,
            options, tf_dataset, is_train=False, limit_samples=False)
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

def _eval(session, towers, squad_dataset, options, tf_dataset, is_train, limit_samples):
    text_predictions = []
    ground_truths = []
    run_ops = []
    for tower in towers:
        run_ops.append(tower.get_start_span_probs())
        run_ops.append(tower.get_end_span_probs())
        run_ops.append(tower.get_gnd_truth_spans())
        run_ops.append(tower.get_data_index_iterator())
    dataset = squad_dataset.train_ds if is_train else squad_dataset.dev_ds

    num_samples = dataset.get_size() if not limit_samples else options.num_evaluation_samples
    num_batches = max(1, int(num_samples / options.batch_size)) # close enough
    for _ in range(num_batches):
        feed_dict = get_eval_feed_dict(squad_dataset, tf_dataset, options, towers, is_train=is_train)
        towers_spans_values = session.run(run_ops, feed_dict=feed_dict)

        num_towers = len(towers)
        items_per_tower = int(len(run_ops) / num_towers)
        for z in range(num_towers):
            start_span_probs, end_span_probs, gnd_spans, data_indices = \
                towers_spans_values[items_per_tower * z], \
                towers_spans_values[items_per_tower * z + 1], \
                towers_spans_values[items_per_tower * z + 2], \
                towers_spans_values[items_per_tower * z + 3],
            if start_span_probs.shape != end_span_probs.shape:
                print("start_span_probs shape", start_span_probs.shape,
                      "end_span_probs shape", end_span_probs.shape,
                      "gnd_spans shape", gnd_spans.shape,
                      "data_indices shape", data_indices.shape)
                print("start_span_probs", start_span_probs)
                print("end_span_probs", end_span_probs)
                print("gnd_spans", gnd_spans)
                print("data_indices", gnd_spans)
            assert start_span_probs.shape == end_span_probs.shape
            assert start_span_probs.shape[0] == gnd_spans.shape[0]
            assert start_span_probs.shape[0] == data_indices.shape[0]
            for zz in range(start_span_probs.shape[0]):
                start, end = get_best_start_and_end(start_span_probs[zz],
                        end_span_probs[zz], options)
                example_index = data_indices[zz]
                # These need to be the original sentences from the training/dev
                # sets, without any padding/unique word replacements.
                text_predictions.append(dataset.get_sentence(example_index, start, end))
                gnd_start = gnd_spans[zz, 0]
                gnd_end = gnd_spans[zz, 1]
                ground_truths.append(dataset.get_sentence(example_index, gnd_start, gnd_end))
    if options.verbose_logging:
        print("text_predictions", str(text_predictions).encode("utf-8"),
              "ground_truths", str(ground_truths).encode("utf-8"))
    exact_match = avg_over_list(exact_match_score, text_predictions,
            ground_truths)
    f1 = avg_over_list(f1_score, text_predictions, ground_truths)
    return exact_match, f1, text_predictions, ground_truths
