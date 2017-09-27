"""Functions to help with training and evaluation.
"""

import numpy as np
import tensorflow as tf

def get_feed_dict(squad_data, dataset, options, towers, is_train,
        indices_per_tower=None):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")
    examples_per_tower = int(options.batch_size / len(towers))
    feed_dict = {}
    if indices_per_tower is None:
        rng = np.arange(dataset.get_size())
        rnd_choice = np.random.choice(rng, options.batch_size * len(towers))
        indices_per_tower = np.array_split(rnd_choice, len(towers))
    for i in range(len(towers)):
        tower = towers[i]
        twr_idx = indices_per_tower[i]
        batch_ctx = np.reshape(dataset.ctx[twr_idx], (-1, options.max_ctx_length))
        batch_qst = np.reshape(dataset.qst[twr_idx], (-1, options.max_qst_length))
        batch_spn = np.reshape(dataset.spn[twr_idx], (-1, 2))
        feed_dict[tower.get_contexts_placeholder()] = batch_ctx
        feed_dict[tower.get_questions_placeholder()] = batch_qst
        feed_dict[tower.get_spans_placeholder()] = batch_spn
        feed_dict[tower.get_embedding_placeholder()] = squad_data.embeddings
        feed_dict[tower.get_keep_prob_placeholder()] = 1 if not is_train else 1 - options.dropout
    return feed_dict

def get_train_feed_dict(squad_data, options, towers):
    return get_feed_dict(squad_data, squad_data.train_ds, options, towers, is_train=True)

def get_dev_feed_dict(squad_data, options, towers):
    return get_feed_dict(squad_data, squad_data.dev_ds, options, towers, is_train=False)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
