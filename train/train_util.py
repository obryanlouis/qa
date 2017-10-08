"""Functions to help with training and evaluation.
"""

import numpy as np
import tensorflow as tf

def get_feed_dict(squad_data, tf_dataset, options, towers, is_train):
    if len(towers) < 1:
        raise Exception("There are no models in the list of towers to train")
    examples_per_tower = int(options.batch_size / len(towers))
    feed_dict = {}
    for i in range(len(towers)):
        tower = towers[i]
        feed_dict[tower.get_embedding_placeholder()] = squad_data.embeddings
        feed_dict[tower.get_keep_prob_placeholder()] = 1 if not is_train else 1 - options.dropout
    train_handle = tf_dataset.get_train_handle()
    dev_handle = tf_dataset.get_dev_handle()
    tf_handle = tf_dataset.get_iterator_handle()
    feed_dict[tf_handle] = train_handle if is_train else dev_handle
    return feed_dict

def get_train_feed_dict(squad_data, tf_dataset, options, towers):
    return get_feed_dict(squad_data, tf_dataset, options, towers, is_train=True)

def get_dev_feed_dict(squad_data, tf_dataset, options, towers):
    return get_feed_dict(squad_data, tf_dataset, options, towers, is_train=False)

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
