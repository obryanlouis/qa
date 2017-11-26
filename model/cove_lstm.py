"""Defines an LSTM that loads CoVe weights. Why isn't this easier in tensorflow?
"""

import numpy as np
import os
import preprocessing.constants as constants
import tensorflow as tf


class CoveCellsWrapper:
    def __init__(self, forward_cell_l0, backward_cell_l0, forward_cell_l1,
        backward_cell_l1):
        self.forward_cell_l0 = forward_cell_l0
        self.backward_cell_l0 = backward_cell_l0
        self.forward_cell_l1 = forward_cell_l1
        self.backward_cell_l1 = backward_cell_l1

def _load_cove_np_arr(cove_file_name, options):
    return np.transpose(
        np.load(os.path.join(options.data_dir, constants.COVE_WEIGHTS_FOLDER,
        cove_file_name)))

def load_cove_lstm(options):
    weight_ih_l0 = _load_cove_np_arr("weight_ih_l0.npy", options)
    weight_hh_l0 = _load_cove_np_arr("weight_hh_l0.npy", options)
    bias_ih_l0 = _load_cove_np_arr("bias_ih_l0.npy", options)
    bias_hh_l0 = _load_cove_np_arr("bias_hh_l0.npy", options)
    weight_ih_l1 = _load_cove_np_arr("weight_ih_l1.npy", options)
    weight_hh_l1 = _load_cove_np_arr("weight_hh_l1.npy", options)
    bias_ih_l1 = _load_cove_np_arr("bias_ih_l1.npy", options)
    bias_hh_l1 = _load_cove_np_arr("bias_hh_l1.npy", options)
    weight_ih_l0_reverse = _load_cove_np_arr("weight_ih_l0_reverse.npy", options)
    weight_hh_l0_reverse = _load_cove_np_arr("weight_hh_l0_reverse.npy", options)
    bias_ih_l0_reverse = _load_cove_np_arr("bias_ih_l0_reverse.npy", options)
    bias_hh_l0_reverse = _load_cove_np_arr("bias_hh_l0_reverse.npy", options)
    weight_ih_l1_reverse = _load_cove_np_arr("weight_ih_l1_reverse.npy", options)
    weight_hh_l1_reverse = _load_cove_np_arr("weight_hh_l1_reverse.npy", options)
    bias_ih_l1_reverse = _load_cove_np_arr("bias_ih_l1_reverse.npy", options)
    bias_hh_l1_reverse = _load_cove_np_arr("bias_hh_l1_reverse.npy", options)

    forward_cell_0 = CoveLSTMCell(weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0)
    forward_cell_1 = CoveLSTMCell(weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1)
    backward_cell_0 = CoveLSTMCell(weight_ih_l0_reverse, weight_hh_l0_reverse,
        bias_ih_l0_reverse, bias_hh_l0_reverse)
    backward_cell_1 = CoveLSTMCell(weight_ih_l1_reverse, weight_hh_l1_reverse,
        bias_ih_l1_reverse, bias_hh_l1_reverse)

    return CoveCellsWrapper(forward_cell_0, backward_cell_0, forward_cell_1,
        backward_cell_1)

class CoveLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
    super(CoveLSTMCell, self).__init__()
    self._state_size = tf.contrib.rnn.LSTMStateTuple(constants.WORD_VEC_DIM,
        constants.WORD_VEC_DIM)
    self._output_size = weight_hh.shape[0]
    self.w_ii, self.w_if, self.w_ig, self.w_io = tf.split(weight_ih,
        num_or_size_splits=4, axis=1)
    self.w_hi, self.w_hf, self.w_hg, self.w_ho = tf.split(weight_hh,
        num_or_size_splits=4, axis=1)
    self.b_ii, self.b_if, self.b_ig, self.b_io = tf.split(bias_ih,
        num_or_size_splits=4, axis=0)
    self.b_hi, self.b_hf, self.b_hg, self.b_ho = tf.split(bias_hh,
        num_or_size_splits=4, axis=0)

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    c_prev, m_prev = state

    i = tf.nn.sigmoid(tf.matmul(inputs, self.w_ii) + self.b_ii
        + tf.matmul(m_prev, self.w_hi) + self.b_hi)
    f = tf.nn.sigmoid(tf.matmul(inputs, self.w_if) + self.b_if
        + tf.matmul(m_prev, self.w_hf) + self.b_hf)
    g = tf.tanh(tf.matmul(inputs, self.w_ig) + self.b_ig
        + tf.matmul(m_prev, self.w_hg) + self.b_hg)
    o = tf.nn.sigmoid(tf.matmul(inputs, self.w_io) + self.b_io
        + tf.matmul(m_prev, self.w_ho) + self.b_ho)
    c = f * c_prev + i * g
    m = o * tf.tanh(c)

    new_state = tf.contrib.rnn.LSTMStateTuple(c, m)
    return m, new_state
