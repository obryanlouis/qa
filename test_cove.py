from cove import MTLSTM
from datasets.squad_data import SquadData
from flags import get_options_from_flags
from model.cove_lstm import load_cove_lstm
from model.cudnn_cove_lstm import load_cudnn_cove_lstm
from preprocessing.s3_util import *
from preprocessing.vocab_util import get_vocab
from torch.autograd import Variable
from train.trainer import Trainer
from util.file_util import *
import numpy as np
import os
import preprocessing.constants as constants
import preprocessing.embedding_util as embedding_util
import re
import tensorflow as tf
import torch

def compute_torch_values(inputs, embeddings):
    model = MTLSTM(
        n_vocab=embeddings.shape[0],
        vectors=torch.from_numpy(embeddings.astype(np.float32)))
    model.cuda(0)
    model_inputs = Variable(torch.from_numpy(inputs.astype(np.int64)))
    lengths = torch.from_numpy(
        np.ones((inputs.shape[0],), dtype=np.int64) * inputs.shape[1])
    cove_outputs = model.forward(model_inputs.cuda(), lengths=lengths.cuda())
    torch_output = (cove_outputs.data.cpu().numpy())
    print("Torch output shape", torch_output.shape)
    return torch_output

def compute_tf_values(inputs, embeddings, options):
    with tf.Session() as sess:
        cove_cells = load_cove_lstm(options)
        embedding_placeholder = tf.placeholder(tf.float32,
            shape=embeddings.shape)
        tf_inputs = tf.nn.embedding_lookup(embedding_placeholder, inputs)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cove_cells.forward_cell_l0,
            cove_cells.backward_cell_l0, tf_inputs, dtype=tf.float32)
        fw_outputs, bw_outputs = outputs
        intermediate = tf.concat([fw_outputs, bw_outputs], axis=-1)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cove_cells.forward_cell_l1,
            cove_cells.backward_cell_l1, intermediate, dtype=tf.float32)
        fw_outputs, bw_outputs = outputs
        tf_output = sess.run(tf.concat([fw_outputs, bw_outputs], axis=-1),
                feed_dict={embedding_placeholder:embeddings})
        print("tf output shape", tf_output.shape)
        return tf_output

def compute_tf_cudnn_values(inputs, embeddings, options):
    with tf.Session() as sess:
        cudnn_cove_cell = load_cudnn_cove_lstm(options)
        embedding_placeholder = tf.placeholder(tf.float32,
            shape=embeddings.shape)
        tf_inputs = tf.nn.embedding_lookup(embedding_placeholder, inputs)
        tf_output = sess.run(cudnn_cove_cell(tf_inputs),
                feed_dict={embedding_placeholder:embeddings})
        print("cudnn tf output shape", tf_output.shape)
        return tf_output

def assert_tensors_equal(tensor1, tensor2, name1, name2):
    assert tensor1.shape == tensor2.shape
    max_diff = np.max(np.abs(tensor1 - tensor2))
    sum_diff = np.sum(np.abs(tensor1 - tensor2))
    tolerance = 1e-5
    print("Max diff of %s and %s" % (name1, name2), max_diff)
    print("Sum diff of %s and %s" % (name1, name2), sum_diff)
    if max_diff > tolerance:
        print("Difference is too large between %s and %s!" % (name1, name2))
    else:
        print("Implementations match!")


def main(_):
    tf.set_random_seed(0)
    np.random.seed(0)
    options = get_options_from_flags()

    inputs = np.random.randint(low=0, high=30, size=(1, 1),
        dtype=np.int64)
    embeddings = embedding_util \
        .load_word_embeddings_including_unk_and_padding(options)
    torch_output = compute_torch_values(inputs, embeddings)
    tf_cudnn_output = compute_tf_cudnn_values(inputs, embeddings, options)
    tf_output = compute_tf_values(inputs, embeddings, options)

    assert_tensors_equal(torch_output, tf_output, "Torch output", "TF output")
    assert_tensors_equal(tf_cudnn_output, tf_output, "cuDNN TF output", "TF output")

if __name__ == "__main__":
    tf.app.run()

