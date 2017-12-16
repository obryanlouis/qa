from cove import MTLSTM
from datasets.squad_data import SquadData
from flags import get_options_from_flags
from model.cove_lstm import *
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

def main(_):
    options = get_options_from_flags()

    inputs = np.random.randint(low=0, high=30, size=(10, 10),
        dtype=np.int64)
    embeddings = embedding_util \
        .load_word_embeddings_including_unk_and_padding(options)
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
    assert torch_output.shape == tf_output.shape
    max_diff = np.max(np.abs(torch_output - tf_output))
    sum_diff = np.sum(np.abs(torch_output - tf_output))
    print("Max diff", max_diff)
    print("Sum diff", sum_diff)

if __name__ == "__main__":
    tf.app.run()

