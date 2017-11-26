import numpy as np
import os
import preprocessing.constants as constants
import preprocessing.embedding_util as embedding_util
import torch

from cove import MTLSTM
from preprocessing.vocab_util import get_vocab

def save_cove_weights(options):
    """Saves the weights of the CoVe LSTM for manual TensorFlow initialization.
    """
    folder_name = os.path.join(options.data_dir, constants.COVE_WEIGHTS_FOLDER)
    if all([os.path.exists(os.path.join(folder_name, name + ".npy")) \
        for name in constants.COVE_WEIGHT_NAMES]):
        print("Cove weights already saved")
        return
    os.makedirs(folder_name, exist_ok=True)
    vocab = get_vocab(options.data_dir)
    embeddings = embedding_util.load_word_embeddings_including_unk_and_padding(
        options)
    vec_size = 2 * embeddings.shape[1]
    print("Loading CoVe model")
    model = MTLSTM(
        n_vocab=embeddings.shape[0],
        vectors=torch.from_numpy(embeddings.astype(np.float32)))
    print("Saving CoVe weights")
    for weight_name in constants.COVE_WEIGHT_NAMES:
        tensor = getattr(model.rnn, weight_name)
        np_value = tensor.cpu().data.numpy()
        full_file_name = os.path.join(folder_name, weight_name + ".npy")
        np.save(full_file_name, np_value)
