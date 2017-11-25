import numpy as np
import os
import preprocessing.constants as constants
import torch

from cove import MTLSTM
from datasets.squad_data import SquadData
from torch.autograd import Variable

_BATCH_SIZE = 100

def _create_cove_np_arr(input_np_arr, vec_size, model, options):
    output_arr = np.zeros(input_np_arr.shape + (vec_size,))
    batch_idx = 0
    num_samples = input_np_arr.shape[0]
    while batch_idx < num_samples:
        inputs = input_np_arr[batch_idx:(batch_idx + _BATCH_SIZE), :]
        model_inputs = Variable(torch.from_numpy(inputs.astype(np.int64)))
        lengths = torch.from_numpy(
            np.ones((inputs.shape[0],), dtype=np.int64) * input_np_arr.shape[1])
        cove_outputs = model.forward(
            model_inputs if options.num_gpus == 0 else model_inputs.cuda(),
            lengths=lengths if options.num_gpus == 0 else lengths.cuda())
        np_data = cove_outputs.data.cpu().numpy()
        output_arr[batch_idx:(batch_idx + _BATCH_SIZE), :, :] = np_data
        batch_idx += _BATCH_SIZE
        print("Processed %s / %s" % (min(batch_idx, num_samples), num_samples),
              end="\r")
    print("")
    return output_arr

def maybe_create_cove_vectors(options):
    sq_dataset = SquadData(options)
    files_and_vectors = [
        (constants.DEV_COVE_QST_FILE, sq_dataset.dev_ds.qst),
        (constants.DEV_COVE_CTX_FILE, sq_dataset.dev_ds.ctx),
        (constants.TRAIN_COVE_QST_FILE, sq_dataset.train_ds.qst),
        (constants.TRAIN_COVE_CTX_FILE, sq_dataset.train_ds.ctx),
    ]
    if all([os.path.isfile(os.path.join(options.data_dir, f_name)) \
        for f_name, _ in files_and_vectors]):
        print("Already created CoVe vectors.")
        return
    model = MTLSTM(
        n_vocab=sq_dataset.vocab.get_vocab_size_including_pad_and_unk(),
        vectors=torch.from_numpy(sq_dataset.embeddings.astype(np.float32)))
    vec_size = 2 * sq_dataset.embeddings.shape[1]
    if options.num_gpus > 0:
        model.cuda(0)
    for file_name, vector in files_and_vectors:
        print("Creating CoVe vectors for file:", file_name)
        cove_vectors = _create_cove_np_arr(vector, vec_size, model, options)
        print("Cove vector shape", cove_vectors.shape)
        full_file_name = os.path.join(options.data_dir, file_name)
        np.save(full_file_name, cove_vectors)
