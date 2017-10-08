from preprocessing.vocab_util import *
from datasets.squad_data import SquadData
from flags import get_options_from_flags

import tensorflow as tf

PRINT_LIMIT = 10

def _print_ds(vocab, ds):
    print("Context")
    for z in range(PRINT_LIMIT):
        l = []
        for zz in range(ds.ctx.shape[1]):
            i = ds.ctx[z, zz]
            word = vocab.get_word_for_id(i)
            l.append(word)
        print(" ".join(l))
    print("Questions")
    for z in range(PRINT_LIMIT):
        l = []
        for zz in range(ds.qst.shape[1]):
            i = ds.qst[z, zz]
            word = vocab.get_word_for_id(i)
            l.append(word)
        print(" ".join(l))
    print("Spans")
    print(ds.spn[:PRINT_LIMIT])
    print("")

def main(_):
    options = get_options_from_flags()
    squad_data = SquadData(options)
    print("TRAIN")
    _print_ds(squad_data.vocab, squad_data.train_ds)
    print("DEV")
    _print_ds(squad_data.vocab, squad_data.dev_ds)


if __name__ == "__main__":
    tf.app.run()
