from preprocessing.vocab_util import *
from datasets.squad_data import SquadData
from flags import get_options_from_flags

import tensorflow as tf

PRINT_LIMIT = 10
WORD_PRINT_LIMIT = 5

def _print_qst_or_ctx_np_arr(arr, char_arr, vocab, ds):
    for z in range(PRINT_LIMIT):
        l = []
        for zz in range(arr.shape[1]):
            i = arr[z, zz]
            word = vocab.get_word_for_id(i)
            chars_for_word = []
            for cz in range(WORD_PRINT_LIMIT):
                ch_idx = char_arr[z, zz, cz]
                chars_for_word.append(vocab.get_char_for_id(ch_idx))
            if ds.word_in_question[z, zz] == 1:
                word = "[WIQ:]" + word
            word_from_chars = ''.join(chars_for_word)
            word = word + "[CHARS:" + word_from_chars + "]"
            l.append(word)
        print(" ".join(l))

def _print_gnd_truths(ds, vocab):
    for z in range(PRINT_LIMIT):
        sentences = ds.get_sentences_for_all_gnd_truths(z)
        print(";".join(sentences))

def _print_ds(vocab, ds):
    print("Context")
    _print_qst_or_ctx_np_arr(ds.ctx, ds.ctx_chars, vocab, ds)
    print("Questions")
    _print_qst_or_ctx_np_arr(ds.qst, ds.qst_chars, vocab, ds)
    print("Spans")
    print(ds.spn[:PRINT_LIMIT])
    print("Ground truths")
    _print_gnd_truths(ds, vocab)
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
