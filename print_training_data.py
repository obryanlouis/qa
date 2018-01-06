from preprocessing.vocab_util import *
from datasets.squad_data import SquadData
from flags import get_options_from_flags

import tensorflow as tf

PRINT_LIMIT = 10
WORD_PRINT_LIMIT = 5

def _print_qst_or_ctx_np_arr(arr, vocab, ds, is_ctx, wiq_or_wic,
    question_ids=None, question_ids_to_squad_ids=None):
    for z in range(PRINT_LIMIT):
        l = []
        if question_ids_to_squad_ids is not None:
            question_id = question_ids[z]
            squad_id = question_ids_to_squad_ids[question_id]
            l.append("[SQUAD ID: " + squad_id + "]")
        for zz in range(arr.shape[1]):
            i = arr[z, zz]
            if i == vocab.PAD_ID:
                continue
            word = vocab.get_word_for_id(i)
            if wiq_or_wic[z, zz] == 1:
                word = ("[WIQ:]" if is_ctx else "[WIC:]") + word
            l.append(word)
        print(" ".join(l))
        print("")

def _print_gnd_truths(ds, vocab):
    for z in range(PRINT_LIMIT):
        question_id = ds.qid[z]
        sentences = ds.get_sentences_for_all_gnd_truths(question_id)
        print(";".join(sentences))

def _print_ds(vocab, ds):
    print("Context")
    _print_qst_or_ctx_np_arr(ds.ctx, vocab, ds, is_ctx=True, wiq_or_wic=ds.wiq,
        question_ids=ds.qid)
    print("Questions")
    _print_qst_or_ctx_np_arr(ds.qst, vocab, ds, is_ctx=False, wiq_or_wic=ds.wic,
        question_ids=ds.qid,
        question_ids_to_squad_ids=ds.question_ids_to_squad_ids)
    print("Spans")
    print(ds.spn[:PRINT_LIMIT])
    print("Ground truths")
    _print_gnd_truths(ds, vocab)
    print("")

def main(_):
    options = get_options_from_flags()
    options.num_gpus = 0
    with tf.Session() as sess:
        squad_data = SquadData(options)
        squad_data.setup_with_tf_session(sess)
        print("TRAIN")
        _print_ds(squad_data.vocab, squad_data.train_ds)
        print("DEV")
        _print_ds(squad_data.vocab, squad_data.dev_ds)


if __name__ == "__main__":
    tf.app.run()
