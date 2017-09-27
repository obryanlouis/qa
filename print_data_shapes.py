"""Verification utility to run after creating the training data.
"""
import numpy as np
from preprocessing.vocab_util import get_vocab

vocab = get_vocab("data")

def print_files(context_file_name, question_file_name, span_file_name):
    ctx = np.load(context_file_name)
    qst = np.load(question_file_name)
    spn = np.load(span_file_name)
    num_sample = 3
    print("Context file", context_file_name, "shape", ctx.shape)
    ctx_sentences = vocab.get_sentences(ctx[:num_sample])
    print("Context sentences", ctx_sentences)
    print("Question file", question_file_name, "shape", qst.shape)
    print("Question sentences", vocab.get_sentences(qst[:num_sample]))
    print("Span file", span_file_name, "shape", spn.shape)
    split_ctx_sentences = [s.split(" ") for s in ctx_sentences]
    span_sentences = []
    for z in range(num_sample):
        span_start = spn[z, 0]
        span_end = spn[z, 1]
        print("span_start", span_start, "span_end", span_end)
        s = split_ctx_sentences[z][span_start:span_end + 1]
        span_sentences.append(s)
    print("Span sentences", span_sentences)

print_files("data/train.context.npy", "data/train.question.npy", "data/train.spans.npy")
print_files("data/dev.context.npy", "data/dev.question.npy", "data/dev.spans.npy")
