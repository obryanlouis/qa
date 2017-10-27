"""Implements a memory-based answer pointer from the Mnemonic Reader model
https://arxiv.org/pdf/1705.02798.pdf
"""

import tensorflow as tf

from model.rnn_util import *
from model.semantic_fusion import *
from model.tf_util import *

def _calculate_span_probs(ctx, memory, M, ctx_dim, batch_size, scope):
    with tf.variable_scope(scope):
        ctx_times_memory = ctx * memory # size = [batch_size, M, d]
        memory_repeated = tf.tile(memory, [1, M, 1]) # size = [batch_size, M, d]
        fc_inputs = tf.concat([ctx, ctx_times_memory, memory_repeated], axis=2) # size = [batch_size, M, 3 * d]
        w = tf.get_variable("fc_matrix", dtype=tf.float32, shape=[3 * ctx_dim, ctx_dim]) # size = [3 * d, d]
        b = tf.get_variable("bias", dtype=tf.float32, shape=[ctx_dim]) # size = [d]
        ws = tf.get_variable("w", dtype=tf.float32, shape=[ctx_dim]) # size = [d]
        s = multiply_tensors(fc_inputs, w) + b # size = [batch_size, M, d]
        logits = multiply_tensors(s, ws) # size = [batch_size, M]
        probs = tf.nn.softmax(logits, dim=1) # size = [batch_size, M]
        return probs, logits

def _update_memory(memory, ctx, probs, ctx_dim, batch_size, M, scope):
    evidence = tf.reshape(tf.matmul(
        tf.transpose(ctx, perm=[0, 2, 1]) # size = [batch_size, d, M]
        , tf.reshape(probs, [batch_size, M, 1]))
        , [batch_size, ctx_dim]) # size = [batch_size, d]
    memory_reshaped = tf.reshape(memory, [batch_size, ctx_dim])
    updated_memory = semantic_fusion(memory_reshaped, ctx_dim, [evidence],
        scope) # size = [batch_size, d]
    return tf.reshape(updated_memory, [batch_size, 1, ctx_dim]) # size = [batch_size, 1, d]

def memory_answer_pointer(options, ctx, qst, ctx_dim, spans, sq_dataset, keep_prob):
    """Runs a memory-based answer pointer to get start/end span predictions

       Input:
         ctx: The question-aware passage of shape [batch_size, M, d]
         qst: The question representation of shape [batch_size, N, d]
         ctx_dim: d, from above
         spans: The target spans of shape [batch_size, 2]. spans[:,0] are the
            start spans while spans[:,1] are the end spans.
         sq_dataset: A SquadDataBase object
         keep_prob: Probability used for dropout.

       Output:
         (loss, start_span_probs, end_span_probs)
         loss - a single scalar
         start_span_probs - the probabilities of the start spans of shape
            [batch_size, M]
         end_span_probs - the probabilities of the end spans of shape
            [batch_size, M]
    """
    with tf.variable_scope("memory_answer_pointer"):
        sh_ctx = tf.shape(ctx)
        batch_size, M = sh_ctx[0], sh_ctx[1]
        # Initialize the memory by running a bi-rnn over the question.
        memory = tf.reshape(get_question_attention(options, qst,
                    reduce_size=False) # size = [batch_size, d]
                , [batch_size, 1, ctx_dim]) # size = [batch_size, 1, d]
        start_span_probs = None # size = [batch_size, M]
        end_span_probs = None # size = [batch_size, M]
        start_span_logits = None # size = [batch_size, M]
        end_span_logits = None # size = [batch_size, M]
        assert options.num_memory_answer_pointer_hops > 0
        for z in range(options.num_memory_answer_pointer_hops):
            start_span_probs, start_span_logits = _calculate_span_probs(ctx, memory, M, ctx_dim, batch_size, "start_spans_" + str(z))
            memory = _update_memory(memory, ctx, start_span_probs, ctx_dim, batch_size, M, "update_start_memory_" + str(z))
            end_span_probs, end_span_logits = _calculate_span_probs(ctx, memory, M, ctx_dim, batch_size, "end_spans_" + str(z))
            if z < options.num_memory_answer_pointer_hops - 1:
                memory = _update_memory(memory, ctx, end_span_probs, ctx_dim, batch_size, M, "update_end_memory_" + str(z))
                ctx = run_bidirectional_lstm("bidirectional_ctx_" + str(z), ctx, keep_prob, options)
        casted_batch_size = tf.cast(batch_size, tf.float32)
        # Since each passage has been truncated to at most max_ctx_len words,
        # the labels need to be constrained to that range.
        start_labels = tf.minimum(spans[:,0], sq_dataset.get_max_ctx_len() - 1)
        end_labels = tf.minimum(spans[:,1], sq_dataset.get_max_ctx_len() - 1)
        loss = (tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=start_labels, logits=start_span_logits))
                + tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=end_labels, logits=end_span_logits))) \
               / casted_batch_size
        return loss, start_span_probs, end_span_probs
