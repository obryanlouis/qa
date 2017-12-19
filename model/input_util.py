"""Functions for creating model inputs.
"""

import preprocessing.constants as constants
import tensorflow as tf

from model.cove_lstm import *
from model.cudnn_lstm_wrapper import *
from model.fusion_net_util import *
from model.tf_util import *

def _create_word_fusion(options, sq_dataset, ctx_glove, qst_glove):
    """See the FusionNet paper https://arxiv.org/pdf/1711.07341.pdf.

       Inputs:
            ctx_glove: Context GloVE vectors of size [batch_size, M, d]
            qst_glove: Question GloVE vectors of size [batch_size, N, d]

       Output:
            A tensor of size [batch_size, M, d]
    """
    with tf.variable_scope("word_fusion"):
        vec_dim = sq_dataset.get_word_vec_size()
        W = tf.get_variable("W", shape=[vec_dim, vec_dim],
            dtype=tf.float32) # size = [d, d]
        ctx_times_W = tf.nn.relu(multiply_tensors(ctx_glove, W)) # size = [batch_size, M, d]
        qst_times_W = tf.nn.relu(multiply_tensors(qst_glove, W)) # size = [batch_size, N, d]
        alpha = tf.matmul(ctx_times_W,
            tf.transpose(qst_times_W, perm=[0, 2, 1])) # size = [batch_size, M, N]
        alpha_softmax = tf.nn.softmax(alpha, dim=2) # size = [batch_size, M, N]
        return tf.matmul(alpha_softmax, qst_glove) # size = [batch_size, M, d]

def _create_word_similarity(primary_iterator, secondary_iterator, v_wiq_or_wic,
    batch_size):
    """Creates a word similarity tensor, which is used as an input feature.
       Inputs:
         primary_iterator: Either the contexts or questions shaped [batch_size, N, W]
         secondary_iterator: Vice versa of the primary iterator shaped [batch_size, M, W]
         v_wiq_or_wic: 1-Dimensional vector shaped [W]
       Output:
         A word-similarity vector shaped [batch_size, N, 1]
    """
    sh_prim = primary_iterator.get_shape().as_list()
    sh_sec = secondary_iterator.get_shape().as_list()
    N, W = sh_prim[1], sh_prim[2]
    M = sh_sec[1]
    prim = tf.reshape(primary_iterator, shape=[batch_size * W, N, 1])
    sec = tf.reshape(secondary_iterator, shape=[batch_size * W, 1, M])
    mult = tf.reshape(tf.matmul(prim, sec), shape=[batch_size * N * M, W]) # size = [batch_size * W, N, M]
    similarity =  tf.reshape(
       tf.matmul(mult, tf.reshape(v_wiq_or_wic, shape=[W, 1]))
       , shape=[batch_size, N, M]) # size = [batch_size, N, M]
    sm = tf.nn.softmax(similarity, dim=1) # size = [batch_size, N, M]
    return tf.reshape(
            tf.reduce_sum(sm, axis=2) # size = [batch_size, N]
            , [batch_size, N, 1])

def _create_char_embedding(sq_dataset, options):
    return tf.get_variable("char_embeddings",
            shape=[2**8 + 2, options.character_embedding_size],
            dtype=tf.float32)

def _run_cudnn_char_birnn(sess, scope, embedded_chars_tensor, options,
    sq_dataset, use_dropout):
    """
        Inputs:
            embedded_chars_tensor: Shaped [batch_size, max_(ctx|qst)_length, max_word_length, char_embedding_size]
        Output:
            A tensor shaped [batch_size, max_(ctx|qst)_length, 2 * rnn_size]
    """
    with tf.variable_scope(scope):
        lstm = create_cudnn_lstm(options.character_embedding_size,
            sess, options, "lstm", tf.constant(1.0), num_layers=1,
            bidirectional=True)
        sh = tf.shape(embedded_chars_tensor)
        batch_size, N = sh[0], sh[1]
        rnn_batch_size = batch_size * N
        inputs = tf.reshape(embedded_chars_tensor, [rnn_batch_size,
            sq_dataset.max_word_len, options.character_embedding_size])
        rnn_outputs, _ = run_cudnn_lstm_and_return_hidden_outputs(
                inputs, tf.constant(1.0), options, lstm, rnn_batch_size,
                use_dropout) # size = [batch_size * N,  2, rnn_size]
        return tf.reshape(rnn_outputs, [batch_size, N, 2 * options.rnn_size])

def _add_char_embedding_inputs(sess, scope, char_embedding, char_data, options,
        inputs_list, sq_dataset, use_dropout):
    chars_embedded = tf.nn.embedding_lookup(char_embedding, tf.cast(char_data, dtype=tf.int32)) # size = [batch_size, max_(ctx|qst)_length, max_word_length, char_embedding_size]
    chars_input = _run_cudnn_char_birnn(sess, scope, chars_embedded, options,
        sq_dataset, use_dropout)
    inputs_list.append(chars_input)

def _cast_int32(tensor):
    return tf.cast(tensor, dtype=tf.int32)

def _run_cove(inputs, cove_cells):
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cove_cells.forward_cell_l0,
        cove_cells.backward_cell_l0, inputs, dtype=tf.float32)
    fw_outputs, bw_outputs = outputs
    intermediate = tf.concat([fw_outputs, bw_outputs], axis=-1)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cove_cells.forward_cell_l1,
        cove_cells.backward_cell_l1, intermediate, dtype=tf.float32)
    fw_outputs, bw_outputs = outputs
    return tf.concat([fw_outputs, bw_outputs], axis=-1)

def _get_cove_vectors(options, ctx_glove, qst_glove, cove_cells):
    ctx_outputs = _run_cove(ctx_glove, cove_cells)
    qst_outputs = _run_cove(qst_glove, cove_cells)
    return ctx_outputs, qst_outputs

class ModelInputs:
    def __init__(self, ctx_glove, qst_glove, ctx_cove, qst_cove, ctx_concat,
        qst_concat):
        self.ctx_glove = ctx_glove # The GloVE vectors
        self.qst_glove = qst_glove
        self.ctx_cove = ctx_cove # The Cove vectors
        self.qst_cove = qst_cove
        self.ctx_concat = ctx_concat # The full set of features
        self.qst_concat = qst_concat

def create_model_inputs(sess, words_placeholder, ctx, qst,
        options, wiq, wic, sq_dataset, ctx_pos, qst_pos, ctx_ner, qst_ner,
        word_chars, cove_cells, use_dropout, batch_size):
    with tf.variable_scope("model_inputs"):
        ctx_embedded = tf.nn.embedding_lookup(words_placeholder, ctx)
        qst_embedded = tf.nn.embedding_lookup(words_placeholder, qst)
        ctx_inputs_list = [ctx_embedded]
        qst_inputs_list = [qst_embedded]
        ctx_cove, qst_cove = None, None
        if options.use_cove_vectors:
            ctx_cove, qst_cove = _get_cove_vectors(options, ctx_embedded,
                qst_embedded, cove_cells)
        if options.use_word_fusion_feature:
            ctx_inputs_list.append(_create_word_fusion(options, sq_dataset,
                ctx_embedded, qst_embedded))
        if options.use_word_in_question_feature:
            wiq_sh = tf.shape(wiq)
            wiq_feature_shape = [wiq_sh[0], wiq_sh[1]] + [1]
            wic_sh = tf.shape(wic)
            wic_feature_shape = [wic_sh[0], wic_sh[1]] + [1]
            ctx_inputs_list.append(tf.reshape(tf.cast(wiq, dtype=tf.float32), shape=wiq_feature_shape))
            qst_inputs_list.append(tf.reshape(tf.cast(wic, dtype=tf.float32), shape=wic_feature_shape))
        if options.use_word_similarity_feature:
            v_wiq = tf.get_variable("v_wiq", shape=[sq_dataset.word_vec_size])
            v_wic = tf.get_variable("v_wic", shape=[sq_dataset.word_vec_size])
            ctx_inputs_list.append(_create_word_similarity(ctx_embedded, qst_embedded, v_wiq, batch_size))
            qst_inputs_list.append(_create_word_similarity(qst_embedded, ctx_embedded, v_wic, batch_size))
        if options.use_character_data:
            char_embedding = _create_char_embedding(sq_dataset, options)
            ctx_chars = tf.nn.embedding_lookup(word_chars, ctx) # size = [batch_size, max_ctx_length, max_word_length]
            qst_chars = tf.nn.embedding_lookup(word_chars, qst) # size = [batch_size, max_qst_length, max_word_length]
            _add_char_embedding_inputs(sess, "ctx_embedding", char_embedding,
                    ctx_chars, options, ctx_inputs_list, sq_dataset,
                    use_dropout)
            _add_char_embedding_inputs(sess, "qst_embedding", char_embedding,
                    qst_chars, options, qst_inputs_list, sq_dataset,
                    use_dropout)
        if options.use_pos_tagging_feature:
            pos_embedding = tf.get_variable("pos_embedding", shape=[2**8, options.pos_embedding_size])
            ctx_inputs_list.append(tf.nn.embedding_lookup(pos_embedding, _cast_int32(ctx_pos)))
            qst_inputs_list.append(tf.nn.embedding_lookup(pos_embedding, _cast_int32(qst_pos)))
        if options.use_ner_feature:
            ner_embedding = tf.get_variable("ner_embedding", shape=[2**8, options.ner_embedding_size])
            ctx_inputs_list.append(tf.nn.embedding_lookup(ner_embedding, _cast_int32(ctx_ner)))
            qst_inputs_list.append(tf.nn.embedding_lookup(ner_embedding, _cast_int32(qst_ner)))
        if len(ctx_inputs_list) == 1:
            return ModelInputs(ctx_embedded, qst_embedded, ctx_cove, qst_cove,
                ctx_embedded, qst_embedded)
        else:
            return ModelInputs(ctx_embedded, qst_embedded, ctx_cove, qst_cove,
                tf.concat(ctx_inputs_list, axis=-1),
                tf.concat(qst_inputs_list, axis=-1))
