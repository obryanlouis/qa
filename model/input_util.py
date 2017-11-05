"""Functions for creating model inputs.
"""

import preprocessing.constants as constants
import tensorflow as tf

def _create_word_similarity(primary_iterator, secondary_iterator, v_wiq_or_wic):
    """Creates a word similarity tensor, which is used as an input feature.
       Inputs:
         primary_iterator: Either the contexts or questions shaped [batch_size, N, W]
         secondary_iterator: Vice versa of the primary iterator shaped [batch_size, M, W]
         v_wiq_or_wic: 1-Dimensional vector shaped [W]
       Output:
         A word-similarity vector shaped [batch_size, N, 1]
    """
    sh_prim = tf.shape(primary_iterator)
    sh_sec = tf.shape(secondary_iterator)
    batch_size, N, W = sh_prim[0], sh_prim[1], sh_prim[2]
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
            shape=[sq_dataset.vocab.get_num_chars_including_padding(),
                   options.character_embedding_size],
            dtype=tf.float32)

def _run_char_birnn(scope, embedded_chars_tensor, options, sq_dataset):
    with tf.variable_scope(scope):
        with tf.variable_scope("forward"):
            rnn_cell_fw = tf.nn.rnn_cell.GRUCell(options.rnn_size)
        with tf.variable_scope("backward"):
            rnn_cell_bw = tf.nn.rnn_cell.GRUCell(options.rnn_size)
        sh = tf.shape(embedded_chars_tensor)
        batch_size, N = sh[0], sh[1]
        _, final_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, 
                tf.reshape(embedded_chars_tensor, [batch_size * N, sq_dataset.max_word_len, options.character_embedding_size]),
                dtype=tf.float32)
        final_state_fw, final_state_bw = final_states # sizes = [batch_size * N, rnn_size]
        desired_shape = [batch_size, N, options.rnn_size]
        state_fw = tf.reshape(final_state_fw, desired_shape)
        state_bw = tf.reshape(final_state_bw, desired_shape)
        return tf.reshape(
                 tf.concat([state_fw, state_bw], axis=2)
             , [batch_size, N, 2 * options.rnn_size])

def _add_char_embedding_inputs(scope, char_embedding, char_data, options,
        inputs_list, sq_dataset):
    chars_embedded = tf.nn.embedding_lookup(char_embedding, tf.cast(char_data, dtype=tf.int32))
    chars_input = _run_char_birnn(scope, chars_embedded, options, sq_dataset)
    inputs_list.append(chars_input)

def _cast_int32(tensor):
    return tf.cast(tensor, dtype=tf.int32)

def create_model_inputs(words_placeholder, ctx, qst, ctx_chars, qst_chars,
        options, wiq, wic, sq_dataset, ctx_pos, qst_pos, ctx_ner, qst_ner):
    with tf.device("/cpu:0"):
        with tf.variable_scope("model_inputs"):
            v_wiq = tf.get_variable("v_wiq", shape=[sq_dataset.word_vec_size])
            v_wic = tf.get_variable("v_wic", shape=[sq_dataset.word_vec_size])
            ctx_embedded = tf.nn.embedding_lookup(words_placeholder, ctx)
            qst_embedded = tf.nn.embedding_lookup(words_placeholder, qst)
            ctx_inputs_list = [ctx_embedded]
            qst_inputs_list = [qst_embedded]
            wiq_sh = tf.shape(wiq)
            wiq_feature_shape = [wiq_sh[0], wiq_sh[1]] + [1]
            wic_sh = tf.shape(wic)
            wic_feature_shape = [wic_sh[0], wic_sh[1]] + [1]
            if options.use_word_in_question_feature:
                ctx_inputs_list.append(tf.reshape(tf.cast(wiq, dtype=tf.float32), shape=wiq_feature_shape))
                qst_inputs_list.append(tf.reshape(tf.cast(wic, dtype=tf.float32), shape=wic_feature_shape))
            if options.use_word_similarity_feature:
                ctx_inputs_list.append(_create_word_similarity(ctx_embedded, qst_embedded, v_wiq))
                qst_inputs_list.append(_create_word_similarity(qst_embedded, ctx_embedded, v_wic))
            if options.use_character_data:
                char_embedding = _create_char_embedding(sq_dataset, options)
                _add_char_embedding_inputs("ctx_embedding", char_embedding,
                        ctx_chars, options, ctx_inputs_list, sq_dataset)
                _add_char_embedding_inputs("qst_embedding", char_embedding,
                        qst_chars, options, qst_inputs_list, sq_dataset)
            if options.use_pos_tagging_feature:
                pos_embedding = tf.get_variable("pos_embedding", shape=[2**8, options.pos_embedding_size])
                ctx_inputs_list.append(tf.nn.embedding_lookup(pos_embedding, _cast_int32(ctx_pos)))
                qst_inputs_list.append(tf.nn.embedding_lookup(pos_embedding, _cast_int32(qst_pos)))
            if options.use_ner_feature:
                ner_embedding = tf.get_variable("ner_embedding", shape=[2**8, options.ner_embedding_size])
                ctx_inputs_list.append(tf.nn.embedding_lookup(ner_embedding, _cast_int32(ctx_ner)))
                qst_inputs_list.append(tf.nn.embedding_lookup(ner_embedding, _cast_int32(qst_ner)))
            if len(ctx_inputs_list) == 1:
                return ctx_inputs_list[0], qst_inputs_list[0]
            else:
                return tf.concat(ctx_inputs_list, axis=-1), tf.concat(qst_inputs_list, axis=-1)
