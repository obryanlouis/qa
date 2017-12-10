import tensorflow as tf

from model.rnn_util import *
from model.tf_util import *

def _run_n_lstms(seq, seq_list, n, scope, keep_prob, options, sess,
    use_dropout, batch_size):
    with tf.variable_scope(scope):
        for z in range(n):
            new_seq = (run_bidirectional_cudnn_lstm("seq_" + str(z),
                tf.concat(seq_list, axis=2), keep_prob, options,
                batch_size, sess, use_dropout))
            seq_list.append(new_seq)

def run_qa(options, ctx, qst, keep_prob, use_dropout, batch_size, sess):
    """
        Inputs:
            ctx: Passage tensor of shape [batch_size, max_ctx_length, 2 * rnn_size]
            qst: Question tensor of shape [batch_size, max_qst_length, 2 * rnn_size]
        Outputs:
            (qa_ctx, qa_qst):
                qa_ctx: Updated passage tensor of size [batch_size,
                    max_ctx_length, 2 * rnn_size]
                qa_qst: Updated question tensor of size [batch_size,
                    max_qst_length, 2 * rnn_size]
    """
    ctx_reprs = [ctx]
    qst_reprs = [qst]

    _run_n_lstms(ctx, ctx_reprs, 2, "ctx_preprocess", keep_prob, options, sess,
        use_dropout, batch_size)
    _run_n_lstms(qst, qst_reprs, 2, "qst_preprocess", keep_prob, options, sess,
        use_dropout, batch_size)

    for z in range(options.num_qa_loops):
        iter_c, iter_q = _iterate_combinations("loop_" + str(z),
            ctx_reprs, qst_reprs,
            options, use_dropout, batch_size, keep_prob, sess)
        ctx_reprs.append(iter_c)
        qst_reprs.append(iter_q)
    qa_ctx = tf.concat(ctx_reprs, axis=2)
    qa_qst = tf.concat(qst_reprs, axis=2)
    assert qa_ctx.get_shape()[2] == qa_qst.get_shape()[2]
    qa_ctx = run_bidirectional_cudnn_lstm("qa_ctx_final",
        qa_ctx, keep_prob, options, batch_size, sess, use_dropout)
    qa_qst = run_bidirectional_cudnn_lstm("qa_qst_final",
        qa_qst, keep_prob, options, batch_size, sess, use_dropout)
    return qa_ctx, qa_qst

def _iterate_combinations(scope, ctx_reprs, qst_reprs, options, use_dropout,
    batch_size, keep_prob, sess):
    with tf.variable_scope(scope):
        assert len(qst_reprs) == len(qst_reprs)
        C = tf.concat(ctx_reprs, axis=2) # size = [batch_size, max_ctx_len, 2 * rnn_size * len(ctx_reprs)]
        Q = tf.concat(qst_reprs, axis=2) # size = [batch_size, max_qst_len, 2 * rnn_size * len(qst_reprs)]
        cc_reprs = _create_new_reprs(ctx_reprs, C, C, "ctx_ctx", options,
            keep_prob, batch_size, sess, use_dropout) # size(each) = [batch_size, max_ctx_len, 2 * rnn_size]
        qq_reprs = _create_new_reprs(qst_reprs, Q, Q, "qst_qst", options,
            keep_prob, batch_size, sess, use_dropout) # size(each) = [batch_size, max_qst_len, 2 * rnn_size]
        qc_reprs = _create_new_reprs(ctx_reprs, Q, C, "qst_ctx", options,
            keep_prob, batch_size, sess, use_dropout) # size(each) = [batch_size, max_qst_len, 2 * rnn_size]
        cq_reprs = _create_new_reprs(qst_reprs, C, Q, "ctx_qst", options,
            keep_prob, batch_size, sess, use_dropout) # size(each) = [batch_size, max_ctx_len, 2 * rnn_size]
        c_concat = tf.concat(cc_reprs + cq_reprs, axis=2) # size = [batch_size, max_ctx_len, *]
        q_concat = tf.concat(qq_reprs + qc_reprs, axis=2) # size = [batch_size, max_qst_len, 2 * rnn_size]
        new_c = run_bidirectional_cudnn_lstm("c_iter",
            c_concat, keep_prob, options, batch_size, sess, use_dropout)
        new_q = run_bidirectional_cudnn_lstm("q_iter",
            q_concat, keep_prob, options, batch_size, sess, use_dropout)
        return new_c, new_q

def _create_new_reprs(reprs, A, B, scope, options, keep_prob, batch_size,
        sess, use_dropout):
    """
        Inputs:
            reprs: A list of tensors shaped [batch_size, b, 2 * rnn_size]
            A: A tensor shaped [batch_size, a, d]
            B: A tensor shaped [batch_size, b, d]

        Outputs:
            A list of tensors of size [batch_size, a, d]
    """
    with tf.variable_scope(scope):
        fused_reprs = []
#        weights = create_weights(A, B, "shared_fusion_weights", options) # size = [batch_size, shape(A)[1], shape(B)[1]]
        for z in range(len(reprs)):
            fused_reprs.append(get_fusion(reprs[z], A, B, "fusion_" + str(z),
#                options, weights=weights))
                options))
        return fused_reprs

def get_fusion(b, A, B, scope, options, weights=None):
    with tf.variable_scope(scope):
        if weights is None:
            weights = create_weights(A, B, "fusion_weights", options) # size = [batch_size, shape(A)[1], shape(B)[1]]
        weights = tf.nn.softmax(weights, dim=2)
        return tf.matmul(weights, b) # size = [batch_size, shape(A)[1], 2 * rnn_size]

def create_weights(A, B, scope, options):
    with tf.variable_scope(scope):
        d = A.get_shape()[2]
        k = options.qa_diag_dim
        U = tf.get_variable("U", shape=[d, k])
        diag_values = tf.get_variable("diag_values", shape=[k], dtype=tf.float32)
        D = tf.diag(diag_values)
        AU = tf.nn.relu(multiply_tensors(A, U)) # size = [batch_size, shape(A)[1], k]
        BU = tf.nn.relu(multiply_tensors(B, U)) # size = [batch_size, shape(B)[1], k]
        AUD = multiply_tensors(AU, D) # size = [batch_size, shape(A)[1], k]
        return tf.matmul(AUD, tf.transpose(BU, perm=[0, 2, 1])) # size = [batch_size, shape(A)[1], shape(B)[1]]
