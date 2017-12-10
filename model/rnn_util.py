"""Functions for building model components that deal with RNN structures.
"""

import tensorflow as tf

from model.cudnn_lstm_wrapper import *
from model.tf_util import *

def get_question_attention(options, question_rnn_outputs, reduce_size):
    '''Gets an attention pooling vector of the question to be used as the
       initial input to the answer recurrent network.

       Inputs:
            question_rnn_outputs: Tensor of the question rnn outputs of
                size [batch_size, Q, 2 * rnn_size].
       Output:
            A tensor of size [batch_size, rnn_size].
    '''
    with tf.variable_scope("decode_question_attention"):
        sh = tf.shape(question_rnn_outputs)
        batch_size = sh[0]
        Q = sh[1]

        W = 2 * options.rnn_size
        W_question = tf.get_variable("W", dtype=tf.float32, shape=[W, options.rnn_size])
        W_param = tf.get_variable("W_param", dtype=tf.float32, shape=[options.rnn_size, options.rnn_size])
        V_param = tf.get_variable("V_param", dtype=tf.float32, shape=[options.rnn_size, 1])
        v = tf.get_variable("v", dtype=tf.float32, shape=[options.rnn_size, 1])

        s = multiply_tensors(
                    multiply_tensors(question_rnn_outputs,
                              W_question) # size = [batch_size, Q, rnn_size]
                    + tf.squeeze(tf.matmul(W_param, V_param)) # size = [rnn_size]
                    , v) # size = [batch_size, Q, 1]
        a = tf.nn.softmax(s, dim=1) # size = [batch_size, Q, 1]
        reduced_sum = tf.reduce_sum(
                  a * question_rnn_outputs # size = [batch_size, Q, W]
                  , axis=1) # size = [batch_size, W]
        if not reduce_size:
            return reduced_sum
        W_reduce_size = tf.get_variable("W_reduce_size",
            dtype=tf.float32, shape=[W, options.rnn_size])
        return tf.matmul(reduced_sum, W_reduce_size)

def create_multi_rnn_cell(options, scope, keep_prob, num_rnn_layers=None, layer_size=None):
    num_rnn_layers = options.num_rnn_layers if num_rnn_layers is None else num_rnn_layers
    layer_size = options.rnn_size if layer_size is None else layer_size
    with tf.variable_scope(scope):
        cells = []
        for z in range(num_rnn_layers):
            with tf.variable_scope("cell_" + str(z)):
                cell = tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.GRUCell(layer_size),
                        input_keep_prob=keep_prob,
                        output_keep_prob=keep_prob)
                cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)

def run_bidirectional_cudnn_lstm(scope, inputs, keep_prob, options, batch_size,
        sess, use_dropout, num_layers=None):
    '''
        Input:
            inputs: A tensor of size [batch_size, seq_len, input_size]
        Output:
            A tensor of size [batch_size, seq_len, 2 * rnn_size]
    '''
    lstm = create_cudnn_lstm(inputs.get_shape()[2],
            sess, options, scope, keep_prob, bidirectional=True,
            num_layers=num_layers)
    return run_cudnn_lstm_and_return_outputs(inputs, keep_prob, options,
        lstm, batch_size, use_dropout)

def _get_or_create_attention_variables(options, attention_input, input_dim,
        attention_dim, scope, attention_length,
        num_rnn_layers, keep_prob):
    with tf.variable_scope(scope):
        W_gate = tf.get_variable("W_gate", shape=[2 * input_dim, 2 * input_dim])
        Wq = tf.get_variable("Wq", shape=[attention_dim, options.rnn_size])
        Wp = tf.get_variable("Wp", shape=[input_dim, options.rnn_size])
        Wr = tf.get_variable("Wr", shape=[options.rnn_size, options.rnn_size])
        w = tf.get_variable("w", shape=[options.rnn_size, 1])
        bp = tf.get_variable("bp", shape=[1, options.rnn_size])
        b = tf.get_variable("b", shape=[1])
        match_lstm_cell = create_multi_rnn_cell(options, "match_lstm_birnn_cell", keep_prob, num_rnn_layers=num_rnn_layers)
        WqHq = multiply_tensors(attention_input, Wq) # shape = [batch_size, max_qst_length, rnn_size]
        return {
            "W_gate": W_gate,
            "Wq": Wq,
            "Wp": Wp,
            "Wr": Wr,
            "w": w,
            "bp": bp,
            "b": b,
            "match_lstm_cell": match_lstm_cell,
            "WqHq": WqHq,
        }

def run_attention(options, inputs, input_dim, attention_input, attention_dim,
        scope, batch_size, attention_length, keep_prob,
        input_length, num_rnn_layers=1):
    '''Runs an attention RNN over the inputs given the attention.
       Inputs:
            inputs: sized [batch_size, input_length, input_dim]
            attention_input: sized [batch_size, *, attention_dim]
       Outputs:
            output tensor sized [batch_size, max_ctx_length, 2 * rnn_size]
    '''
    with tf.variable_scope(scope):
        # Need to share variables, as is done in the paper.
        lstm_vars = _get_or_create_attention_variables(options,
                attention_input, input_dim, attention_dim,
                "birnn_forward", attention_length,
                num_rnn_layers, keep_prob)

        zeros = tf.zeros([batch_size, options.rnn_size])
        forward_state  = (zeros,) * num_rnn_layers
        backward_state = (zeros,) * num_rnn_layers
        forward_outputs  = []
        backward_outputs = []
        rnn_scope = "rnn_scope"
        for z in range(input_length):
            forward_state = _build_match_lstm(
                    options,
                    batch_size,
                    lstm_vars["match_lstm_cell"],
                    z,
                    lstm_vars["Wp"],
                    lstm_vars["Wr"],
                    lstm_vars["WqHq"], lstm_vars["w"],
                    lstm_vars["bp"], lstm_vars["b"],
                    lstm_vars["W_gate"],
                    attention_input, inputs,
                    forward_state, forward_outputs, input_dim,
                    reuse_vars=z > 0, scope=rnn_scope,
                    use_last_hidden=True)

            # Make sure to always reuse variables where possible
            backward_state = _build_match_lstm(
                    options,
                    batch_size,
                    lstm_vars["match_lstm_cell"],
                    input_length - z - 1,
                    lstm_vars["Wp"],
                    lstm_vars["Wr"],
                    lstm_vars["WqHq"], lstm_vars["w"],
                    lstm_vars["bp"], lstm_vars["b"],
                    lstm_vars["W_gate"],
                    attention_input, inputs,
                    backward_state, backward_outputs, input_dim,
                    reuse_vars=True, scope=rnn_scope,
                    use_last_hidden=True)

        Hr_forward  = tf.stack(forward_outputs , axis=1) # size = [batch_size, max_ctx_length, rnn_size]
        Hr_backward = tf.stack(backward_outputs, axis=1) # size = [batch_size, max_ctx_length, rnn_size]
        return tf.concat([Hr_forward, Hr_backward], axis=2) # size = [batch_size, max_ctx_length, 2 * rnn_size]

def _build_match_lstm(options, batch_size, lstm_cell,
        input_index, Wp, Wr,
        WqHq, w, bp, b, W_gate, attention_inputs, inputs, state, outputs,
        input_dim, reuse_vars=False, scope="", use_last_hidden=True):
    '''Builds a Match LSTM next state and output given the current state
       and output (and other relevant model variables). Adds to the current
       outputs variable.

       Inputs:
            use_last_hidden: Whether to add the last hidden state into the
                input of tanh. Should be False for self matching attention,
                although it doesn't seem to make much difference.
            state: GRU cell state

       Output:
            state: The new state configuration.
    '''
    Hpi = inputs[:, input_index, :] # size = [batch_size, rnn_size]
    WpHpi = tf.matmul(Hpi, Wp) # [batch_size, rnn_size] . [rnn_size, rnn_size] = [batch_size, rnn_size]
    Eq = WpHpi # size = [batch_size, rnn_size]
    if use_last_hidden:
        for s in state:
            Eq += tf.matmul(s, Wr) # hidden_state * matrix
    Eq += bp
    G = tf.tanh(WqHq + tf.reshape(Eq, [batch_size, 1, options.rnn_size])) # size = [batch_size, attention_length, rnn_size]
    wG = tf.squeeze(multiply_tensors(G, w)) # size = [batch_size, attention_length]
    alpha = tf.nn.softmax(wG + b) # size = [batch_size, attention_length]
    Yalpha = tf.reshape(
                tf.matmul(tf.reshape(alpha, [batch_size, 1, -1]), attention_inputs)
                , tf.shape(Hpi)) # size = [batch_size, attention_dim]
    z = tf.concat([Hpi, Yalpha], axis=1) # size = [batch_size, 2 * attention_dim]
    smd = tf.sigmoid(tf.matmul(z, W_gate)) # size = [batch_size, 2 * attention_dim]
    z = z * smd
    z = tf.reshape(z, [batch_size, 2 * input_dim])

    with tf.variable_scope(scope, reuse=reuse_vars):
        new_output, new_state = lstm_cell(z, state) # shape = [batch_size, rnn_size]
    outputs.append(new_output)
    return new_state
