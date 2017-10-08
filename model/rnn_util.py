"""Functions for building model components that deal with RNN structures.
"""

import tensorflow as tf

from model.tf_util import *

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

def _get_or_create_attention_variables(options, attention_input, input_dim,
        attention_dim, scope, attention_length,
        num_rnn_layers, keep_prob):
    with tf.variable_scope(scope):
        Wq = tf.get_variable("Wq", shape=[attention_dim, options.rnn_size])
        Wp = tf.get_variable("Wp", shape=[input_dim, options.rnn_size])
        Wr = tf.get_variable("Wr", shape=[options.rnn_size, options.rnn_size])
        w = tf.get_variable("w", shape=[options.rnn_size, 1])
        bp = tf.get_variable("bp", shape=[1, options.rnn_size])
        b = tf.get_variable("b", shape=[1])
        match_lstm_cell = create_multi_rnn_cell(options, "match_lstm_birnn_cell", keep_prob, num_rnn_layers=num_rnn_layers)
        WqHq = multiply_3d_and_2d_tensor(attention_input, Wq) # shape = [batch_size, max_qst_length, rnn_size]
        return {
            "Wq": Wq,
            "Wp": Wp,
            "Wr": Wr,
            "w": w,
            "bp": bp,
            "b": b,
            "match_lstm_cell": match_lstm_cell,
            "WqHq": WqHq,
        }

def run_attention(sq_dataset, options, inputs, input_dim, attention_input, attention_dim,
        scope, batch_size, attention_length, keep_prob,
        num_rnn_layers=1):
    '''Runs an attention RNN over the inputs given the attention.
       Inputs:
            inputs: sized [batch_size, max_ctx_length, input_dim]
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
        for z in range(sq_dataset.get_max_ctx_len()):
            forward_state = _build_match_lstm(
                    options,
                    batch_size,
                    lstm_vars["match_lstm_cell"],
                    z,
                    lstm_vars["Wp"],
                    lstm_vars["Wr"],
                    lstm_vars["WqHq"], lstm_vars["w"],
                    lstm_vars["bp"], lstm_vars["b"],
                    attention_input, inputs,
                    forward_state, forward_outputs, input_dim,
                    reuse_vars=z > 0, scope=rnn_scope,
                    use_last_hidden=True)

            # Make sure to always reuse variables where possible
            backward_state = _build_match_lstm(
                    options,
                    batch_size,
                    lstm_vars["match_lstm_cell"],
                    sq_dataset.get_max_ctx_len() - z - 1,
                    lstm_vars["Wp"],
                    lstm_vars["Wr"],
                    lstm_vars["WqHq"], lstm_vars["w"],
                    lstm_vars["bp"], lstm_vars["b"],
                    attention_input, inputs,
                    backward_state, backward_outputs, input_dim,
                    reuse_vars=True, scope=rnn_scope,
                    use_last_hidden=True)

        Hr_forward  = tf.stack(forward_outputs , axis=1) # size = [batch_size, max_ctx_length, rnn_size]
        Hr_backward = tf.stack(backward_outputs, axis=1) # size = [batch_size, max_ctx_length, rnn_size]
        return tf.concat([Hr_forward, Hr_backward], axis=2) # size = [batch_size, max_ctx_length, 2 * rnn_size]

def _build_match_lstm(options, batch_size, lstm_cell,
        input_index, Wp, Wr,
        WqHq, w, bp, b, attention_inputs, inputs, state, outputs,
        input_dim, reuse_vars=False, scope="", use_last_hidden=True):
    '''Builds a Match LSTM next state and output given the current state
       and output (and other relevant model variables). Adds to the current
       outputs variable.

       Inputs:
            use_last_hidden: Whether to add the last hidden state into the
                input of tanh. Should be False for self matching attention.
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
    G = tf.tanh(WqHq + tf.reshape(Eq, [batch_size, 1, options.rnn_size])) # size = [batch_size, attention_dim, rnn_size]
    wG = tf.squeeze(multiply_3d_and_2d_tensor(G, w)) # size = [batch_size, attention_dim]
    alpha = tf.nn.softmax(wG + b) # size = [batch_size, attention_dim]
    Yalpha = tf.reshape(
                tf.matmul(tf.reshape(alpha, [batch_size, 1, -1]), attention_inputs)
                , tf.shape(Hpi)) # size = [batch_size, rnn_size]
    z = tf.concat([Hpi, Yalpha], axis=1)
    z = tf.reshape(z, [batch_size, 2 * input_dim])

    with tf.variable_scope(scope, reuse=reuse_vars):
        new_output, new_state = lstm_cell(z, state) # shape = [batch_size, rnn_size]
    outputs.append(new_output)
    return new_state
