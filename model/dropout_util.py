"""Utility methods for dropout.
"""

import tensorflow as tf


def sequence_dropout(seq, keep_prob):
    """Applies dropout to a sequence with the same mask at each timestep.

       Inputs:
            seq: A tensor of size [*, num_timesteps, *]

       Output:
            A tensor of the same size as the input.
    """
    assert len(seq.get_shape()) == 3
    sh = tf.shape(seq)
    noise_shape = [sh[0], 1, sh[2]]
    return tf.nn.dropout(seq, keep_prob, noise_shape=noise_shape)
