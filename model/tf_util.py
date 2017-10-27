"""Functions for building tensorflow models.
"""

import tensorflow as tf

def multiply_tensors(tensor1, tensor2):
    """Multiplies two tensors in a matrix-like multiplication based on the
       last dimension of the first tensor and first dimension of the second
       tensor.

       Inputs:
            tensor1: A tensor of shape [a, b, c, .., x]
            tensor2: A tensor of shape [x, d, e, f, ...]

       Outputs:
            A tensor of shape [a, b, c, ..., d, e, f, ...]
    """
    sh1 = tf.shape(tensor1)
    sh2 = tf.shape(tensor2)
    len_sh1 = len(tensor1.get_shape())
    len_sh2 = len(tensor2.get_shape())
    prod1 = tf.constant(1, dtype=tf.int32)
    sh1_list = []
    for z in range(len_sh1 - 1):
        sh1_z = sh1[z]
        prod1 *= sh1_z
        sh1_list.append(sh1_z)
    prod2 = tf.constant(1, dtype=tf.int32)
    sh2_list = []
    for z in range(len_sh2 - 1):
        sh2_z = sh2[len_sh2 - 1 - z]
        prod2 *= sh2_z
        sh2_list.append(sh2_z)
    reshape_1 = tf.reshape(tensor1, [prod1, sh1[len_sh1 - 1]])
    reshape_2 = tf.reshape(tensor2, [sh2[0], prod2])
    result = tf.reshape(tf.matmul(reshape_1, reshape_2), sh1_list + sh2_list)
    assert len(result.get_shape()) == len_sh1 + len_sh2 - 2
    return result
