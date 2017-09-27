"""Functions for building tensorflow models.
"""

import tensorflow as tf

def multiply_3d_and_2d_tensor(tensor_3d, tensor_2d):
    '''Multiplies tensors of shapes [A, B, C] and [C, D].

       Inputs:
            tensor_3d: A 3D tensor of size [A, B, C].
            tensor_2d: A 2D tensor of size [C, D].
       Outputs:
            A tensor of size [A, B, D] that is the result of
            multiplying the 2D tensor with each batch element of the 3D
            tensor.
    '''
    assert len(tensor_3d.get_shape()) == 3
    assert len(tensor_2d.get_shape()) == 2
    sh = tf.shape(tensor_3d)
    batch_size, A, B = sh[0], sh[1], sh[2]
    reshaped_3d = tf.reshape(tensor_3d, [batch_size * A, B])
    mult = tf.matmul(reshaped_3d, tensor_2d)
    return tf.reshape(mult, [batch_size, A, -1])

def multiply_3d_and_3d_tensor(tensor_a, tensor_b):
    '''Multiplies tensors of shapes [A, B, C] and [C, D, E].

       Inputs:
            tensor_3d: A 3D tensor of size [A, B, C].
            tensor_3d: A 3D tensor of size [C, D, E].
       Outputs:
            A tensor of size [A, B, D, E] that is the result of
            resizing the tensors as if it was a single matrix product
            [A * B, C] . [C, D * E].
    '''
    assert len(tensor_a.get_shape()) == 3
    assert len(tensor_b.get_shape()) == 3
    shape_a = tf.shape(tensor_a)
    shape_b = tf.shape(tensor_b)
    A, B, C = shape_a[0], shape_a[1], shape_a[2]
    D, E = shape_b[1], shape_b[2]
    reshaped_a = tf.reshape(tensor_a, [A * B, C])
    reshaped_b = tf.reshape(tensor_b, [C, D * E])
    mult = tf.matmul(reshaped_a, reshaped_b)
    return tf.reshape(mult, [A, B, D, E])
