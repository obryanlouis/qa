"""Functions for dealing with printing various things.
"""

import tensorflow as tf

def readable_time(eta):
    if eta >= 3600:
        return "(hr) " + str(eta / 3600)
    if eta >= 60:
        return "(min) " + str(eta / 60)
    return "(sec) " + str(eta)

def readable_eta(eta):
    return "ETA" + readable_time(eta)

def maybe_print_model_parameters(options):
    """Debugging function to print the model parameters.
    """
    if not options.verbose_logging:
        return
    total_parameters = 0
    params_list = []
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        params_list.append((variable_parameters, variable, shape, len(shape)))
    params_list.sort(key=lambda x:x[0])
    for variable_parameters, variable, shape, l in params_list:
        print("variable_parameters", variable_parameters, "variable", variable, "shape", shape, "len(shape)", l)
    print("total_parameters", total_parameters)

