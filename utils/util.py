import torch.nn as nn

def get_active_func(activation_func):
    """Get the activation function.
    Args:
        activation_func: (str) activation function
    Returns:
        activation function: (torch.nn) activation function
    """
    if activation_func == "sigmoid":
        return nn.Sigmoid()
    elif activation_func == "gelu":
        return nn.GELU()
    elif activation_func == "tanh":
        return nn.Tanh()
    elif activation_func == "prelu":
        return nn.PReLU()
    elif activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_func == "selu":
        return nn.SELU()
    elif activation_func == "hardswish":
        return nn.Hardswish()
    elif activation_func == "identity":
        return nn.Identity()
    else:
        assert False, "activation function not supported!"