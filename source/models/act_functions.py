
import torch.nn as nn

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    raise ValueError("activation should be relu/gelu, not {}".format(activation))