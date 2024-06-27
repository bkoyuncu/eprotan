from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer, LayerNorm
from source.models.act_functions import _get_activation_fn


#from https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py



def _get_normalization_layer(normalization, d_model):
    if normalization == "BatchNorm1d":
        return BatchNorm1d(d_model, eps=1e-5)
    elif normalization == "LayerNorm":
        return LayerNorm(d_model, eps=1e-5)
    raise ValueError("norms should be BatchNorm1d/LayerNorm, not {}".format(normalization))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first: bool = True, normalization="BatchNorm1d"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        # self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.norm1 = _get_normalization_layer(normalization, d_model)
        self.norm2 = _get_normalization_layer(normalization, d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    #set normalization

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: Currently a dummy argument because we stack BNEncoderLayers with torch.nn.TransformerEncoder, which passes the is_causal argument to the BNEncoderLayer
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = src + self.dropout1(src2) # [bs, T, d]
        src = src.permute(0, 2, 1)  # [bs, d, T]
        src = self.norm1(src)
        src = src.permute(0, 2, 1)  # [bs, T, d]
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # [bs, T, d]
        src = src.permute(0, 2, 1)  # [bs, d, T]
        src = self.norm2(src)
        src = src.permute(0, 2, 1)  # [bs, T, d]
        return src