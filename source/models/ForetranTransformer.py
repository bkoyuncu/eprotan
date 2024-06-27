import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Literal, Optional
from source.models.act_functions import _get_activation_fn

def _get_normalization_layer(normalization, d_model):
    if normalization == "BatchNorm1d":
        return nn.BatchNorm1d(d_model, eps=1e-5)
    elif normalization == "LayerNorm":
        return nn.LayerNorm(d_model, eps=1e-5)
    else:
        raise ValueError("norms should be BatchNorm1d/LayerNorm, not {}".format(normalization))

class ForetranLayer(nn.Module):
    """
    Class for the Foretran layer, which allows to include all possible attentions (to the first layer, to the previous layer, autoregressive attention to the previously generated embedding).
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: List[int],
        dropout: float = 0.1, 
        activation: str ="relu", 
        normalization: str = "LayerNorm",
        kind_attentions: List[Literal['layer', 'input', 'autoregressive']] = ['layer']
    ):
        """
        Args:
            d_model (int): the number of expected features in the input (required).
            nhead (int): the number of heads in the multiheadattention models (required).
            dim_feedforward (List[int]): the dimensions of the feedforward network model (required).
            dropout (float): the dropout value (default=0.1).
            activation (str): the activation function of intermediate layer, relu or gelu (default=relu).
            kind_attentions (List[Literal['layer', 'input', 'autoregressive']]): list of kinds of attentions to include. Possible values: 'layer', 'input', 'autoregressive'.
        """
        assert len(kind_attentions) > 0, "At least one attention should be included"
        assert dim_feedforward[0] == dim_feedforward[-1] == d_model, "The first and last dimensions of the feedforward network should be equal to d_model"
        for kind in kind_attentions:
            assert kind in ['layer', 'input', 'autoregressive'], f"Attention kind {kind} is not supported"
        super(ForetranLayer, self).__init__()
        self.kind_attentions = kind_attentions
        self.kind_norm = normalization
        # if we don't do autoregressive stuff in the same layer, we can do the forward pass in parallel
        self.parallel = 'autoregressive' not in kind_attentions
        
        # 3 possible attentions
        self.layer_attention = None
        self.input_attention = None
        self.same_layer_attention = None
        if 'layer' in kind_attentions:
            self.layer_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if 'input' in kind_attentions:
            self.input_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if 'autoregressive' in kind_attentions:
            self.same_layer_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            
        # Implementation of Feedforward model
        modules_mlp = []     
        for i in range(len(dim_feedforward)-1):
            if i+1 == len(dim_feedforward)-1:
                modules_mlp.append(nn.Linear(dim_feedforward[i], dim_feedforward[i+1]))    
            else:
                modules_mlp.append(nn.Linear(dim_feedforward[i], dim_feedforward[i+1]))
                activation_layer = _get_activation_fn(activation)
                modules_mlp.append(activation_layer)        
        self.mlp = nn.Sequential(*modules_mlp)
        
        # norms and dropout for each attention
        self.layer_norm = None
        self.layer_dropout = None
        self.input_norm = None
        self.same_layer_norm = None 
        self.input_dropout = None
        self.same_layer_dropout = None
        
        if 'layer' in kind_attentions:
            self.layer_norm = _get_normalization_layer(normalization, d_model) # do not confuse with layer normalization
            self.layer_dropout = nn.Dropout(dropout)
        if 'input' in kind_attentions:
            self.input_norm = _get_normalization_layer(normalization, d_model)
            self.input_dropout = nn.Dropout(dropout)
        if 'autoregressive' in kind_attentions:
            self.same_layer_norm = _get_normalization_layer(normalization, d_model)
            self.same_layer_dropout = nn.Dropout(dropout)
            
        # Learnable representation of h_0 for autoregressive attention
        self.h_0 = None,
        if 'autoregressive' in kind_attentions:
            self.h_0 = nn.Parameter(torch.normal(0, 1, size=(1, d_model)))
        self.activation = _get_activation_fn(activation)
        
    def forward(
        self,
        h_lower: torch.Tensor, # lower layer
        device: torch.device,
        input_embeddings: Optional[Tensor] = None, # input projection, needed for input attention
        mask_layer: Optional[Tensor] = None,
        key_padding_mask_layer: Optional[Tensor] = None,
        mask_input: Optional[Tensor] = None,
        key_padding_mask_input: Optional[Tensor] = None,
    ):  
        """
        Args:
            h_lower (torch.Tensor): lower layer output [bs, Tm, d]. This can be None for the first layer.
            input_embeddings (Optional[Tensor], optional): input projection [bs, Tm, d]. Defaults to None. Has to be provided if 'input' in kind_attentions.
            mask_layer (Optional[Tensor], optional): Mask for attention to the layer [Tm, Tm].  Defaults to None. If used with auto-regressive attention, the rows are used one by one. Broadcasted across batch.
            key_padding_mask_layer (Optional[Tensor], optional): Key padding mask for attention to the layer [bs, Tm]. Defaults to None.
            mask_input (Optional[Tensor], optional): Mask for attention to the input (projection) [Tm, Tm]. Defaults to None. If used with auto-regressive attention, the rows are used one by one. Broadcasted across batch.
            key_padding_mask_input (Optional[Tensor], optional): Key padding mask for attention to the input (projection) [bs, Tm]. Defaults to None.

        Returns:
            h (torch.Tensor): output of the layer [bs, Tm, d]
        """
        if 'input' in self.kind_attentions and input_embeddings is None:
            raise ValueError("Input embeddings are required for input attention.")
        if 'layer' in self.kind_attentions and h_lower is None:
            raise ValueError("Lower layer output is required for layer attention.")
        
       
        # fast path if we don't need to do autoregressive stuff in the same layer
        if self.parallel:
            
            # lower layer attention
            if 'layer' in self.kind_attentions:
                h_layer = self.layer_attention(h_lower, h_lower, h_lower, attn_mask=mask_layer, key_padding_mask=key_padding_mask_layer, need_weights=False)[0]
                h_layer = self.layer_dropout(h_layer)             
                if self.kind_norm == "LayerNorm":
                    h_layer = self.layer_norm(h_layer + h_lower)
                elif self.kind_norm == "BatchNorm1d":
                    h_layer = h_layer + h_lower
                    h_layer = h_layer.permute(0, 2, 1)  # [bs, d, T]
                    h_layer = self.layer_norm(h_layer)
                    h_layer = h_layer.permute(0, 2, 1)  # [bs, T, d]
            else:
                h_layer = h_lower
            
            # input attention    
            if 'input' in self.kind_attentions:
                if h_layer is None: # first layer
                    # one could consider having an h_0 here instead of querying the inputs with themelves
                    h_layer = input_embeddings
                # consider .clone
                h_input = self.input_attention(h_layer, input_embeddings, input_embeddings, attn_mask=mask_input, key_padding_mask=key_padding_mask_input, need_weights=False)[0]
                h_input = self.input_dropout(h_input)
                if self.kind_norm == "LayerNorm":
                    h_input = self.input_norm(h_input + h_layer)
                elif self.kind_norm == "BatchNorm1d":
                    h_input = h_input + h_layer
                    h_input = h_input.permute(0, 2, 1) 
                    h_input = self.input_norm(h_input)
                    h_input = h_input.permute(0, 2, 1)
            else:
                h_input = h_layer
                
            # FFNN
            h = self.mlp(h_input)
            return h
        
        # slow path
        else:
            if h_lower is not None:
                batch_size, Tm, d = h_lower.shape
            else:
                batch_size, Tm, d = input_embeddings.shape
            h =  torch.zeros((batch_size, Tm, d)).to(device) # [bs, Tm, d] for output
            
            # stack h_0 to size [bs, 1, d] for batch processing
            h_t = self.h_0.repeat(batch_size, 1).unsqueeze(1) # [bs, 1, d]
            
            for t in range(Tm):
                # lower layer attention
                if 'layer' in self.kind_attentions:
                    # extract the t-th row of the mask if it is not None
                    if mask_layer is not None:
                        mask_layer_t = mask_layer[t].unsqueeze(0) # [1, Tm] = [L, S]; broadcasted across batch
                    else:
                        mask_layer_t = None
                    
                    # ensure that the mask is not all 1s or all -infs; otherwise we don't want to do the attention
                    # flag_full_mask = mask_layer_t is not None
                    # flag_full_mask = flag_full_mask and (torch.all(mask_layer_t == 1) or torch.all(mask_layer_t == -torch.inf))
                    # if not flag_full_mask:  
                    out_att = self.layer_attention(h_t, h_lower, h_lower, attn_mask=mask_layer_t, key_padding_mask=key_padding_mask_layer, need_weights=False)[0] # [bs, 1, d]
                    h_layer_t = self.layer_dropout(out_att)
                    if self.kind_norm == "LayerNorm":
                        h_layer_t = self.layer_norm(h_layer_t + h_t)
                    elif self.kind_norm == "BatchNorm1d":
                        h_layer_t = h_layer_t + h_t
                        h_layer_t = h_layer_t.permute(0, 2, 1)
                        h_layer_t = self.layer_norm(h_layer_t)
                        h_layer_t = h_layer_t.permute(0, 2, 1)   
                else:
                    h_layer_t = h_t
                    
                # same layer attention
                if 'autoregressive' in self.kind_attentions and t > 0:
                    key_padding_mask_same_layer = torch.arange(Tm).to(device).expand(batch_size, Tm) >= t # [bs, Tm], mask for future
                    out_att = self.same_layer_attention(h_layer_t, h, h, attn_mask=None, key_padding_mask=key_padding_mask_same_layer, need_weights=False)[0] # [bs, 1, d]
                    h_auto_t = self.same_layer_dropout(out_att)
                    if self.kind_norm == "LayerNorm":
                        h_auto_t = self.same_layer_norm(h_auto_t + h_layer_t)
                    elif self.kind_norm == "BatchNorm1d":
                        h_auto_t = h_auto_t + h_layer_t
                        h_auto_t = h_auto_t.permute(0, 2, 1)
                        h_auto_t = self.same_layer_norm(h_auto_t)
                        h_auto_t = h_auto_t.permute(0, 2, 1)
                else:
                    h_auto_t = h_layer_t
                    
                # input attention
                if 'input' in self.kind_attentions:
                    # extract the t-th row of the mask if it is not None
                    if mask_input is not None:
                        mask_input_t = mask_input[t].unsqueeze(0) # [1, Tm] = [L, S]; broadcasted across batch
                    else:
                        mask_input_t = None
                        
                    # ensure that the mask is not all 1s or all -infs; otherwise we don't want to do the attention
                    # flag_full_mask = mask_input_t is not None
                    # flag_full_mask = flag_full_mask and (torch.all(mask_input_t == 1) or torch.all(mask_input_t == -torch.inf))
                    # if not flag_full_mask:
                    out_att = self.input_attention(h_auto_t, input_embeddings, input_embeddings, attn_mask=mask_input_t, key_padding_mask=key_padding_mask_input, need_weights=False)[0]
                    h_input_t = self.input_dropout(out_att)
                    if self.kind_norm == "LayerNorm":
                        h_input_t = self.input_norm(h_input_t + h_auto_t)
                    elif self.kind_norm == "BatchNorm1d":
                        h_input_t = h_input_t + h_auto_t
                        h_input_t = h_input_t.permute(0, 2, 1)
                        h_input_t = self.input_norm(h_input_t)
                        h_input_t = h_input_t.permute(0, 2, 1)
                else:
                    h_input_t = h_auto_t
                    
                # FFNN
                h_t = self.mlp(h_input_t)
                
                # store the result
                mask_h = torch.ones_like(h).bool()
                mask_h[:,t,:] = False
                h = h + torch.masked_fill(h_t, mask_h, 0)
            
            return h
            
            
class ForetranTransformer(nn.Module):
    """Stack of Foretran layers"""
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int,
        dim_feedforward: List[int],
        dropout:int = 0.1, 
        activation: str = "relu", 
        normalization: str = "LayerNorm",
        kind_attentions: List[Literal['layer', 'input', 'autoregressive']] = ['layer'],
    ):
        """
        Args:
            d_model (int): Dimension of the hidden stats/embeddings.
            nhead (int): Number of heads in the multiheadattention models.
            num_layers (int): Number of ForeTran layers in the transformer.
            dim_feedforward (List[int]): Dimensionality list of the FF-NN.
            dropout (int, optional): Dropout probability. Defaults to 0.1.
            activation (str, optional): Kind of Activation function. Defaults to "relu".
            normalization (str, optional): Kind of normalization. Defaults to "LayerNorm".
        """
        assert len(kind_attentions) > 0, "At least one attention should be included"
        for kind in kind_attentions:
            assert kind in ['layer', 'input', 'autoregressive'], f"Attention kind {kind} is not supported"
          
        super(ForetranTransformer, self).__init__()
        self.kind = 'ForetranTransformer'
        self.kind_attentions = kind_attentions
        
        # exclude lower layer attention in the first layer
        if 'layer' in kind_attentions:
            lowest_layer_attentions = [att for att in kind_attentions if att != 'layer']
        else:
            lowest_layer_attentions = kind_attentions.copy()
        # lowest layer must include the input attention
        if 'input' not in lowest_layer_attentions:
            lowest_layer_attentions.append('input')
        self.layers = nn.ModuleList()
        self.layers.append(ForetranLayer(d_model, nhead, dim_feedforward, dropout, activation, normalization, lowest_layer_attentions))
        for _ in range(1, num_layers):
            self.layers.append(ForetranLayer(d_model, nhead, dim_feedforward, dropout, activation, normalization, kind_attentions))
            
        
    def forward(
        self, 
        input_projections: torch.Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask_layer: Optional[Tensor] = None,
        mask_input: Optional[Tensor] = None,
        key_padding_mask_input: Optional[Tensor] = None,
    ):
        """
        Args:
            input_projections (torch.Tensor): input projections [bs, Tm, d]
            kind_attentions (List[Literal['layer', 'input', 'autoregressive']], optional): Kinds of attentions to include. Defaults to ['layer'].
            mask(_layer) (Optional[Tensor], optional): Mask for attention to the layer [Tm, Tm].  Defaults to None. If used with auto-regressive attention, the rows are used one by one. Broadcasted across batch.
            key_padding_mask_layer (Optional[Tensor], optional): Key padding mask for attention to the layer [bs, Tm]. Defaults to None.
            mask_input (Optional[Tensor], optional): Mask for attention to the input (projection) [Tm, Tm]. Defaults to None. If used with auto-regressive attention, the rows are used one by one. Broadcasted across batch.
            key_padding_mask_input (Optional[Tensor], optional): Key padding mask for attention to the input (projection) [bs, Tm]. Defaults to None.

        Returns:
            torch.Tensor: output of the transformer [bs, Tm, d]
        """
        # first layer
        h = self.layers[0](None, input_projections.device, input_projections, mask, key_padding_mask_layer, mask_input, key_padding_mask_input)
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                if 'layer' in self.kind_attentions and 'input' in self.kind_attentions:
                    h = layer(h, h.device, input_projections, mask, key_padding_mask_layer, mask_input, key_padding_mask_input)
                elif 'layer' in self.kind_attentions:
                    h = layer(h, h.device, None, mask, key_padding_mask_layer, mask_input, key_padding_mask_input)
                elif 'input' in self.kind_attentions:
                    h = layer(None, input_projections.device, input_projections, mask, key_padding_mask_layer, mask_input, key_padding_mask_input)
                    
        return h