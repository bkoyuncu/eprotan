from torch import nn

from typing import List, Tuple

import torch
import torch.distributions as D
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Dropout, LayerNorm
from source.models.act_functions import _get_activation_fn



class ProTranEncoderLayer(nn.Module):
    r"""
    Implements one hierarchical latent layer of the encoder of the ProTran model (https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html)
    Args:
        d_model (int): Dimensionality of the representations w_t of the latent space, which must have the same dimensionality as the input projections.
        mlp_dim_list_latent (List[int]): List of dimensions of the MLP used to create the mean and variance of z_t from \hat{w}_t concatenated with attention over the projections. Note that we calculate mean and variance with the same MLP.
        nhead (int): Number of heads in the multihead attention layer.
        dropout (float): Dropout probability used for all dropout layers.
        activation (Literal['relu', 'gelu']): Activation function used in the MLP.  """
    
    def __init__(
        self, 
        d_model: int, 
        mlp_dim_list_latent: List[int], 
        nhead: int,
        dropout: float = 0.1,
        activation: str = 'relu',
    ) -> None:
        
        # assert activation in ['relu', 'gelu'], f"activation should be relu/gelu, not {activation}"
        super(ProTranEncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.att_input_projection = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MLP used to create the mean and variance of z_t from \hat{w}_t and the full history of the projected inputs
        modules_mlp_latent = []     
        for i in range(len(mlp_dim_list_latent)-1):
            if i+1 == len(mlp_dim_list_latent)-1:
                modules_mlp_latent.append(nn.Linear(mlp_dim_list_latent[i], 2*mlp_dim_list_latent[i+1]))   
            elif i == 0: 
                modules_mlp_latent.append(nn.Linear(2*mlp_dim_list_latent[i], mlp_dim_list_latent[i+1]))
                activation_layer = _get_activation_fn(activation) #activation_layer = nn.ReLU() if activation == 'relu' else nn.GELU()
                modules_mlp_latent.append(activation_layer) 
            else:
                modules_mlp_latent.append(nn.Linear(mlp_dim_list_latent[i], mlp_dim_list_latent[i+1]))
                activation_layer = _get_activation_fn(activation)
                modules_mlp_latent.append(activation_layer)        
        self.mlp_latent = nn.Sequential(*modules_mlp_latent)
        
        # input attention layer output that we do not need to recompute every time forward is called as it is only dependent on the input projectionss
        self.k = None
        
    
    def forward(
        self,
        time_step: int,             # current time step t
        w_hat: Tensor,              # deterministic representation of the latent variable z_t
        input_projection: Tensor,   # projections of inputs x_{1:T}
        key_padding_mask: Tensor,   # attention mask ensuring that the attention is not using padding elements T < t <= Tm
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            time_step (int): Current time step t.
            w_hat (Tensor): Deterministic representation of the latent variable z_t of shape [batch_size, d_model].
            input_projection (Tensor): Projections of inputs x_{1:Tm} of shape [batch_size, Tm, d_model].
            key_padding_mask (Tensor): Attention mask ensuring that the attention is not using padding elements T < t <= Tm of shape [batch_size, Tm].
        Returns:
            mu (Tensor): Mean of the latent variable z_t of shape [batch_size, d_latent].
            var (Tensor): Variance of the latent variable z_t of shape [batch_size, d_latent].
        """
        if self.k is None:
            # store the output of the input attention layer to avoid recomputing it every time forward is called, needs to be reset after each forward pass in ProTran Hierarchy
            self.k = self.att_input_projection(input_projection, input_projection, input_projection, key_padding_mask=key_padding_mask, need_weights=False)[0] # [batch_size, Tm, d_model]
        
        k_t = self.k[:, time_step, :] # [batch_size, d_model]
        w_hat_and_k = torch.cat((w_hat, k_t), dim=-1) # [batch_size, 2*d_model]
        mlp_out = self.mlp_latent(w_hat_and_k) # [batch_size, 2*dim_latent]
        mu, var = torch.chunk(mlp_out, 2, dim=-1) # [batch_size, dim_latent] [batch_size, dim_latent]
        var = F.softplus(var) # [batch_size, dim_latent]
        
        return mu, var
    
    def reset_k(self):
        """Should be called after each forward pass in ProTran Hierarchy to reset the input attention layer output k."""
        self.k = None

class ProTranLayer(nn.modules.Module):
    r"""
    Implements one hierarchical latent layer of the conditional prior of the ProTran model (https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html)
    Note that it shares parameters (up to w_hat) with the encoder layer and thus we concatenate both representations in the forward pass.
    Args:
        d_model (int): Dimensionality of the representations w_t of the latent space, which must have the same dimensionality as the input projections.
        dim_latent (int): Dimensionality of the latent variable z_t.
        mlp_dim_list (List[int]): List of dimensions of the MLP used to create the mean and variance of z_t from \hat{w}_t. Note that we calculate mean and variance with the same MLP.
        mlp_dim_list_latent_to_w (List[int]): List of dimensions of the MLPs used to create w_t from z_t.
        nhead (int): Number of heads in the multihead attention layers.
        encoder_layer (ProTranEncoderLayer): Encoder layer used to create mu_q and var_q.
        positional_encoding (nn.Module): Positional encoding used to create w_t.
        dropout (float): Dropout probability used for all dropout layers.
        activation (Literal['relu', 'gelu']): Activation function used in the MLPs.
        lowest_hierarchy (bool): Whether the layer is the lowest layer in the hierarchy.
    """
    def __init__(
        self, 
        d_model: int, 
        dim_latent: int,
        mlp_dim_list_latent: List[int], 
        mlp_dim_list_latent_to_w: List[int],
        nhead: int,
        encoder_layer: ProTranEncoderLayer,
        positional_encoding: nn.Module,
        dropout: float = 0.1,
        activation: str = 'relu',
        lowest_hierarchy: bool = False,
    ) -> None:
        
        # assert activation in ['relu', 'gelu'], f"activation should be relu/gelu, not {activation}"
        
        # Assure the correct dimensionality of the MLPs. Latent dimension does not need to match the dimensionality of the input projections/w_t
        assert mlp_dim_list_latent[0] == mlp_dim_list_latent_to_w[-1] == d_model, f"mlp_dim_list_latent[0] should be equal to mlp_dim_list_latent_to_w[-1] and d_model"
        assert mlp_dim_list_latent[-1] == mlp_dim_list_latent_to_w[0] == dim_latent, f"mlp_dim_list_latent[-1] should be equal to mlp_dim_list_latent_to_w[0]"
        
        assert positional_encoding is not None, f"positional_encoding should be given"
        super(ProTranLayer, self).__init__()
        
        self.d_model = d_model
        self.dim_latent = dim_latent
        self.lowest_hierarchy = lowest_hierarchy
        # Learnable representation of w_0
        self.w_0 = nn.Parameter(torch.normal(0, 1, size=(1, d_model)))
        
        # Positional encoding added to create w_t
        self.positional_encoding = positional_encoding
        
        # encoder layer used to create mu_q and var_q
        self.encoder_layer = encoder_layer
        
        # Multihead attention layers
        self.att_lower_layer = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) if not lowest_hierarchy else None
        self.att_same_layer = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.att_input_projection = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # create the MLPs used in the layer
        # MLP used to create the mean and variance of z_t from \hat{w}_t
        modules_mlp_latent = []     
        for i in range(len(mlp_dim_list_latent)-1):
            if i+1 == len(mlp_dim_list_latent)-1:
                modules_mlp_latent.append(nn.Linear(mlp_dim_list_latent[i], 2*mlp_dim_list_latent[i+1]))    
            else:
                modules_mlp_latent.append(nn.Linear(mlp_dim_list_latent[i], mlp_dim_list_latent[i+1]))
                activation_layer = _get_activation_fn(activation) #nn.ReLU() if activation == 'relu' else nn.GELU()
                modules_mlp_latent.append(activation_layer)        
        self.mlp_latent = nn.Sequential(*modules_mlp_latent)
        
        # MLP used to create w_t from z_t
        modules_mlp_latent_to_w = []
        for i in range(len(mlp_dim_list_latent_to_w)-1):
            if i+1 == len(mlp_dim_list_latent_to_w)-1:
                modules_mlp_latent_to_w.append(nn.Linear(mlp_dim_list_latent_to_w[i], mlp_dim_list_latent_to_w[i+1]))    
            else:
                modules_mlp_latent_to_w.append(nn.Linear(mlp_dim_list_latent_to_w[i], mlp_dim_list_latent_to_w[i+1]))
                activation_layer = _get_activation_fn(activation)
                modules_mlp_latent_to_w.append(activation_layer) 
        self.mlp_latent_to_w = nn.Sequential(*modules_mlp_latent_to_w)
        
        self.ln1 = LayerNorm(d_model, eps=1e-5) if not lowest_hierarchy else None
        self.ln2 = LayerNorm(d_model, eps=1e-5)
        self.ln3 = LayerNorm(d_model, eps=1e-5)
        self.ln4 = LayerNorm(d_model, eps=1e-5)
        
        self.dropout1 = Dropout(dropout) if not lowest_hierarchy else None
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
            
    def forward(
        self, 
        input_projection_p: Tensor,
        input_projection_q: Tensor,
        batch_size: int,
        Tm: int,
        key_padding_mask_history: Tensor,
        device: torch.device,
        lower_layer_p: Tensor = None,
        lower_layer_q: Tensor = None,
        key_padding_mask_padding: Tensor = None,
        use_sampling:bool = True,                       
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            input_projection_p: [bs, Tm, d] Projections h_{1:Tm} of the input at timesteps 1:Tm, where h_{T0:Tm} have been masked
            input_projection_q: [bs, Tm, d] Projections h_{1:Tm} of the input at timesteps 1:Tm
            batch_size: batch size
            Tm: maximum sequence length	
            key_padding_mask_history: [bs, Tm] Attention mask ensuring that the attention is only over the history of the input sequence h_{1:T0} for the conditional prior layer (and in general w_hat).
            device: Device used for the model.
            lower_layer_p: [bs, T, d] Input w^(l-1)_{1:T} from the lower layer (l-1) in the latent variable hierarchy.
            lower_layer_q: [bs, T, d] Input w_q^(l-1)_{1:T} from the lower layer (l-1) in the encoder hierarchy.
            key_padding_mask_padding: [bs, Tm] Attention mask ensuring that the attention is only over w^(l-1)_{1:T} of the lower layer (and does not include padding for batches for T < t <= Tm), and only over h_{1:T} for the input attention of the encoder layer.
            use_sampling: Whether to sample z_t from the mean and variance of the latent variable z_t. If False, we use the mean of the latent variable z_t.
        
        Returns:
            w_p: [bs, Tm, d] Sequence of latent representations w^(l)_{1:Tm} at the current layer l. Note that we don't stop the generation for T < t <= Tm, this has to be addressed when computing the loss and the next layer.
            w_p_hat: [bs, Tm, d] Sequence of latent representations \hat{w}^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}.
            mu_p_z: [bs, Tm, d] Sequence of latent means mu^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}. mu_p_z is used to calculate KL divergence.
            var_p_z: [bs, Tm, d] Sequence of latent variances var^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}. var_p_z is used to calculate KL divergence.
            w_q: [bs, Tm, d] Sequence of latent representations w^(l)_{1:Tm} at the current layer l for the encoder.
            w_q_hat: [bs, Tm, d] Sequence of latent representations \hat{w}^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm} for the encoder.
            mu_q_z: [bs, Tm, d] Sequence of latent means mu^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm} for the encoder. mu_q_z is used to calculate KL divergence.
            var_q_z: [bs, Tm, d] Sequence of latent variances var^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm} for the encoder. var_q_z is used to calculate KL divergence.
            
        """
        # CAUTION: Note that we don't stop the generation of w_t, w_hat_t and z_t for T < t <= Tm, this has to be addressed when computing the loss and the next layer. --> This is why we use key padding masks for the attention layers.
        
        if not self.lowest_hierarchy:
            assert lower_layer_p is not None, f"lower_layer should be given"
            assert lower_layer_q is not None, f"lower_layer should be given"
        
        # conditional prior layer     
        w_p = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        w_p_hat = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        
        mu_p_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        var_p_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        
        # encoder layer
        w_q = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        w_q_hat = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        
        mu_q_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        var_q_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        
        # stack w_0 to size [bs, 1, d] for batch processing
        w_p_t = self.w_0.repeat(batch_size, 1).unsqueeze(1) # [bs, 1, d]
        w_q_t = self.w_0.repeat(batch_size, 1).unsqueeze(1) # [bs, 1, d]
        
        # concat lower_layer_p and lower_layer_q for fast attention
        if lower_layer_p is not None and lower_layer_q is not None:
            lower_layer = torch.cat((lower_layer_p, lower_layer_q), dim=0) # [2*bs, T, d] first is for conditional prior, second is for encoder
            key_padding_mask_lower_layer = torch.cat((key_padding_mask_padding, key_padding_mask_padding), dim=0) # [2*bs, Tm], the mask is the same for encoder and conditional prior
        
        # stack input_projection to size [2*bs, Tm, d] for batch processing
        # note that we do p twice because these steps are shared according to the paper
        input_projection_cat = torch.cat((input_projection_p, input_projection_p), dim=0) # [2*bs, Tm, d] first is for conditional prior, second is for encoder
        key_padding_mask_projection = torch.cat((key_padding_mask_history, key_padding_mask_history), dim=0) # [2*bs, Tm], the mask is the same for encoder and conditional prior
        
        # iterative generation of w_t, w_hat_t and z_t at this layer (l)
        for t in range(Tm):
            # attention to lower hierarchy w^(l-1)_{1:T}
            if self.lowest_hierarchy:
                w_p_tilde_t = w_p_t
                w_q_tilde_t = w_q_t
            else:
                # concat for fast attention
                w_t = torch.cat((w_p_t, w_q_t), dim=0) # [2*bs, 1, d] first is for conditional prior, second is for encoder
                
                out_att = self.att_lower_layer(w_t, lower_layer, lower_layer, key_padding_mask=key_padding_mask_lower_layer, need_weights=False)[0] # [bs, 1, d]
                out_att = self.ln1(w_t + self.dropout1(out_att)) # [bs, 1, d]
                
                # split back to w_p_tilde_t and w_q_tilde_t
                w_p_tilde_t, w_q_tilde_t = torch.chunk(out_att, 2, dim=0) # [bs, 1, d] and [bs, 1, d]
                       
            # attention to the history w_{1:t-1} in the same layer
            if t == 0:
                w_p_bar_t = w_p_tilde_t
                w_q_bar_t = w_q_tilde_t
            else:
                key_padding_mask_same_layer = torch.arange(Tm).to(device).expand(batch_size, Tm) >= t # [bs, Tm]
                out_att_p = self.att_same_layer(w_p_tilde_t, w_p, w_p, key_padding_mask=key_padding_mask_same_layer, need_weights=False)[0] # [bs, 1, d]
                w_p_bar_t = self.ln2(w_p_tilde_t + self.dropout2(out_att_p)) # [bs, 1, d]
                
                out_att_q = self.att_same_layer(w_q_tilde_t, w_q, w_q, key_padding_mask=key_padding_mask_same_layer, need_weights=False)[0] # [bs, 1, d]
                w_q_bar_t = self.ln2(w_q_tilde_t + self.dropout2(out_att_q)) # [bs, 1, d]
            
            # attention to the input projection h_{1:T0}
            # concat for fast attention
            w_bar_t = torch.cat((w_p_bar_t, w_q_bar_t), dim=0) # [2*bs, 1, d] first is for conditional prior, second is for encoder
            out_att = self.att_input_projection(w_bar_t, input_projection_cat, input_projection_cat, key_padding_mask=key_padding_mask_projection, need_weights=False)[0] # [bs, 1, d]
            w_hat_t = self.ln3(w_bar_t + self.dropout3(out_att)) # [bs, 1, d]
            
            # split w_hat_t to w_p_hat_t and w_q_hat_t
            w_p_hat_t, w_q_hat_t = torch.chunk(w_hat_t, 2, dim=0) # [bs, 1, d] and [bs, 1, d]
            
            # predict mu_p_t and var_p_t from w_p_hat_t
            out_mlp = self.mlp_latent(w_p_hat_t.squeeze(1)) # [bs, 2*dim_latent]
            mu_p_t, var_p_t = torch.chunk(out_mlp, 2, -1) # [bs, dim_latent] and [bs, dim_latent]
            var_p_t = F.softplus(var_p_t) # [bs, dim_latent]
            
            # predict mu_q_t and var_q_t from w_q_hat_t
            mu_q_t, var_q_t = self.encoder_layer(time_step=t, w_hat=w_q_hat_t.squeeze(1), input_projection=input_projection_q, key_padding_mask=key_padding_mask_padding) # [bs, dim_latent] and [bs, dim_latent]
            
            # sample z_t from mu_t and var_t
            if use_sampling:
                z_p_t = D.Normal(mu_p_t, torch.sqrt(var_p_t)).rsample() # [bs, dim_latent]
                z_q_t = D.Normal(mu_q_t, torch.sqrt(var_q_t)).rsample() # [bs, dim_latent]
            else:
                z_p_t = mu_p_t
                z_q_t = mu_q_t
            
            # create w_t from z_t
            # concat for faster processing
            z_t = torch.cat((z_p_t, z_q_t), dim=0) # [2*bs, dim_latent] first is for conditional prior, second is for encoder
            out_mlp_2 = self.mlp_latent_to_w(z_t).unsqueeze(1) # [2*bs, 1, d]
            mlp_2_w_hat_t_pos = self.positional_encoding.get_single_positional_encoding(out_mlp_2 + torch.cat((w_p_hat_t, w_q_hat_t), dim=0), t) # add potisional encoding [2*bs, 1, d]
            w_t = self.ln4(mlp_2_w_hat_t_pos)
            # split w_t to w_p_t and w_q_t
            w_p_t, w_q_t = torch.chunk(w_t, 2, dim=0) # [bs, 1, d] and [bs, 1, d]
            
            # save w_hat_t, w_t, mu_t and var_t
            
            mask_w = torch.ones_like(w_p_hat).bool()
            mask_w[:, t, :] = 0

            w_p_hat = w_p_hat + torch.masked_fill(w_p_hat_t, mask_w, 0) # broadcasts w_p_hat_t to w_p_hat
            w_p = w_p + torch.masked_fill(w_p_t, mask_w, 0) # broadcasts w_p_t to w_p
            w_q_hat = w_q_hat + torch.masked_fill(w_q_hat_t, mask_w, 0) # broadcasts w_q_hat_t to w_q_hat
            w_q = w_q + torch.masked_fill(w_q_t, mask_w, 0) # broadcasts w_q_t to w_q
            
            # w_p_hat[:, t, :] = w_hat_t.squeeze(1)
            # w_p[:, t, :] = w_t.squeeze(1)
            # w_q_hat[:, t, :] = w_q_hat_t.squeeze(1)
            # w_q[:, t, :] = w_q_t.squeeze(1)
            
            mu_p_z[:, t, :] = mu_p_t
            var_p_z[:, t, :] = var_p_t
            mu_q_z[:, t, :] = mu_q_t
            var_q_z[:, t, :] = var_q_t
            
        return w_p, w_p_hat, mu_p_z, var_p_z, w_q, w_q_hat, mu_q_z, var_q_z  
    
    def reset_k_encoder_layer(self):
        """Should be called after each forward pass in ProTran Hierarchy to reset the input attention layer output k."""
        self.encoder_layer.reset_k()
            
          
    def predict(
        self, 
        input_projection_p: Tensor, 
        batch_size: int,
        Tm: int,
        key_padding_mask_history: Tensor,
        device: torch.device,
        lower_layer_p: Tensor = None,
        key_padding_mask_padding: Tensor = None,
        use_sampling:bool = True,                       
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Same as forward, but used for prediction, i.e., without calculating encoder.
        
        Args:
            input_projection_p: [bs, Tm, d] Projections h_{1:Tm} of the input at timesteps 1:Tm, where h_{T0:Tm} have been masked
            batch_size: batch size
            Tm: maximum sequence length	
            key_padding_mask_history: [bs, Tm] Attention mask ensuring that the attention is only over the history of the input sequence h_{1:T0} for the conditional prior layer (and in general w_hat).
            device: Device used for the model.
            lower_layer_p: [bs, T, d] Input w^(l-1)_{1:T} from the lower layer (l-1) in the latent variable hierarchy.
            key_padding_mask_padding: [bs, Tm] Attention mask ensuring that the attention is only over w^(l-1)_{1:T} of the lower layer (and does not include padding for batches for T < t <= Tm), and only over h_{1:T} for the input attention of the encoder layer.
            use_sampling: Whether to sample z_t from the mean and variance of the latent variable z_t. If False, we use the mean of the latent variable z_t.
        
        Returns:
            w_p: [bs, Tm, d] Sequence of latent representations w^(l)_{1:Tm} at the current layer l. Note that we don't stop the generation for T < t <= Tm, this has to be addressed when computing the loss and the next layer.
            w_p_hat: [bs, Tm, d] Sequence of latent representations \hat{w}^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}.
            mu_p_z: [bs, Tm, d] Sequence of latent means mu^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}. mu_p_z is used to calculate KL divergence.
            var_p_z: [bs, Tm, d] Sequence of latent variances var^(l)_{1:Tm} at the current layer that are used to sample the latent variables z^(l)_{1:Tm}. var_p_z is used to calculate KL divergence.           
        """
        # CAUTION: Note that we don't stop the generation of w_t, w_hat_t and z_t for T < t <= Tm, this has to be addressed when computing the loss and the next layer. --> This is why we use key padding masks for the attention layers.
        
        if not self.lowest_hierarchy:
            assert lower_layer_p is not None, f"lower_layer should be given"
        
        # conditional prior layer     
        w_p = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        w_p_hat = torch.zeros((batch_size, Tm, self.d_model)).to(device) # [bs, Tm, d] for output
        
        mu_p_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        var_p_z = torch.zeros((batch_size, Tm, self.dim_latent)).to(device) # [bs, Tm, dim_latent] for loss
        
        
        # stack w_0 to size [bs, 1, d] for batch processing
        w_p_t = self.w_0.repeat(batch_size, 1).unsqueeze(1) # [bs, 1, d]
        
        # iterative generation of w_t, w_hat_t and z_t at this layer (l)
        for t in range(Tm):
            # attention to lower hierarchy w^(l-1)_{1:T}
            if self.lowest_hierarchy:
                w_p_tilde_t = w_p_t
            else:
                out_att = self.att_lower_layer(w_p_t, lower_layer_p, lower_layer_p, key_padding_mask=key_padding_mask_padding, need_weights=False)[0] # [bs, 1, d]
                w_p_tilde_t = self.ln1(w_p_t + self.dropout1(out_att)) # [bs, 1, d]
                       
            # attention to the history w_{1:t-1} in the same layer
            if t == 0:
                w_p_bar_t = w_p_tilde_t
            else:
                key_padding_mask_same_layer = torch.arange(Tm).to(device).expand(batch_size, Tm) >= t # [bs, Tm]
                out_att = self.att_same_layer(w_p_tilde_t, w_p, w_p, key_padding_mask=key_padding_mask_same_layer, need_weights=False)[0] # [bs, 1, d]
                w_p_bar_t = self.ln2(w_p_tilde_t + self.dropout2(out_att)) # [bs, 1, d]
                

            # attention to the input projection h_{1:T0}
            out_att = self.att_input_projection(w_p_bar_t, input_projection_p, input_projection_p, key_padding_mask=key_padding_mask_history, need_weights=False)[0] # [bs, 1, d]
            w_p_hat_t = self.ln3(w_p_bar_t + self.dropout3(out_att)) # [bs, 1, d]
            
            # predict mu_p_t and var_p_t from w_p_hat_t
            out_mlp = self.mlp_latent(w_p_hat_t.squeeze(1)) # [bs, 2*dim_latent]
            mu_p_t, var_p_t = torch.chunk(out_mlp, 2, -1) # [bs, dim_latent] and [bs, dim_latent]
            var_p_t = F.softplus(var_p_t) # [bs, dim_latent]
            
            # sample z_t from mu_t and var_t
            if use_sampling:
                z_p_t = D.Normal(mu_p_t, torch.sqrt(var_p_t)).rsample() # [bs, dim_latent]
            else:
                z_p_t = mu_p_t

            # create w_t from z_t
            out_mlp_2 = self.mlp_latent_to_w(z_p_t).unsqueeze(1) # [2*bs, 1, d]
            mlp_2_w_hat_t_pos = self.positional_encoding.get_single_positional_encoding(out_mlp_2 + w_p_hat_t, t) # add potisional encoding [2*bs, 1, d]
            w_p_t = self.ln4(mlp_2_w_hat_t_pos)
            
            
            # save w_hat_t, w_t, mu_t and var_t
            
            mask_w = torch.ones_like(w_p_hat).bool()
            mask_w[:, t, :] = 0

            w_p_hat = w_p_hat + torch.masked_fill(w_p_hat_t, mask_w, 0) # broadcasts w_p_hat_t to w_p_hat
            w_p = w_p + torch.masked_fill(w_p_t, mask_w, 0) # broadcasts w_p_t to w_p
            
            mu_p_z[:, t, :] = mu_p_t
            var_p_z[:, t, :] = var_p_t
            
        return w_p, w_p_hat, mu_p_z, var_p_z
      

class ProTranHierarchy(nn.Module):
    """
    Combines the ProTranConditionalPriorLayer and ProTranEncoderLayer to create the ProTran model.
    """
    
    def __init__(self, num_layers: int, d_model: int, dim_latent: int, protran_layer_list: nn.ModuleList):
        assert num_layers == len(protran_layer_list), f"num_layers should be equal to the number of layers in the protran_layer_list"
        super(ProTranHierarchy, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.dim_latent = dim_latent
        self.protran_layers = protran_layer_list
        
    def forward(
        self, 
        input_projection_p: Tensor, 
        input_projection_q: Tensor,                       
        batch_size: int,                                 	
        Tm: int, 
        key_padding_mask_history: Tensor, 
        key_padding_mask_sequence: Tensor,
        device: torch.device,
        use_sampling: bool = True, 
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            input_projection_p (Tensor): Projections of inputs h_{1:T}, where h_{T0:Tm} have been masked of shape [batch_size, Tm, d_model].
            input_projection_q (Tensor): Projections of inputs h_{1:T} of shape [batch_size, Tm, d_model].
            batch_size (int): batch size
            Tm (int): maximum sequence length (in batch)
            key_padding_mask_history (Tensor): [bs, Tm] Attention mask ensuring that the attention in the conditional prior (w_hat) is only over the history h_{1:T0} of the projected inputs.
            key_padding_mask_sequence (Tensor): [bs, Tm] Attention mask ensuring that the attention is only over the sequence x_{1:T} of the inputs, excluding padding elements T < t <= Tm.
            device (torch.device): Device used for the model.
            use_sampling (bool): Whether to sample z_t from the mean and variance of the latent variable z_t in the ProTranConPriorLayers. If False, we use the mean of the latent variable z_t.
            
        Returns:
            mu_p (Tensor): Mean of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_laten].
            var_p (Tensor): Variance of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_latent].
            mu_q (Tensor): Mean of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_latent].
            var_q (Tensor): Variance of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_latent].
            w_p_hat (Tensor): Latent representations w_hat_t of the last conditional prior layer of the hierarchy of shape [batch_size, Tm, d_model].
            w_p (Tensor): Latent representations w_t of the last conditional prior layer of the hierarchy of shape [batch_size, Tm, d_model].
            w_q_hat (Tensor): Latent representations w_hat_t of the last encoder layer of the hierarchy of shape [batch_size, Tm, d_model].
            w_q (Tensor): Latent representations w_t of the last encoder layer of the hierarchy of shape [batch_size, Tm, d_model].
        """
        
        mu_p = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device)  # [num_layers, batch_size, Tm, dim_latent]
        var_p = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device) # [num_layers, batch_size, Tm, dim_latent]
        mu_q = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device)  # [num_layers, batch_size, Tm, dim_latent]
        var_q = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device) # [num_layers, batch_size, Tm, dim_latent]
        
        w_p, w_p_hat, w_q, w_q_hat = None, None, None, None
        for i in range (self.num_layers):
            w_p, w_p_hat, mu_p[i], var_p[i], w_q, w_q_hat, mu_q[i], var_q[i] = self.protran_layers[i](
                input_projection_p = input_projection_p,
                input_projection_q = input_projection_q,
                batch_size = batch_size,
                Tm = Tm,
                key_padding_mask_history = key_padding_mask_history,
                device = device,
                lower_layer_p = w_p,
                lower_layer_q = w_q,
                key_padding_mask_padding = key_padding_mask_sequence,
                use_sampling = use_sampling
            )
            # reset k in encoder layer
            self.protran_layers[i].reset_k_encoder_layer()
        
        return mu_p, var_p, mu_q, var_q, w_p_hat, w_p, w_q_hat, w_q
    
    
    def get_w_samples(
        self,
        w_hat: Tensor,
        z: Tensor,
    ) -> Tensor:
        r"""
        Args:
            w_hat (Tensor): Latent representations w_hat_t of the last layer of the hierarchy of shape [batch_size, Tm, d_model].
            z (Tensor): Latent variable z_t of shape [batch_size, samples, Tm, dim_latent] | [batch_size, Tm, dim_latent].
        
        Returns:
            w (Tensor): Latent representations w_t of the last layer of the hierarchy of shape [batch_size, samples, Tm, d_model] | [batch_size, Tm, d_model].
        """
        if z.dim() == 4:
            w_hat = w_hat.unsqueeze(1).expand(-1, z.shape[1], -1, -1) # [batch_size, samples, Tm, d_model]
        w = self.protran_layers[-1].mlp_latent_to_w(z) + w_hat
        w = w + self.protran_layers[-1].positional_encoding(w)
        w = self.protran_layers[-1].ln4(w)
        
        return w
      
    
    def predict(
        self, 
        input_projection_p: Tensor,                        
        batch_size: int,                                 	
        Tm: int, 
        key_padding_mask_history: Tensor, 
        key_padding_mask_sequence: Tensor,
        device: torch.device,
        use_sampling: bool = True, 
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Same as forward, but used for prediction, i.e., without calculating encoder.
        Args:
            input_projection_p (Tensor): Projections of inputs h_{1:T}, where h_{T0:Tm} have been masked of shape [batch_size, Tm, d_model].
            batch_size (int): batch size
            Tm (int): maximum sequence length (in batch)
            key_padding_mask_history (Tensor): [bs, Tm] Attention mask ensuring that the attention in the conditional prior (w_hat) is only over the history h_{1:T0} of the projected inputs.
            key_padding_mask_sequence (Tensor): [bs, Tm] Attention mask ensuring that the attention is only over the sequence x_{1:T} of the inputs, excluding padding elements T < t <= Tm.
            device (torch.device): Device used for the model.
            use_sampling (bool): Whether to sample z_t from the mean and variance of the latent variable z_t in the ProTranConPriorLayers. If False, we use the mean of the latent variable z_t.
            
        Returns:
            mu_p (Tensor): Mean of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_laten].
            var_p (Tensor): Variance of the latent variable z_t of shape [num_layers, batch_size, Tm, dim_latent].
            w_p_hat (Tensor): Latent representations w_hat_t of the last conditional prior layer of the hierarchy of shape [batch_size, Tm, d_model].
            w_p (Tensor): Latent representations w_t of the last conditional prior layer of the hierarchy of shape [batch_size, Tm, d_model].
        """
        
        mu_p = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device)  # [num_layers, batch_size, Tm, dim_latent]
        var_p = torch.zeros(self.num_layers, batch_size, Tm, self.dim_latent).to(device) # [num_layers, batch_size, Tm, dim_latent]
        
        w_p, w_p_hat = None, None
        for i in range (self.num_layers):
            w_p, w_p_hat, mu_p[i], var_p[i] = self.protran_layers[i].predict(
                input_projection_p = input_projection_p,
                batch_size = batch_size,
                Tm = Tm,
                key_padding_mask_history = key_padding_mask_history,
                device = device,
                lower_layer_p = w_p,
                key_padding_mask_padding = key_padding_mask_sequence,
                use_sampling = use_sampling
            )
        
        return mu_p, var_p, w_p_hat, w_p
    
        

def get_protran_hierarchy(
    num_layers: int, 
    d_model: int,
    dim_latent: int, 
    dim_list_latent_cond_prior: List[int], 
    dim_list_latent_to_w: List[int],
    dim_list_latent_encoder: List[int],
    num_heads: int,
    positional_encoding,
    **kwargs
) -> ProTranHierarchy:
    r"""
    Args:
        num_layers (int): Number of layers of hierarchical latent layers
        d_model (int): Dimensionality of the deterministic representations w_t of the latent space, which must have the same dimensionality as the input projections.
        dim_latent (int): Dimensionality of the latent space.
        dim_list_latent (List[int]): List of dimensions of the MLP used to create the mean and variance of z_t from \hat{w}_t.
        dim_list_latent_to_w (List[int]): List of dimensions of the MLP used to create w_t from z_t.
        dim_list_latent_encoder (List[int]): List of dimensions of the MLP used to create the mean and variance of z_t from \hat{w}_t concatenated with attention over the projections.
        nhead (int): Number of heads in the multihead attention layers.
        positional_encoding: Positional encoding.
        **kwargs: dropout and activation for the MLPs
    """
    assert num_layers > 0, f"num_layers should be > 0, not {num_layers}"
    
    protran_layers = nn.ModuleList()
    
    for i in range(num_layers):
        lowest_hierarchy = True if i == 0 else False
        
        encoder_layer = ProTranEncoderLayer(
            d_model = d_model, 
            mlp_dim_list_latent = dim_list_latent_encoder, 
            nhead=num_heads, 
            **kwargs
        )
        
        protran_layers.append(
            ProTranLayer(
                d_model = d_model,
                dim_latent = dim_latent,
                mlp_dim_list_latent = dim_list_latent_cond_prior,
                mlp_dim_list_latent_to_w = dim_list_latent_to_w,
                nhead = num_heads,
                encoder_layer = encoder_layer,
                positional_encoding = positional_encoding,
                lowest_hierarchy = lowest_hierarchy,
                **kwargs
            )
        )

    return ProTranHierarchy(num_layers, d_model, dim_latent, protran_layers)
    