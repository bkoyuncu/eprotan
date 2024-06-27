
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

class Encoder(nn.Module):
    """
    Implements a Conditional Prior distribution
    """
    def __init__(self, type: str='ConditionalPrior', tm: int=2, tmf:int =1, tmh: int=1):
        """
        Prior initialization
        Args:
            type (str, optional): prior type. Defaults to 'ConditionalPrior'.
            tm (int): maximum input length
            tmh (int): maximum history length
            tmf (int): maximum forecasting length
            tm=tmh+tmf
        """
        super(Encoder, self).__init__()
        self.type=type
        self.tm=tm
        self.tmf=tmf
        self.tmh=tmh
    
    def forward():
        pass
            

    def logq(self, z, mu, logvar):
        """
        computes q(z|x) where z is coming from q
        size for z is expected [bs, L, T,dim]
        size for mu is expected [bs, T,dim]
        size for logvar is expected [bs, T,dim]
        """

        mu_z, logvar_z = mu, logvar
        mu_z = mu_z.unsqueeze(1) # // INFO these are adding the latent sample dimension which is now 1
        logvar_z = logvar_z.unsqueeze(1)
        cnt = mu_z.shape[-1] * np.log(2 * np.pi) + torch.sum(logvar_z, dim=-1)
        logqz = -0.5 * (cnt + (z - mu_z)**2 * torch.exp(-logvar_z))

        return logqz



class MLPEncoder(Encoder):
    """
    Implements a MLP network for modeling conditional distribution
    """
    def __init__(self, type='MLP', network=None):
        """
        Prior initialization
        Args:
            type (str, optional): prior network
            network (nn.Module): prior network type
        """
        super(MLPEncoder, self).__init__(type)
        self.network=network
    
    def forward(self, x: torch.Tensor, causal_mask) -> torch.Tensor:
        phi = self.network(x)
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar


class TransformerEncoder(Encoder):
    """
    Implements a Transformer network for modeling conditional distribution
    """
    def __init__(self, type='Transformer', transformer_network=None, mlp_network=None,  tm: int=2, tmf:int =1, tmh: int=1):
        super(TransformerEncoder, self).__init__(type, tm,tmf,tmh)
        self.transformer_network=transformer_network
        self.mlp_network=mlp_network
        

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor= None, input_mask: torch.Tensor = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        computes input -> h (deterministic) -> z (random)
        Args:
            x (torch.Tensor): input with shape [bs, T, d]
        """
        if causal_mask is not None and input_mask is not None:
            if self.transformer_network.kind == "ForetranTransformer":
                h = self.transformer_network(x, mask=causal_mask, key_padding_mask_input=input_mask, **kwargs)
            else:
                h = self.transformer_network(x, mask=causal_mask, src_key_padding_mask=input_mask)
        elif causal_mask is not None:
            if self.transformer_network.kind == "ForetranTransformer":
                h = self.transformer_network(x, mask=causal_mask, **kwargs)
            else:
                h = self.transformer_network(x, mask=causal_mask)
        elif input_mask is not None:
            if self.transformer_network.kind == "ForetranTransformer":
                h = self.transformer_network(x, key_padding_mask_input=input_mask, **kwargs)
            else:
                h = self.transformer_network(x, src_key_padding_mask=input_mask)
        else:
            if self.transformer_network.kind == "ForetranTransformer":
                h = self.transformer_network(x, **kwargs)
            else:
                h = self.transformer_network(x)
        phi = self.mlp_network(h)
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar, h
    

    def regularizer(self, mu, logvar, observed):
        kl = -0.5 * torch.sum(1. + logvar - mu ** 2 - torch.exp(logvar), dim=-1, keepdim=True)
        kl = kl * observed
        return kl.T
    
    def regularizer_arbitrary_prior(self, mu0, mu1, logvar0, logvar1, observed):
        '''
        computes KL(N(mu0,logvar0)||N(mu1,logvar1))
        '''
        kl = -0.5 * torch.sum(1. + (logvar0-logvar1) - (mu1-mu0) *  torch.exp(-logvar1) * (mu1-mu0) - torch.exp(logvar0-logvar1), dim=-1, keepdim=True)
        kl = kl * observed
        return kl.T


class EncoderHead(Encoder):
    """
    Implements an encoder head that is meant to be used on top of a conditional prior transformer
    """
    def __init__(self, type='EncoderHead', input_attention_head=None, mlp_network=None, tm: int=None, tmf:int =None, tmh: int=None):
        super(EncoderHead, self).__init__(type, tm,tmf,tmh)
        self.mlp_network=mlp_network
        self.input_attention_head=input_attention_head
    
    def forward(self, h_input: torch.Tensor, h_cond_prior: torch.Tensor, input_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_attention, _ = self.input_attention_head(h_input, h_input, h_input, key_padding_mask=input_mask, need_weights=False)
        h = torch.cat((h_cond_prior, input_attention), dim=-1)
        phi = self.mlp_network(h)
        mu, logvar = torch.chunk(phi, 2, -1)
        return mu, logvar