import torch
import torch.nn as nn

from source.models.likelihood import Likelihood


class Decoder(nn.Module):
    """
    Implements a Conditional Prior distribution
    """
    def __init__(self, likelihood:str, type: str='Decoder', variance=0.1 ):
        """
        Prior initialization
        Args:
            likelihood (str): likelihood function
            type (str, optional): prior type. Defaults to 'ConditionalPrior'.
            tm=tmh+tmf
        """
        super(Decoder, self).__init__()
        self.type=type
        self.likelihood = Likelihood(likelihood, variance) # //INFO call likelihood function
    
    def forward(): 
        pass
            

    def logp(self, x: torch.Tensor, theta: torch.Tensor=None, variance: float=None) -> torch.Tensor:
        """
        Compute likelihood of data x given parameters theta and hyperparametere variance
        Args:  
            x:data  (batch_size, time steps, dim_data)
            theta:parameters (batch_size, latent_samples, time steps, dim_data)
        """
        
        logp = self.likelihood(theta, x)
        return logp




class MLPDecoder(Decoder):
    """
    Implements a MLP network for modeling conditional distribution
    """
    def __init__(self, likelihood:str, type='MLP', network=None, variance=None):
        """
        Prior initialization
        Args:
            likelihood (str): likelihood function
            type (str, optional): prior network
            network (nn.Module): prior network type
        """
        super(MLPDecoder, self).__init__(type=type, likelihood=likelihood, variance=variance)
        self.network=network
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward computation for MLP
        Args: 
            z (torch.Tensor): input with shape [Bs, T, d]
        Returns:
            mu (torch.Tensor): mu with shape [Bs, T, d]
            logvar (torch.Tensor): logvar with shape [Bs, T, d]
        """
        theta = self.network(z)
        mu, logvar = torch.chunk(theta, 2, -1)
        return mu, logvar

