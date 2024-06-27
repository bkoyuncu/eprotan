from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
import torch.nn as nn
import numpy as np


# ============= VAE submodules ============= # *from Peis

class Likelihood(nn.Module):
    """
    Implements the likelihood functions
    """
    def __init__(self, type='gaussian', variance=0.01):
        """
        Likelihood initialization
        Args:
            type (str, optional): likelihood type ('gaussian', 'categorical', 'loggaussian' or 'bernoulli'). Defaults to 'gaussian'.
            variance (float, optional): fixed variance for gaussian/loggaussian variables. Defaults to 0.1.
        """
        super(Likelihood, self).__init__()
        self.type=type
        self.variance=variance # only for Gaussian or Laplace

        print(f"Likelihood class is {self.type} with scale {self.variance}")
    
    def forward(self, theta: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of a given data under parameters theta
        Note that we assume a fixed scale for the likelihood (e.g. variance for Gaussian)
        // INFO it is for a single point so not for an image for instance where there are NxN elements to compute likelihood.
        Args:
            theta (torch.Tensor): tensor with params                (batch_size, latent_samples, time steps, dim_data)
            data (torch.Tensor): tensor with data                   (batch_size, time steps, dim_data)
        Returns:
            torch.Tensor: tensor with the log probs                 (batch_size, latent_samples, time steps, dim_data)
        """

        # ============= Gaussian ============= #
        if self.type in ['gaussian', 'loggaussian']:
            # If variance is not specified we use the predefined
            
            variance = torch.ones_like(theta) * self.variance
            # print("variance of gaussian Likelihood class is ", variance)
            logvar = torch.log(variance)
            
            # Add dimension for latent samples
            data = data.unsqueeze(1)
            mu = theta
            
            cnt = (np.log(2 * np.pi) + logvar)
            logp = -0.5 * (cnt + (data - mu) * torch.exp(-logvar) * (data - mu)) # // INFO [bs, L, T, d]

        # ============= Laplace ============= #
        elif self.type == 'laplace':
            # //  INFO theta has shape [bs, latent sample, T, dim=1] probs for x=1


            b = torch.ones_like(theta) * self.variance

            # Add dimension for latent samples
            data = data.unsqueeze(1)
            mu = theta

            logp = -torch.abs(data - mu) / b - torch.log(2 * b) # - torch.exp(-torch.abs(data - mu) / b) # // INFO [bs, L, T, d]

        # print(self.type)
        return logp

    def sample(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Returns samples or modes from the likelihood distributions
        Args:
            theta (torch.Tensor): tensor with the probs             (B, latent_samples, T, dim_data)
        Returns:
            torch.Tensor: samples                                   (B, latent_samples, T, dim_data) for Gaussian, Laplace, and Bernoulli
                          or modes                                  (B, latent_samples, T, 1)        for Categorical
        """
        # ============= Gaussian ============= #
        if self.type in ['gaussian', 'loggaussian']:
            var = torch.ones_like(theta) * self.variance
            x = reparameterize(theta, var)
    
        # ============= Laplace ============= #
        if self.type == 'laplace':
            b = torch.ones_like(theta) * self.variance
            x = reparameterize_laplace(theta, b)

        return x


def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterized samples from a Gaussian distribution
    Args:
        mu (torch.Tensor): means                    (batch_size, ..., dim)
        var (torch.Tensor): variances               (batch_size, ..., dim)
    Returns:
        torch.Tensor: samples                       (batch_size, ..., dim)
    """
    std = var**0.5
    eps = torch.randn_like(std)
    return mu + eps*std


def reparameterize_laplace(mu: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Reparameterized samples from a Laplace distribution
    Args:
        mu (torch.Tensor): means                    (batch_size, ..., dim)
        b (torch.Tensor): scale parameter           (batch_size, ..., dim)
    Returns:
        torch.Tensor: samples                       (batch_size, ..., dim)
    """
    u = torch.rand_like(mu) - 0.5 # in [-0.5, 0.5]
    return mu - b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))