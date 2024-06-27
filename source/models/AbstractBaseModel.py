from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import pytorch_lightning as pl
import torch
import torch.distributions as D
from torch import zeros
import numpy as np
from tqdm import tqdm

from source.datasets_gluonts import get_gluonts_data_loader, GluonTSDataset

class AbstractBaseModel(ABC, pl.LightningModule):
    r"""
    Abstract class to implement shared methods among models.
    """
    
    def __init__(self, model_type:str, reconstruct:bool, epsilon=1e-7):
        super(AbstractBaseModel, self).__init__()
        # ABC.__init__(self)
        self.model_type = model_type
        self.reconstruct = reconstruct
        self.epsilon = epsilon
        self.test_step_outputs = []
        
    @abstractmethod
    def _forward(self, batch:tuple, samples: int, reconstruct: bool, *args, **kwargs):
        """
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [bs, Tm, dim_c]
                T0: length of history sequence [bs, 1]
                T: total length of sequence [bs, 1]
            samples (int): number of latent samples in bottleneck to be sampled from // q //
            reconstruct (bool): If True, the loss is computed over the whole sequence, else only the forecasted part is considered.

        Returns (x, x_hat, ELBO, logpx_z, KL, mask_temporal):
            x (torch.Tensor): input data [bs, Tm, dim_x]
            x_p_hat (torch.Tensor): reconstructed data by decoding encoder latent z_q [bs, samples, Tm, dim_x]
            x_q_hat (torch.Tensor): reconstructed data by decoding conditional prior latent z_p [bs, samples, Tm, dim_x]
            ELBO (torch.Tensor): ELBO [bs, samples, Tm]
            logpx_z (torch.Tensor): logpx_z [bs, samples, Tm]
            KL (torch.Tensor): KL [bs, samples, Tm]
            mask_temporal (torch.Tensor): temporal mask to only consider the parts we are interested in [bs, Tm]
        """
        pass
    
    @abstractmethod
    def predict_and_gt_denormalized(self, batch:tuple, samples: int, reconstruct: bool, denormalize=True):
        """
        Given the batch (x,c) computes forecasting x_pred and x_gt denormalized
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [Bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [Bs, Tm, dim_c]
                T0: length of history sequence
                T: total length of sequence
            samples (int): number of samples
            reconstruct (bool): Whether to mask the history in the predictions or not
        """
        pass
    
    
    def forward(self, batch:tuple, samples: int = 1, *args, **kwargs):
        """
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [Bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [Bs, Tm, dim_c]
                T0: length of history sequence [bs, 1]
                T: total length of sequence [bs, 1]
            samples (int): number of latent samples in bottleneck to be sampled from // q //

        return (-ELBO, logp_x, KL, x, x_hat, mask_temporal)
            -ELBO (torch.Tensor): EvidenceLowerBound shape: [bs]
            logp_x (torch.Tensor): logp(x|z) shape: [bs]
            KL (torch.Tensor): KL( q(z|x)|| p(z|x)) shape: [bs]
            x (torch.Tensor): input data shape: [bs, Tm, dim_x]
            x_p_hat (torch.Tensor): predicted data shape: [bs, samples, Tm, dim_x]
            x_q_hat (torch.Tensor): predicted from the encoder shape: [bs, samples, Tm, dim_x]
            mask_temporal (torch.Tensor): temporal mask of shape [bs, Tm] for masking parts that are not used in ELBO (padding; and history if reconstruct=False)
        """
        assert samples == 1, "Only one sample is supported here, else see metric_timeseries_L or predict_and_gt_denormalized"
        
        x, x_p_hat,  x_q_hat, ELBO, logpx_z_sliced, KL_sliced, mask_temporal = self._forward(batch, samples=samples, reconstruct=self.reconstruct, *args, **kwargs)
        
        # reduce the sample dimension
        ELBO = ELBO.mean(1) # [bs, Tm]
        logpx_z_sliced = logpx_z_sliced.mean(1) # [bs, Tm]
        KL_sliced = KL_sliced.mean(1) # [bs, Tm]

        #taking mean over the part we are interested in
        ELBO_mean = ELBO.sum(-1) / (~mask_temporal).sum(-1) # [bs]
        logpx_z_sliced_mean = logpx_z_sliced.sum(-1) / (~mask_temporal).sum(-1) # [bs]
        KL_sliced_mean = KL_sliced.sum(-1) / (~mask_temporal).sum(-1) # [bs]

        return -ELBO_mean, logpx_z_sliced_mean, KL_sliced_mean, x, x_p_hat, x_q_hat, mask_temporal
    
    
    def metric_timeseries_L(self, batch: tuple, batch_idx: int, samples: int=5, window_len:int=5, stride:int=5, *args, **kwargs):
        """Compute test metrics
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [Bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [Bs, Tm, dim_c]
                T0: length of history sequence
                T: total length of sequence
            samples (int): number of samples to be drawn from the bottleneck
            window_len (int): length of window for ELBO, RMSE and cross correlation
            stride (int): stride of window
            reconstruct (bool): if True, the metrics are computed over the whole sequence, else only the forecasted part is considered
        Returns:
            Dictionary with metrics.
        """
        # do multiple forward passes for ProTran, as it samples in each layer and hence, we cannot just sample from the bottleneck
        if self.model_type == "protran":
            x_p_hat_samples = []
            ELBO_samples = []
            for _ in range(samples):
                x, x_p_hat, x_q_hat, ELBO, logpx_z_sliced, KL_sliced, mask_temporal = self._forward(batch, samples=1, reconstruct=self.reconstruct, *args, **kwargs)
                x_p_hat_samples.append(x_p_hat)
                ELBO_samples.append(ELBO)
            # stack the samples
            x_p_hat = torch.concatenate(x_p_hat_samples, dim=1) # [bs, samples, Tm, dim_x]
            ELBO = torch.concatenate(ELBO_samples, dim=1) # [bs, samples, Tm]
        else:
            x, x_p_hat,  x_q_hat, ELBO, logpx_z_sliced, KL_sliced, mask_temporal = self._forward(batch, samples=samples, reconstruct=self.reconstruct, *args, **kwargs)
        
        # denormalize the data
        x = self.denormalize_x(x)
        x_p_hat = self.denormalize_x(x_p_hat)
        
        x_L = x[:,None].repeat(1,samples,1,1) #[bs, samples, Tm, dim_x]
        
        mask_temporal_L = mask_temporal.unsqueeze(1) # [bs,1,T] it broadcast to L
        metric_elbo = self.elbo_temporal_window(ELBO,mask_temporal_L,window_len, stride) # List[List[Tensor]], len(list_1) = bs * L; len(list_2) = T/window_len
        metric_rmse = self.prediction_metric_rmse_temporal(x_p_hat, x_L, mask_temporal_L,window_len, stride) # List[List[Tensor]], len(list_1) = bs * L; len(list_2) = T/window_len
        metric_cross_corr = self.prediction_metric_cross_corr_temporal(x_p_hat, x_L, mask_temporal_L,window_len, stride) # # List[List[Tensor]], len(list_1) = bs * L; len(list_2) = T/window_len

        return {'metric_elbo': metric_elbo, 'metric_rmse': metric_rmse, 'metric_cross_cor': metric_cross_corr}
        
    
    def training_step(self, batch: tuple, batch_idx: int, samples: int = 1, *args, **kwargs) -> torch.Tensor:
        """
        Returns the loss (negative ELBO) for the minimization
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [Bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [Bs, Tm, dim_c]
                T0: length of history sequence
                T: total length of sequence
            batch_idx (int): index of batch (required by pytorch-lightning)
    
        Returns:
            torch.Tensor: mean loss (negative ELBO)                                  
        """
        loss, logpx_z, kl, x, x_p_hat, x_q_hat, mask_temporal = self(batch, samples=samples, *args, **kwargs)
        
        with torch.no_grad():
            # denormalize the data
            x = self.denormalize_x(x)
            x_p_hat = self.denormalize_x(x_p_hat)
            metric_rmse = self.prediction_metric_rmse_temporal(theta_x=x_p_hat, x=x, mask=mask_temporal, temporal=False) # [bs]
            # metric_cross_corr = self.prediction_metric_cross_corr_temporal(theta_x=x_p_hat, x=x, mask=mask_temporal, temporal=False) # [bs]
        

        if (torch.isnan(loss).any() or torch.isinf(loss).any())  == True:
            print("nan is caught")
            pass
        
        loss = loss.mean() #negative elbo to minimize

        # ELBO/ KL / logpx_z are on the normalized data, RMSE and CrossCorr are on the denormalized data
        self.log('ELBO', -loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('KL', kl.mean(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('logpx_z', logpx_z.mean(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('metric_rmse', metric_rmse.mean(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('metric_cross_corr', metric_cross_corr.mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        return loss
    
        
    def test_step(self, batch: tuple, batch_idx: int, samples: int = 100, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [Bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [Bs, Tm, dim_c]
                T0: length of history sequence [bs, 1]
                T: total length of sequence [bs, 1]
            batch_idx (int): index of batch (required by pytorch-lightning)

        return Avg. of (ELBO, KL, RMSE, CrossCorr)
            elbo_test (torch.Tensor): EvidenceLowerBound, shape: [1]
            KL_slicde_mean (torch.Tensor): KL( q(z|x)|| p(z|x)), shape: [1]
            metric_rmse (torch.Tensor): RMSE, shape: [1]
            metric_cross_cor shape (torch.Tensor): CrossCorr, shape: [1]
        """
        ELBO_mean_samples = []
        KL_sliced_mean_samples = []
        x_p_hat_samples = []
        
        # create multiple samples to estimate the metrics
        # since ProTran samples in each layer, we need to do multiple forward passes
        for _ in tqdm(range(samples), desc=f"Generating {samples} test samples"):
            ELBO_mean, _, KL_sliced_mean, x, x_p_hat, x_q_hat, _ = self(batch, samples=1, *args, **kwargs)
            ELBO_mean_samples.append(ELBO_mean)
            KL_sliced_mean_samples.append(KL_sliced_mean)
            x_p_hat_samples.append(x_p_hat)
        
        ELBO_mean = torch.stack(ELBO_mean_samples, dim=1).mean(1) # [bs]
        KL_sliced_mean = torch.stack(KL_sliced_mean_samples, dim=1).mean(1) # [bs]
        x_p_hat = torch.concatenate(x_p_hat_samples, dim=1).mean(1, keepdim=True) # [bs, 1, Tm, dim_x] 
        # here we ignore the masking that is induced by the model and rather calculate the metrics for the history and forecasted part
        # note that reconstruct only affects the loss, not the other outputs of the forward pass
        _, _, T0, T = batch
        bs, Tm, _ = x.shape
        full_mask = self.create_temporal_mask(bs, Tm, T0, T, reconstruct=True)
        forecasting_mask = self.create_temporal_mask(bs, Tm, T0, T, reconstruct=False)
        history_mask = self.create_history_mask(T0, Tm)
        
        # denormalize the data
        x = self.denormalize_x(x)
        x_p_hat = self.denormalize_x(x_p_hat)
        rmse = self.prediction_metric_rmse_temporal(theta_x=x_p_hat, x=x, mask=full_mask, temporal=False) # [bs]
        rmse_hist = self.prediction_metric_rmse_temporal(theta_x=x_p_hat, x=x, mask=history_mask, temporal=False) # [bs]
        rmse_forecast = self.prediction_metric_rmse_temporal(theta_x=x_p_hat, x=x, mask=forecasting_mask, temporal=False) # [bs]
        cross_cor = self.prediction_metric_cross_corr_temporal(theta_x=x_p_hat, x=x, mask=full_mask, temporal=False) # [bs]
        cross_cor_hist = self.prediction_metric_cross_corr_temporal(theta_x=x_p_hat, x=x, mask=history_mask, temporal=False) # [bs]
        cross_cor_forecast = self.prediction_metric_cross_corr_temporal(theta_x=x_p_hat, x=x, mask=forecasting_mask, temporal=False) # [bs]  
        
        output_test = {
            'elbo_test': -ELBO_mean, 
            'KL_sliced_mean': KL_sliced_mean, 
            'metric_rmse_hist': rmse_hist,
            'metric_rmse_forecast': rmse_forecast,
            'metric_cross_cor_hist': cross_cor_hist,
            'metric_cross_cor_forecast': cross_cor_forecast,
            'metric_rmse': rmse,
            'metric_cross_cor': cross_cor,
            'test_samples': samples
        }
        self.test_step_outputs.append(output_test)
        return output_test
    
    
    def on_test_epoch_end(self, save=True) -> None:
        """
        Compute mean epoch test metrics from the batches
        """
        #outputs are formed by computing the test step for each batch
        
        elbo_mean =  torch.mean(torch.hstack([o['elbo_test'] for o in self.test_step_outputs]))
        KL_mean =  torch.mean(torch.hstack([o['KL_sliced_mean'] for o in self.test_step_outputs]))
        metric_rmse_mean =  torch.mean(torch.hstack([o['metric_rmse'] for o in self.test_step_outputs]))
        metric_cross_cor_mean =  torch.mean(torch.hstack([o['metric_cross_cor'] for o in self.test_step_outputs]))
        
        metric_rmse_hist_mean = torch.mean(torch.hstack([o['metric_rmse_hist'] for o in self.test_step_outputs]))
        metric_rmse_forecast_mean = torch.mean(torch.hstack([o['metric_rmse_forecast'] for o in self.test_step_outputs]))
        metric_cross_cor_hist_mean = torch.mean(torch.hstack([o['metric_cross_cor_hist'] for o in self.test_step_outputs]))
        metric_cross_cor_forecast_mean = torch.mean(torch.hstack([o['metric_cross_cor_forecast'] for o in self.test_step_outputs]))
        
        elbo_min = torch.hstack([o['elbo_test'] for o in self.test_step_outputs]).min()
        elbo_max = torch.hstack([o['elbo_test'] for o in self.test_step_outputs]).max()
        rmse_min = torch.hstack([o['metric_rmse'] for o in self.test_step_outputs]).min()
        rmse_max = torch.hstack([o['metric_rmse'] for o in self.test_step_outputs]).max()
        cross_cor_min = torch.hstack([o['metric_cross_cor'] for o in self.test_step_outputs]).min()
        cross_cor_max = torch.hstack([o['metric_cross_cor'] for o in self.test_step_outputs]).max()

        metrics = {
            'test_elbo_mean': elbo_mean, 
            'test_KL_mean': KL_mean,
            'test_metric_rmse_mean_hist': metric_rmse_hist_mean,
            'test_metric_rmse_mean_forecast': metric_rmse_forecast_mean,
            'test_metric_rmse_mean': metric_rmse_mean, 
            'test_metric_cross_cor_mean_hist': metric_cross_cor_hist_mean,
            'test_metric_cross_cor_mean_forecast': metric_cross_cor_forecast_mean,
            'test_metric_cross_cor_mean': metric_cross_cor_mean,
            'test_elbo_min': elbo_min,
            'test_elbo_max': elbo_max,
            'test_rmse_min': rmse_min,
            'test_rmse_max': rmse_max,
            'test_cross_cor_min': cross_cor_min,
            'test_cross_cor_max': cross_cor_max,
            'test_samples': self.test_step_outputs[0]['test_samples']
        }
        self.log_dict(metrics, logger=True)
        
        
        print(f"ELBO: {elbo_mean:.2f} (min: {elbo_min:.2f}, max: {elbo_max:.2f})")
        print(f"RMSE: {metric_rmse_mean:.2f} (min: {rmse_min:.2f}, max: {rmse_max:.2f})")
        print(f"CrossCorr: {metric_cross_cor_mean:.2f} (min: {cross_cor_min:.2f}, max: {cross_cor_max:.2f})")
        print(f"RMSE_hist: {metric_rmse_hist_mean:.2f}, RMSE_forecast: {metric_rmse_forecast_mean:.2f}")
        print(f"CrossCorr_hist: {metric_cross_cor_hist_mean:.2f}, CrossCorr_forecast: {metric_cross_cor_forecast_mean:.2f}")
        
        self.test_step_outputs = []
        
        # log number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        self.log('n_params', n_params, logger=True)
        return metrics


    def on_after_backward(self):
        metric_gradients = self._compute_grad_norm()
        self.log('metric_gradients', metric_gradients, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    
    def initialize_buffer(self):
        self.register_buffer('means_x', torch.Tensor(zeros((1, self.tm, self.dim_x))))
        self.register_buffer('stds_x', torch.Tensor(zeros((1, self.tm, self.dim_x))))
        self.register_buffer('means_c', torch.Tensor(zeros((1, self.tm, self.dim_c-self.dim_c_lag))))
        self.register_buffer('stds_c', torch.Tensor(zeros((1, self.tm, self.dim_c-self.dim_c_lag))))
        print("Buffer initialized.")
    
    
    def train_dataloader(self):
            
        if self.dataset_name in ['electricity_nips', 'solar_nips', 'wiki2000_nips']:
            window_offset = self.tmh + self.tmf
            if self.dataset_name == 'wiki2000_nips':
                window_offset = 4
            loader = get_gluonts_data_loader(
                self.dataset_name,
                split='train',
                prediction_length=self.tmf,
                history_length=self.tmh,
                window_offset=window_offset,
                random_offset=True,
                batch_size=self.batch_size,
                num_workers=1, # change to more if needed
                shuffling=True,
                persistent_workers=True,
            )
            self.register_buffer('means_x', torch.Tensor(loader.dataset.data_scaler.means).repeat(1,self.tm,1).to(self.device))
            self.register_buffer('stds_x', torch.Tensor(loader.dataset.data_scaler.stds).repeat(1,self.tm,1).to(self.device))
            self.register_buffer('means_c', torch.Tensor(loader.dataset.covariates_scaler.means).repeat(1,self.tm,1).to(self.device))
            self.register_buffer('stds_c', torch.Tensor(loader.dataset.covariates_scaler.stds).repeat(1,self.tm,1).to(self.device))
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        return loader

    
    def test_dataloader(self):
        if self.dataset_name in ['electricity_nips', 'solar_nips', 'wiki2000_nips']:
            loader = get_gluonts_data_loader(
                self.dataset_name,
                split='test',
                prediction_length=self.tmf,
                history_length=self.tmh,
                window_offset=None,
                random_offset=False,
                batch_size=self.batch_size,
                num_workers=1,
                shuffling=False,
                persistent_workers=True,
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")
        return loader
    
      
    def build_x_hat(self, theta_x:torch.Tensor):
        """
        Compute x_hat given parameters theta_x of shape [batch_size, samples, Tm, dim_x]
        Args:
            theta_x parameters of p(x) of shape [batch_size, samples, Tm, dim_x]
        Returns:
            x_hat (torch.Tensor): shape [batch_size, samples, Tm, dim_x]
        """
        if self.likelihood_x=="gaussian":
            xu = theta_x
        elif self.likelihood_x=="laplace":
            xu = theta_x
        x_hat = xu
        
        return x_hat
    

    def sample_z(self, mu:torch.Tensor, var:torch.Tensor, samples=1):
        r"""
        Args:
            mu (torch.Tensor): mean of latent variable z at the last layer [batch_size, Tm, d_model]
            var (torch.Tensor): variance of latent variable z at the last layer [batch_size, Tm, d_model]
            samples (int): number of samples
        Returns:
            z (torch.Tensor): latent variable z [batch_size, samples, Tm, d_model] sample from p(z|x,c)
        """
        assert mu.shape == var.shape, "mu and var should have the same shape"
        assert len(mu.shape) == 3, "mu should have 3 dimensions"
        z = D.Normal(mu, torch.sqrt(var)).rsample((samples,)) # [samples, batch_size, Tm, d_model]
        z = z.permute(1,0,2,3) # [batch_size, samples, Tm, d_model]
        return z
    
    
    def denormalize_x(self, sequences_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data with precomputed mean and std
        Args:
            xn (torch.Tensor): input normalized data      (batch_size, ..., dim_x)
        Returns:
            torch.Tensor: denormalized data      (batch_size, ..., dim_x)
        """
        
        sequences_denormalized = sequences_normalized * (self.stds_x + self.epsilon) + self.means_x
        return sequences_denormalized

    
    def denormalize_c(self, sequences_normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data with precomputed mean and std
        Args:
            cn (torch.Tensor): input normalized data      (batch_size, ..., dim_c)
        Returns:
            torch.Tensor: denormalized data      (batch_size, ..., dim_c)
        """
        # assume that the first dimensions are c_feat
        sequence_lags, sequence_features = torch.split(sequences_normalized, [self.dim_c - self.dim_c_lag, self.dim_c_lag], dim=-1)
        
        sequences_denormalized = sequence_features * (self.stds_c + self.epsilon) + self.means_c    
        sequences_denormalized = torch.cat([sequence_lags, sequences_denormalized], dim=-1)    
        return sequences_denormalized

    
    def elbo_temporal_window(self, elbo: torch.Tensor, mask:torch.Tensor, window_len=5, stride=3) -> List[torch.Tensor]:
        """
        Given ELBO (elbo) and mask computes the mean of ELBO in a temporal window.
        Args:
            ELBO [bs, samples, T]
            mask [bs, 1, T]
        returns:
            returns a list
        """
        if len(elbo.shape) == 3:
            mask = mask.repeat(1,elbo.shape[1],1) #torch.Size([64, 5, 40]), why reapeat if we only reapeat 1 time in each dim
            bs, samples, Tm = elbo.shape
        else:  
            print("elbo should have 3 dimensions")
            raise NotImplementedError
        
        list_elbo = list(elbo.reshape(bs*samples, Tm))
        list_elbo_masked = []
        mask_temporal_L_reshape = mask.reshape(bs*samples,Tm) #[bs*L, Tm]
        
        for i in range(len(list_elbo)):
            list_elbo_masked.append([torch.mean(list_elbo[i][~mask_temporal_L_reshape[i]][:x])
            for x in range(window_len, len(list_elbo[i][~mask_temporal_L_reshape[i]])+window_len,window_len)])
        
        return list_elbo_masked
    

    def prediction_metric_rmse_temporal(self, theta_x: torch.Tensor, x: torch.Tensor, mask:torch.Tensor, window_len=5, stride=3, temporal=True)-> torch.Tensor:
        """
        Given theta_x (x_pred), x (x_gt) computes RMSE metric after masking them with temporal mask
        Args:
            theta_x [bs, L, T, dim_x]
            x [bs, L, T, dim_x] or x [bs, T, dim_x] 
            mask [bs, L, T]
            temporal : bool
        returns:
            if temporal metric returns a list of convolution results
            if not temporal  
        """
        # theta_x_denorm = self.denormalize_x(theta_x)
        # x_denorm = self.denormalize_x(x)
        # x is already denormalized in training/test step
        theta_x_denorm = theta_x
        x_denorm = x

        if len(x.shape) == 3 and temporal==True:
            print("x should have 4 dimensions for temporal case")
            raise NotImplementedError
        
        if len(x.shape) == 4 and temporal==False:
            print("x should have 3 dimensions for non-temporal case")
            raise NotImplementedError

        if len(x.shape) == 4 and temporal==True:

            mask = mask.unsqueeze(-1).repeat(1,x.shape[1],1,x.shape[-1]) #[bs,L,T,dim_x]
            bs, samples, Tm, dim_x = x.shape

            metric = torch.nn.MSELoss(reduction="none")(theta_x_denorm.masked_fill(mask, 0), x_denorm.masked_fill(mask, 0)) #[bs, L, T, dim_x]
            list_se = list(metric.reshape(bs*samples, Tm, dim_x))
            mask_temporal_L_reshape = mask.reshape(bs*samples,Tm,dim_x)
            list_se_masked = []

            for i in range(len(list_se)):
                cx = list_se[i][~mask_temporal_L_reshape[i]] #current x
                list_se_masked.append([torch.sqrt(torch.mean(cx.reshape(-1,dim_x)[:k], dim=0)) for k in range(window_len, len(cx.reshape(-1,dim_x))+window_len, window_len)])
            
            return(list_se_masked)
        
        if len(x.shape) == 3 and temporal==False:
            mask = mask.squeeze(1) #[bs,T]
            mask = mask.unsqueeze(-1).repeat(1,1,x.shape[-1]) #[bs,T,dim_x]
            bs, Tm, dim_x = x.shape
            theta_x_denorm = theta_x_denorm.squeeze(1) #[bs,1,T,dim_x] -> [bs,T,dim_x]
            #Squared loss summed over T and dim_x
            se = ((theta_x_denorm.masked_fill(mask, 0) - x_denorm.masked_fill(mask, 0))**2).sum((1,2)) #[bs]
            #Gets mean over T and dim_x
            metric = torch.sqrt(se/(~mask).sum((1,2))) #[bs]
            return metric

    
    def prediction_metric_cross_corr_temporal(self, theta_x: torch.Tensor, x: torch.Tensor, mask:torch.Tensor, window_len=5, stride=3, temporal=True)-> torch.Tensor:
        """
        Given theta_x (x_pred), x (x_gt) computes CROSS CORR metric after masking them with temporal mask
        Args:
            theta_x [bs, L, T, dim_x]
            x [bs, L, T, dim_x]
            mask [bs, L, T]
        returns:
            metric [bs, L, T, dim_x]
        """
        # theta_x_denorm = self.denormalize_x(theta_x)
        # x_denorm = self.denormalize_x(x)
        # x is already denormalized in training/test step
        theta_x_denorm = theta_x
        x_denorm = x

        if len(x.shape) == 3 and temporal==True:
            print("x should have 4 dimensions for temporal case")
            raise NotImplementedError
        
        if len(x.shape) == 4 and temporal==False:
            print("x should have 3 dimensions for non-temporal case")
            raise NotImplementedError

        if len(x.shape) == 4 and temporal==True:
            mask = mask.unsqueeze(-1).repeat(1,x.shape[1],1,x.shape[-1]) #[bs,L,T,d]
            bs, L,  T, dim_x = x.shape
            list1 = list(theta_x_denorm.reshape(-1,T,dim_x)) # list of BS*L: each jas tensor [T,d]
            list2 = list(x_denorm.reshape(-1,T,dim_x)) # list of BS*L: each jas tensor [T,d]
            mask = mask.reshape(-1,T,dim_x) # BS*L,T,d

            list1_masked, list2_masked = cross_cor_temporal(list1,list2,mask,dim_x,window_len) #each has bs*L,

            cross_cor = []

            for i,j in zip(list1_masked,list2_masked): #each has list1(list2([window_len,dim_x])) list1 len bs*L list2 len #windows
                _per_batch =[]
                for a,b in zip(i,j):
                    metric_all = []
                    for dim_x_s in range(dim_x):
                        metric = np.correlate(a[:,dim_x_s].flatten().cpu().detach().numpy(),b[:,dim_x_s].flatten().cpu().detach().numpy(),mode="same").max()/a[:,dim_x_s].shape[0]
                        metric_all.append(metric)
                    _per_batch.append(np.stack(metric_all,axis=0))
                cross_cor.append(_per_batch)
            return (cross_cor)

        if len(x.shape) == 3 and temporal==False:
            mask = mask.squeeze(1) #looks [bs,T] -> [bs,T]
            mask = mask.unsqueeze(-1).repeat(1,1,x.shape[-1])
            bs, T, dim_x = x.shape

            theta_x_denorm = theta_x_denorm.squeeze(1)
            
            list1 = list(theta_x_denorm)
            list2 = list(x_denorm)

            list1_masked, list2_masked = cross_cor_temporal(list1,list2,mask,dim_x,None)

            cross_cor = []

            for i,j in zip(list1_masked,list2_masked):
                #takes the max value of the correlation between x, theta_x
                metric = np.correlate(i.flatten().cpu().detach().numpy(),j.flatten().cpu().detach().numpy(),mode="same").max()/i.shape[0]
                cross_cor.append(metric)

            cross_cor_array = np.array(cross_cor,  dtype=np.float32).reshape(bs) #[bs]

            return torch.Tensor(cross_cor_array)
        
        
    def create_temporal_mask(self, bs: int, Tm: int, T0: torch.Tensor, T: torch.Tensor, reconstruct: bool=False)-> torch.Tensor:
        """
        Args:
            bs (int): bs
            Tm (int): maximum sequence length
            T0 (torch.Tensor): length of history sequence (batch_size, 1)
            T (torch.Tensor): length of forecasting sequence (history included) (batch_size, 1)
            reconstruct (boolean): choice if reconstructing data or not
        Returns:
            mask_temporal (torch.Tensor): boolean tensor for slicing elbo (batch_size, Tm) #Trues are dropped
        """
        
        if reconstruct:
            pair_index = torch.cat((T,torch.ones_like(T)*Tm),dim=1).to(self.device)
            cols = torch.LongTensor (range (Tm)).repeat (bs, 1).to(self.device)
            beg = pair_index[:,0].unsqueeze (1).repeat (1, Tm) #(lots of T)
            end = pair_index[:,1].unsqueeze (1).repeat (1, Tm) #(lots of Tm)
            mask_temporal = cols.ge (beg) & cols.lt (end) # =>T and <Tm

        else: 
            pair_index = torch.cat((T0,T),dim=1).to(self.device)
            cols = torch.LongTensor (range (Tm)).repeat (bs, 1).to(self.device)
            beg = pair_index[:,0].unsqueeze (1).repeat (1, Tm)
            end = pair_index[:,1].unsqueeze (1).repeat (1, Tm)
            mask = cols.ge (beg) & cols.lt (end) # =>T0 and <T
            mask_temporal = torch.logical_not(mask) # <T0 and >=T
            

        return mask_temporal #[10,14] #T is included, Tm is not included
    

    def create_history_mask(self, T0:torch.Tensor, Tm:int) -> torch.Tensor:
        r"""
        Args:
            T0 (torch.Tensor): length of history sequence (batch_size, 1)
            Tm (int): length of whole sequence (Hist+Forecasting+Padding)
        Returns:
            mask (torch.Tensor): boolean tensor that is False for the history part and True for the forecasting part and padding (batch_size, Tm)
        """
        # uses some 'wild' broadcasting
        mask = torch.arange(Tm).to(self.device).unsqueeze(0) >= T0 # [bs, Tm]
        return mask
        
        
    def create_prior_input_mask(self, bs: int, Tm: int, T0: torch.Tensor, T: torch.Tensor, x_dim: int)-> torch.Tensor:
        """
        Creates mask for creating the input for conditional prior network
        Args:
            bs (int): bs
            Tm (int): maximum sequence length
            x_dim (int): dimension of x data
            T0 (torch.Tensor): length of history sequence (batch_size, 1)
            T (torch.Tensor): total length of sequence (batch_size, 1) (history + forecasting)
            
        Returns:
            mask_repeat (torch.Tensor): boolean tensor for slicing elbo (batch_size, Tm)
        """
        pair_index = torch.cat((T0,T),dim=1).to(self.device)
        cols = torch.LongTensor (range (Tm)).repeat (bs, 1).to(self.device)
        beg = pair_index[:,0].unsqueeze (1).repeat (1, Tm)
        end = pair_index[:,1].unsqueeze (1).repeat (1, Tm)
        mask = cols.ge (beg) & cols.lt (end)
        mask_repeat = mask[:,:,None].repeat((1,1,x_dim)) #[10,14,11]
         
        return mask_repeat #[10,14,11] (T0 included T is not included)
    
    def prediction_metric_crps_batch(self, batch:tuple, samples: int) -> List[List[List[float]]]:
        """
        Get the CRPS metric for the forecasted samples.

        Args:
            batch (tuple): Input data.
            samples (int): Number of samples to be drawn from the bottleneck and estimate the CDF in CRPS with.
            
        Returns:
            List[List[List[float]]]: List of CRPS values for each batch sample, each dimension of x and at every time step.
                First dimension is the batch size, second dimension is dim_x, and third dimension is the time step.
        """
        x, _, T0, T = batch
        bs, Tm, dim_x = x.shape
        
        _, x_pred_l, x = self.predict_and_gt_denormalized(batch, samples, reconstruct=self.reconstruct) # [bs, L, Tm, dim_x], [bs, Tm, dim_x]
        batch_crps = []
        for i in range(bs):
            crps_dims = []
            for dim in tqdm(range(dim_x), desc=f"Calculating CRPS for sample {i}"):
                x_pred = x_pred_l[i,:,:,dim] # [L, T]
                x_true = x[i,:,dim] # [T]
                # only consider the forecasted part
                x_pred = x_pred[:,T0[i]:T[i]] # [L, T - T0]
                x_true = x_true[T0[i]:T[i]] # [T - T0]
                crps_list = crps_from_tensor_sequence(x_pred, x_true) # List of length [T - T0]
                crps_dims.append(crps_list)
            
            # crps_dims is now [dim_x, T - T0]
            batch_crps.append(crps_dims)
            
        # batch_crps is now [bs, dim_x, T - T0]   
        return batch_crps
    
    
    def prediction_metric_crps_sum_batch(self, batch:tuple, samples: int) -> List[List[float]]:
        """
        Get the CRPS metric for forecasted samples, where the ground truth and predicted samples are first summed over the dimensions of x.

        Args:
            batch (tuple): Input data.
            samples (int): Number of samples to be drawn from the bottleneck and estimate the CDF in CRPS with.

        Returns:
            List[List[float]]: List of (summed) CRPS values for each batch sample and at every time step.
                First dimension is the batch size, and second dimension is the time step.
        """
        x, _, T0, T = batch
        bs, Tm, dim_x = x.shape
        
        _, x_pred_l, x = self.predict_and_gt_denormalized(batch, samples, reconstruct=self.reconstruct) # [bs, L, Tm, dim_x], [bs, Tm, dim_x]
        batch_crps_sum = []
        for i in tqdm(range(bs), desc="Calculating CRPS sum"):
            # sum over the dimensions of x
            x_pred = x_pred_l[i].sum(dim=-1) # [L, T]
            x_true = x[i].sum(dim=-1) # [T]
            # only consider the forecasted part
            x_pred = x_pred[:,T0[i]:T[i]] # [L, T - T0]
            x_true = x_true[T0[i]:T[i]] # [T - T0]
            crps_list = crps_from_tensor_sequence(x_pred, x_true) # List of length [T - T0]
            batch_crps_sum.append(crps_list)
        
        # batch_crps_sum is now [bs, T - T0]
        return batch_crps_sum

    
    def aggregate_crps_dataset(self, data_loader: torch.utils.data.DataLoader, samples: int) -> Tuple[float, float, float, float]:
        """Aggregate the CRPS by calculating it for each timeseries and each time step separately, then calculating the mean over the sequences, dimensions and time steps.
        Assumes that each timeseries contains the same amount of predictions!

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the dataset.
            samples (int): Number of samples to be drawn from the bottleneck and estimate the CDF in CRPS with.
            
        Returns:
            float: Mean CRPS over the dataset, the dimensions and the time steps.
            float: Mean CRPS_Sum over the dataset and the time steps.
            float: Normalized mean CRPS over the dataset, the dimensions and the time steps. The normalization is done by dividing by the mean of the absolute values of the true forecasting sequence. Note that this mean is over the dataset, the dimensions and the time steps.
            float: Normalized mean CRPS_Sum over the dataset and the time steps. The normalization is done by dividing by the mean of the absolute SUMMED values of the true forecasting sequence. Note that this mean is over the dataset and the time steps.
        """
        crps_list_dataset = []
        crps_sum_list_dataset = []
        
        target_aggregate = []
        summed_target_aggregate = []
        for batch in data_loader:
            x, _, _, _ = batch
            x = x.to(self.device)
            x = self.denormalize_x(x)
            target_aggregate.append(x.flatten().cpu().detach().numpy())
            summed_target_aggregate.append(x.sum(dim=-1).flatten().cpu().detach().numpy())
            
            batch_crps = self.prediction_metric_crps_batch(batch, samples)
            crps_list_dataset.extend(batch_crps)
            
            batch_crps_sum = self.prediction_metric_crps_sum_batch(batch, samples)
            crps_sum_list_dataset.extend(batch_crps_sum)
        
        # mean over the dataset (N), the dimensions (dim_x) and the time steps (T - T0)      
        crps_list_dataset = np.array(crps_list_dataset)
        crps_mean = crps_list_dataset.mean()
        target_aggregate = np.concatenate(target_aggregate)
        normalized_crps_mean = crps_mean / np.abs(target_aggregate).mean()
        
        # Crps_sum mean over the dataset (N) and the time steps (T - T0)
        crps_sum_list_dataset = np.array(crps_sum_list_dataset)
        crps_sum_mean = crps_sum_list_dataset.mean()
        summed_target_aggregate = np.concatenate(summed_target_aggregate)
        normalized_crps_sum_mean = crps_sum_mean / np.abs(summed_target_aggregate).mean()
        
        return crps_mean, crps_sum_mean, normalized_crps_mean, normalized_crps_sum_mean
    
    
    def validate_gluonts_metrics(
        self,
        dataset: GluonTSDataset,
        samples: int,
        train: bool = False,
    ) -> Dict[str, float]:
        """ 
        Calculate the metrics with gluonts library to validate our calculations.
        
        Args:
            dataset (GluonTSDataset): GluonTSDataset object.
            samples (int): Number of samples to be drawn from the bottleneck and estimate the CDF in CRPS with.
            train (bool): Whether to use the training or test set.
            
        Returns:
            Dict[str, float]: Dictionary of the calculated metrics.
        """
        assert dataset.dataset_name == self.dataset_name, "Dataset name does not match the model's dataset name."
        from gluonts.evaluation import MultivariateEvaluator
        from gluonts.model.forecast import SampleForecast
        self.eval()

        evaluator = MultivariateEvaluator(
            quantiles=(np.arange(100) / 100.0)[1:],
            target_agg_funcs={"sum": np.sum},
        )
        if train:
            kind = "train"
        else:
            kind = "test"
        unnormalized_target_iterator, prediction_start_dates = dataset.create_pandas_evaluation_iterator(kind)
        
        # Create forecasts
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        num_seqs = len(data_loader)
        with torch.no_grad():
            forecasts = []
            for i, batch in enumerate(data_loader):
                x, _, T0, T = batch
                x_mu_pred, x_samples_pred, x = self.predict_and_gt_denormalized(batch, samples=samples, reconstruct=self.reconstruct)
                # only consider the forecasted part
                x_samples_pred = x_samples_pred[0, :, T0[0]:T[0]] # [L, T - T0, dim_x]
                if i == 0:
                    assert self.tmf == x_samples_pred.shape[1], "The forecast length does not match the model's forecast length."
                forecast = SampleForecast(samples=x_samples_pred.cpu().numpy(), start_date=prediction_start_dates[i])
                forecasts.append(forecast)
             
        
        # Calculate metrics
        agg_metric, ts_wise_metrics = evaluator(unnormalized_target_iterator, iter(forecasts))
        metrics = {
            "CRPS_normalized": agg_metric.get("mean_wQuantileLoss", float("nan")),             # same as agg_metric['mean_absolute_QuantileLoss'] / agg_metric['abs_target_sum']
            "CRPS": agg_metric.get("mean_absolute_QuantileLoss", float("nan")) / (self.dim_x * self.tmf * num_seqs),
            "ND": agg_metric.get("ND", float("nan")),
            "NRMSE": agg_metric.get("NRMSE", float("nan")),
            "MSE": agg_metric.get("MSE", float("nan")),
            "RMSE": agg_metric.get("RMSE", float("nan")),
            "CRPS_sum_normalized": agg_metric.get("m_sum_mean_wQuantileLoss", float("nan")),   # same as agg_metric['m_sum_mean_absolute_QuantileLoss'] / agg_metric['m_sum_abs_target_sum']
            "CRPS_sum": agg_metric.get("m_sum_mean_absolute_QuantileLoss", float("nan")) / (self.tmf * num_seqs),
            "ND_sum": agg_metric.get("m_sum_ND", float("nan")),
            "NRMSE_sum": agg_metric.get("m_sum_NRMSE", float("nan")),
            "MSE_sum": agg_metric.get("m_sum_MSE", float("nan")),
            "RMSE_sum": agg_metric.get("m_sum_RMSE", float("nan"))
        }
        return metrics
            

    def _compute_grad_norm(self):
        grads = [param.grad.detach().flatten() for param in self.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm()
        return norm.item()


def crps_from_tensor_sequence(x_samples: torch.Tensor, x_true: torch.Tensor) -> List[float]:
    """Returns the list of CRPS values at each time step for a sequence of forecast samples and true values.

    Args:
        x_samples (torch.Tensor): Samples from the forecast model with shape [samples, pred_len]
        x_true (torch.Tensor): True values with shape [pred_len]

    Returns:
        List[float]: List of CRPS values at each time step.
    """
    x_samples = x_samples.cpu().detach().numpy()
    x_true = x_true.cpu().detach().numpy()
    crps_list = []
    for i in range(x_samples.shape[1]):
        crps_list.append(crps(x_samples[:,i], x_true[i]))
    return crps_list


def crps(x_samples: np.ndarray, x_true: float) -> float:
    """Calculates the Continuous Ranked Probability Score (CRPS) for a single forecast.
    To estimate the CDF of the forecast, the empirical CDF is calculated using the forecast samples. 

    Args:
        x_samples (np.ndarray): 1-D array of forecast samples from the model.
        x_true (float): The true value of the target variable.

    Returns:
        float: The CRPS of the forecast.
    """
    # ensure float
    x_true = float(x_true)
    x_samples = x_samples.astype(float)
    
    num_samples = len(x_samples)
    x_samples = np.sort(x_samples)
    y_cdf = np.arange(0, num_samples + 1) / num_samples # [0, 1/n, 2/n, ..., 1], len = n+1
    crps = 0
    
    if x_true >= x_samples[-1]:
        x_samples = np.append(x_samples, x_true) # [x0, x1, ..., x(n-1), x_true], len = n+1
        x_upper = x_samples[1:] # [x1, x2, ..., x(n-1), x_true], len = n
        x_lower = x_samples[:-1] # [x0, x1, ..., x(n-1)], len = n
        
        # x[1] - x[0] * (1/n)^2 + x[2] - x[1] * (2/n)^2 + ... + x[n-1] - x[n-2] * ((n-1)/n)^2 + x_true - x[n-1] * (n/n)^2
        crps = ((x_upper - x_lower) * y_cdf[1:]**2).sum() 
        
    elif x_true <= x_samples[0]:
        x_samples = np.insert(x_samples, 0, x_true) # [x_true, x0, x1, ..., x(n-1)], len = n+1
        x_upper = x_samples[1:] # [x0, x1, ..., x(n-1)], len = n
        x_lower = x_samples[:-1] # [x_true, x0, x1, ..., x(n-1)], len = n
        
        # x[0] - x_true * (1-0/n)^2 + x[1] - x[0] * (1-1/n)^2 + ... + x[n-1] - x[n-2] * (1-(n-1)/n)^2
        crps = ((x_upper - x_lower) * (1 - y_cdf[:-1])**2).sum()
    
    else:
        split_index = np.searchsorted(x_samples, x_true, side="right")
        x_smaller = x_samples[:split_index]
        x_smaller = np.append(x_smaller, x_true) # [x0, x1, ..., x(split_index-1), x_true], len = split_index+1
        x_larger = x_samples[split_index:]
        x_larger = np.insert(x_larger, 0, x_true) # [x_true, x(split_index), x(split_index+1), ..., x(n-1)], len = n-split_index+1
        
        y_cdf_smaller = y_cdf[:split_index+1] # [0, 1/n, 2/n, ..., split_index/n], len = split_index+1
        y_cdf_larger = y_cdf[split_index:] # [split_index/n, (split_index+1)/n, ..., 1], len = n-split_index+1
        
        x_lower_s = x_smaller[:-1] # [x0, x1, ..., x(split_index-1)], len = split_index
        x_upper_s = x_smaller[1:] # [x1, x2, ..., x(split_index-1), x_true], len = split_index
        crps += ((x_upper_s - x_lower_s) * y_cdf_smaller[1:]**2).sum() # sum of (x[i] - x[i-1]) * (i/n)^2, i=1 to split_index
        
        x_lower_l = x_larger[:-1] # [x_true, x(split_index), x(split_index+1), ..., x(n-2)], len = n-split_index
        x_upper_l = x_larger[1:] # [x(split_index), x(split_index+1), ..., x(n-1)], len = n-split_index
        crps += ((x_upper_l - x_lower_l) * (1 - y_cdf_larger[:-1])**2).sum() # sum of (x[i] - x[i-1]) * (1-(i/n))^2, i=split_index to n-1
        
    return crps
        

def cross_cor_temporal(list1,list2,mask,dim_x,window_len=None):
    """
    mask [bs*L,T,dim_x] selects the part we want to discard
    lists has the shape [bs*L,T,dim_x]

    cross corr is computed with using a windows with increased length.
    """
    list1_masked=[]
    list2_masked=[]
    epsilon = 1e-6

    
    for i in range(len(list1)):
        cm = mask[i] #current mask
        ctx = list1[i][~cm].reshape(-1,dim_x) #current_theta_x
        cx = list2[i][~cm].reshape(-1,dim_x) #current_x
        
        if window_len is not None:
            list1_masked.append([(ctx[:k]- ctx[:k].mean(axis=0)[None])/(ctx[:k].std(axis=0)[None]+epsilon) for k in range(window_len, len(ctx)+window_len,window_len)])
            list2_masked.append([(cx[:k]- cx[:k].mean(axis=0)[None])/(cx[:k].std(axis=0)[None]+epsilon) for k in range(window_len, len(cx)+window_len,window_len)])

        else:
            list1_masked.append((ctx-ctx.mean(axis=0)[None])/(ctx.std(axis=0)[None]+epsilon))
            list2_masked.append((cx-cx.mean(axis=0)[None])/(cx.std(axis=0)[None]+epsilon))
    
    return list1_masked,list2_masked
    # list2_masked.append([(cx[~cm].reshape(-1,dim_x)[:k]- cx[~cm].reshape(-1,dim_x)[:k].mean(axis=0)[None])/cx[~cm].reshape(-1,dim_x)[:k].std(axis=0)[None] for k in range(window_len, len(cx[~cm])+window_len,window_len)])