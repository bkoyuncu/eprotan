from typing import Any, Literal

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback, LightningModule, Trainer

import torch
from torch import Tensor
from torch.optim import Optimizer

import numpy as np
import time
import math
from itertools import cycle
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px        
        
class plotPredictions_mean_var_plotly(Callback):
    """
    Plot the mean and variance of the predictions and the ground truth for each batch in the test set or 12 if the batch size is larger.
    """

    def __init__(
        self,
        samples=1,
        reconstruct=False
    ) -> None:
        """
        Args:
            samples: Number of samples used to calculate the std and mean of the predictions.
            reconstruct: If the model is reconstructing the history sequence or not.
        """

        super().__init__()
        self.samples = samples
        self.reconstruct = reconstruct
    
    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # // this is loading twice we could use the existing one.
        pl_module.eval()
        batch = next(iter(pl_module.test_dataloader()))
        x, c, T0, T = batch
        bs = x.shape[0] #x has the shape [Bs, Tm, dim_x]
        x_dim = x.shape[-1]
        rows = min(math.ceil(bs/2), 6)
        num_plots = min(bs, 12)

        x_hat_mu_z, x_hat_L, x = pl_module.predict_and_gt_denormalized(batch=batch,samples=self.samples, reconstruct=self.reconstruct)
        x_hat_mu_z, x_hat_L, x = x_hat_mu_z.detach().cpu(), x_hat_L.detach().cpu(), x.detach().cpu()
        
        x_hat_mean = torch.mean(x_hat_L,dim=1).numpy()
        x_hat_std = torch.std(x_hat_L,dim=1).numpy()

        #check the dimensions
        assert x_hat_mu_z.shape==x.shape

        L = x_hat_L.shape[1]

        #PLOT PRED GT

        colors_ = px.colors.qualitative.Bold
        colors = []
        colors_pred = []
        for i in range(len(colors_)):
            colors.append(colors_[i].replace("rgb","rgba").replace(")",", 0.8)"))
            colors_pred.append(colors_[i].replace("rgb","rgba").replace(")",", 0.4)"))
        fig = make_subplots(rows=rows, cols=2)

        for i in range(num_plots):
            showlegend = True if i == 0 else False
            for d in range(x_dim):
                if self.reconstruct:
                    x_axis_ticks = np.arange(0, T[i].item())
                    fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[i, :T[i], d], mode='lines', name=f'GT_{d}', line=dict(color=colors[d]), showlegend=showlegend),
                                             row = i // 2 + 1, 
                                             col = i % 2 + 1)
                    fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[i, :T[i], d], mode='lines', name=f'Pred_{d}',
                                            error_y=dict(type='data', array=x_hat_std[i, :T[i], d], visible=True, color=colors_pred[d]),
                                            line=dict(color=colors_pred[d]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)
                else:
                    x_axis_ticks = np.arange(T0[i].item(), T[i].item())
                    fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[i, T0[i]:T[i], d], mode='lines', name=f'GT_{d}', line=dict(color=colors[d]), showlegend=showlegend),
                                             row = i // 2 + 1, 
                                             col = i % 2 + 1)
                    fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[i, T0[i]:T[i], d], mode='lines', name=f'Pred_{d}',
                                            error_y=dict(type='data', array=x_hat_std[i, T0[i]:T[i], d], visible=True, color=colors_pred[d]),
                                            line=dict(color=colors_pred[d]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)

    
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"mean_variance_L_{L}_bs_{bs}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})
        
        
class plotPredictions_mean_var_plotly_dims(Callback):
    """
    Plot the mean and variance of the predictions and the ground truth for the first 12 dimensions of the data.
    """

    def __init__(
        self,
        samples=1,
        reconstruct=False
    ) -> None:
        """
        Args:
            samples (int): Number of samples used to calculate the std and mean of the predictions.
            reconstruct (bool): If the model is reconstructing the history sequence or not.
        """

        super().__init__()
        self.samples = samples
        self.reconstruct = reconstruct
    
    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # // this is loading twice we could use the existing one.
        pl_module.eval()
        batch = next(iter(pl_module.test_dataloader()))
        x, c, T0, T = batch
        old_bs = x.shape[0]
        new_batch = (x[0:1], c[0:1], T0[0:1], T[0:1])
        #x has the shape [bs, Tm, dim_x]
        x_dim = x.shape[-1]
        rows = min(math.ceil(x_dim/2), 6)
        num_plots = min(x_dim, 12)

        x_hat_mu_z, x_hat_L, x = pl_module.predict_and_gt_denormalized(batch=new_batch,samples=self.samples, reconstruct=self.reconstruct) # [bs, Tm, dim_x], [bs, L, Tm, dim_x], [bs, Tm, dim_x]
        x_hat_mu_z, x_hat_L, x = x_hat_mu_z.detach().cpu(), x_hat_L.detach().cpu(), x.detach().cpu()
        
        x_hat_mean = torch.mean(x_hat_L,dim=1).numpy()
        x_hat_std = torch.std(x_hat_L,dim=1).numpy()

        #check the dimensions
        assert x_hat_mu_z.shape==x.shape

        L = x_hat_L.shape[1]

        #PLOT PRED GT

        colors_ = px.colors.qualitative.Bold
        colors = []
        colors_pred = []
        for i in range(len(colors_)):
            colors.append(colors_[i].replace("rgb","rgba").replace(")",", 0.8)"))
            colors_pred.append(colors_[i].replace("rgb","rgba").replace(")",", 0.4)"))
        fig = make_subplots(rows=rows, cols=2)

        for dim in range(num_plots):
            showlegend = True if dim == 0 else False         
            if self.reconstruct:
                x_axis_ticks = np.arange(0, T[0].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[0, :T[0], dim], mode='lines', name=f'GT', line=dict(color=colors[0]), showlegend=showlegend),
                                            row = dim // 2 + 1, 
                                            col = dim % 2 + 1)
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[0, :T[0], dim], mode='lines', name=f'Pred',
                                        error_y=dict(type='data', array=x_hat_std[0, :T[0], dim], visible=True, color=colors_pred[0]),
                                        line=dict(color=colors_pred[0]), showlegend=showlegend),
                                        row = dim // 2 + 1, 
                                        col = dim % 2 + 1)
            else:
                x_axis_ticks = np.arange(T0[0].item(), T[0].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x[0, T0[0]:T[0], dim], mode='lines', name=f'GT', line=dict(color=colors[0]), showlegend=showlegend),
                                            row = dim // 2 + 1, 
                                            col = dim % 2 + 1)
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_mean[0, T0[0]:T[0], dim], mode='lines', name=f'Pred',
                                        error_y=dict(type='data', array=x_hat_std[0, T0[0]:T[0], dim], visible=True, color=colors_pred[0]),
                                        line=dict(color=colors_pred[0]), showlegend=showlegend),
                                        row = dim // 2 + 1, 
                                        col = dim % 2 + 1)

    
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"mean_variance_L_{L}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})

        
        # plot the sum over all dimensions
        rows = min(math.ceil(old_bs/2), 6)
        num_plots = min(old_bs, 12)
        
        x_hat_mu_z, x_hat_L, x = pl_module.predict_and_gt_denormalized(batch=batch,samples=self.samples, reconstruct=self.reconstruct) # [bs, Tm, dim_x], [bs, L, Tm, dim_x], [bs, Tm, dim_x]
        # sum along the last dimension
        x_hat_mu_z_sum = torch.sum(x_hat_mu_z, dim=-1) # [bs, Tm]
        x_hat_L_sum = torch.sum(x_hat_L, dim=-1) # [bs, L, Tm]
        x_sum = torch.sum(x, dim=-1) # [bs, Tm]
        x_hat_mu_z_sum, x_hat_L_sum, x_sum = x_hat_mu_z_sum.detach().cpu(), x_hat_L_sum.detach().cpu(), x_sum.detach().cpu()
        
        x_hat_sum_mean = torch.mean(x_hat_L_sum,dim=1).numpy() # [bs, Tm] 
        x_hat_sum_std = torch.std(x_hat_L_sum,dim=1).numpy() # [bs, Tm]

        #check the dimensions
        assert x_hat_mu_z_sum.shape==x_sum.shape

        L = x_hat_L_sum.shape[1]
        fig = make_subplots(rows=rows, cols=2)
        
        for i in range(num_plots):
            showlegend = True if i == 0 else False
            if self.reconstruct:
                x_axis_ticks = np.arange(0, T[i].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_sum[i, :T[i]], mode='lines', name=f'GT_sum_dims_{x_dim}', line=dict(color=colors[0]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_sum_mean[i, :T[i]], mode='lines', name=f'Pred_sum_dims_{x_dim}',
                                        error_y=dict(type='data', array=x_hat_sum_std[i, :T[i]], visible=True, color=colors_pred[0]),
                                        line=dict(color=colors_pred[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)
            else:
                x_axis_ticks = np.arange(T0[i].item(), T[i].item())
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_sum[i, T0[i]:T[i]], mode='lines', name=f'GT_sum_dims_{x_dim}', line=dict(color=colors[0]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)
                fig.add_trace(go.Scatter(x=x_axis_ticks, y=x_hat_sum_mean[i, T0[i]:T[i]], mode='lines', name=f'Pred_sum_dims_{x_dim}',
                                        error_y=dict(type='data', array=x_hat_sum_std[i, T0[i]:T[i]], visible=True, color=colors_pred[0]),
                                        line=dict(color=colors_pred[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)

    
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"mean_variance_sum_dims_{x_dim}_L_{L}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})

class plotMetric_timeseries_plotly(Callback):
    """
    Plot the metrics (ELBO, RMSE, Cross-Correlation) over time for each batch in the test set or 12 if the batch size is larger.
    """

    def __init__(
        self,
        samples=1,
        reconstruct=False,
        reduction: Literal['none', 'mean', 'sum', 'max', 'min', 'first_n'] = 'none'
    ) -> None:
        """
        Args:
            samples (int): Number of samples used to calculate the std and mean of the metrics.
            reconstruct (bool): If the model is reconstructing the history sequence or not.
            reduction (str): Reduction method for the metrics, reduce over the data dimension. Note that ELBO is already summed over the data dimension.
        """
        check_1 = reduction in ['none', 'mean', 'sum', 'max', 'min']
        check_2 = reduction.split('_')[0] == 'first' and reduction.split('_')[1].isdigit()
        if not check_1 and not check_2:
            raise ValueError(f"Reduction method {reduction} not supported. Choose from ['none', 'mean', 'sum', 'max', 'min', 'first_n']")
        super().__init__()
        self.samples = samples
        self.reconstruct = reconstruct
        self.reduction = reduction
        
    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print("ENTERING PLOT METRIC")
        # // this is loading twice we could use the existing one.
        pl_module.eval()
        batch = next(iter(pl_module.test_dataloader()))
        x, c, T0, T = batch
        x_dim = x.shape[-1]
        L = self.samples
        bs = x.shape[0]
        
        rows = min(math.ceil(bs/2), 6)
        num_plots = min(bs, 12)
        reduction_fn = None
        idx_fn = None
        if self.reduction == 'mean':
            reduction_fn = np.mean
        elif self.reduction == 'sum':
            reduction_fn = np.sum
        elif self.reduction == 'max':
            reduction_fn = np.max
            idx_fn = np.argmax
        elif self.reduction == 'min':
            reduction_fn = np.min
            idx_fn = np.argmin
            
        window_len=5
        stride=5
        metric_dict = pl_module.metric_timeseries_L(batch=batch, batch_idx=0, samples=self.samples, window_len=window_len, stride=stride)
        
        #these are becoming list of lists
        elbo = metric_dict["metric_elbo"] #[bs, L, T]
        elbo = [list(torch.stack(tensor_list).detach().cpu()) for tensor_list in elbo]
        metric_se = metric_dict["metric_rmse"]  #[bs,L,T,d] #this is only square error
        metric_se = [list(torch.stack(tensor_list).detach().cpu()) for tensor_list in metric_se]
        metric_cross_cor = metric_dict["metric_cross_cor"] #[bs, L, number of windows,d]

        #deal with lists now#
        #every list has bs* sample len

        new_elbo_list = [np.array(elbo[x:x+self.samples]) for x in range(0,bs*self.samples,self.samples)] #returns a list of BS elements: each has (L,#windows)
        new_metric_se_list = [np.array(metric_se[x:x+self.samples]) for x in range(0,bs*self.samples,self.samples)] #returns a list of BS elements: each has (L,#windows)
        new_metric_cross_cor_list = [np.array(metric_cross_cor[x:x+self.samples])  for x in range(0,bs*self.samples,self.samples)] #returns a list of BS elements: each has (L,#windows)
        # new_metric_cross_cor_list = [np.array(metric_cross_cor[x:x+window_len]).reshape(window_len,-1)  for x in range(0,bs*self.samples,window_len)]


        # set plotly color 
        colors_ = px.colors.qualitative.Bold
        colors = []
        for i in range(len(colors_)):
            colors.append(colors_[i].replace("rgb","rgba").replace(")",", 0.8)"))
        
        #PLOT ELBO
        elbo_mean = [i.mean(axis=0) for i in new_elbo_list]
        elbo_std = [i.std(axis=0) for i in new_elbo_list]

        fig = make_subplots(rows=rows, cols=2)
        for i in range(num_plots):
            showlegend = True if i == 0 else False
            if self.reconstruct:
                x_axis_ticks = np.arange(0,T[i].item(),1)
            else:   
                x_axis_ticks = np.arange(T0[i].item(),T[i].item(),1)
            data_ticks = [int(x_axis_ticks[x:x+window_len].mean()) for x in np.arange(0,len(x_axis_ticks),stride)]
            
            fig.add_trace(go.Scatter(x=data_ticks, y=elbo_mean[i], mode='lines', name='ELBO',
                                    error_y=dict(type='data', array=elbo_std[i], visible=True, color=colors[0]),
                                    line=dict(color=colors[0]), showlegend=showlegend),
                                    row = i // 2 + 1, 
                                    col = i % 2 + 1)
            # fig.update_xaxes(tickmode="array", tickvals=x_axis_ticks, row = i // 2 + 1, col = i % 2 + 1)
            
        
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
              
        name = f"elbo_temporal_L_{L}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})
        

        #PLOT SE
        se_mean = [i.mean(axis=0) for i in new_metric_se_list]
        se_std = [i.std(axis=0) for i in new_metric_se_list]
        
        fig = make_subplots(rows=rows, cols=2)
        for i in range(num_plots):
            showlegend = True if i == 0 else False
            if self.reconstruct:
                x_axis_ticks = np.arange(0,T[i].item(),1)
            else:   
                x_axis_ticks = np.arange(T0[i].item(),T[i].item(),1)
            data_ticks = [int(x_axis_ticks[x:x+window_len].mean()) for x in np.arange(0,len(x_axis_ticks),stride)]
            
            if self.reduction == 'none' or self.reduction.startswith('first'):
                final_dim = int(self.reduction.split('_')[1]) if self.reduction.startswith('first') else x_dim
                for d in range(final_dim):
                    fig.add_trace(go.Scatter(x=data_ticks, y=se_mean[i][:,d], mode='lines', name='RMSE_dim_' + str(d),
                                            error_y=dict(type='data', array=se_std[i][:,d], visible=True, color=colors[d]),
                                            line=dict(color=colors[d]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)
            elif self.reduction == 'max' or self.reduction == 'min':
                idx = idx_fn(se_mean[i], axis=1)
                time_enum = np.arange(se_mean[i].shape[0])
                fig.add_trace(go.Scatter(x=data_ticks, y=se_mean[i][time_enum, idx], mode='lines', name='RMSE_' + self.reduction + '_dims_' + str(x_dim),
                                        error_y=dict(type='data', array=se_std[i][time_enum, idx], visible=True, color=colors[0]),
                                        line=dict(color=colors[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)
                
            elif self.reduction == 'mean' or self.reduction == 'sum':
                # Get the sum/mean of the stds assuming that the covariance is diagonal
                # std(X+Y) = sqrt(Var(X) + Var(Y) + 2*Cov(X,Y)) --> sqrt(Var(X) + Var(Y)) if X and Y are independent
                se_std_current = se_std[i] ** 2         # square the std to get the variance
                var_sum = se_std_current.sum(axis=1)    # Var(x_1) + Var(x_2) + ... + Var(x_d)
                se_std_current = np.sqrt(var_sum)       # std(x_1 + x_2 + ... + x_d)
                if self.reduction == 'mean':
                    se_std_current /= se_std[i].shape[-1]
                
                fig.add_trace(go.Scatter(x=data_ticks, y=reduction_fn(se_mean[i], axis=1), mode='lines', name='RMSE_' + self.reduction + '_dims_' + str(x_dim),
                                        error_y=dict(type='data', array=se_std_current, visible=True, color=colors[0]),
                                        line=dict(color=colors[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)
            
            # fig.update_xaxes(tickmode="array", tickvals=x_axis_ticks, row = i // 2 + 1, col = i % 2 + 1)
            
        
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"rmse_temporal_L_{L}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})


        #PLOT CC 
        cc_mean = [i.mean(axis=0) for i in new_metric_cross_cor_list]
        cc_std = [i.std(axis=0) for i in new_metric_cross_cor_list]

        fig = make_subplots(rows=rows, cols=2)
        for i in range(num_plots):
            showlegend = True if i == 0 else False
            if self.reconstruct:
                x_axis_ticks = np.arange(0,T[i].item(),1)
            else:   
                x_axis_ticks = np.arange(T0[i].item(),T[i].item(),1)
            data_ticks = [int(x_axis_ticks[x:x+window_len].mean()) for x in np.arange(0,len(x_axis_ticks),stride)]
            
            
            if self.reduction == 'none' or self.reduction.startswith('first'):
                final_dim = int(self.reduction.split('_')[1]) if self.reduction.startswith('first') else x_dim
                
                for d in range(final_dim):
                    fig.add_trace(go.Scatter(x=data_ticks, y=cc_mean[i][:,d], mode='lines', name='CC_dim_' + str(d),
                                            error_y=dict(type='data', array=cc_std[i][:,d], visible=True, color=colors[d]),
                                            line=dict(color=colors[d]), showlegend=showlegend),
                                            row = i // 2 + 1, 
                                            col = i % 2 + 1)
                    
            elif self.reduction == 'max' or self.reduction == 'min':
                idx = idx_fn(cc_mean[i], axis=1)
                time_enum = np.arange(cc_mean[i].shape[0])
                fig.add_trace(go.Scatter(x=data_ticks, y=cc_mean[i][time_enum, idx], mode='lines', name='CC_' + self.reduction + '_dims_' + str(x_dim),
                                        error_y=dict(type='data', array=cc_std[i][time_enum, idx], visible=True, color=colors[0]),
                                        line=dict(color=colors[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)
            
            elif self.reduction == 'mean' or self.reduction == 'sum':
                # Get the sum/mean of the stds assuming that the covariance is diagonal
                # std(X+Y) = sqrt(Var(X) + Var(Y) + 2*Cov(X,Y)) --> sqrt(Var(X) + Var(Y)) if X and Y are independent
                cc_std_current = cc_std[i] ** 2
                var_sum = cc_std_current.sum(axis=1)
                cc_std_current = np.sqrt(var_sum)
                if self.reduction == 'mean':
                    cc_std_current /= cc_std[i].shape[-1]
                fig.add_trace(go.Scatter(x=data_ticks, y=reduction_fn(cc_mean[i], axis=1), mode='lines', name='CC_' + self.reduction + '_dims_' + str(x_dim),
                                        error_y=dict(type='data', array=cc_std_current, visible=True, color=colors[0]),
                                        line=dict(color=colors[0]), showlegend=showlegend),
                                        row = i // 2 + 1, 
                                        col = i % 2 + 1)
                
                
            # fig.update_xaxes(tickmode="array", tickvals=x_axis_ticks, row = i // 2 + 1, col = i % 2 + 1)
            
        
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='value')
        
        name = f"cc_temporal_L_{L}"
        fig.update_layout(showlegend=True, height=3000, width=1500, title_text=name)
        fig.show()
        # pl_module.logger.experiment.log({name: fig})
        
        
class CRPSCallback(Callback):
    """
    Callback to invoke the calculation of the CRPS metrics.
    
    Args:
        samples (int): Number of samples used to estimate the the predictive CDF for the CRPS calculation.
    """
    
    def __init__(self, samples: int = 100) -> None:
        super().__init__()
        self.samples = samples
    
    @torch.no_grad() 
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.eval()
        test_loader = pl_module.test_dataloader()
        crps_mean, crps_sum_mean, crps_mean_normalized, crps_sum_mean_normalized = pl_module.aggregate_crps_dataset(test_loader, samples=self.samples)
        pl_module.log("test_crps_mean", crps_mean)
        pl_module.log("test_crps_sum_mean", crps_sum_mean)
        pl_module.log("test_crps_mean_normalized", crps_mean_normalized)
        pl_module.log("test_crps_sum_mean_normalized", crps_sum_mean_normalized)


class GradCheck(Callback):
    """
    Callback to invoke the calculation of the Grad Norm 2 metrics.
    """
    
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad() 
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        grad_norm = pl_module._compute_grad_norm()
        pl_module.log("train_grad_norm", grad_norm)
    
    
class TrainingTimer(Callback):
    """
    Callback to measure the time for forward passes, backward passes + gradient descent step, and the times per epoch in training.
    See https://lightning.ai/docs/pytorch/stable/common/lightning_module.html under section 'Hooks' to see when exactly these API methods are called.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.forward_start_time = 0
        self.backward_start_time = 0
        self.train_epoch_start_time = 0
        self.test_epoch_start_time = 0
        
        self.forward_times = []
        self.backward_times = []
        self.iteration_times = []
        
        self.all_forward_times = []
        self.all_backward_times = []
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        self.forward_start_time = time.time()
        
    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:  
        # before zero grad is after the forward pass (training_step) and before the backward pass
        current_time = time.time()
        forward_time = current_time - self.forward_start_time
        self.forward_times.append(forward_time)
        self.all_forward_times.append(forward_time)
    
    def on_before_backward(self, trainer: Trainer, pl_module: LightningModule, loss: Tensor) -> None:
        # Note: Calling this here excludes the time needed to zero the gradients!
        self.backward_start_time = time.time()
        
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # calculate the time needed for performing the forward pass and backward pass on a batch
        # forward + backward
        current_time = time.time()
        iteration_time = current_time - self.forward_start_time
        self.iteration_times.append(iteration_time)
        
        # backward + gradient step in optimizer
        backward_time = current_time - self.backward_start_time
        self.backward_times.append(backward_time)
        self.all_backward_times.append(backward_time)
        
        # reset the timers
        self.forward_start_time = 0
        self.backward_start_time = 0
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_epoch_start_time = time.time()
        self.forward_times = []
        self.backward_times = []
        self.iteration_times = []
        

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log the times via the model
        train_epoch_time = time.time() - self.train_epoch_start_time
        avg_forward_time = np.mean(self.forward_times)
        avg_backward_time = np.mean(self.backward_times)
        avg_iteration_time = np.mean(self.iteration_times)
        
        pl_module.log("train_epoch_time", train_epoch_time)
        pl_module.log("avg_forward_time", avg_forward_time)
        pl_module.log("avg_backward_time", avg_backward_time)
        pl_module.log("avg_iteration_time", avg_iteration_time)
        
        
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log the total times for forward and backward passes
        total_forward_time_mean = np.mean(self.all_forward_times)
        total_backward_time_mean = np.mean(self.all_backward_times)
        total_forward_time_std = np.std(self.all_forward_times)
        total_backward_time_std = np.std(self.all_backward_times)
        
        # not compatible if not ran with wandb
        # trainer.logger.experiment.log({"total_forward_time_mean": total_forward_time_mean})
        # trainer.logger.experiment.log({"total_backward_time_mean": total_backward_time_mean})
        # trainer.logger.experiment.log({"total_forward_time_std": total_forward_time_std})
        # trainer.logger.experiment.log({"total_backward_time_std": total_backward_time_std})
        print(f"Total forward time: {total_forward_time_mean} +- {total_forward_time_std}")
        print(f"Total backward time: {total_backward_time_mean} +- {total_backward_time_std}")
        
        self.all_forward_times = []
        self.all_backward_times = []
        
    def on_test_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        self.forward_start_time = time.time()
        
        
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # calculate the time needed for performing the forward pass on a test batch
        current_time = time.time()
        forward_time = current_time - self.forward_start_time
        self.forward_times.append(forward_time)
        
        # reset the timer
        self.forward_start_time = 0
        
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_epoch_start_time = time.time()
        self.forward_times = []

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log the times via the model
        test_epoch_time = time.time() - self.test_epoch_start_time
        avg_forward_time = np.mean(self.forward_times)
        
        pl_module.log("test_epoch_time", test_epoch_time)
        pl_module.log("avg_forward_time_test", avg_forward_time)
        
        
        
    
class InferenceTimerCallback(Callback):
    """
    Callback to measure the time for forward passes in inference.
    """
    
    def __init__(self, num_trials) -> None:
        super().__init__()
        self.num_trials = num_trials
    
    @torch.no_grad()
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.eval()
        data_loader = pl_module.train_dataloader()
        cyclic_loader = cycle(data_loader)
        
        # forward pass num_trial times and measure the inference times
        times = []
        for i in tqdm(range(self.num_trials), desc="Testing inference time"):
            batch = next(cyclic_loader)
            forward_start_time = time.time()
            # note that reconstruct = True only changes masking here, nothing else
            pl_module.predict_and_gt_denormalized(batch=batch, samples=1, reconstruct=True)
            forward_time = time.time() - forward_start_time
            times.append(forward_time)
            
        # get the mean and the std of the forward passes and log them
        mean_time = np.mean(times)
        std_time = np.std(times)
        max_time = np.max(times)
        min_time = np.min(times)
        pl_module.log("inference_time_mean", mean_time)
        pl_module.log("inference_time_std", std_time)
        pl_module.log("inference_time_max", max_time)
        pl_module.log("inference_time_min", min_time)