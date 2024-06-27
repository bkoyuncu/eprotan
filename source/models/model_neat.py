import torch
from torch import autograd
autograd.set_detect_anomaly(True)

from source.models.archs import *
from source.models.AbstractBaseModel import AbstractBaseModel

class BaseModel(AbstractBaseModel):
    def __init__(self, dataset, train, model, optim, projection_args, embedding_args, conditional_prior_args, encoder_args, decoder_args, **kwargs) -> None:
        """
        Args:
            log_dir: (str) path to project folder currently empty
            dataset:
                name: (str) name of the dataset
                split: (None) split parameter for dataset splitting
                dim_x: (int) dimension for x
                dim_c: (int) dimension for c, c = c_lag + c_feat
                dim_c_lag: (int) dimension for c_lag
                tm: (int) length of whole sequence (Hist+Forecasting)
                tmh: (int) length of history sequence
                tmf: (int) length of forecasting sequence
                kind: (str) type of dataset, e.g., 'mini' for a small subset, 'full' for the whole dataset
            train:
                batch_size: (int) batch size
                ckpt_period: (int) period for checkpointing in epochs
                share_arch: (bool) to share architectures for cond. prior and encoder
            model:
                type: (str) type of the model, here 'foretran'
                dim_z: (int) dimension for latent z
                likelihood: (str) type of likelihood function
                reconstruction: (bool) wether to reconstruct the history or not
                epsilon (float): epsilon for numerical stability
            optim:
                max_epoch: (int) number of epochs
                base_lr: (float) learning rate
                optimizer: (str) optimizer name
                scheduler_gamma: (float) gamma for the exponential scheduler
                scheduler_n_epochs: (int) number of epochs for the scheduler
            projection_args:
                dim_list: (list) number of hidden units for the projection layer
            embedding_args:
                arch_name: (str) name of the embedding layer
                max_len: (int) maximum length of the embedding layer should be equal to Tm
                mode: (str) add or concat
                embedding_dim: (int) dimension of the embedding
            conditional_prior_args:
                arch_name: (str) name of the cond prior block
                dim_ff_tran: (int | List[int]) dimension(s) of the feed forward network in the transformer
                num_heads: (int) number of heads for the transformer
                num_layers: (int) number of layers for the transformer
                causal: (bool) to make the architecture causal # currently UNUSED
                dim_list: (list) number of hidden units for the MLP (on top of transformer or on its own)
                kind_attentions (List[str]): Kind of attentions to use, only used with arch_name foretran_transformer
                dim_list_latent_to_w: (list) unused here, only used with ProTran
                use_sampling (bool): unused here, only used with ProTran
            encoder_args:
                arch_name: (str) name of the encoder block
                dim_ff_tran: (int | List[int]) dimension(s) of the feed forward network in the transformer
                num_heads: (int) number of head for the transformer
                num_layers: (int) number of layers for the transformer
                causal: (bool) to make the architecture causal # currently UNUSED
                kind_attentions (List[str]): Kind of attentions to use, only used with arch_name foretran_transformer
                dim_list: (list) number of hidden units for the MLP (on top of transformer or on its own)
            decoder_args
                arch_name: 'MLP'
                dim_list: (list) number of hidden units of decoder
                variance: (float) variance for the likelihood function, only used for gaussian and laplace
            testing:
                samples: (int) number of samples used in test mode

            mode:
                training: (bool) to state if train the model
                testing: (bool) to state if test the model
                version: (int) version number to use for testing
            
        """
        epsilon = model["epsilon"]
        reconstruct = model["reconstruct"]
        super(BaseModel, self).__init__(model_type='foretran', reconstruct=reconstruct, epsilon=epsilon)
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

        self.dataset_name = dataset["name"]
        self.dim_x = dataset["dim_x"]
        self.dim_c = dataset["dim_c"]
        self.dim_c_lag = dataset["dim_c_lag"]
        self.tm = dataset["tm"]
        self.tmh = dataset["tmh"]
        self.tmf = dataset["tmf"]
        assert self.tm == self.tmh + self.tmf, "tm should be equal to tmh + tmf"
        self.t_ratio = (dataset["tmh"]/dataset["tm"])
        self.kind_dataset = dataset["kind"]

        self.dim_latent= model["dim_z"]
        self.likelihood_x= model["likelihood"]

        self.batch_size= train["batch_size"]
        self.ckpt_period = train["ckpt_period"]
        self.share_arch = train["share_arch"]

        self.lr =  optim["base_lr"]
        self.scheduler_gamma = optim["scheduler_gamma"]
        self.scheduler_n_epochs = optim["scheduler_n_epochs"]

        self.architecture_cprior = conditional_prior_args["arch_name"]
        self.architecture_encoder = encoder_args["arch_name"]

        #self.prediction_metric_name = None

        self.initialize_buffer() #this is for saving means and stds

        assert projection_args['dim_list'][0]==self.dim_x+self.dim_c

        if embedding_args["mode"]=="MODE_ADD":
            assert embedding_args["embedding_dim"] == projection_args['dim_list'][-1]
            self.dim_h = projection_args['dim_list'][-1]
        
        if embedding_args["mode"]=="MODE_CONCAT":
            self.dim_h = embedding_args["embedding_dim"] + projection_args['dim_list'][-1]

        assert embedding_args["max_len"] == dataset["tm"]

        assert conditional_prior_args["dim_list"][0] == self.dim_h
        assert conditional_prior_args["dim_list"][-1] == self.dim_latent
        
        if encoder_args["arch_name"] == 'encoder_head':
            assert train['share_arch'] == True, 'If an encoder head is used, the architecture is shared by default. Please set share_arch to True.'

        assert encoder_args["dim_list"][0] == self.dim_h
        assert encoder_args["dim_list"][-1] == self.dim_latent

        assert decoder_args["dim_list"][0] == self.dim_latent
        assert decoder_args["dim_list"][-1] == self.dim_x


        self.projection_layer = get_projection_layer('projectionLayer', **projection_args) 
        self.embedding_layer = get_embedding_layer( **embedding_args)
        self.ln = torch.nn.LayerNorm(self.dim_h)

        if self.share_arch and encoder_args["arch_name"] != 'encoder_head':
            self.conditional_prior = get_conditional_prior(**conditional_prior_args)
            self.encoder = self.conditional_prior
        else:
            self.conditional_prior= get_conditional_prior(**conditional_prior_args)
            self.encoder= get_encoder(**encoder_args)

        self.decoder= get_decoder(likelihood=self.likelihood_x,**decoder_args)

        self.mask_causal = generate_square_subsequent_mask(self.tm).to(self.device) 
        self.save_hyperparameters()

    def _forward(self, batch:tuple, samples: int = 1, reconstruct = False) -> tuple:
            """
            Args:
                batch (tuple): has elements (X,C,T0,T)
                    X (torch.Tensor): input data with shape [bs, Tm, dim_x]
                    C (torch.Tensor): input data with shape [bs, Tm, dim_c]
                    T0: length of history sequence [bs, 1]
                    T: total length of sequence [bs, 1]
                samples (int): number of latent samples in bottleneck to be sampled from // q //
                reconstruct (bool): If True, reconstructs the data, i.e., the loss is computed on the whole sequence instead of only the forecasting part

            Returns (x, x_hat, ELBO, logpx_z, KL, mask_temporal):
                x (torch.Tensor): input data [bs, Tm, dim_x]
                x_p_hat (torch.Tensor): reconstructed data by decoding encoder latent z_q [bs, L, Tm, dim_x]
                x_q_hat (torch.Tensor): reconstructed data by decoding conditional prior latent z_p [bs, L, Tm, dim_x]
                ELBO (torch.Tensor): ELBO [bs, L, Tm]
                logpx_z (torch.Tensor): logpx_z [bs, L, Tm]
                KL (torch.Tensor): KL [bs, L, Tm]
                mask_temporal (torch.Tensor): temporal mask to only consider the parts we are interested in
            """
            # print("self.device", self.device)
            assert samples > 0, "samples should be greater than 0"
            
            x, c, T0, T = batch
            bs, Tm, dim_x = x.shape
            
            if not (x.device == c.device == T0.device == T.device == self.device):
                x, c, T0, T = x.to(self.device), c.to(self.device), T0.to(self.device), T.to(self.device)

            #input for the posterior
            xc = torch.concat((x,c),dim=-1) # [bs, Tm, dim_x + dim_c]
            xc = self.projection_layer(xc)
            xc_embedding = self.embedding_layer(xc) #[bs, Tm, dim_xc] if the mode is add
            xc_embedding = self.ln(xc_embedding) #[bs, Tm, dim_xc]
            
            #input for the conditional prior
            prior_mask = self.create_prior_input_mask(bs, Tm, T0, T, dim_x) #creating mask for creating the input for conditional prior network
            # note that padding is already done with the data, so we only need to mask the data after T0
            x_prior = x.masked_fill(prior_mask, 0) # [bs, Tm, dim_x] 
            xc_prior = torch.concat((x_prior,c),dim=-1) # [bs, Tm, dim_x + dim_c] 
            xc_prior = self.projection_layer(xc_prior)
            xc_prior_embedding = self.embedding_layer(xc_prior) #[bs, Tm, dim_xc] if the mode is add
            xc_prior_embedding = self.ln(xc_prior_embedding) #[bs, Tm, dim_xc]
            
            padding_mask = torch.arange(Tm).to(self.device).unsqueeze(0) >= T # [bs, Tm]
            
            #---CONDITIONAL PRIOR p(z|x)---#
            #computes [x,c] -> h (deterministic) -> mu_z, var_z #the input is masked after time horizon T0
            self.mask_causal = self.mask_causal.to(self.device) #causal mask for attention
            if self.architecture_cprior == 'foretran_transformer':
                # mask for attention to the input
                key_mask_input = torch.arange(Tm).to(self.device).unsqueeze(0) >= T0 # [bs, Tm]
                mu_p_z, logvar_p_z, h_p = self.conditional_prior(xc_prior_embedding, causal_mask = self.mask_causal, input_mask=key_mask_input | padding_mask, key_padding_mask_layer = padding_mask)
            else:
                mu_p_z, logvar_p_z, h_p = self.conditional_prior(xc_prior_embedding, causal_mask = self.mask_causal, input_mask = padding_mask) #  [bs, Tm, dim_latent] [bs, Tm, dim_latent] [bs, Tm, dim_h]
            var_p_z = F.softplus(logvar_p_z)
            z_p = self.sample_z(mu=mu_p_z, var=var_p_z, samples=samples) #[bs, L, Tm, dim_latent]
            
            
            
            #---ENCODER q(z|x)---#
            #computes [x,c] -> h (deterministic) -> mu_z, var_z #the input is whole data
            if self.architecture_encoder == 'encoder_head':
                mu_q_z, logvar_q_z = self.encoder(xc_embedding, h_p, input_mask = padding_mask) # [bs, Tm, dim_latent] [bs, Tm, dim_latent]
                var_q_z = F.softplus(logvar_q_z)
            else:
                mu_q_z, logvar_q_z, h_q = self.encoder(xc_embedding, causal_mask= None, input_mask = padding_mask, key_padding_mask_layer = padding_mask) # [bs, Tm, dim_latent] [bs, Tm, dim_latent], [bs, Tm, dim_h]
                # z_q = self.sample_z(mu=mu_q_z, var=torch.exp(logvar_q_z), samples=samples) #[bs, L, Tm, dim_latent] 
                var_q_z = F.softplus(logvar_q_z)
            z_q = self.sample_z(mu=mu_q_z, var=var_q_z, samples=samples) #[bs, L, Tm, dim_latent]
            
            #--- DECODER p(x|z)---#
            #computes x_hat: z (comes from posterior) -> mu_x, var_x -> x_hat
            #computes logpx_z: mu_x, x -> logpx_z 
            
            # stack z_p and z_q for faster computation
            z = torch.cat((z_p, z_q), dim=0) # [2*bs, L, Tm, dim_latent]    
            mu_x, logvar_x = self.decoder(z) # [2*bs, L, Tm, dim_x]
            var_x = torch.exp(logvar_x)      # [batch_size, L, Tm, dim_x]
            
            # split mu_x and var_x
            mu_p_x, mu_q_x = torch.chunk(mu_x, 2, dim=0) # [bs, L, Tm, dim_x]
            var_p_x, var_q_x = torch.chunk(var_x, 2, dim=0) # [bs, L, Tm, dim_x]
            
            # we fix the scaling (variance) for stability, hence we ignore the variance here
            x_p_hat = self.build_x_hat(mu_p_x) # [bs, L, Tm, dim_x] for gaussian or laplace it just returns the same mu_x
            x_q_hat = self.build_x_hat(mu_q_x) # [bs, L, Tm, dim_x] for gaussian or laplace it just returns the same mu_x

            # E_q[log p(x|z)] 
            logpx_z_samples = self.decoder.logp(x=x, theta=x_q_hat).sum(-1) # [bs, L, Tm]
            logpx_z = logpx_z_samples.squeeze(1) # [bs, Tm] or [bs, L, Tm] if L > 1
            
            #temporal mask on the model output x^hat, decide if you want to reconstruct or not
            mask_temporal = self.create_temporal_mask(bs, Tm, T0, T, reconstruct=reconstruct) # [bs, Tm]

            ELBO, logpx_z_sliced, KL_sliced = self.compute_elbo(mask_temporal, logpx_z, mu_q_z, mu_p_z, torch.log(var_q_z), torch.log(var_p_z)) # [bs, L, Tm] [bs, L, Tm] [bs, L, Tm]
        
            return x, x_p_hat, x_q_hat, ELBO, logpx_z_sliced, KL_sliced, mask_temporal
        

    def KL_compute(self, mu0, mu1, logvar0, logvar1): #OK
        '''
        computes KL(N(mu0,logvar0)||N(mu1,logvar1))
        '''
        kl = -0.5 * torch.sum(1. + (logvar0-logvar1) - (mu1-mu0) *  torch.exp(-logvar1) * (mu1-mu0) - torch.exp(logvar0-logvar1), dim=-1, keepdim=False)
        kl = kl

        return kl # // output dimension [bs, Tm]


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
            reconstruct (bool): Whether to mask the history or not
         Returns:
            x_pred_mu_z_masked (torch.Tensor): forecasting data means [bs, Tm, dim_x]
            x_pred_L_masked (torch.Tensor): forecasting data [bs, samples, Tm, dim_x]
            x_gt_masked (torch.Tensor): ground truth data [bs, Tm, dim_x]
        """
        # samples = 1
        x, c, T0, T = batch
        bs, Tm, dim_x = x.shape
        
        if not (x.device == c.device == T0.device == T.device == self.device):
            x, c, T0, T = x.to(self.device), c.to(self.device), T0.to(self.device), T.to(self.device)

        #creating mask
        prior_mask = self.create_prior_input_mask(bs, Tm, T0, T, dim_x)
        x_prior = x.masked_fill(prior_mask, 0) # [bs, Tm, dim_x]
        xc_prior = torch.concat((x_prior,c),dim=-1) # [bs, Tm, dim_x + dim_c]
        xc_prior = self.projection_layer(xc_prior)
        xc_prior_embedding = self.embedding_layer(xc_prior) #[bs, Tm, dim_xc]
        xc_prior_embedding = self.ln(xc_prior_embedding) #[bs, Tm, dim_xc]


        padding_mask = torch.arange(Tm).to(self.device).unsqueeze(0) >= T # [bs, Tm]
        if self.architecture_cprior == 'foretran_transformer':
            # mask for attention to the input
            key_mask_input = torch.arange(Tm).to(self.device).unsqueeze(0) >= T0 # [bs, Tm]     
            mu_p_z, logvar_p_z, _ = self.conditional_prior(xc_prior_embedding, causal_mask = self.mask_causal, input_mask=key_mask_input | padding_mask, key_padding_mask_layer = padding_mask)
        else:
            mu_p_z, logvar_p_z, _ = self.conditional_prior(xc_prior_embedding, causal_mask = self.mask_causal, input_mask = padding_mask) #  [bs, Tm, dim_latent] [bs, Tm, dim_latent]


        z = self.sample_z(mu_p_z, F.softplus(logvar_p_z), samples) #[bs, L, Tm, dim_latent]

        theta_x_mu_z, _ = self.decoder(mu_p_z) # [bs, Tm, dim_x]
        theta_x, _ = self.decoder(z) # [bs, L, Tm, dim_x]

        x_pred_mu_z = self.build_x_hat(theta_x_mu_z) # [bs, Tm, dim_x] for gaussian or laplace it just returns the same theta_x

        x_pred = self.build_x_hat(theta_x) # [bs, L, Tm, dim_x] for gaussian or laplace it just returns the same theta_x

        mask_temporal = self.create_temporal_mask(bs, Tm, T0, T, reconstruct=reconstruct) #decide if you want to reconstruct or not
        mask = mask_temporal.unsqueeze(-1).repeat(1,1,x_pred_mu_z.shape[-1])
        # mask_L = mask[:,None].repeat(1,x_pred.shape[1],1,1)
        mask_L = mask.unsqueeze(1) #it broadcast to L

        if denormalize==True:
            x_pred_mu_z=self.denormalize_x(x_pred_mu_z)
            x_pred=self.denormalize_x(x_pred)
            x=self.denormalize_x(x)

        #masking
        x_pred_mu_z_masked=x_pred_mu_z.masked_fill(mask,0) # [bs, Tm, dim_x]
        x_pred_L_masked=x_pred.masked_fill(mask_L,0) # [bs, L, Tm, dim_x]
        x_gt_masked=x.masked_fill(mask,0) # [bs, Tm, dim_x]

        return x_pred_mu_z_masked, x_pred_L_masked, x_gt_masked


    def prediction_metric_quantile_loss_temporal(self, theta_x: torch.Tensor, x: torch.Tensor, mask:torch.Tensor,window_len=5, stride=3, temporal=True)-> torch.Tensor:
        """
        Given theta_x (x_pred), x (x_gt) computes quantiles loss given the quantiles in self.quantiles
        Args:
            self.quantules
            theta_x [bs, L, T, dim_x]
            x [bs, L, T, dim_x]
            mask [bs, L, T]
            temporal : True
        returns:
            if temporal metric returns a list with the shape of inputs
        """

        theta_x_denorm = self.denormalize_x(theta_x)
        x_denorm = self.denormalize_x(x)

        if len(x.shape) == 3 and temporal==True:
            print("x should have 4 dimensions for temporal case")
            raise NotImplementedError

        if len(x.shape) == 4 and temporal==True:
            mask = mask.unsqueeze(-1).repeat(1,1,1,x.shape[-1]) #torch.Size([64, 5, 40, 1])
            bs, samples, Tm, dim_x = x.shape
            mask_temporal_L_reshape = mask.reshape(bs*samples,Tm,dim_x)
            #how to mask this one with the mask 
            q_loss = quantile_loss(self.quantiles, theta_x_denorm, x_denorm)
            
            return(q_loss)
        

    def compute_elbo(self, mask_temporal, logpx_z, mu_q_z, mu_p_z, logvar_q_z, logvar_p_z):
        """
        Computes ELBO given
        Args:
            mask_temporal [bs, Tm]
            logpx_z  [bs, L, Tm] or [bs, Tm] takes the shape from it
            mu_q_z, logvar_q_z [bs, Tm, dim_latent] [bs, Tm, dim_latent]
            mu_p_z, logvar_p_z [bs, Tm, dim_latent] [bs, Tm, dim_latent]
        Returns:
            ELBO (torch.Tensor): ELBO [bs, L, Tm] where L=1 if logpx_z has shape [bs, Tm]
            logpx_z_sliced (torch.Tensor): logp(x|z) [bs, L, Tm] where L=1 if logpx_z has shape [bs, Tm]
            KL_sliced (torch.Tensor): KL[q(z|x)||p(z|x)] [bs, L, Tm] where L=1 if logpx_z has shape [bs, Tm]
        """

        if len(logpx_z.shape)==3:
            bs,L,Tm = logpx_z.shape
            sampling=True
            mask_temporal_L = mask_temporal[:,None,:].repeat(1,L,1) #[bs, L, T]
        if len(logpx_z.shape)==2:
            bs,Tm = logpx_z.shape
            sampling=False
            mask_temporal_L = mask_temporal        

        #--KL--#
        #computes KL(q||p):  mu_q_z,mu_p_z,logvar_q_z,logvar_p_z -> KL
        #computes ELBO: logpx_z - KL
        KL = self.KL_compute(mu_q_z,mu_p_z,logvar_q_z,logvar_p_z) #(N(mu_q_z, logvar_q_z)||N(mu_q_z, logvar_q_z))
        if sampling:
            KL = KL[:,None,:].repeat(1,L,1) # [bs,L,Tm] torch.Size([64, 5, 40])

        if (torch.isnan(KL).any() or torch.isinf(KL).any())  == True:
            print("nan is catched")
            pass

        if (torch.isnan(logpx_z).any() or torch.isinf(logpx_z).any())  == True:
            print("nan is catched")
            pass
    
        #slicing
        logpx_z_sliced = logpx_z.masked_fill(mask_temporal_L, 0) # [bs, L, Tm]
        KL_sliced = KL.masked_fill(mask_temporal_L, 0) # [bs, L, Tm]

        if (torch.isnan(logpx_z_sliced).any() or torch.isinf(logpx_z_sliced).any())  == True:
            print("nan is catched")
            pass
        
        if (torch.isnan(KL_sliced).any() or torch.isinf(KL_sliced).any())  == True:
            print("nan is catched")
            pass
        
        #---ELBO---#
        ELBO = logpx_z_sliced - KL_sliced # [bs, L, Tm]
        
        # if the input is not sampled (logpx_z has shape [bs, Tm]) then expand the dimension of the metrics to be consistent with the sampled case
        if sampling==False:
            ELBO = ELBO.unsqueeze(1)
            logpx_z_sliced = logpx_z_sliced.unsqueeze(1)
            KL_sliced = KL_sliced.unsqueeze(1)

        return ELBO, logpx_z_sliced, KL_sliced
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)  # use AdamW in case we want to use weight decay?
        lr_scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.scheduler_gamma, verbose=True),
            'interval': 'epoch',
            'frequency': self.scheduler_n_epochs
        }
        return [opt], [lr_scheduler_config]

    
# ============= Extra functions ============= #
class View(nn.Module):
    """
    Reshape tensor inside Sequential objects. Use as: nn.Sequential(...,  View(shape), ...)
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def quantile_loss(quantiles, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # calculate quantile loss
    losses = []
    for q in quantiles:
        errors = target - y_pred
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = 1 * torch.cat(losses, dim=2)

    return losses