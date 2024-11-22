import torch
import torch.distributions as D

from source.models.archs import *
from source.models.ProTranHierarchyDeprecated import (
    get_protran_hierarchy as get_protran_hierarchy_deprecated,
)
from source.models.ProTranHierarchy import (
    get_protran_hierarchy
)
from source.models.likelihood import *
from source.models.AbstractBaseModel import AbstractBaseModel

class ProTranModel(AbstractBaseModel):
    def __init__(self, dataset, train, model, optim, projection_args, embedding_args, conditional_prior_args, encoder_args, decoder_args, deprecated: bool = True, **kwargs) -> None:
        """
        Args:
            deprecated (bool): Whether to use the deprecated ProTranHierarchy or not; the deprecated version is the version used in the original paper
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
                type: (str) type of the model (should be 'protran' here)
                dim_z: (int) dimension for latent z
                likelihood: (str) type of likelihood function
                reconstruct (bool): Whether to reconstruct the history of the sequence or not
                epsilon: (float) epsilon for numerical stability when normalizing sequences
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
                embedding_dim: (int) embedding dimension       
            conditional_prior_args:
                arch_name: (str) name of the cond prior block
                dim_ff_tran: (int | List[int]) unused with this architecture
                num_heads: (int) number of heads for the transformer
                num_layers: (int) number of layers for the transformer
                causal: (bool) to make the architecture causal # currently UNUSED
                dim_list: (list) MLP to generate z^(l)_t from \hat{w}^(l)_t, used in every layer, not just the last one
                dim_list_latent_to_w: (list) MLP to generate w^(l)_t from z^(l)_t 
                use_sampling (bool): Whether to sample z_t from the mean and variance of the latent variable z_t in the ProTranConPriorLayers. 
                    If False, we use the mean of the latent variable z_t.
                kind_attentions (List[str]): Unused here, only for foretran
            encoder_args:
                dim_list: (list) MLP to generate z^(l)_t from \hat{w}^(l)_t AND full history 
                rest unused, as the architecture is shared with the conditional prior: Always the same as conditional prior          
            decoder_args
                arch_name: 'MLP'
                dim_list: (list) number of hidden units of decoder
                variance: (float) variance for the likelihood function, only used for gaussian and laplace
            testing:
                samples: (int) number of samples used in test mode
            mode:
                training: (bool) to state if train the model
                testing: (bool) to state if test the model
                wandb_ckpt_name: (str) name of the checkpoint for wandb logging
            
        """
        epsilon = model["epsilon"]
        reconstruct = model["reconstruct"]
        assert reconstruct == True, "The Protran model always reconstructs the history of the sequence."
        assert train["share_arch"] == True, "The Protran model always shares architectures for conditional prior and encoder."
        assert conditional_prior_args["arch_name"] == "transformer", "The Protran model always uses transformer for conditional prior."
        super(ProTranModel, self).__init__(model_type='protran', reconstruct=reconstruct, epsilon=epsilon)
        
        # maybe add a causal version with masking? --> kind of defeats the purpose of the original paper and their hierachies

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

        self.dim_latent = model["dim_z"]
        self.use_sampling = conditional_prior_args["use_sampling"]
        self.likelihood_x = model["likelihood"]

        self.batch_size= train["batch_size"]
        self.ckpt_period = train["ckpt_period"]
        self.share_arch = train["share_arch"]

        self.lr =  optim["base_lr"]
        self.scheduler_gamma = optim["scheduler_gamma"]
        self.scheduler_n_epochs = optim["scheduler_n_epochs"]

        # self.prediction_metric_name = None

        self.initialize_buffer() #this is for saving means and stds

        assert projection_args['dim_list'][0]==self.dim_x+self.dim_c
        
        assert embedding_args["mode"]=="MODE_ADD", "only MODE_ADD is supported for ProTranModel"

        if embedding_args["mode"]=="MODE_ADD":
            assert embedding_args["embedding_dim"] == projection_args['dim_list'][-1]
            self.dim_h = projection_args['dim_list'][-1]

        assert embedding_args["max_len"] == dataset["tm"]

        assert conditional_prior_args["dim_list"][0] == conditional_prior_args["dim_list_latent_to_w"][-1] == encoder_args["dim_list"][0] == self.dim_h
        assert conditional_prior_args["dim_list"][-1] == conditional_prior_args["dim_list_latent_to_w"][0] == encoder_args["dim_list"][-1] == self.dim_latent    
        
        # assert decoder_args["dim_list"][0] == self.dim_latent
        assert decoder_args["dim_list"][0] == self.dim_h # as we decode w instead of z
        assert decoder_args["dim_list"][-1] == self.dim_x


        self.projection_layer = get_projection_layer('projectionLayer', **projection_args) 
        self.embedding_layer = get_embedding_layer( **embedding_args)
        self.ln = torch.nn.LayerNorm(self.dim_h)

        conditional_prior_dropout = conditional_prior_args.get("dropout", None)
        encoder_dropout = encoder_args.get("dropout", None)
        assert conditional_prior_dropout == encoder_dropout, "dropout should be the same for conditional prior and encoder"
        conditional_prior_activation = conditional_prior_args.get("activation", None)
        encoder_activation = encoder_args.get("activation", None)
        assert conditional_prior_activation == encoder_activation, "activation should be the same for conditional prior and encoder"
        add_args = {}
        if conditional_prior_dropout is not None:
            add_args["dropout"] = conditional_prior_dropout
        if conditional_prior_activation is not None:
            add_args["activation"] = conditional_prior_activation
        
        if deprecated:
            self.deprecated = True
            self.protran_hierarchy = get_protran_hierarchy_deprecated(
                num_layers=conditional_prior_args["num_layers"],
                d_model= self.dim_h, 
                dim_latent= self.dim_latent,
                dim_list_latent_cond_prior = conditional_prior_args["dim_list"],
                dim_list_latent_to_w = conditional_prior_args["dim_list_latent_to_w"],
                dim_list_latent_encoder = encoder_args["dim_list"],
                num_heads=conditional_prior_args["num_heads"],
                positional_encoding = self.embedding_layer, 
                **add_args
            )
        else:
            self.deprecated = False
            self.protran_hierarchy = get_protran_hierarchy(
                num_layers=conditional_prior_args["num_layers"],
                d_model= self.dim_h, 
                dim_latent= self.dim_latent,
                dim_list_latent_cond_prior = conditional_prior_args["dim_list"],
                dim_list_latent_to_w = conditional_prior_args["dim_list_latent_to_w"],
                dim_list_latent_encoder = encoder_args["dim_list"],
                num_heads=conditional_prior_args["num_heads"],
                positional_encoding = self.embedding_layer, 
                **add_args
            )

        self.decoder = get_decoder(likelihood=self.likelihood_x,**decoder_args)
        
        self.save_hyperparameters()
        
    def _forward(self, batch:tuple, samples: int = 1, reconstruct = True) -> tuple:
        """
        Args:
            batch (tuple): has elements (X,C,T0,T)
                X (torch.Tensor): input data with shape [bs, Tm, dim_x]
                C (torch.Tensor): input data with shape [bs, Tm, dim_c]
                T0: length of history sequence [bs, 1]
                T: total length of sequence [bs, 1]
            samples (int): number of latent samples in bottleneck to be sampled from // q //
            
        Returns (x, x_hat, ELBO, logpx_z, KL, mask_temporal):
            x (torch.Tensor): input data [bs, Tm, dim_x]
            x_p_hat (torch.Tensor): reconstructed data by decoding encoder latent z_q [bs, samples, Tm, dim_x]
            x_q_hat (torch.Tensor): reconstructed data by decoding conditional prior latent z_p [bs, samples, Tm, dim_x]   
            ELBO (torch.Tensor): ELBO [bs, samples, Tm]
            logpx_z (torch.Tensor): logpx_z [bs, samples, Tm]
            KL (torch.Tensor): KL [bs, samples, Tm]
            mask_temporal (torch.Tensor): temporal mask to only consider the parts we are interested in
        """
        assert samples > 0, "samples should be greater than 0"
        assert reconstruct == True, "The Protran model always reconstructs the history of the sequence."
        if samples > 1:
            print("WARNING: in _forward we only sample from the bottleneck. However, since Protran has dependencies of the latent samples througout it's layers, the samples generated here do not reflect this dependency. See predict_and_gt_denormalized for the correct sampling.")
        
        x, c, T0, T = batch
        bs, Tm, dim_x = x.shape
        
        if not (x.device == c.device == T0.device == T.device == self.device):
            x, c, T0, T = x.to(self.device), c.to(self.device), T0.to(self.device), T.to(self.device)
        
        history_mask = self.create_history_mask(T0, Tm) # [bs, Tm]
        padding_mask = self.create_padding_mask(T, Tm) # [bs, Tm]
        history_mask = history_mask | padding_mask # [bs, Tm]

        xc = torch.concat((x,c),dim=-1) # [bs, Tm, dim_x + dim_c]
        xc = self.projection_layer(xc)
        xc_embedding = self.embedding_layer(xc) #[bs, Tm, dim_xc]
        xc_embedding = self.ln(xc_embedding) #[bs, Tm, dim_xc]
        
        
        #input for the conditional prior
        prior_mask = self.create_prior_input_mask(bs, Tm, T0, T, dim_x) #creating mask for creating the input for conditional prior network
        # note that padding is already done with the data, so we only need to mask the data after T0
        x_prior = x.masked_fill(prior_mask, 0) # [bs, Tm, dim_x] 
        xc_prior = torch.concat((x_prior,c),dim=-1) # [bs, Tm, dim_x + dim_c] 
        xc_prior = self.projection_layer(xc_prior)
        xc_prior_embedding = self.embedding_layer(xc_prior) #[bs, Tm, dim_xc] if the mode is add
        xc_prior_embedding = self.ln(xc_prior_embedding) #[bs, Tm, dim_xc]
        
        # --- Conditional Prior p(z|x) and Encoder q(z|x) at once ---
        if self.deprecated:
            mu_p_z, var_p_z, mu_q_z, var_q_z, w_p_hat, w_p, w_q_hat, w_q = self.protran_hierarchy(xc_prior_embedding, xc_embedding, bs, Tm, history_mask, padding_mask, self.device, self.use_sampling) # all [num_layers, batch_size, Tm, dim_latent], except w, w_hat [batch_size, Tm, d_model]
        else:
            # Adjustment to fix wrong ELBO
            mu_p_z, var_p_z, mu_p_z_loss, var_p_z_loss, mu_q_z, var_q_z, w_p_hat, w_p, w_q_hat, w_q = self.protran_hierarchy(xc_prior_embedding, xc_embedding, bs, Tm, history_mask, padding_mask, self.device, self.use_sampling) # all [num_layers, batch_size, Tm, dim_latent], except w, w_hat [batch_size, Tm, d_model]
        # avoid resampling z if we only need one sample
        if samples == 1:
            w_q = w_q[:,None,:,:] # [bs, samples, Tm, d_model] with samples=1
            w_p = w_p[:,None,:,:] # [bs, samples, Tm, d_model] with samples=1
        else:
            z_q = self.sample_z(mu_q_z[-1], var_q_z[-1], samples) # [batch_size, samples, Tm, dim_latent]
            w_q = self.protran_hierarchy.get_w_samples(w_q_hat, z_q) # [batch_size, samples, Tm, d_model]
            z_p = self.sample_z(mu_p_z[-1], var_p_z[-1], samples) # [batch_size, samples, Tm, dim_latent]
            w_p = self.protran_hierarchy.get_w_samples(w_p_hat, z_p) # [batch_size, samples, Tm, d_model]

        # --- Decoder p(x|z) ---
        # NOT mu_x, logvar_x = self.decoder(z) # [batch_size, samples, Tm, dim_x] [batch_size, samples, Tm, dim_x]
        # BECAUSE BASELINE DECODES W INSTEAD OF Z
        # stack w_q and w_p to process them at once
        w = torch.cat((w_p, w_q), dim=0)    # [2*bs, samples, Tm, d_model]
        mu_x, logvar_x = self.decoder(w)    # [2*batch_size, samples, Tm, dim_x] [2*batch_size, samples, Tm, dim_x]
        var_x = torch.exp(logvar_x)         # [2*batch_size, samples, Tm, dim_x]
        # split mu_x and var_x
        mu_p_x, mu_q_x = torch.chunk(mu_x, 2, dim=0)    # [batch_size, samples, Tm, dim_x] [batch_size, samples, Tm, dim_x]
        var_p_x, var_q_x = torch.chunk(var_x, 2, dim=0) # [batch_size, samples, Tm, dim_x] [batch_size, samples, Tm, dim_x]
        
        # we fix the scaling (variance) for stability, hence we ignore the variance here
        x_p_hat = self.build_x_hat(mu_p_x) # [bs, samples, Tm, dim_x] for gaussian or laplace it just returns the same mu_x
        x_q_hat = self.build_x_hat(mu_q_x) # [bs, samples, Tm, dim_x] for gaussian or laplace it just returns the same mu_x
        
        # E_q[log p(x|z)] 
        logpx_z_samples = self.decoder.logp(x=x, theta=x_q_hat).sum(-1) # [bs, samples, Tm]
        logpx_z = logpx_z_samples.squeeze(1) # [bs, Tm] or [bs, samples, Tm] if samples > 1
        
        mask_temporal = padding_mask # [bs, Tm]
        
        if self.deprecated:
            ELBO, logpx_z_sliced, KL_sliced = self.compute_elbo(mask_temporal, logpx_z, mu_q_z, mu_p_z, var_q_z, var_p_z)
        else:
            ELBO, logpx_z_sliced, KL_sliced = self.compute_elbo(mask_temporal, logpx_z, mu_q_z, mu_p_z_loss, var_q_z, var_p_z_loss)
        
        return x, x_p_hat, x_q_hat, ELBO, logpx_z_sliced, KL_sliced, mask_temporal
       
       
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
        assert reconstruct == True, "The Protran model always reconstructs the history of the sequence."
        x, c, T0, T = batch
        bs, Tm, dim_x = x.shape
        
        if not (x.device == c.device == T0.device == T.device == self.device):
            x, c, T0, T = x.to(self.device), c.to(self.device), T0.to(self.device), T.to(self.device)
        
        history_mask = self.create_history_mask(T0, Tm) # [bs, Tm]
        padding_mask = self.create_padding_mask(T, Tm) # [bs, Tm]
        history_mask = history_mask | padding_mask # [bs, Tm]

        #creating mask
        prior_mask = self.create_prior_input_mask(bs, Tm, T0, T, dim_x)
        x_prior = x.masked_fill(prior_mask, 0) # [bs, Tm, dim_x]
        xc_prior = torch.concat((x_prior,c),dim=-1) # [bs, Tm, dim_x + dim_c]
        xc_prior = self.projection_layer(xc_prior)
        xc_prior_embedding = self.embedding_layer(xc_prior) #[bs, Tm, dim_xc]
        xc_prior_embedding = self.ln(xc_prior_embedding) #[bs, Tm, dim_xc]
        
        # Run the forward pass samples times to ensure that we get all dependencies of the latent samples
        
        # --- Conditional Prior p(z|x) ---
        w_samples = []
        w_mu_samples = []
        for _ in range(samples):
            mu_p_z, var_p_z, w_p_hat, w_p = self.protran_hierarchy.predict(xc_prior_embedding, bs, Tm, history_mask, padding_mask, self.device, self.use_sampling) # all [num_layers, batch_size, Tm, dim_latent], except w, w_hat [batch_size, Tm, d_model]
        
            w_mu = self.protran_hierarchy.get_w_samples(w_p_hat, mu_p_z[-1]) # [batch_size, Tm, d_model]
            w_samples.append(w_p)
            w_mu_samples.append(w_mu)
        
        w = torch.stack(w_samples, dim=1) # [batch_size, samples, Tm, d_model]
        w_mu = torch.stack(w_mu_samples, dim=1).mean(dim=1) # [batch_size, Tm, d_model]
        
        # theta_x_mu_z, _ = self.decoder(mu_p_z[-1]) # [bs, Tm, dim_x]
        # theta_x, _ = self.decoder(z) # [bs, samples, Tm, dim_x]  
        # BECAUSE BASELINE DECODES W INSTEAD OF Z
        theta_x_mu_z, _ = self.decoder(w_mu) # [bs, Tm, dim_x]
        theta_x, _ = self.decoder(w) # [batch_size, samples, Tm, dim_x]
        
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
    
     
        
    def compute_elbo(self, mask_temporal:Tensor, logpx_z:Tensor, mu_q_z:Tensor, mu_p_z:Tensor, var_q_z:Tensor, var_p_z:Tensor) -> tuple:
        r"""
        Args:
            mask_temporal (torch.Tensor): mask for temporal dimension [bs, 1, Tm]
            logpx_z (torch.Tensor): logp(x|z) [bs, samples, Tm] or [bs, Tm]
            mu_q_z (torch.Tensor): mean of q(z|x) [num_layers, batch_size, Tm, d_model]
            mu_p_z (torch.Tensor): mean of p(z|x) [num_layers, batch_size, Tm, d_model]
            var_q_z (torch.Tensor): variance of q(z|x) [num_layers, batch_size, Tm, d_model]
            var_p_z (torch.Tensor): variance of p(z|x) [num_layers, batch_size, Tm, d_model]
        Returns:
            ELBO (torch.Tensor): ELBO [bs, samples, Tm] where samples=1 if logpx_z is [bs, Tm]
            logpx_z_sliced (torch.Tensor): logp(x|z) [bs, samples, Tm] where samples=1 if logpx_z is [bs, Tm]
            KL_sliced (torch.Tensor): KL[q(z|x)||p(z|x)] [bs, samples, Tm] where samples=1 if logpx_z is [bs, Tm]
        """
        if len(logpx_z.shape)==3:
            bs, samples, Tm = logpx_z.shape
            sampling = True
            mask_temporal_L = mask_temporal.unsqueeze(1).repeat(1,samples,1) #[bs, L, T]
        if len(logpx_z.shape)==2:
            bs, Tm = logpx_z.shape
            sampling = False
            mask_temporal_L = mask_temporal        

        #--KL--#
        #computes KL(q||p):  mu_q_z, mu_p_z, logvar_q_z, logvar_p_z -> KL for each layer of the ProTranHierarchy
        #computes ELBO: logpx_z - KL
        KL = self.KL_layered_compute(mu_q_z, mu_p_z, var_q_z, var_p_z) #(N(mu_q_z, logvar_q_z)||N(mu_q_z, logvar_q_z)) for each layer of the ProTranHierarchy
        # [bs, Tm]
        if sampling:
            KL = KL[:,None,:].repeat(1,samples,1) # [bs, samples, Tm] 

        if (torch.isnan(KL).any() or torch.isinf(KL).any())  == True:
            print("nan is caught")
            pass

        if (torch.isnan(logpx_z).any() or torch.isinf(logpx_z).any())  == True:
            print("nan is caught")
            pass
    
        #slicing
        logpx_z_sliced = logpx_z.masked_fill(mask_temporal_L, 0) # [bs, samples, Tm]
        KL_sliced = KL.masked_fill(mask_temporal_L, 0) # [bs, samples, Tm]

        if (torch.isnan(logpx_z_sliced).any() or torch.isinf(logpx_z_sliced).any())  == True:
            print("nan is caught")
            pass
        
        if (torch.isnan(KL_sliced).any() or torch.isinf(KL_sliced).any())  == True:
            print("nan is caught")
            pass
        
        #---ELBO---#
        ELBO = logpx_z_sliced - KL_sliced # [bs, samples, Tm]
        
        # if the input is not sampled (logpx_z has shape [bs, Tm]) then expand the dimension of the metrics to be consistent with the sampled case
        if sampling==False:
            ELBO = ELBO.unsqueeze(1)
            logpx_z_sliced = logpx_z_sliced.unsqueeze(1)
            KL_sliced = KL_sliced.unsqueeze(1)

        return ELBO, logpx_z_sliced, KL_sliced
    
        
    def KL_layered_compute(self, mu0:Tensor, mu1:Tensor, var0:Tensor, var1:Tensor):
        '''
        computes KL(N(mu0,logvar0)||N(mu1,logvar1)) over the layers of the ProTranHierarchy
        Args:
            mu0 (torch.Tensor): mean of q(z|x) [num_layers, batch_size, Tm, d_model]
            mu1 (torch.Tensor): mean of p(z|x) [num_layers, batch_size, Tm, d_model]
            var0 (torch.Tensor): variance of q(z|x) [num_layers, batch_size, Tm, d_model]
            var1 (torch.Tensor): variance of p(z|x) [num_layers, batch_size, Tm, d_model]
        Returns:
            kl (torch.Tensor): KL[q(z|x)||p(z|x)] [batch_size, Tm]
        '''
        
        kl = -0.5 * torch.sum(1. + torch.log(var0 / var1) - (mu1-mu0) * (mu1-mu0) / var1 - var0/var1, dim=-1, keepdim=False) # [num_layers, batch_size, Tm]
        kl = kl.sum(dim=0) # [batch_size, Tm], sum over layers

        return kl # // output dimension [bs, Tm]
    
    
    def create_history_mask(self, T0:Tensor, Tm:int) -> Tensor:
        r"""
        Args:
            T0 (torch.Tensor): length of history sequence (batch_size, 1)
            Tm (int): length of whole sequence (Hist+Forecasting+Padding)
        Returns:
            history_mask (torch.Tensor): boolean tensor for only employing attention on history sequence (batch_size, Tm)
        """
        key_padding_mask = torch.arange(Tm).to(self.device).unsqueeze(0) >= T0 # [bs, Tm]
        return key_padding_mask
    
    
    def create_padding_mask(self, T:Tensor, Tm:int) -> Tensor:
        r"""
        Args:
            T (torch.Tensor): total length of sequence (batch_size, 1) (Hist+Forecasting)
            Tm (int): length of whole sequence (Hist+Forecasting+Padding)
        Returns:
            padding_mask (torch.Tensor): boolean tensor for only employing attention on forecasting sequence, not padding (batch_size, Tm)
        """
        padding_mask = torch.arange(Tm).to(self.device).unsqueeze(0) >= T # [bs, Tm]
        return padding_mask
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)  # use AdamW in case we want to use weight decay?
        lr_scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.scheduler_gamma, verbose=True),
            'interval': 'epoch',
            'frequency': self.scheduler_n_epochs
        }
        return [opt], [lr_scheduler_config]
    
    
    