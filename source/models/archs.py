from torch import nn
from source.models.TransformerBNEncoderLayer import *
from source.models.embedding import *
from source.models.conditional_prior import *
from source.models.encoder import *
from source.models.decoder import *
from source.models.ForetranTransformer import *
from source.models.act_functions import _get_activation_fn

# You can add more complex architectures here

def generate_square_subsequent_mask(size): # Generate mask covering the top right triangle of a matrix
        # mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = (torch.triu(torch.ones(size, size)) == 0).transpose(0,1)
        return mask

def get_projection_layer(arch_name: str="projectionLayer", dim_list = None, non_linearity=False):
    """
    Gets projection layer for input
    Args:
        arch_name (str, optional): name of the architecture. Defaults to 'projectionLayer'
        dim_list (list, int): weights of the layers
        non_linearity (bool): boolean for turn on/off non_linearity
    """

    if arch_name=="projectionLayer":
        projection_layer = ProjectionLayer(dim_list, non_linearity)
    return projection_layer


def get_embedding_layer(arch_name: str="learnable", max_len:int=5, embedding_dim:int=20,mode='MODE_ADD'):
    """
    Gets embedding layer for input
    Args:
        arch_name (str, optional): name of the architecture. Defaults to 'base'.
        max_len: length of the embedding, should be equal to Tm
        embedding_dim: dimension of the embedding
        mode: mode of the embedding to combine with input
            MODE_ADD: sums input with the embedding
            MODE_CONCAT: concatenates the input with the embedding in the last dim 
    """

    if arch_name=="fixed":
        embedding_network = get_pos_encoder("fixed")(max_len, embedding_dim, 0)
    
    if arch_name=="learnable":
        embedding_network = get_pos_encoder("learnable")(max_len, embedding_dim, mode)

    return embedding_network
    
def get_conditional_prior(dim_ff_tran: int | List[int], arch_name: str='transformer', tm:int =None, tmh: int =None, tmf: int = None,  num_heads: int= 8, num_layers: int= 6, dim_list=[], **kwargs):
    
    """
    Get NNs for conditional prior to model p(z|x,c) 
    Args:
        dim_ff_tran (int| List[int]): dimension(s) of the feed forward network in the transformer
        arch_name (str, optional): name of the architecture. Defaults to 'transformer'.
        dim_list: dimensions for the MLP
            first one is dim_h
            last one is dim_latent
            dim_h (int): dimension of the input and hidden state of transformer
            dim_latent (int): dimension of the latent state (z)
        num_heads (int): number of heads of the transformer
        num_layers (int): number of layers of the transformer
        causal (bool): mask of the transformer
    Returns:
        p(z|h)p(h|x,c)    #x is masked
            torch.nn.TransformerEncoder:   p(h|x,c)
            torch.nn.Sequential:    p(z|h)
    """
    try:
        activation = kwargs['activation']
    except:
        activation = 'relu'

    if arch_name=='transformer':
        assert not isinstance(dim_ff_tran, list), "dim_ff_tran is not supported as list for transformer"
        print(f"number of heads {num_heads}, layers {num_layers}")
        layer = TransformerBatchNormEncoderLayer(d_model=dim_list[0], nhead=num_heads, dim_feedforward=dim_ff_tran, batch_first=True)
        transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)

        network = TransformerConditionalPrior(type='Transformer', transformer_network=transformer, mlp_network=mlp, tm=tm, tmh=tmh, tmf=tmf)
        print("Transformer model is created")
        
    if arch_name=='foretran_transformer':
        assert isinstance(dim_ff_tran, list), "dim_ff_tran is not supported as int for foretran_transformer"
        print(f"number of heads {num_heads}, layers {num_layers}")
        transformer = ForetranTransformer(d_model=dim_list[0], nhead=num_heads, dim_feedforward=dim_ff_tran, num_layers=num_layers, activation=activation, 
                                          kind_attentions=kwargs['kind_attentions'])
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)

        network = TransformerConditionalPrior(type='Transformer', transformer_network=transformer, mlp_network=mlp, tm=tm, tmh=tmh, tmf=tmf)
        print("Foretran Transformer is created")

    if arch_name=='MLP':
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)

        network = MLPConditionalPrior(type='MLP', network=mlp)
        print("MLP model is created")
        
        
    # You can add more complex architectures here
    return network

def get_encoder(dim_ff_tran: int | List[int]=None, arch_name: str='transformer', tm:int =None, tmh: int =None, tmf: int = None, num_heads: int= 8, num_layers: int= 6, dim_list= [], **kwargs):
    
    """
    Get NNs for encoder to model q(z|x,c) 
    Args:
        dim_ff_tran (int | List[int], optional): dimension(s) of the feed forward network in the transformer
        arch_name (str, optional): name of the architecture. Defaults to 'transformer'.
        dim_list: dimensions for the MLP
            first one is dim_h
            last one is dim_latent
            dim_h (int): dimension of the input and hidden state of transformer
            dim_latent (int): dimension of the latent state (z)
        num_heads (int): number of heads of the transformer
        num_layers (int): number of layers of the transformer
        causal (bool): mask of the transformer
    Returns:
        q(z|h)q(h|x,c)    
            torch.nn.TransformerEncoder:   q(h|x,c)
            torch.nn.Sequential:    q(z|h)
    """

    try:
        activation = kwargs['activation']
    except:
        activation = 'relu'

    
    if arch_name=='transformer':
        assert not isinstance(dim_ff_tran, list), "dim_ff_tran is not supported as list for transformer"
        layer = TransformerBatchNormEncoderLayer(d_model=dim_list[0], nhead=num_heads, dim_feedforward=dim_ff_tran, batch_first=True)
        transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)
        print("Transformer model is created")
        network = TransformerEncoder(type='Transformer', transformer_network=transformer, mlp_network=mlp, tm=tm, tmh=tmh, tmf=tmf)
        
        
    if arch_name=='foretran_transformer':
        assert isinstance(dim_ff_tran, list), "dim_ff_tran is not supported as int for foretran_transformer"
        print(f"number of heads {num_heads}, layers {num_layers}")
        transformer = ForetranTransformer(d_model=dim_list[0], nhead=num_heads, dim_feedforward=dim_ff_tran, num_layers=num_layers, activation=activation, 
                                          kind_attentions=kwargs['kind_attentions'])
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)

        network = TransformerEncoder(type='Transformer', transformer_network=transformer, mlp_network=mlp, tm=tm, tmh=tmh, tmf=tmf)
        print("Foretran Transformer is created")

    if arch_name=='MLP':
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)

        network = MLPEncoder(type='MLP', network=mlp)
        print("MLP model is created")
        
        
    if arch_name=='encoder_head':
        attention = nn.MultiheadAttention(embed_dim=dim_list[0], num_heads=num_heads, dropout=0.1, batch_first=True)
        
        modules = []
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))  
            elif i == 0:
                modules.append(nn.Linear(2 * dim_list[i], dim_list[i+1]))
                modules.append(nn.ReLU())  
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                modules.append(_get_activation_fn(activation))
        mlp = nn.Sequential(*modules)
        
        
        network = EncoderHead(input_attention_head=attention, mlp_network=mlp)
        print("EncoderHead model is created")
    # You can add more complex architectures here
    return network

def get_decoder(likelihood:str, arch_name: str='MLP', dim_list: list=[], variance=None, **kwargs):
    """
    Get NNs for likelihood to model p(x|z), specifically p(x_t|z_t)
    Args:
        likelihood (str): likelihood function
        arch_name (str, optional): name of the architecture. Defaults to 'mlp'.
        dim_list (list): weight of the MLPs
            first element is dim_latent (int): dimension of the latent state (z)
            last element is dim_output (int): dimension of the output (x)
    Returns:
        torch.nn.Sequential:    p(x|z)
    """
    try:
        activation = kwargs['activation']
    except:
        activation = 'relu'

    
    if arch_name=='MLP' or arch_name=="mlp":
        modules = []
        non_linearity = False
        
        for i in range(len(dim_list)-1):
            if i+1 == len(dim_list)-1:
                modules.append(nn.Linear(dim_list[i], 2*dim_list[i+1]))    
            else:
                modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
                if non_linearity:
                    modules.append(_get_activation_fn(activation))

        sequential = nn.Sequential(*modules)
        decoder_network =MLPDecoder(likelihood=likelihood, type="MLP", network = sequential, variance= variance)

    return decoder_network