import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class LearnablePositionalEncoding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 max_len:int=5,
                 embedding_dim:int=20,
                 mode=MODE_ADD):
        """
        Initialization
        Note: it learns an embedding independent of the input (?)
        Args:
            max_len (int): length of the embeddings (temporal) // INFO this should be fixed to maximum length Tm
            embedding_dim (int): dimension of the output of the embedding to be fed into model.
            mode (embedding mode): 
                MODE_ADD adds the embedding on input (dim_embedding should be equal to dim_x)
                MODE_CONCAT concats the embedding with input
        """

        super(LearnablePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(self.max_len * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.max_len, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(input, -self.max_len, self.max_len) + self.num_embeddings # BUG: but unused, num_embeddings is not defined
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len, dim_input = input.size()[:3]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            assert dim_input == self.embedding_dim
            return input + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((input, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'max_len={}, embedding_dim={}, mode={}'.format(
            self.max_len, self.embedding_dim, self.mode,
        )
        
    def get_single_positional_encoding(self, x:torch.Tensor, pos:int) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape [batch_size, 1, dim_h]
            pos: position of the embedding
        """
        assert x.shape[1] == 1
        if self.mode == self.MODE_EXPAND:
            raise ValueError ("MODE_EXPAND not implemented for single positional encoding")
        embeddings = self.weight[pos:pos+1, :].view(1, 1, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            assert x.shape[2] == self.embedding_dim
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(x.shape[0], 1, 1)), dim=-1)
        


class FixedPositionalEncoding(nn.Module):

    def __init__(self, max_len: int = 50,
                    embedding_dim: int= 12, 
                    dropout: float = 0.1):
        """
        Initialization
        Args:
            max_len (int): length of the embeddings (temporal) // INFO this should be fixed to maximum length Tm
            embedding_dim (int): dimension of the output of the embedding to be fed into model.
            mode (): additive by default.
        """


        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :,0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape [batch_size, seq_len, dim_h] | [batch_size, samples, seq_len, dim_h]
        """
        # x = torch.permute(x,(1,0,2))
        # print(self.pe.shape)
        x = x + self.pe[:, :x.size(-2)] # second last dimension is the sequence length
        return self.dropout(x)
    
    def get_single_positional_encoding(self, x: torch.Tensor, pos:int) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape [batch_size, 1, dim_h]
            pos (int): position of the embedding
        """
        assert x.shape[1] == 1
        assert x.dim() == 3
        x = x + self.pe[:, pos:pos+1]
        return self.dropout(x)

class ProjectionLayer(nn.Module):

    def __init__(self, dim_list: list= [] , non_linearity: bool= False):
        """
        It projects the input and embedding to the model dim aka dim_h
        Args:
            dim_list (list): weights of layer for Linear network
            non_linearity:  boolean to turn on/off ReLU #activation
        """

        super().__init__()
        self.dim_input = dim_list[0]
        self.dim_h = dim_list[-1]

        self.modules = []
        self.non_linearity = non_linearity
        
        for i in range(len(dim_list)-1):
            self.modules.append(nn.Linear(dim_list[i], dim_list[i+1]))
            if self.non_linearity:
                self.modules.append(torch.nn.Relu())

        self.sequential = nn.Sequential(*self.modules)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, dim_input]
        
        Returns:
            output: Tensor, shape [batch_size, seq_len, dim_h]
        """

        output = self.sequential(x)

        return output