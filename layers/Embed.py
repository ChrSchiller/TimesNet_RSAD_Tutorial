import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def doy_to_month(doy):
    ### approximate mapping of day of the year to month
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = torch.zeros_like(doy)
    cumulative_days = 0
    for i, days in enumerate(month_days):
        cumulative_days += days
        month += (doy <= cumulative_days).int() * (month == 0).int() * (i + 1)
    return month - 1 # adjust to 0-based indexing

def normalize_doy(x_mark):
    while x_mark.max() > 365:
        x_mark = x_mark - 365 * (x_mark > 365).int()
    return x_mark

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1462):
        super(PositionalEmbedding, self).__init__()
        ### compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

### reconsider the TokenEmbedding class: kernel_size = 3 for embeddings of the satellite channels?
### does that mean only three neighboring channels can be considered?
class TokenEmbeddingCNN(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=10):
        super(TokenEmbeddingCNN, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        ### note that we have changed the default kernel_size from 3 to 10
        ### because we have 10 satellite channels and a kernel of size 3 
        ### would only consider the three neighboring channels
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=kernel_size, padding='same', padding_mode='circular', bias=False)
        ### since we changed the kernel_size to always be equal to the number of input features, 
        ### we don't need any padding anymore
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                            kernel_size=1, padding=0, bias=False)
        ### explanation of the above: 
        ### the kernel_size parameter essentially defines the number of timesteps
        ### of the moving window of the Conv1d layer, 
        ### and as we only want to tokenize satellite bands and not timesteps at all, 
        ### we should set it to 1 (padding=0 is no problem then)
        ### kernel_size is used as a hyperparameter here because my understanding was not entirely correct
        ### for the final code version, we can probably remove that and keep kernel_size=1 all the time
        ### that is also why Benny said "a Conv1d layer with kernel_size of 1 is essentially 
        ### a token embedding using a MLP, but more efficient"
        ### here's the explanation that really helped me: 
        ### https://pub.aimind.so/lets-understand-the-1d-convolution-operation-in-pytorch-541426f01448
        ### note that in_channels is the relevant parameter here to operate across all input channels
        ### that's why it is always set to the number of channels in the input data
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

### this class provides TRAINABLE (not fixed) sine and cosine embeddings
### this is the class that we should use for our DOY embeddings
### it accounts for the cyclical nature of our data (forest time series)
### and should be able to capture the yearly seasonality
class TrainableSinCosEmbedding(nn.Module):
    # num_embeddings = size of vocabulary
    def __init__(self, num_embeddings, embedding_dim):
        super(TrainableSinCosEmbedding, self).__init__()

        ### initialize the embeddings using sine and cosine functions
        position = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        embeddings = torch.zeros(num_embeddings, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        ### create an nn.Embedding layer and set its weights to the initialized embeddings
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed.weight = nn.Parameter(embeddings)
        self.embed.weight.requires_grad = True  # make the embeddings trainable

    def forward(self, x):
        return self.embed(x)

### make sure that embed_type is NOT "fixed"
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='learnedsincos', freq='d'):
        super(TemporalEmbedding, self).__init__()

        doy_size = 366 # each of the four years get the normal doy values

        ### three options: 
        ### TrainableSinCosEmbedding: trainable + sin/cos embeddings account for cyclical nature of data
        ### FixedEmbedding: sin/cos but fixed (not updated during training)
        ### nn.Embedding: not fixed, but not sin/cos embeddings (not accounting for cyclical nature)
        Embed = TrainableSinCosEmbedding if embed_type == 'learnedsincos' else nn.Embedding

        ### initialize day of the year embedding
        self.doy_embed = Embed(doy_size, d_model)

    def forward(self, x):
        x = x.long()
        doy_x = self.doy_embed(x)

        ### this would be the place where we can switch to concatenation of the embeddings
        ### instead of summing them up
        return doy_x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='learnedsincos', dropout=0.5, kernel_size=10, freq='h'):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbeddingCNN(c_in=c_in, d_model=d_model, kernel_size=kernel_size) # c_in = input channels
        ### alternatively, we can use other embedding methods (not used in the final code):
        # self.value_embedding = TokenEmbeddingMLP(c_in=c_in, d_model=d_model) # c_in = input channels
        # self.value_embedding = TokenEmbeddingAttention(c_in=c_in, d_model=d_model, nhead=8)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        ### since we do NOT pass "timeF", we never use TimeFeatureEmbedding (which is not useful for our use case)
        
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    ### x_mark is the timestamp information
    ### that means that we can use this to embed the day of year
    def forward(self, x, x_mark):
        ### add temporal embeddings to the value embeddings
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)