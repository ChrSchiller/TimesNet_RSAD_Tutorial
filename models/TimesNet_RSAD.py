import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

### this class avoids an error that occurs because the GELU activation function does not accept two arguments
### but the Inception_Block expects two arguments for the activation function
### so the GELUwrapper essentially just drops the doy argument
class GELUWrapper(nn.Module):
    def __init__(self):
        super(GELUWrapper, self).__init__()
        self.gelu = nn.GELU()
    
    def forward(self, x, _):
        return self.gelu(x)

def apply_sine_embedding(time_tensor, max_time=366):
    """
    This function applies the sine/cosine embedding to the time tensor.
    Assumes the time tensor is in days and the maximum time value is max_time (e.g., 365 for days of the year).
    """

    ### normalize to range [0, 1]
    time_tensor = time_tensor / max_time  

    ### apply sine transformation
    # range [0, 2 * pi] -> [-1, 1]
    sin_embedding = torch.sin(2 * torch.pi * time_tensor) 

    ### return time embedding
    return sin_embedding

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = 0 # configs.pred_len
        ### Inception_Block_V1 uses symmetrical convolutions, meaning that the kernels always span all years
        ### but differ in the time dimension
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            GELUWrapper(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        
        
    def forward(self, x, doy):
        B, T, N = x.size()
        # N is d_model


        ### we dropped Fourier transforms here, because it does not make sense in irregular time series
        res = []

        ### we re-define period, which is always the length of the whole sequence divided by 4
        ### this is because we have a 4-year time series to run the convolutions across
        ### note that in case the model gets adjusted to take more years of data, this will likely have to be adjusted
        period = x.shape[1] // 4 ### seq_len is 200, so this will be 50

        ### padding
        if (self.seq_len + self.pred_len) % period != 0:
            length = (
                                ((self.seq_len + self.pred_len) // period) + 1) * period
            padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = (self.seq_len + self.pred_len)
            out = x
        
        ### reshape
        out = out.reshape(B, length // period, period,
                            N).permute(0, 3, 1, 2).contiguous()
        ### shape is [B, d_model, seq_len // period, period]
        doy = doy.reshape(B, length // period, period)  ### reshape DOY for convolution
        ### shape is [B, seq_len // period, period]

        ### 2D conv: from 1d Variation to 2d Variation
        ### apply each layer in the sequential container individually 
        ### (otherwise error occurs: does not accept two arguments)
        for layer in self.conv:
            out = layer(out, doy)
        
        ### reshape back
        out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
        ### results in shape [batch_size, seq_len + pred_len, d_model]

        ### the following two lines can be simplified (or even dropped?) because they stem from top_k > 1 from Fourier transform
        ### but in the latest code we only have 1 period, so top_k == 1
        res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1).squeeze(-1)
        ### the squeeze operation on the last dimension stems from the fact that we only have top_k == 1
        ### because it is the number of list entries from the list "res"
        ### in the original code, the following commented out lines are used for top_k > 1
        ### which contain an aggregation to remove this last dimension
        ### we don't need that, but we need to squeeze instead 
        ### (or use the aggregation step as well, 
        ### but it is not necessary for top_k == 1)

        ### residual connection
        res = res + x
        return res # [B, seq_len, d_model]


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

	### initialization of the class
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = 0 # configs.pred_len
        ### this sequential nature of the TimesBlock stack means that the consecutive TimesBlock gets the 
        ### output of the previous one, and only the first TimesBlock gets the real input data as input
        ### since there is no pooling for higher-level feature extraction, this is more akin to 
        ### a refinement of the features learned from the previous layers/blocks
        ### nevertheless: each block essentially has access to the output from the previous block, 
        ### allowing it to learn higher-level abstractions over time. So, the model does not explicitly create a "hierarchical" abstraction 
        ### in the sense of pooling layers in CNNs; 
        ### it refines the same sequence progressively across layers
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, # configs.freq,
                                           configs.dropout, len(configs.indices_bands))
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

	### defining the anomaly detection workflow
    ### we added x_mark_enc here, which constitutes the day-of-the-year information
    ### it was not present in the original code, which uses regular time series for anomaly detection
    def anomaly_detection(self, x_enc, x_mark_enc): 
                
        ### we need to exclude the padding values (end padding with zeros)
        ### thus, we acquire a mask
        mask = (x_enc != 0)

        ### embedding step
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C] 
        ### enc_out.shape does not change despite the explicit time information, but the embeddings are different
        
        ### TimesNet RSAD
        ### for loop through the model layers
        for i in range(self.layer):
            ### apply the mask to the input of the model
            # select the first "column" of mask and reshape it for broadcasting
            mask_broadcast = mask[:, :, 0].unsqueeze(-1)
            # broadcast mask_broadcast to match the shape of enc_out and multiply
            enc_out = enc_out * mask_broadcast

            ### normalize the x_mark_enc time tensor and mask it to make sure 0 values are not considered
            x_mark_enc = apply_sine_embedding(x_mark_enc, max_time=366)
            ### mask x_mark_enc to make sure 0 values are not considered
            x_mark_enc = x_mark_enc * mask[:, :, 0]

            enc_out = self.layer_norm(self.model[i](enc_out, x_mark_enc))

        ### project back
        dec_out = self.projection(enc_out) # shape [B, T, C] again

        ### apply the mask to the output to make sure padding values are not considered
        dec_out = dec_out * mask
        return dec_out # reconstructed time series [B, T, C]

    # def classification(self, x_enc, x_mark_enc):
    #     # embedding
    #     enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
    #     # TimesNet
    #     for i in range(self.layer):
    #         enc_out = self.layer_norm(self.model[i](enc_out))

    #     # Output
    #     # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = self.act(enc_out)
    #     output = self.dropout(output)
    #     # zero-out padding embeddings
    #     output = output * x_mark_enc.unsqueeze(-1)
    #     # (batch_size, seq_length * d_model)
    #     output = output.reshape(output.shape[0], -1)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    ### apply the operations defined above to the input data
    ### this is the forward function that is executed when the model is called
    ### it is the main entry point for the model, where the input data is passed through the model
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        ### this is the "real" forward function that passes the input data for the model
        ### it is what is being executed when executing self.model(...)
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc)
        #     return dec_out  # [B, N]
        return None
