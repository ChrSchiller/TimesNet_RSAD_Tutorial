import torch
import torch.nn as nn
import math

### this class models the time-aware convolutional layer
### the following is the parallel version
class TimeAwareConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_sequences=4, padding=0):
        super(TimeAwareConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_sequences = num_sequences
        self.padding = padding

        ### learnable kernel weights shaped for input embeddings, sequences, and kernel size
        self.weights = nn.Parameter(
            torch.randn(out_channels, in_channels, num_sequences, kernel_size)
        )

    def forward(self, x, time_tensor):
        """
        x: Input tensor of shape (batch_size, in_channels, num_sequences, sequence_length).
        time_tensor: Time information tensor of shape (batch_size, num_sequences, sequence_length).
        """
        batch_size, in_channels, num_sequences, sequence_length = x.shape

        ### ensure input matches the expected dimensions
        assert in_channels == self.in_channels, "Input in_channels does not match the initialized value."
        assert num_sequences == self.num_sequences, "Input num_sequences does not match the initialized value."

        ### apply padding to maintain sequence length
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x, 
                (self.padding, self.padding),  ### apply padding along the sequence dimension
                mode='constant', 
                value=0
            )
            time_tensor = torch.nn.functional.pad(
                time_tensor, 
                (self.padding, self.padding), 
                mode='constant', 
                value=0
            )
            sequence_length += 2 * self.padding

        ### unfold the input tensor to get sliding windows
        x_unfolded = x.unfold(-1, self.kernel_size, 1)  # [batch_size, in_channels, num_sequences, sequence_length - kernel_size + 1, kernel_size]
        x_unfolded = x_unfolded.permute(0, 2, 1, 4, 3)  # [batch_size, num_sequences, in_channels, kernel_size, sequence_length - kernel_size + 1]

        ### we also have to apply padding on x_unfolded
        ### because otherwise the convolution will not be able to slide over the entire sequence
        ### and the dimensions do not match with dynamic_weights
        if self.padding > 0:
            x_unfolded = torch.nn.functional.pad(
                x_unfolded, 
                (self.padding, self.padding),  ### apply padding along the sequence dimension
                mode='constant', 
                value=0
            )

        ### calculate dynamic weights for the entire sequence
        dynamic_weights = self.weights.unsqueeze(0).unsqueeze(-1) * time_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        ### dynamic_weights.shape: [batch_size, out_channels, in_channels, num_sequences, kernel_size, sequence_length]

        ### perform convolution using einsum
        ### note that x_unfolded is the input tensor from the observations
		### and dynamic_weights are the time-parameterized kernel weights
        ### for future users of this function: one typical error is to forget to mention each dimension of the tensors
        ### e.g. if "boick,bcikl->bocl" is used, it will throw an error because dynamic_weights has 6 dimensions
        conv_result = torch.einsum("boickl,bcikl->bocl", dynamic_weights, x_unfolded)
        ### output dimensions: [batch_size, out_channels, num_sequences, sequence_length]
        
        ### remove padding from the result if necessary
        if self.padding > 0:
            conv_result = conv_result[..., self.padding:-self.padding]
        
        return conv_result

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            # kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            kernels.append(TimeAwareConv1d(in_channels, out_channels, kernel_size=2 * i + 1, num_sequences=4, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ### attribute is called "weight" not "weights"
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, TimeAwareConv1d):
                ### atribute is called "weights" not "weight"
                nn.init.kaiming_normal_(m.weights, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, doy):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x, doy))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res