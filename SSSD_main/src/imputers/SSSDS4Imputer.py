import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../../../SSSD_main')
from SSSD_main.src.utils.util import calc_diffusion_step_embedding
from SSSD_main.src.imputers.S4Model import S4Layer


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # print(f"{2*self.res_channels=} {s4_lmax=} {s4_d_state=} {s4_dropout=} {s4_bidirectional=} {s4_layernorm=}")
        self.S41 = S4Layer(features=2*self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        # Unpack the input data
        x, cond, diffusion_step_embed = input_data

        # Initialize h with the input tensor x
        h = x

        # Get the shape of the input tensor
        B, C, L = x.shape

        # Assert that the number of channels in the input matches the expected number
        assert C == self.res_channels

        # Apply the fully connected layer to the diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)

        # Reshape part_t to match the expected shape
        part_t = part_t.view([B, self.res_channels, 1])

        # Add part_t to h
        h = h + part_t

        # Apply the convolutional layer to h
        h = self.conv_layer(h)

        # Permute the dimensions of h and apply the S41 layer
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)

        # Assert that the condition tensor is not None
        assert cond is not None

        # Apply the conditional convolution to the condition tensor
        cond = self.cond_conv(cond)

        # Add the condition tensor to h
        h += cond

        # Permute the dimensions of h and apply the S42 layer
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)

        # Apply the tanh and sigmoid activation functions to h
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # Apply the residual convolution to out
        res = self.res_conv(out)

        # Assert that the shapes of x and res match
        assert x.shape == res.shape

        # Apply the skip convolution to out
        skip = self.skip_conv(out)

        # Return the sum of x and res, scaled by sqrt(0.5), and the skip tensor
        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability

class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))


    def forward(self, input_data):
        # Unpack the input data
        noise, conditional, diffusion_steps = input_data

        # Calculate the diffusion step embedding
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)

        # Apply the swish activation function after passing through the first fully connected layer
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))

        # Apply the swish activation function again after passing through the second fully connected layer
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # Initialize h with the noise tensor
        h = noise

        # Initialize the skip connection accumulator
        skip = 0

        # Loop over the residual blocks
        for n in range(self.num_res_layers):
            # Pass the input through the n-th residual block and accumulate the skip connections
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            skip += skip_n

            # Return the accumulated skip connections, scaled by the square root of the inverse number of residual layers
        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDS4Imputer(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        noisy_state, conditional, mask, diffusion_steps = input_data

        # Conditional masking focuses error computations on available data
        conditional = torch.cat([conditional * mask, mask.float()], dim=1)

        # extract initial features from the noise data
        embed_noisy_state = self.init_conv(noisy_state)

        # learn and apply complex transformations to the data, taking into account the conditional data and the current diffusion step
        learned_noisy_state = self.residual_layer((embed_noisy_state, conditional, diffusion_steps))
        
        # map the transformed data back to the original data space, producing the final output
        y = self.final_conv(learned_noisy_state)

        return y
  
