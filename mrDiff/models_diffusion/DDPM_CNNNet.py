

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Callable, Optional
from torch import Tensor

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def noise_mask(X, masking_ratio=0.15, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) tensor of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked sequences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean tensor with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = torch.ones(X.shape, dtype=torch.bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = torch.tile(torch.unsqueeze(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), (1, X.shape[1]))
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = torch.rand(X.shape) > masking_ratio
        else:
            mask = torch.rand(X.shape[0], 1) < (1 - masking_ratio)
            mask = mask.repeat(1, X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean tensor intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = torch.ones(L, dtype=torch.bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(torch.rand(1).item() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if torch.rand(1).item() < p[state]:
            state = 1 - state

    return keep_mask

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half #         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Conv1dWithInitialization(nn.Module):
    def __init__(self, decode=False, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        if decode:
            self.conv1d = nn.ConvTranspose1d(**kwargs)
        else:
            self.conv1d = nn.Conv1d(**kwargs)
        nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)

class DynamicUNet1D(nn.Module):
    def __init__(self, din, dout, num_layers, encoder_only=False):
        super(DynamicUNet1D, self).__init__()
        self.num_blocks = num_layers
        self.encoder_only = encoder_only
        self.encoder_blocks = nn.ModuleList()
        self.default_init_params = {
            0: {"kernel_size": 3, "stride": 1, "padding": 1},
            1: {"kernel_size": 5, "stride": 3, "padding": 1},
            2: {"kernel_size": 5, "stride": 3, "padding": 2},
            3: {"kernel_size": 5, "stride": 3, "padding": 2},
            4: {"kernel_size": 5, "stride": 4, "padding": 2},
            5: {"kernel_size": 5, "stride": 3, "padding": 2},
            6: {"kernel_size": 5, "stride": 3, "padding": 2},
            7: {"kernel_size": 9, "stride": 4, "padding": 1},
            8: {"kernel_size": 9, "stride": 4, "padding": 1},
            9: {"kernel_size": 9, "stride": 3, "padding": 1},
            10: {"kernel_size": 10, "stride": 1, "padding": 2},
        }
        
        # dims = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        dims = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        
        self.dims = None
        if din > 64:
            #get position of 64 in dims
            pos = dims.index(64)
            # add a layer to reduce the input dimension to 64
            self.dims = [din] + dims[pos : pos+num_layers]
        else:  
            for i, e in enumerate(dims):
                if e < din:
                    continue
                elif e == din:
                    self.dims = dims[i : i+num_layers + 1]
                    break
                else:  # e > din
                    self.dims = dims[i : i+num_layers + 1]
                    #append din to start of dims
                    self.dims[0] = din
                    break
        
        if self.dims is None:
            raise ValueError(f"Input dimension {din} not found in the list of supported dimensions.")
        
        if self.encoder_only:
            self.dims[-1] = dout
        
        params = {i: self.default_init_params[i] for i in range(num_layers)}
        
        # Encoder
        for i in range(self.num_blocks):
            self.encoder_blocks.append(
                self.conv_block_1d(self.dims[i], self.dims[i + 1], **params[i], encoder_only=self.encoder_only)
            )
            
        self.final_conv = None

        if not self.encoder_only:
            # Bottleneck
            bottleneck_params = params[self.num_blocks - 1]
            self.bottleneck = self.conv_block_1d(self.dims[self.num_blocks], self.dims[self.num_blocks] * 2, **bottleneck_params)

            # Decoder
            self.decoder_blocks = nn.ModuleList()
            for i in range(self.num_blocks - 1, -1, -1):
                self.decoder_blocks.append(
                    self.conv_block_1d(self.dims[i + 1] * 3, self.dims[i + 1], **params[i], decode=True)
                )

            # Final layer
            self.final_conv = Conv1dWithInitialization(in_channels=self.dims[1], out_channels=dout, kernel_size=1)
        else:
            # self.final_conv = Conv1dWithInitialization(in_channels=self.dims[-1], out_channels=dout, kernel_size=1)
            pass

    @staticmethod
    def conv_block_1d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, decode=False, encoder_only=False):
        if encoder_only:
            return Conv1dWithInitialization(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                decode=decode
            )
        else:
            return nn.Sequential(
                # Main convolution
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    decode=decode
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.3)
            )

    def forward(self, x):
        original_length = x.shape[-1]
        
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)

        if not self.encoder_only:
            # Bottleneck
            x = self.bottleneck(x)

            # Decoder
            for i, decoder in enumerate(self.decoder_blocks):
                skip = skip_connections[-(i + 1)]
                x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=True)
                x = torch.cat([x, skip], dim=1)
                x = decoder(x)

            # Final output
            x = self.final_conv(x)
        
        # Interpolate to match the original length
        x = F.interpolate(x, size=original_length, mode="linear", align_corners=True)

        return x
    
class DiffusionEmbedding(nn.Module):
    """Generates embeddings for diffusion steps using sinusoidal functions.

    This class creates a set of embeddings based on the diffusion steps, which can be projected into a specified dimensionality.
    It utilizes sine and cosine functions to create a rich representation of the diffusion steps, allowing for effective learning in diffusion models.

    Args:
        num_steps (int): The total number of diffusion steps.
        embedding_dim (int, optional): The dimensionality of the embedding. Defaults to 128.
        projection_dim (int, optional): The dimensionality of the projected output. If None, it defaults to embedding_dim.

    Returns:
        Tensor: The projected embedding corresponding to the given diffusion step.

    Examples:
        >>> diffusion_embedding = DiffusionEmbedding(num_steps=1000)
        >>> output = diffusion_embedding(diffusion_step=5)
    """
    
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        
        layers = []
        layers.append(nn.Linear(embedding_dim, projection_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Linear(projection_dim, projection_dim))
        layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        # self.projection1 = nn.Linear(embedding_dim, projection_dim)
        # self.projection2 = nn.Linear(projection_dim, projection_dim)
        
        

    def forward(self, diffusion_step):
        out = self.embedding[diffusion_step]
        out = self.net(out)
            
        # x = self.embedding[diffusion_step]
        # # print("1", np.shape(x))
        # x = self.projection1(x)
        # # print("2", np.shape(x))
        # x = F.silu(x)
        # x = self.projection2(x)
        # # print("3", np.shape(x))
        # x = F.silu(x)
        # # 1 torch.Size([64, 128])
        # # 2 torch.Size([64, 128])
        # # 3 torch.Size([64, 128])
        # return x
        
        return out

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class My_DiffusionUnet_v0(nn.Module):
    """A diffusion model based on a U-Net architecture for processing time series data.

    This class implements a U-Net style architecture specifically designed for diffusion processes.

    Args:
        args (Namespace): Configuration parameters for the model.
        num_vars (int): The number of variables in the input data.
        seq_len (int): The length of the input sequences.
        pred_len (int): The length of the prediction sequences.
        net_id (int, optional): Identifier for the network instance. Defaults to 0.

    Returns:
        Tensor: The output of the model after processing the input through the U-Net architecture.

    Examples:
        >>> model = My_DiffusionUnet_v0(args, num_vars=5, seq_len=20, pred_len=10)
        >>> output = model(xt, timesteps, cond)
    """
    
    def __init__(self, args, num_vars, seq_len, pred_len, net_id=0, use_residual=None):
        super(My_DiffusionUnet_v0, self).__init__()

        self.args = args
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.label_len = args.training.sequence.label_len
        self.pred_len = pred_len
        self.net_id = net_id
        self.smoothed_factors = args.training.smoothing.smoothed_factors 
        self.num_bridges = len(self.smoothed_factors) + 1 if not self.args.emd.use_emd else self.args.emd.num_sifts + 1
        self.dim_diff_step = args.training.diffusion.ddpm_dim_diff_steps
        self.channels = args.training.diffusion.ddpm_inp_embed
        self.use_residual = self.args.data.use_residual if use_residual is None else use_residual

        # Diffusion embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=args.training.diffusion.diff_steps,
            embedding_dim=self.dim_diff_step,
        )
        self.act = lambda x: x * torch.sigmoid(x)

        # Feature projection
        self.use_features_proj = False
        if self.use_features_proj:
            self.feature_projection = nn.Sequential(
                linear(self.num_vars, self.channels),
            )
            self.input_unet = DynamicUNet1D(
                                    din=self.channels, 
                                    dout=self.channels, 
                                    num_layers=args.training.diffusion.ddpm_layers_inp,
                                    encoder_only=True,
                                    )
        else:
            self.input_unet = DynamicUNet1D(
                                    din=self.num_vars, 
                                    dout=self.channels,
                                    num_layers=args.training.diffusion.ddpm_layers_inp,
                                    encoder_only=True,
                                    )

        # Encoder
        self.input_and_t_step_unet = DynamicUNet1D(
                                        din = self.channels + self.dim_diff_step,
                                        dout = args.training.diffusion.ddpm_channels_fusion_I, 
                                        num_layers=args.training.diffusion.ddpm_layers_I,
                                        encoder_only=True,
                                        )

        # Conditional projections
        if self.args.data.individual:
            self.cond_projections = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.num_vars)])
        else:
            self.cond_projection = DynamicUNet1D(
                                        din = self.num_vars,
                                        dout = self.num_vars, 
                                        num_layers=5,
                                        encoder_only=True,
                                        )
                                    #nn.Linear(self.seq_len, self.pred_len)

        if args.training.ablation_study.ablation_study_F_type == "CNN":
            self.cnn_cond_projections = DynamicUNet1D(
                                            # din=self.num_vars, 
                                            dout=self.num_vars, 
                                            num_layers=args.training.diffusion.cond_ddpm_num_layers,
                                            encoder_only=True, 
                                            )
            
            self.cnn_linear = nn.Linear(self.seq_len, self.num_vars)

        # Combine convolution
        if self.net_id == self.num_bridges - 1:
            if args.data.use_ar_init:
                self.input_t_cond_unet = DynamicUNet1D(
                                                din=args.training.diffusion.ddpm_channels_fusion_I + 2 * self.num_vars, 
                                                dout=self.num_vars, 
                                                num_layers=args.training.diffusion.ddpm_layers_II, 
                                                encoder_only=False,
                                                )
            else:
                self.input_t_cond_unet = DynamicUNet1D(
                                                din=args.training.diffusion.ddpm_channels_fusion_I + self.num_vars, 
                                                dout=self.num_vars, 
                                                num_layers=args.training.diffusion.ddpm_layers_II,
                                                encoder_only=False,
                                                )
        else:
            if args.data.use_ar_init:
                self.input_t_cond_unet = DynamicUNet1D(
                                                din=args.training.diffusion.ddpm_channels_fusion_I + 3 * self.num_vars, 
                                                dout=self.num_vars, 
                                                num_layers=args.training.diffusion.ddpm_layers_II,
                                                encoder_only=False,
                                                )
            else:
                self.input_t_cond_unet = DynamicUNet1D(
                                                din=args.training.diffusion.ddpm_channels_fusion_I + 2 * self.num_vars, 
                                                dout=self.num_vars, 
                                                num_layers=args.training.diffusion.ddpm_layers_II,
                                                encoder_only=False,
                                                )

    def forward(self, xt, timesteps, cond=None, ar_init=None, future_gt=None, mask=None):
        if ar_init is None:
            ar_init = cond[:, :, -self.pred_len:]

        prev_scale_out = None if self.net_id == self.num_bridges - 1 else cond[:, :, self.seq_len:self.seq_len + self.pred_len]
        cond = cond[:, :, :self.seq_len]

        # Embedding
        diffusion_emb = self.diffusion_embedding(timesteps.long())
        diffusion_emb = self.act(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1).expand(-1, -1, xt.size(-1))

        if self.use_features_proj:
            xt = self.feature_projection(xt.permute(0, 2, 1)).permute(0, 2, 1)

        input_proj_out = self.input_unet(xt)

        # Debug prints to check tensor shapes
        # print(f"diffusion_emb shape: {diffusion_emb.shape}")
        # print(f"input_proj_out shape: {input_proj_out.shape}")
        # print(f"cond shape: {cond.shape}")

        enc_x_and_t = self.input_and_t_step_unet(torch.cat([diffusion_emb, input_proj_out], dim=1))

        # Conditional projection
        if self.args.data.individual:
            enc_cond = torch.zeros([xt.size(0), self.num_vars, self.pred_len], dtype=xt.dtype).to(xt.device)
            for i in range(self.num_vars):
                enc_cond[:, i, :] = self.cond_projections[i](cond[:, i, :])
        else:
            # Ensure cond has correct shape before projection
            if cond.shape[-1] != self.seq_len:
                cond = F.interpolate(cond, size=self.seq_len, mode='linear', align_corners=True)
            enc_cond = self.cond_projection(cond)
            # Ensure output has correct shape
            if enc_cond.shape[-1] != self.pred_len:
                enc_cond = F.interpolate(enc_cond, size=self.pred_len, mode='linear', align_corners=True)
            
        if self.args.training.ablation_study.ablation_study_F_type == "CNN":
            temp_out = self.cnn_cond_projections(cond)
            enc_cond += self.cnn_linear(temp_out).permute(0, 2, 1)

        # Mixing matrix
        if self.args.training.analysis.use_future_mixup and future_gt is not None:
            y_clean = future_gt[:, :, -self.pred_len:]
            if self.args.training.ablation_study.beta_dist_alpha > 0:
                beta_dist = torch.distributions.Beta(self.args.training.ablation_study.beta_dist_alpha, self.args.training.ablation_study.beta_dist_alpha)
                rand_for_mask = beta_dist.sample(y_clean.shape).to(xt.device).long()
            else:
                rand_for_mask = torch.rand_like(y_clean).to(xt.device)
                if self.args.training.ablation_study.ablation_study_masking_type == "hard":
                    tau = self.args.training.ablation_study.ablation_study_masking_tau
                    hard_indices = rand_for_mask > tau
                    data_random_hard_masking = rand_for_mask
                    data_random_hard_masking[hard_indices] = 1
                    data_random_hard_masking[~hard_indices] = 0
                    rand_for_mask = data_random_hard_masking
                if self.args.training.ablation_study.ablation_study_masking_type == "segment":
                    tau = self.args.training.ablation_study.ablation_study_masking_tau
                    segment_mask = torch.from_numpy(noise_mask(enc_cond[:, 0, :], masking_ratio=tau, lm=24)).to(xt.device)
                    segment_mask = segment_mask.unsqueeze(1).repeat(1, enc_cond.shape[1], 1)
                    rand_for_mask = segment_mask.float()

            enc_cond = rand_for_mask * enc_cond + (1 - rand_for_mask) * future_gt[:, :, -self.pred_len:]

        # Concatenate outputs
        if self.args.data.use_ar_init:
            if self.net_id == self.num_bridges - 1:
                concat_out = torch.cat([enc_x_and_t, enc_cond, ar_init], dim=1)
            else:
                concat_out = torch.cat([enc_x_and_t, enc_cond, ar_init, prev_scale_out], dim=1)
        else:
            if self.net_id == self.num_bridges - 1:
                concat_out = torch.cat([enc_x_and_t, enc_cond], dim=1)
            else:
                concat_out = torch.cat([enc_x_and_t, enc_cond, prev_scale_out], dim=1)

        # Residual connection
        if self.use_residual and self.net_id != 0:
            final_out = self.input_t_cond_unet(concat_out) + enc_cond
        else:
            final_out = self.input_t_cond_unet(concat_out)

        return final_out  #shape: (batch_size, num_vars, pred_len)
    


