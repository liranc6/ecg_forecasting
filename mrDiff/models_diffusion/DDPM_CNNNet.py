

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
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

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
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
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
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)

class InputConvNetwork(nn.Module):

    def __init__(self, args, inp_num_channel, out_num_channel, num_layers=3, ddpm_channels_conv=None):
        super(InputConvNetwork, self).__init__()

        self.args = args

        self.inp_num_channel = inp_num_channel
        self.out_num_channel = out_num_channel

        kernel_size = 3
        padding = 1
        if ddpm_channels_conv is None:
            self.channels = args.training.diffusion.ddpm_channels_conv
        else:
            self.channels = ddpm_channels_conv
        self.num_layers = num_layers

        self.net = nn.ModuleList()

        if num_layers == 1:
            self.net.append(Conv1dWithInitialization(
                                            in_channels=self.inp_num_channel,
                                            out_channels=self.out_num_channel,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        )
                                    )
        else:
            for i in range(self.num_layers-1):
                if i == 0:
                    dim_inp = self.inp_num_channel
                else:
                    dim_inp = self.channels
                self.net.append(Conv1dWithInitialization(
                                            in_channels=dim_inp,
                                            out_channels=self.channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        ))
                self.net.append(torch.nn.BatchNorm1d(self.channels)), 
                self.net.append(torch.nn.LeakyReLU(0.1)),
                self.net.append(torch.nn.Dropout(0.1, inplace = True))

            self.net.append(Conv1dWithInitialization(
                                            in_channels=self.channels,
                                            out_channels=self.out_num_channel,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        )
                                    )

    def forward(self, x=None):

        out = x
        for m in self.net:
            out = m(out)

        return out

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):

        x = self.embedding[diffusion_step]
        # print("1", np.shape(x))
        x = self.projection1(x)
        # print("2", np.shape(x))
        x = F.silu(x)
        x = self.projection2(x)
        # print("3", np.shape(x))
        x = F.silu(x)
        # 1 torch.Size([64, 128])
        # 2 torch.Size([64, 128])
        # 3 torch.Size([64, 128])
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class My_DiffusionUnet_v0(nn.Module):

    def __init__(self, args, num_vars, seq_len, pred_len, net_id=0):
        super(My_DiffusionUnet_v0, self).__init__()

        self.args = args

        self.num_vars = num_vars
        
        self.seq_len = seq_len
        self.label_len = args.training.sequence.label_len
        self.pred_len = pred_len
        
        self.device = args.device

        self.net_id = net_id
        self.smoothed_factors = args.training.smoothing.smoothed_factors 
        self.num_bridges = len(self.smoothed_factors) + 1

        self.dim_diff_step = args.training.diffusion.ddpm_dim_diff_steps
        time_embed_dim = self.dim_diff_step
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=args.training.diffusion.diff_steps,
            embedding_dim=self.dim_diff_step,
        )
        self.act = lambda x: x * torch.sigmoid(x)
        
        self.use_features_proj = False
        self.channels = args.training.diffusion.ddpm_inp_embed
        if self.use_features_proj:
            self.feature_projection = nn.Sequential(
                                        linear(self.num_vars, self.channels),
                                    )
            self.input_projection = InputConvNetwork(args, self.channels, self.channels, num_layers=args.training.diffusion.ddpm_layers_inp)
        else:
            self.input_projection = InputConvNetwork(args, self.num_vars, self.channels, num_layers=args.training.diffusion.ddpm_layers_inp)

        self.dim_intermediate_enc = args.training.diffusion.ddpm_channels_fusion_I
        self.enc_conv = InputConvNetwork(args, self.channels+self.dim_diff_step, self.dim_intermediate_enc, num_layers=args.training.diffusion.ddpm_layers_I)
        
        if self.args.data.individual:
            self.cond_projections = nn.ModuleList()

        if args.training.ablation_study.ablation_study_F_type == "Linear":
            if self.args.data.individual:
                for i in range(self.num_vars):
                    self.cond_projections.append(nn.Linear(self.seq_len,self.pred_len))
                    self.cond_projections[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.cond_projection = nn.Linear(self.seq_len,self.pred_len)

        elif args.training.ablation_study.ablation_study_F_type == "CNN":
            if self.args.data.individual:
                for i in range(self.num_vars):
                    self.cond_projections.append(nn.Linear(self.seq_len,self.pred_len))
                    self.cond_projections[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.cond_projection = nn.Linear(self.seq_len,self.pred_len)

            self.cnn_cond_projections = InputConvNetwork(args, self.num_vars, self.pred_len, num_layers=args.training.diffusion.cond_ddpm_num_layers, ddpm_channels_conv=args.training.diffusion.cond_ddpm_channels_conv)
            self.cnn_linear = nn.Linear(self.seq_len, self.num_vars)
        
        if self.net_id == self.num_bridges-1:
            if args.data.use_ar_init:
                self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+2*self.num_vars, self.num_vars, num_layers=args.training.diffusion.ddpm_layers_II, ddpm_channels_conv=args.training.diffusion.dec_channel_nums)
            else:
                self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+1*self.num_vars, self.num_vars, num_layers=args.training.diffusion.ddpm_layers_II, ddpm_channels_conv=args.training.diffusion.dec_channel_nums)
        else:
            if args.data.use_ar_init:
                self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+3*self.num_vars, self.num_vars, num_layers=args.training.diffusion.ddpm_layers_II, ddpm_channels_conv=args.training.diffusion.dec_channel_nums)
            else:
                self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+2*self.num_vars, self.num_vars, num_layers=args.training.diffusion.ddpm_layers_II, ddpm_channels_conv=args.training.diffusion.dec_channel_nums)
           
        args.modes1 = 10
        args.compression = 0
        args.ratio = 1
        args.mode_type = 0

    def forward(self, xt, timesteps, cond=None, ar_init=None, future_gt=None, mask=None):
        
        # print("cond", np.shape(cond))
        if ar_init is None:
            ar_init = cond[:,:,-self.pred_len:]

        if self.net_id == self.num_bridges-1:
            prev_scale_out = None
        else:
            prev_scale_out = cond[:,:,self.seq_len:self.seq_len+self.pred_len]    
        cond = cond[:,:,:self.seq_len]
        
        # xt： B, N, H
        # timesteps: torch.Size([64])
        # cond B, N, L

        # diffusion_emb = timestep_embedding(timesteps, self.dim_diff_step)
        # diffusion_emb = self.time_embed(diffusion_emb)
        diffusion_emb = self.diffusion_embedding(timesteps.long())
        diffusion_emb = self.act(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1).repeat(1,1,np.shape(xt)[-1])
        
        if self.use_features_proj:
            xt = self.feature_projection(xt.permute(0,2,1)).permute(0,2,1)

        out = self.input_projection(xt)  
        out = self.enc_conv(torch.cat([diffusion_emb, out], dim=1))
        
        if self.args.data.individual:
            pred_out = torch.zeros([xt.size(0),self.num_vars,self.pred_len],dtype=xt.dtype).to(xt.device)
            for i in range(self.num_vars):
                pred_out[:,i,:] = self.cond_projections[i](cond[:,i,:])
        else:
            pred_out = self.cond_projection(cond)

        if self.args.training.ablation_study.ablation_study_F_type == "CNN":
            temp_out = self.cnn_cond_projections(cond)
            pred_out += self.cnn_linear(temp_out).permute(0,2,1)
        
        # =====================================================================        
        # mixing matrix

        if self.args.training.analysis.use_future_mixup and future_gt is not None:
            
            y_clean = future_gt[:,:,-self.pred_len:]
            if self.args.training.ablation_study.beta_dist_alpha > 0:
                rand_for_mask = np.random.beta(self.args.training.ablation_study.beta_dist_alpha, self.args.training.ablation_study.beta_dist_alpha, size=np.shape(y_clean))
                rand_for_mask = torch.tensor(rand_for_mask, dtype=torch.long).to(xt.device)
            else:
                rand_for_mask = torch.rand_like(y_clean).to(xt.device)
                if self.args.training.ablation_study.ablation_study_masking_type == "hard":
                    tau = self.args.training.ablation_study.ablation_study_masking_tau
                    hard_indcies = rand_for_mask > tau
                    data_random_hard_making = rand_for_mask
                    data_random_hard_making[hard_indcies] = 1
                    data_random_hard_making[~hard_indcies] = 0
                    rand_for_mask = data_random_hard_making
                if self.args.training.ablation_study.ablation_study_masking_type == "segment":
                    tau = self.args.training.ablation_study.ablation_study_masking_tau
                    segment_mask = torch.from_numpy(noise_mask(pred_out[:,0,:], masking_ratio=tau, lm=24)).to(yn.device)
                    # print("masking_ratio", tau, torch.sum(segment_mask))
                    segment_mask = segment_mask.unsqueeze(1).repeat(1, np.shape(pred_out)[1], 1)
                    rand_for_mask = segment_mask.float()

            
            pred_out = rand_for_mask * pred_out + (1-rand_for_mask) * future_gt[:,:,-self.pred_len:]
        
        # ar_init = None
        if self.args.data.use_ar_init:
            if self.net_id == self.num_bridges-1:
                out = torch.cat([out, pred_out, ar_init], dim=1)
            else:
                out = torch.cat([out, pred_out, ar_init, prev_scale_out], dim=1)
        else:
            if self.net_id == self.num_bridges-1:
                out = torch.cat([out, pred_out], dim=1)
            else:
                out = torch.cat([out, pred_out, prev_scale_out], dim=1)

        if self.args.data.use_residual:
            out = self.combine_conv(out) + pred_out
        else:
            out = self.combine_conv(out)

        # SHOULD BE  B, N, L
        return out
