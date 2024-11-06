
from typing import List, Optional, Tuple, Union
from typing import Any, Dict
from functools import partial
from inspect import isfunction
import math
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import loss
import numpy as np

import importlib

from torch import optim
from utils.losses import mape_loss, mase_loss, smape_loss

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from liran_project.utils.common import *

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    t = t.to(a.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Diffusion_Worker(nn.Module):
    
    def __init__(self, args, u_net=None):
        super(Diffusion_Worker, self).__init__()

        self.args = args
        self.device = args.device
        
        self.parameterization = args.training.sampler.parameterization
        assert self.parameterization in ["noise", "x_start"], 'currently only supporting "eps" and "x0"'
        
        self.net = u_net

        self.diff_train_steps = args.training.diffusion.diff_steps
        self.diff_test_steps = self.diff_train_steps

        # self.beta_start = 1e-4 # 1e4
        # self.beta_end = 2e-2

        try:
            self.beta_start = args.training.diffusion.beta_start # 1e4
            self.beta_end = args.training.diffusion.beta_end
        except:
            self.beta_start = 1e-4 # 1e4
            self.beta_end = 2e-2

        try:
            self.beta_schedule = args.training.ablation_study.beta_schedule # default "cosine"
        except:
            self.beta_schedule = "cosine"

        self.v_posterior = 0.0
        self.original_elbo_weight = 0.0
        self.l_simple_weight = 1

        self.loss_type = "l2"

        self.set_new_noise_schedule(None, self.beta_schedule, self.diff_train_steps, self.beta_start, self.beta_end)

        self.clip_denoised = True

        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5

        self.nn = u_net

    def set_new_noise_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  
        if exists(given_betas):
            betas = given_betas
        else:
            if beta_schedule == "linear":
                betas = np.linspace(beta_start, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - beta_start) / (np.exp(-betas) + 1) + beta_start
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "noise":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x_start":
            lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss  

    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def train_forward(self, x0, x1, mask=None, condition=None, ar_init=None, return_mean=True):
        
        x = x0
        cond_ts = condition

        # Feed normalized inputs ith shape of [bsz, feas, seqlen]
        # both x and cond are two time seires 

        B = np.shape(x)[0]
        t = torch.randint(0, self.num_timesteps, size=[(B+1)//2,]).long().to(self.device)
        t = torch.cat([t, self.num_timesteps-1-t], dim=0)
        t = t[:B]

        noise = torch.randn_like(x)
        x_k = self.q_sample(x_start=x, t=t, noise=noise)
        
        if self.args.hardware.print_gpu_memory_usage:
            print(f"after 'x_k = self.q_sample(x_start=x, t=t, noise=noise)"\
                    f"{check_gpu_memory_usage()}"
                    )
            
       
        model_out = self.net(x_k, t, cond=condition, ar_init=ar_init, future_gt=x0, mask=None)  # a bit less than a quarter of the memory
        
        cond_ts = x_k = condition = ar_init = x0 =None
        
        
        if self.parameterization == "noise":
            target = noise 
        elif self.parameterization == "x_start":
            target = x
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        
        # in the submission version, we calculate time first, and then calculate variable
        # f_dim = -1 if self.args.general.features == 'MS' else 0

        # if self.args.general.features == "S":
        model_out = model_out.permute(0,2,1)
        target = target.permute(0,2,1)

        if return_mean:
            loss = self.get_loss(model_out[:,:,:], target[:,:,:], mean=False).mean(dim=2).mean(dim=1)
            loss_simple = loss.mean() * self.l_simple_weight

            loss_vlb = (self.lvlb_weights[t] * loss).mean()
            loss = loss_simple + self.original_elbo_weight * loss_vlb

        else:
            # loss = self.get_loss(model_out[:,:,:], target[:,:,:], mean=False).mean(dim=-1).mean(dim=0)
            # print(">>>", np.shape(loss))

            if self.args.training.model_info.opt_loss_type == "mse":
                loss = F.mse_loss(model_out[:,:,:], target[:,:,:])
            elif self.args.training.model_info.opt_loss_type == "smape":
                criterion = smape_loss()
                loss = criterion(model_out[:,:,:], target[:,:,:])
                
        # Free memory of local variables
        # x = x0 = x1 = mask = condition = ar_init = cond_ts = None
        x = cond_ts = t = noise = x_k = model_out = target = loss_simple = loss_vlb = None

        return loss


    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond_ts, ar_init, clip_denoised: bool):

        model_out = self.nn(x, t, cond=cond_ts, ar_init=ar_init, future_gt=None, mask=None)
        
        if self.parameterization == "noise":
            # model_out.clamp_(-10., 10.)
            model_out = self.predict_start_from_noise(x, t=t, noise=model_out)

        x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-self.args.training.sampler.our_ddpm_clip, self.args.training.sampler.our_ddpm_clip)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond=None, ar_init=None, clip_denoised=True, repeat_noise=False):

        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond_ts=cond, ar_init=ar_init, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def ddpm_sampling(self, x1=None, mask=None, cond=None, ar_init=None, store_intermediate_states=False):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        
        x = x1
        b, d, l = np.shape(x1)
        shape = (b, d, np.shape(x)[-1])
        timeseries = torch.randn(shape, device=self.device)
        intermediates = [timeseries.permute(0,2,1)] # return bsz, seqlen, fea_dim

        for i in reversed(range(0, self.num_timesteps)):

            timeseries = self.p_sample(timeseries, torch.full((b,), i, device=self.device, dtype=torch.long), 
                cond=cond, ar_init=ar_init, clip_denoised=self.clip_denoised)
            if store_intermediate_states:
                intermediates.append(timeseries.permute(0,2,1))

        outs = timeseries

        return outs


