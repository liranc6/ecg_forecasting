import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models_diffusion.DDPM_CNNNet import *
from models_diffusion.DDPM_diffusion_worker import *

from layers.RevIN import *

from .samplers.dpm_sampler import DPMSolverSampler

import sys
import yaml
CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)
from liran_project.EMD import np_sift as my_np_sift_file

class BaseMapping(nn.Module):
    """
    Decomposition-Linear.
    BaseMapping models the trend component of the data using linear layers.
    In train_forward, loss is calculated based on the difference between predicted trends (outputs)
    and ground truth (x_dec).
    """
    def __init__(self, args, seq_len=None, pred_len=None):
        """
        Initializes with parameters like seq_len, pred_len, and configurations from args.
	    Sets up linear layers (Linear_Trend) to model the trend component. If individual is True,
        it creates separate layers for each channel; otherwise, a single layer for all channels.
        """
        
        super(BaseMapping, self).__init__()

        self.args = args
        
        if seq_len is None:
            self.seq_len = args.training.sequence.label_len
            self.pred_len = args.training.sequence.pred_len
        else:
            self.seq_len = seq_len
            self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = args.training.diffusion.kernel_size
        self.individual = args.data.individual
        self.channels = args.data.num_vars
        
        if self.individual:
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        
        self.rev = RevIN(args, args.data.num_vars, affine=args.training.misc.affine, subtract_last=args.training.misc.subtract_last) if args.training.analysis.use_window_normalization else None

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train_val=None):
        """
        During train_forward and test_forward, input data (x_enc) is optionally normalized using RevIN.
        Data is permuted to match the input shape for linear layers.
	    Linear layers model the trend component.
	    Output is permuted back and optionally denormalized.
        """
        # return [Batch, Output length, Channel]

        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')  # 
            x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec

        x = x_enc_i

        trend_init = x
        trend_init = trend_init.permute(0,2,1)
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            trend_output = self.Linear_Trend(trend_init)
        
        x = trend_output
        outputs = x.permute(0,2,1)

        outputs = self.rev(outputs, 'denorm') if self.args.training.analysis.use_window_normalization else outputs

        f_dim = -1 if self.args.general.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        ground_truth = x_dec[:, -self.pred_len:, f_dim:]

        if self.args.training.model_info.opt_loss_type == "mse":
            loss = F.mse_loss(outputs, ground_truth)
        elif self.args.training.model_info.opt_loss_type == "smape":
            criterion = smape_loss()
            loss = criterion(outputs, ground_truth)
        return loss

    def test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')  
            # x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            # x_dec_i = x_dec

        x = x_enc_i

        trend_init = x
        trend_init = trend_init.permute(0,2,1)
        if self.individual:
            
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            trend_output = self.Linear_Trend(trend_init)
        
        x = trend_output
        outputs = x.permute(0,2,1)
        
        outputs = self.rev(outputs, 'denorm') if self.args.training.analysis.use_window_normalization else outputs

        f_dim = -1 if self.args.general.features == 'MS' else 0
        return outputs[:, -self.pred_len:, f_dim:]


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    A PyTorch module for series decomposition.

    This class decomposes a time series into its trend (moving average) and residual components.

    Attributes:
        kernel_size (int): The size of the kernel for the moving average.
        moving_avg (nn.Module): A module to compute the moving average of the input series.

    Methods:
        forward(x):
            Decomposes the input series into residual and moving average components.
            Args:
                x (torch.Tensor): The input time series.
            Returns:
                tuple: A tuple containing the residual component and the moving average component.
    """
    def __init__(self, kernel_size):
        """
        Initializes the series_decomp module with the given kernel size.

        Args:
            kernel_size (int): The size of the kernel for the moving average.
        """
        self.kernel_size = kernel_size
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Forward pass for the series decomposition.

        Decomposes the input series into residual and moving average components.

        Args:
            x (torch.Tensor): The input time series.

        Returns:
            tuple: A tuple containing the residual component and the moving average component.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args, device=None):
        super(Model, self).__init__()

        self.args = args
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seq_len = args.training.sequence.label_len
        self.label_len = args.training.sequence.label_len
        self.pred_len = args.training.sequence.pred_len
        
        self.num_vars = args.data.num_vars
        self.smoothed_factors = args.training.smoothing.smoothed_factors 
        self.linear_history_len = 0 #args.training.sequence.context_len # args.linear_history_lens
        
        self.num_bridges = len(self.smoothed_factors) + 1 if not self.args.emd.use_emd else self.args.emd.num_sifts + 1
        
        self.base_models = nn.ModuleList([BaseMapping(args, seq_len=self.seq_len, pred_len=self.pred_len) for i in range(self.num_bridges)])
        # self.decompsitions = nn.ModuleList([series_decomp(i) for i in self.smoothed_factors])
        
        if args.training.model_info.u_net_type == "v0":
            self.u_nets = nn.ModuleList([My_DiffusionUnet_v0(args, self.num_vars, self.seq_len, self.pred_len, net_id=i) for i in range(self.num_bridges)])
         
        self.diffusion_workers = nn.ModuleList([Diffusion_Worker(args, self.device, self.u_nets[i]) for i in range(self.num_bridges)])

        self.rev = RevIN(args, args.data.num_vars, affine=args.training.misc.affine, subtract_last=args.training.misc.subtract_last) if args.training.analysis.use_window_normalization else None
         
        if args.training.diffusion.diff_steps < 100:
            args.training.sampler.type_sampler == "none"
        if args.training.sampler.type_sampler == "none":
            pass
        elif args.training.sampler.type_sampler == "dpm":
            assert self.args.training.sampler.parameterization == "x_start"
            self.samplers = [DPMSolverSampler(self.u_nets[i], self.diffusion_workers[i]) for i in range(self.num_bridges)]

    def obatin_multi_trends(self, batch_x):
        # batch_x: (B, N, L)
        batch_x_trends = []
        avg_layer = None
        if self.args.emd.use_emd:
            batch_x = batch_x.squeeze()
            batch_x_np = batch_x.cpu().numpy()
            for sample_idx, sample_tensor in enumerate(batch_x):
                sample_np = batch_x_np[sample_idx]
                imfs = my_np_sift_file.sift(sample_np, max_imfs=self.args.emd.num_sifts)
                imfs_tensor = torch.tensor(imfs).float()
                sample_components = []
                for n in range(imfs_tensor.shape[1]-1, 0, -1):
                    comp = sample_tensor - torch.sum(imfs_tensor[:, n:], dim=1) 
                    comp = comp.unsqueeze(0)
                    sample_components.append(comp)
                    
                last_comp = torch.sum(imfs_tensor, dim=1).unsqueeze(0)
                sample_components.append(last_comp)
                sample_components = torch.stack(sample_components)
                batch_x_trends.append(sample_components)
                
            batch = torch.stack(batch_x_trends)
            batch = batch.permute(1,0,2,3)
            batch_x_trends = []
            for t in batch:
                batch_x_trends.append(t)
            
            batch_x_trends.reverse()
            
            
        else:     
            batch_x = batch_x.permute(0,2,1) # batch_x: (B, L, N)
            # print("self.smoothed_factors", self.smoothed_factors)
            
            batch_x_trend_0 = batch_x
            for i in range(self.num_bridges-1):
                kernel_size = self.smoothed_factors[i]
                avg_layer = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
                front = batch_x_trend_0[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
                end = batch_x_trend_0[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
                batch_x_trend_0 = torch.cat([front, batch_x_trend_0, end], dim=1)
                batch_x_trend_0 = avg_layer(batch_x_trend_0.permute(0, 2, 1)).permute(0, 2, 1)
                # moving_mean = batch_x_trend_0
                # res = x - moving_mean
                batch_x_trend = batch_x_trend_0
                
                
                
                
                
                # _, batch_x_trend = self.decompsitions[i](batch_x_trend_0)
                # print("batch_x_trend", np.shape(batch_x_trend))
                
                # plt.plot(batch_x_trend[0,0,:].cpu().numpy())

                batch_x_trends.append(batch_x_trend.permute(0,2,1))
                batch_x_trend_0 = batch_x_trend
                
        if self.args.training.smoothing.reverse_order:
            batch_x_trends.reverse()
            
        del batch_x
        del avg_layer
        torch.cuda.empty_cache()
        # plt.savefig("demo_haha.png")
        return batch_x_trends

    def pretrain_test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')
            x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec

        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.training.sequence.pred_len:,:]

        # ==================================
        # history trends
        future_trends = self.obatin_multi_trends(x_future.permute(0,2,1))
        # each trend: B, N, L
        future_trends = [trend_i.permute(0,2,1) for trend_i in future_trends]
        # each trend: B, L, N

        import matplotlib.pyplot as plt
        # ==================================
        # history trends
        past_trends = self.obatin_multi_trends(x_past.permute(0,2,1))
        past_trends = [trend_i.permute(0,2,1) for trend_i in past_trends]

        total_loss = [] 
        j = 0 # self.num_bridges-1
        for i in range(self.num_bridges):
            if i == j:
                if i == 0:
                    future_trend_i = x_future
                    past_trend_i = x_past
                    
                    future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                    past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                        
                    linear_guess = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec)
                else:
                    future_trend_i = future_trends[i-1]
                    past_trend_i = past_trends[i-1]

                    future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                    past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                        
                    linear_guess = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec)
                    
                return linear_guess

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_mean=True, meta_weights=None, train_val=False, check_gpu_memory_usage=None):
        
        gpu_prints = 0
        if check_gpu_memory_usage is not None:
            print(f"{gpu_prints=}\n"\
                  f"check_gpu_memory_usage():\n" \
                  f"{check_gpu_memory_usage()}")
            gpu_prints += 1
            
        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')
            x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec
        
        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.training.sequence.pred_len:,:]
            
        # print(">>>>>", np.shape(x_past), np.shape(x_future))

        future_trends = self.obatin_multi_trends(x_future.permute(0,2,1))  # from finest to coarsest
        # each trend: B, N, L
        future_trends = [trend_i.permute(0,2,1) for trend_i in future_trends]
        # each trend: B, L, N
        future_xT = torch.randn_like(x_future)
        future_trends.append(future_xT)

        # ==================================
        # history trends
        past_trends = self.obatin_multi_trends(x_past.permute(0,2,1))  # from finest to coarsest
        past_trends = [trend_i.permute(0,2,1) for trend_i in past_trends]
        
        if check_gpu_memory_usage is not None:
            print(f"{gpu_prints=}\n"\
                  f"check_gpu_memory_usage():\n" \
                  f"{check_gpu_memory_usage()}")
            gpu_prints += 1

        # ==================================
        # ar-init trends from finest to coarsest
        ar_init_trends = []
        linear_guess, ar_init_trends = self._compute_trends_and_guesses(x_past, x_future, future_trends, past_trends, x_mark_enc, x_mark_dec)
        
        if check_gpu_memory_usage is not None:
            print(f"linear_guess, ar_init_trends = self._compute_trends_and_guesses(: \n {gpu_prints=}\n"\
                  f"check_gpu_memory_usage():\n" \
                  f"{check_gpu_memory_usage()}")
            gpu_prints += 1
            
        total_loss = self._compute_total_loss(x_past, x_future, future_trends, past_trends, linear_guess, ar_init_trends, return_mean=return_mean)
        if return_mean:
            total_loss = torch.stack(total_loss).mean()
        else:
            if meta_weights is not None:
                total_loss = torch.stack(total_loss).reshape(-1)
                train_loss_tmp = [w * total_loss[i] for i, w in enumerate(meta_weights)]
                total_loss = sum(train_loss_tmp)
            else:
                total_loss = torch.stack(total_loss).reshape(-1)
                
        if check_gpu_memory_usage is not None:
            print(f"{gpu_prints=}\n"\
                  f"check_gpu_memory_usage():\n" \
                  f"{check_gpu_memory_usage()}")
            gpu_prints += 1

        return total_loss

    def test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')
            x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec
        
        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.training.sequence.pred_len:,:]
        
        future_trends = self.obatin_multi_trends(x_future.permute(0,2,1))
        # each trend: B, N, L
        future_trends = [trend_i.permute(0,2,1) for trend_i in future_trends]
        # each trend: B, L, N
        future_xT = torch.randn_like(x_future)
        future_trends.append(future_xT)

        # ==================================
        # history trends
        past_trends = self.obatin_multi_trends(x_past.permute(0,2,1))
        past_trends = [trend_i.permute(0,2,1) for trend_i in past_trends]

        # ==================================
        # ar-init trends
        ar_init_trends = []
        for i in range(self.num_bridges):
            if i == 0:
                future_trend_i = x_future
                past_trend_i = x_past
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess = self.base_models[i].test_forward(past_trend_i[:,:,:], x_mark_enc, future_trend_i, x_mark_dec)
                linear_guess = self.rev(linear_guess, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess
            else:
                future_trend_i = future_trends[i-1]
                past_trend_i = past_trends[i-1]
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess_i = self.base_models[i].test_forward(past_trend_i[:,:,:], x_mark_enc, future_trend_i, x_mark_dec)
                linear_guess_i = self.rev(linear_guess_i, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess_i
                ar_init_trends.append(linear_guess_i)

        
        B, nF, nL = x_past.shape[0], self.num_vars, self.pred_len
        if self.args.general.features in ['MS']:
            nF = 1
        shape = [nF, nL]

        # ==================================
        # history trends

        all_outs = []
        for i in range(self.args.training.iterations.sample_times):
            
            X1 = future_xT.permute(0,2,1)
            res_out = X1

            for j in reversed(range(0, self.num_bridges)):
                
                MASK = torch.ones((x_past.shape[0], self.num_vars, self.pred_len))
                
                if self.args.training.sampler.type_sampler == "none":
                    if j == 0:
                        cond = torch.cat([x_past.permute(0,2,1), X1], dim=-1)
                        X1 = self.diffusion_workers[j].ddpm_sampling(X1, mask=MASK, cond=cond, ar_init=linear_guess.permute(0,2,1))
                    else:
                        if j == self.num_bridges-1:
                            cond = past_trends[j-1].permute(0,2,1)
                        else:
                            # cond = torch.cat([x_past, X1], dim=-1)
                            if self.args.training.analysis.use_X0_in_THiDi:
                                cond = torch.cat([x_past.permute(0,2,1), X1], dim=-1)
                            else:
                                cond = torch.cat([past_trends[j-1].permute(0,2,1), X1], dim=-1)
                            
                        X1 = self.diffusion_workers[j].ddpm_sampling(X1, mask=MASK, cond=cond, ar_init=ar_init_trends[j-1].permute(0,2,1))
                
                else:
                    start_code = torch.randn((B, nF, nL))
                    if j == 0:
                        cond = torch.cat([x_past.permute(0,2,1), X1], dim=-1)
                        cA = cond
                        cB = linear_guess.permute(0,2,1)
                    else:
                        if j == self.num_bridges-1:
                            cond = past_trends[j-1].permute(0,2,1)
                        else:
                            if self.args.training.analysis.use_X0_in_THiDi:
                                cond = torch.cat([x_past.permute(0,2,1), X1], dim=-1)
                            else:
                                cond = torch.cat([past_trends[j-1].permute(0,2,1), X1], dim=-1)
                        cA = cond
                        cB = ar_init_trends[j-1].permute(0,2,1)

                    if self.args.general.dataset == 'lorenz':
                        S = 50
                    else:
                        S = 20
                    samples_ddim, _ = self.samplers[j].sample(S=S,
                                                conditioning=torch.cat([cA, cB], dim=-1),
                                                batch_size=B,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=1.0,
                                                unconditional_conditioning=None,
                                                eta=0.,
                                                x_T=start_code)
                    X1 = samples_ddim

                if j == 0:
                    res_out = X1

            outs_i = res_out.permute(0,2,1)

            outs_i = self.rev(outs_i, 'denorm') if self.args.training.analysis.use_window_normalization else outs_i
            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=0)
        outputs = all_outs.mean(0)

        f_dim = -1 if self.args.general.features == 'MS' else 0
        outputs = outputs[:, -self.args.training.sequence.pred_len:, f_dim:]

        return outputs
     
    def _compute_total_loss(self, x_past, x_future, future_trends, past_trends, linear_guess, ar_init_trends, return_mean=False):
        """
        Compute the total loss for the diffusion model training.

        Args:
            x_past (torch.Tensor): The past input tensor.
            x_future (torch.Tensor): The future input tensor.
            future_trends (list of torch.Tensor): List of future trend tensors.
            past_trends (list of torch.Tensor): List of past trend tensors.
            linear_guess (torch.Tensor): Initial guess tensor for autoregression.
            ar_init_trends (list of torch.Tensor): List of initial trends for autoregression.
            return_mean (bool, optional): Whether to return the mean of the loss. Defaults to False.

        Returns:
            list: List of loss values for each bridge.
        """
        total_loss = []
        for i in range(self.num_bridges - 1, -1, -1):  # from coarsest to finest
            
            # X0 clean
            # X1: occupied [bsz, fea, seq_len]    
            
            if i == 0:
                # For the first bridge, use the first future trend and the future input
                X1 = future_trends[0].permute(0, 2, 1)  # most fined
                X0 = x_future.permute(0, 2, 1)
                
                # Concatenate the past input with the first future trend
                cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)

                # Create a mask tensor
                MASK = torch.ones((X1.shape[0], self.num_vars, self.pred_len))
                
                # Compute the loss for the first bridge
                loss_i = self.diffusion_workers[i].train_forward(X0, X1, mask=MASK, condition=cond, ar_init=linear_guess.permute(0, 2, 1), return_mean=return_mean)
                total_loss.append(loss_i)

            else:
                # For subsequent bridges, use the corresponding future trend
                X1 = future_trends[i].permute(0, 2, 1)

                if i == self.num_bridges - 1:
                    # For the last bridge, use only the past trend as the condition
                    cond = past_trends[i-1].permute(0, 2, 1)
                else:
                    # For intermediate bridges, concatenate the past trend with the current future trend
                    cond = torch.cat([past_trends[i-1].permute(0, 2, 1), X1], dim=-1)

                # Use the previous future trend as X0
                X0 = future_trends[i-1].permute(0, 2, 1)

                # Create a mask tensor
                MASK = torch.ones((np.shape(X1)[0], self.num_vars, self.pred_len))
                
                # Compute the loss for the current bridge
                loss_i = self.diffusion_workers[i].train_forward(X0, X1, mask=MASK, condition=cond, ar_init=ar_init_trends[i-1].permute(0, 2, 1), return_mean=return_mean)
                total_loss.append(loss_i)
        
        return total_loss

    def _compute_trends_and_guesses(self, x_past, x_future, future_trends, past_trends, x_mark_enc, x_mark_dec):
        """
        Compute trends and linear guesses for each bridge in the diffusion model.

        Args:
            x_past (torch.Tensor): The past input tensor.
            x_future (torch.Tensor): The future input tensor.
            future_trends (list of torch.Tensor): List of future trend tensors.
            past_trends (list of torch.Tensor): List of past trend tensors.
            x_mark_enc (torch.Tensor): The encoder marker tensor.
            x_mark_dec (torch.Tensor): The decoder marker tensor.

        Returns:
            tuple: A tuple containing the linear guess for the first bridge and a list of initial trends for autoregression.
        """
        ar_init_trends = []

        for i in range(self.num_bridges):
            if i == 0:
                # For the first bridge, use the future and past input tensors
                future_trend_i = x_future
                past_trend_i = x_past

                # Apply reverse normalization if window normalization is used
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i

                # Compute the linear guess for the first bridge
                linear_guess = self.base_models[i].test_forward(past_trend_i[:,:, :], x_mark_enc, future_trend_i, x_mark_dec).detach()
                linear_guess = self.rev(linear_guess, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess

            else:
                # For subsequent bridges, use the corresponding future trend
                future_trend_i = future_trends[i-1]

                # Determine the past trend based on the use_X0_in_THiDi flag
                if self.args.training.analysis.use_X0_in_THiDi:
                    past_trend_i = x_past
                else:
                    past_trend_i = past_trends[i-1]

                # Apply reverse normalization if window normalization is used
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i

                # Compute the linear guess for the current bridge
                linear_guess_i = self.base_models[i].test_forward(past_trend_i[:, :, :], x_mark_enc, future_trend_i, x_mark_dec).detach()
                linear_guess_i = self.rev(linear_guess_i, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess_i

                # Append the linear guess to the list of initial trends for autoregression
                ar_init_trends.append(linear_guess_i)

        return linear_guess, ar_init_trends

    def forward(self, x_enc, x_mark_enc=None):
        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')
        else:
            x_enc_i = x_enc

        x_past = x_enc_i

        # Obtain future trends
        future_xT = torch.randn_like(x_past)
        future_trends = self.obatin_multi_trends(future_xT.permute(0, 2, 1))
        future_trends = [trend_i.permute(0, 2, 1) for trend_i in future_trends]
        future_trends.append(future_xT)

        # Obtain past trends
        past_trends = self.obatin_multi_trends(x_past.permute(0, 2, 1))
        past_trends = [trend_i.permute(0, 2, 1) for trend_i in past_trends]

        # AR-init trends
        ar_init_trends = []
        for i in range(self.num_bridges):
            if i == 0:
                future_trend_i = future_xT
                past_trend_i = x_past
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess = self.base_models[i].test_forward(past_trend_i, x_mark_enc, future_trend_i, x_mark_enc)
                linear_guess = self.rev(linear_guess, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess
            else:
                future_trend_i = future_trends[i - 1]
                past_trend_i = past_trends[i - 1]
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess_i = self.base_models[i].test_forward(past_trend_i, x_mark_enc, future_trend_i, x_mark_enc)
                linear_guess_i = self.rev(linear_guess_i, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess_i
                ar_init_trends.append(linear_guess_i)

        B, nF, nL = x_past.shape[0], self.num_vars, self.pred_len
        if self.args.general.features in ['MS']:
            nF = 1
        shape = [nF, nL]

        all_outs = []
        for i in range(self.args.training.iterations.sample_times):
            X1 = future_xT.permute(0, 2, 1)
            res_out = X1

            for j in reversed(range(0, self.num_bridges)):
                MASK = torch.ones((x_past.shape[0], self.num_vars, self.pred_len))

                if self.args.training.sampler.type_sampler == "none":
                    if j == 0:
                        cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                        X1 = self.diffusion_workers[j].ddpm_sampling(X1, mask=MASK, cond=cond, ar_init=linear_guess.permute(0, 2, 1))
                    else:
                        if j == self.num_bridges - 1:
                            cond = past_trends[j - 1].permute(0, 2, 1)
                        else:
                            if self.args.training.analysis.use_X0_in_THiDi:
                                cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                            else:
                                cond = torch.cat([past_trends[j - 1].permute(0, 2, 1), X1], dim=-1)
                        X1 = self.diffusion_workers[j].ddpm_sampling(X1, mask=MASK, cond=cond, ar_init=ar_init_trends[j - 1].permute(0, 2, 1))
                else:
                    start_code = torch.randn((B, nF, nL))
                    if j == 0:
                        cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                        cA = cond
                        cB = linear_guess.permute(0, 2, 1)
                    else:
                        if j == self.num_bridges - 1:
                            cond = past_trends[j - 1].permute(0, 2, 1)
                        else:
                            if self.args.training.analysis.use_X0_in_THiDi:
                                cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                            else:
                                cond = torch.cat([past_trends[j - 1].permute(0, 2, 1), X1], dim=-1)
                        cA = cond
                        cB = ar_init_trends[j - 1].permute(0, 2, 1)

                    S = 50 if self.args.general.dataset == 'lorenz' else 20
                    samples_ddim, _ = self.samplers[j].sample(S=S,
                                                            conditioning=torch.cat([cA, cB], dim=-1),
                                                            batch_size=B,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=1.0,
                                                            unconditional_conditioning=None,
                                                            eta=0.,
                                                            x_T=start_code)
                    X1 = samples_ddim

                if j == 0:
                    res_out = X1

            outs_i = res_out.permute(0, 2, 1)
            outs_i = self.rev(outs_i, 'denorm') if self.args.training.analysis.use_window_normalization else outs_i
            all_outs.append(outs_i)

        all_outs = torch.stack(all_outs, dim=0)
        outputs = all_outs.mean(0)

        f_dim = -1 if self.args.general.features == 'MS' else 0
        outputs = outputs[:, -self.args.training.sequence.pred_len:, f_dim:]

        return outputs






