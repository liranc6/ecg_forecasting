import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models_diffusion.DDPM_CNNNet import *
from models_diffusion.DDPM_diffusion_worker import *

from layers.RevIN import *

from .samplers.dpm_sampler import DPMSolverSampler


class BaseMapping(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args, seq_len=None, pred_len=None):
        super(BaseMapping, self).__init__()

        self.args = args
        self.device = args.device
        
        if seq_len is None:
            self.seq_len = args.training.sequence.seq_len
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
        
        self.rev = RevIN(args, args.data.num_vars, affine=args.training.misc.affine, subtract_last=args.training.misc.subtract_last).to(args.device) if args.training.analysis.use_window_normalization else None

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train_val=None):

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
        ground_truth = x_dec[:, -self.pred_len:, f_dim:].to(self.device)

        if self.args.training.model.opt_loss_type == "mse":
            loss = F.mse_loss(outputs, ground_truth)
        elif self.args.training.model.opt_loss_type == "smape":
            criterion = smape_loss()
            loss = criterion(outputs, ground_truth)
        return loss

    def test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')  
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
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)

        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device

        self.seq_len = args.training.sequence.seq_len
        self.label_len = args.training.sequence.label_len
        self.pred_len = args.training.sequence.pred_len
        
        self.num_vars = args.data.num_vars
        self.smoothed_factors = args.training.smoothing.smoothed_factors 
        self.linear_history_len = args.training.sequence.seq_len # args.linear_history_lens
        
        self.num_bridges = len(args.training.smoothing.smoothed_factors) + 1
        
        self.base_models = nn.ModuleList([BaseMapping(args, seq_len=self.linear_history_len, pred_len=self.pred_len) for i in range(self.num_bridges)])
        self.decompsitions = nn.ModuleList([series_decomp(i) for i in self.smoothed_factors])
        
        if args.training.model.u_net_type == "v0":
            self.u_nets = nn.ModuleList([My_DiffusionUnet_v0(args, self.num_vars, self.seq_len, self.pred_len, net_id=i) for i in range(self.num_bridges)])
         
        self.diffusion_workers = nn.ModuleList([Diffusion_Worker(args, self.u_nets[i]) for i in range(self.num_bridges)])

        self.rev = RevIN(args, args.data.num_vars, affine=args.training.misc.affine, subtract_last=args.training.misc.subtract_last).to(args.device) if args.training.analysis.use_window_normalization else None
         
        if args.training.diffusion.diff_steps < 100:
            args.training.sampler.type_sampler == "none"
        if args.training.sampler.type_sampler == "none":
            pass
        elif args.training.sampler.type_sampler == "dpm":
            assert self.args.training.sampler.parameterization == "x_start"
            self.samplers = [DPMSolverSampler(self.u_nets[i], self.diffusion_workers[i]) for i in range(self.num_bridges)]

    def obatin_multi_trends(self, batch_x):

        # batch_x: (B, N, L)

        batch_x = batch_x.permute(0,2,1)

        # batch_x: (B, L, N)
        # print("self.smoothed_factors", self.smoothed_factors)
        
        batch_x_trends = []
        batch_x_trend_0 = batch_x
        for i in range(self.num_bridges-1):
            _, batch_x_trend = self.decompsitions[i](batch_x_trend_0)
            # print("batch_x_trend", np.shape(batch_x_trend))
            
            # plt.plot(batch_x_trend[0,0,:].cpu().numpy())

            batch_x_trends.append(batch_x_trend.permute(0,2,1))
            batch_x_trend_0 = batch_x_trend

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

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_mean=True, meta_weights=None, train_val=False):
      
        
        if self.args.training.analysis.use_window_normalization:
            x_enc_i = self.rev(x_enc, 'norm')
            x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec
        
        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.training.sequence.pred_len:,:]
            
        # print(">>>>>", np.shape(x_past), np.shape(x_future))

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
                linear_guess = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec).detach()
                linear_guess = self.rev(linear_guess, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess
            else:
                future_trend_i = future_trends[i-1]
                if self.args.training.analysis.use_X0_in_THiDi:
                    past_trend_i = x_past
                else:
                    past_trend_i = past_trends[i-1]
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess_i = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec).detach()
                linear_guess_i = self.rev(linear_guess_i, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess_i
                ar_init_trends.append(linear_guess_i)

        total_loss = []
        for i in range(self.num_bridges):
            
            # X0 clean
            # X1: occupied [bsz, fea, seq_len]    
             
            if i == 0:
                X1 = future_trends[0].permute(0,2,1)
                X0 = x_future.permute(0,2,1) 
                
                cond = torch.cat([x_past.permute(0,2,1), X1], dim=-1)

                MASK = torch.ones((np.shape(X1)[0], self.num_vars, self.pred_len)).to(self.device)
                
                loss_i = self.diffusion_workers[i].train_forward(X0, X1, mask=MASK, condition=cond, ar_init=linear_guess.permute(0,2,1), return_mean=return_mean)
                total_loss.append(loss_i)

            else:
                X1 = future_trends[i].permute(0,2,1)

                if i == self.num_bridges-1:
                    cond = past_trends[i-1].permute(0,2,1)
                else:
                    cond = torch.cat([past_trends[i-1].permute(0,2,1), X1], dim=-1)

                X0 = future_trends[i-1].permute(0,2,1) 

                MASK = torch.ones((np.shape(X1)[0], self.num_vars, self.pred_len)).to(self.device)
                # print("past_trends[i-1]", np.shape(past_trends[i-1]), np.shape(ar_init_trends[i-1]))

                loss_i = self.diffusion_workers[i].train_forward(X0, X1, mask=MASK, condition=cond, ar_init=ar_init_trends[i-1].permute(0,2,1), return_mean=return_mean)
                total_loss.append(loss_i)
        
        if return_mean:
            total_loss = torch.stack(total_loss).mean()
        else:
            if meta_weights is not None:
                total_loss = torch.stack(total_loss).reshape(-1)
                train_loss_tmp = [w * total_loss[i] for i, w in enumerate(meta_weights)]
                total_loss = sum(train_loss_tmp)
            else:
                total_loss = torch.stack(total_loss).reshape(-1)

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
                linear_guess = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec)
                linear_guess = self.rev(linear_guess, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess
            else:
                future_trend_i = future_trends[i-1]
                past_trend_i = past_trends[i-1]
                future_trend_i = self.rev(future_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else future_trend_i
                past_trend_i = self.rev(past_trend_i, 'denorm') if self.args.training.analysis.use_window_normalization else past_trend_i
                linear_guess_i = self.base_models[i].test_forward(past_trend_i[:,-self.linear_history_len:,:], x_mark_enc, future_trend_i, x_mark_dec)
                linear_guess_i = self.rev(linear_guess_i, 'test_norm') if self.args.training.analysis.use_window_normalization else linear_guess_i
                ar_init_trends.append(linear_guess_i)

        
        B, nF, nL = np.shape(x_past)[0], self.num_vars, self.pred_len
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
                
                MASK = torch.ones((np.shape(x_past)[0], self.num_vars, self.pred_len)).to(self.device)
                
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
                    start_code = torch.randn((B, nF, nL), device=self.device)
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

            outs_i = res_out.permute(0,2,1).to(self.device) 

            outs_i = self.rev(outs_i, 'denorm') if self.args.training.analysis.use_window_normalization else outs_i
            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=0)
        outputs = all_outs.mean(0)

        f_dim = -1 if self.args.general.features == 'MS' else 0
        outputs = outputs[:, -self.args.training.sequence.pred_len:, f_dim:]

        return outputs
     
    










