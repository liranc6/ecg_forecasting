import argparse
import os
import torch
import random
import numpy as np
import yaml
from box import Box


class Args:
        def __init__(self, config_filename, cmd_args=None):
            assert config_filename.endswith('.yml')
            self.configs_filename = config_filename
            self.configs = self.read_config()
            # if cmd_args is not None:
            #     self.override_with_cmd_args(cmd_args)
            self._analyze_config()

        def read_config(self):
            with open(self.configs_filename, 'r') as file:
                config = yaml.safe_load(file)
            return Box(config)

        def override_with_cmd_args(self, cmd_args):
            assert cmd_args is not None
            cmd_args_dict = vars(cmd_args)
            self.update_config_from_dict(cmd_args_dict)

        def __getattr__(self, name, strict=False):
            if hasattr(self.configs, name):
                return getattr(self.configs, name)
            if strict:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
                        
        def _analyze_config(self):
            
            # Set use_gpu based on availability
            self.configs['use_gpu'] = torch.cuda.is_available()

            # Handle multi-GPU settings
            if self.configs['use_gpu'] and self.configs['hardware']['use_multi_gpu']:
                self.configs['hardware']['devices'] = self.configs['hardware']['devices'].replace(' ', '')
                device_ids = self.configs['hardware']['devices'].split(',')
                self.configs['hardware']['device_ids'] = [int(id_) for id_ in device_ids]
                self.configs['hardware']['gpu'] = self.configs['hardware']['device_ids'][0]
                self.configs['optimization']['patience'] = 30
            else:
                self.configs['hardware']['device_ids'] = [0]  # Default to single GPU or CPU
    
            # Random seed initialization
            if self.configs['general']['random_seed'] <= 0:
                fix_seed = random.randint(0, 10000)
                self.configs['general']['random_seed'] = fix_seed
            fix_seed = self.configs['general']['random_seed']
            random.seed(fix_seed)
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            
            # Load the dataset
            if self.configs['general']['dataset'] == 'ETTm1':
                self.configs['paths']['datasets_dir'] = '/home/liranc6/ecg_forecasting/mrDiff/datasets/SCINet_timeseries/ETT-data/ETT'
                self.configs['data']['data'] = 'ETTm1'
                self.configs['paths']['root_path'] = self.configs['paths']['datasets_dir']
                self.configs['paths']['data_path'] = 'ETTm1.csv'
                self.configs['data']['num_vars'] = 7
    
            if self.configs['general']['features'] == "S":
                self.configs['data']['num_vars'] = 1

        def update_config_from_dict(self, new_dict):
            """update self.config with new_dict

                type(self.config) == box.Box

            Args:
                new_dict (dict): new dictionary to update self.config, can be nested, can be partial
            """
            if isinstance(new_dict, dict):
                new_dict = Box(new_dict)
            
            if not isinstance(new_dict, Box):
                raise ValueError(f"new_dict must be a dictionary or a Box object, got {type(new_dict)}")
            
            assert isinstance(self.configs, Box), f"self.config must be a Box object, got {type(self.configs)}"
            
            self.configs.update(new_dict)
        
                                                
def parse_args(config_filename):
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')

    parser.add_argument('--random_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--pretrain_epochs', type=int, default=2, help='train epochs')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--id_worst', type=int, default=-1)
    parser.add_argument('--focus_variate', type=int, default=-1)
    parser.add_argument('--evaluate', type=bool, default=False)

    parser.add_argument('--tag', type=str, default=None)

    # Datasets
    parser.add_argument('--dataset', type=str, required=False, default='ETTh1', help="['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'solar_AL', 'exchange_rate', 'traffic', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']")
    parser.add_argument('--features', type=str, default='S', choices=['S', 'M'], help='features S is univariate, M is multivariate')

    parser.add_argument('--seq_len', type=int, help='input sequence length of SCINet encoder, look back window')
    parser.add_argument('--label_len', type=int, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, help='prediction sequence length, horizon')

    parser.add_argument('--opt_loss_type', type=str, default='mse') 

    parser.add_argument('--model', type=str, required=False, default='DDPM', help='model of the experiment')
    parser.add_argument('--base_model', type=str, required=False, default='Linear', help='model of the experiment')
    parser.add_argument('--u_net_type', type=str, default='v0') 

    parser.add_argument('--vis_MTS_analysis', type=int, default=0, help='status')

    parser.add_argument('--use_window_normalization', type=bool, default=True)

    parser.add_argument('--use_future_mixup', type=bool, default=True)

    parser.add_argument('--use_X0_in_THiDi', type=bool, default=False)  # Trend and Hierarchical Diffusion
    parser.add_argument('--channel_independence', type=bool, default=False)

    parser.add_argument('--training_mode', type=str, default='ONE')

    # Diffusion models
    parser.add_argument('--smoothed_factors', default=[5, 25, 51], nargs='+', type=int)
    parser.add_argument('--interval', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
    parser.add_argument("--beta-max", type=float, default=1.0, help="max diffusion for the diffusion model")
    parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
    parser.add_argument("--T", type=float, default=0.02, help="sigma end time in network parametrization")
    parser.add_argument('--nfe', type=int, default=20)

    parser.add_argument('--ablation_study_F_type', type=str, default="Linear", help="Linear, CNN")

    parser.add_argument('--beta_schedule', type=str, default="cosine")
    parser.add_argument('--beta_dist_alpha', type=float, default=-1)
    parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
    parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)

    parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
    parser.add_argument('--ddpm_inp_embed', type=int, default=64)
    parser.add_argument('--ddpm_layers_inp', type=int, default=10)
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)
    parser.add_argument('--ddpm_channels_conv', type=int, default=128)
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=10)
    parser.add_argument('--dec_channel_nums', type=int, default=256)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=256)

    parser.add_argument('--type_sampler', type=str, default='dpm', help=["none", "dpm"])
    parser.add_argument('--parameterization', type=str, default='x_start', help=["noise", "x_start"])
    parser.add_argument('--our_ddpm_clip', type=float, default=100)

    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--cols', type=str, nargs='+', help='file list')
    parser.add_argument('--target', type=int, default=-1, help='target feature')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay')
    parser.add_argument('--lradj', type=str, default='3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # GPU
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    parser.add_argument('--individual', default=True, help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--subtract_short_terms', type=int, default=0, help='0: subtract mean; 1: subtract last')

    parser.add_argument('--inverse', type=bool, default=False, help='denorm the output data')
    parser.add_argument('--use_ar_init', type=bool, default=False)
    parser.add_argument('--use_residual', type=bool, default=True)
    parser.add_argument('--uncertainty', type=bool, default=False)

    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--normtype', type=int, default=0)
    
    # wandb args
    parser.add_argument('--wandb_mode', type=str, default='disabled', help='wandb mode')
    parser.add_argument('--wandb_project', type=str, default='mrDiff', help='wandb project')
    parser.add_argument('--wandb_resume', type=bool, help='wandb resume')
    parser.add_argument('--wandb_run_name', type=str, help='wandb run name')
    parser.add_argument('--wandb_id', type=str, help='wandb id')

    # cmd_args = None
    cmd_args, unknown = parser.parse_known_args()

    return Args(config_filename, cmd_args)
