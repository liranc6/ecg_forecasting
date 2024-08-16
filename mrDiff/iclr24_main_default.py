

import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


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
# autoformer_ECL, autoformer_traffic, autoformer_weather, autoformer_exchange, autoformer_wind, caiso, production, caiso_m, production_m
# 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'
parser.add_argument('--dataset', type=str, required=False, default='ETTh1', help="['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'solar_AL', 'exchange_rate', 'traffic', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']")
parser.add_argument('--features', type=str, default='S', choices=['S', 'M'], help='features S is univariate, M is multivariate')

parser.add_argument('--seq_len', type=int, default=336, help='input sequence length of SCINet encoder, look back window')
parser.add_argument('--label_len', type=int, default=336, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length, horizon')

parser.add_argument('--opt_loss_type', type=str, default='mse') 

parser.add_argument('--model', type=str, required=False, default='DDPM', help='model of the experiment')
parser.add_argument('--base_model', type=str, required=False, default='Linear', help='model of the experiment')
parser.add_argument('--u_net_type', type=str, default='v0') 

parser.add_argument('--vis_MTS_analysis', type=int, default=0, help='status')

parser.add_argument('--use_window_normalization', type=bool, default=True)

parser.add_argument('--use_future_mixup', type=bool, default=True)

parser.add_argument('--use_X0_in_THiDi', type=bool, default=False)
parser.add_argument('--channel_independence', type=bool, default=False)

parser.add_argument('--training_mode', type=str, default='ONE')

# =================================================
# Diffusion models
parser.add_argument('--smoothed_factors', default=[5, 25, 51], nargs='+', type=int)
# parser.add_argument('--linear_history_lens', default=[336], nargs='+', type=int)
parser.add_argument('--interval', type=int, default=1000, help='number of diffusion steps')
parser.add_argument('--ot-ode', default=True, help='use OT-ODE model')
parser.add_argument("--beta-max", type=float, default=1.0, help="max diffusion for the diffusion model")
parser.add_argument("--t0", type=float, default=1e-4, help="sigma start time in network parametrization")
parser.add_argument("--T", type=float, default=0.02, help="sigma end time in network parametrization")
parser.add_argument('--nfe', type=int, default=20)

parser.add_argument('--ablation_study_F_type', type=str, default="Linear", help="Linear, CNN")

# =================================================
parser.add_argument('--beta_schedule', type=str, default="cosine")
parser.add_argument('--beta_dist_alpha', type=float, default=-1)  # -1
parser.add_argument('--ablation_study_masking_type', type=str, default="none", help="none, hard, segment")
parser.add_argument('--ablation_study_masking_tau', type=float, default=0.9)
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02) # 0.02

# =================================================
parser.add_argument('--diff_steps', type=int, default=100, help='number of diffusion steps')
parser.add_argument('--ddpm_inp_embed', type=int, default=64) # ddpm_num_channels
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
parser.add_argument('--our_ddpm_clip', type=float, default=100) # 100

parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
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
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')  # '0,1,2,3'

# =====================================
parser.add_argument('--individual', default=True, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
parser.add_argument('--subtract_short_terms', type=int, default=0, help='0: subtract mean; 1: subtract last')


parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--use_ar_init', type=bool, default=False)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--uncertainty', type=bool, default=False)

parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--normtype', type=int, default=0)

args = parser.parse_args()

# random seed
# fix_seed = args.random_seed
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

args.hardware.use_gpu = True if torch.cuda.is_available() and args.hardware.use_gpu else False

if args.hardware.use_gpu and args.hardware.use_multi_gpu:
    args.dvices = args.hardware.devices.replace(' ', '')
    device_ids = args.hardware.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.hardware.gpu = args.device_ids[0]
    args.optimization.patience = 30

print('Args in experiment:')
print(args)


if args.general.dataset == 'ETTh1':
    args.datasets_dir = '../datasets/SCINet_timeseries/ETT-data/ETT'
    args.data = 'ETTh1'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'ETTh1.csv'
    args.num_vars = 7
if args.general.dataset == 'ETTh2':
    args.datasets_dir = '../datasets/SCINet_timeseries/ETT-data/ETT'
    args.data = 'ETTh2'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'ETTh2.csv'
    args.num_vars = 7
if args.general.dataset == 'ETTm1':
    args.datasets_dir = '../datasets/SCINet_timeseries/ETT-data/ETT'
    args.data = 'ETTm1'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'ETTm1.csv'
    args.num_vars = 7
if args.general.dataset == 'ETTm2':
    args.datasets_dir = '../datasets/SCINet_timeseries/ETT-data/ETT'
    args.data = 'ETTm2'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'ETTm2.csv'
    args.num_vars = 7
if args.general.dataset == 'electricity':
    args.datasets_dir = '../datasets/SCINet_timeseries/financial'
    args.data = 'electricity'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'electricity.txt'
    args.num_vars = 321
if args.general.dataset == 'solar_AL':
    args.datasets_dir = '../datasets/SCINet_timeseries/financial'
    args.data = 'solar_AL'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'solar_AL.txt'
    args.num_vars = 137
if args.general.dataset == 'exchange_rate':
    args.datasets_dir = '../datasets/SCINet_timeseries/financial'
    args.data = 'exchange_rate'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'exchange_rate.txt'
    args.num_vars = 8
if args.general.dataset == 'traffic':
    args.datasets_dir = '../datasets/SCINet_timeseries/financial'
    args.data = 'traffic'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'traffic.txt'
    args.num_vars = 862

# ============================================================================
if args.general.dataset == 'autoformer_ECL':
    args.datasets_dir = '../datasets/prediction/data_autoformer/electricity/'
    args.data = 'autoformer_ECL'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'electricity.csv'
    args.num_vars = 321
if args.general.dataset == 'autoformer_traffic':
    args.datasets_dir = '../datasets/prediction/data_autoformer/traffic/'
    args.data = 'autoformer_traffic'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'traffic.csv'
    args.num_vars = 862
if args.general.dataset == 'autoformer_weather':
    args.datasets_dir = '../datasets/prediction/data_autoformer/weather/'
    args.data = 'autoformer_weather'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'weather.csv'
    args.num_vars = 21
if args.general.dataset == 'autoformer_exchange':
    args.datasets_dir = '../datasets/prediction/data_autoformer/exchange_rate/'
    args.data = 'autoformer_exchange'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'exchange_rate.csv'
    args.num_vars = 8
if args.general.dataset == 'autoformer_wind':
    args.datasets_dir = '../datasets/prediction/data_autoformer/wind/'
    args.data = 'autoformer_wind'
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = 'wind.csv'
    args.num_vars = 7

if args.general.dataset in ['caiso', 'caiso_m']:
    args.datasets_dir = '../datasets/prediction/data_depts/caiso/'
    args.data = args.general.dataset
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = ''
    args.num_vars = 10
    if args.general.dataset == "caiso_m":
        assert args.general.features == "M"
if args.general.dataset in ['production', 'production_m']:
    args.datasets_dir = '../datasets/prediction/data_depts/nordpool/'
    args.data = args.general.dataset
    args.paths.root_path = args.datasets_dir
    args.paths.data_path = ''
    args.num_vars = 18
    if args.general.dataset == "production_m":
        assert args.general.features == "M"

if args.general.features == "S":
    args.num_vars = 1

mae_, mse_, rmse_, mape_, mspe_, rse_, corr_, nrmse_ = [], [], [], [], [], [], [], []
if args.evaluate:
    ii =0
    
    d_name = args.data
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_inv{}_itr{}'.format(args.model, d_name, args.general.features, args.training.sequence.seq_len, args.training.sequence.label_len, args.training.sequence.pred_len, args.optimization.learning_rate,args.optimization.batch_size, args.data.inverse, ii)
    if args.tag is not None:
        setting += "_{}".format(args.tag)

    if args.general.dataset == 'm4':
        exp = Exp_Main_M4(args)
    else:
        exp = Exp_Main(args)
    
    if args.general.training_mode == "TWO":
        print('>>>>>>>pretrain testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.pretrain_test(setting, test=1)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, mse, rmse, mape, mspe, rse, corr, nrmse = exp.test(setting, test=1)
    print('Final mean mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, corr:{}, nrmse:{}'.format(mae, mse, rmse, mape, mspe, rse, corr, nrmse))

else:
    if args.training.iterations.itr:
        for ii in range(args.training.iterations.itr):
            # setting record of experiments
            
            # random seed
            if args.training.iterations.itr > 1:
                fix_seed = ii
            else:
                fix_seed = args.random_seed
            
            random.seed(fix_seed)
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
            torch.backends.cudnn.enabled = True

            d_name = args.data
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_inv{}_itr{}'.format(args.model,d_name, args.general.features, args.training.sequence.seq_len, args.training.sequence.label_len, args.training.sequence.pred_len, args.optimization.learning_rate,args.optimization.batch_size, args.data.inverse, ii)
            if args.tag is not None:
                setting += "_{}".format(args.tag)

            exp = Exp_Main(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse, rmse, mape, mspe, rse, corr, nrmse = exp.test(setting, test=1)
            
            mae_.append(mae)
            mse_.append(mse)
            rmse_.append(rmse)
            mape_.append(mape)
            mspe_.append(mspe)
            rse_.append(rse)
            corr_.append(corr)
            nrmse_.append(nrmse)

            torch.cuda.empty_cache()
            
            print('Final mean normed: ')
            print('> mae:{:.4f}, std:{:.4f}'.format(np.mean(mae_), np.std(mae_)))
            print('> mse:{:.4f}, std:{:.4f}'.format(np.mean(mse_), np.std(mse_)))
            print('> rmse:{:.4f}, std:{:.4f}'.format(np.mean(rmse_), np.std(rmse_)))
            print('> mape:{:.4f}, std:{:.4f}'.format(np.mean(mape_), np.std(mape_)))
            print('> rse:{:.4f}, std:{:.4f}'.format(np.mean(rse_), np.std(rse_)))
            print('> corr:{:.4f}, std:{:.4f}'.format(np.mean(corr_), np.std(corr_)))
            print('> nrmse:{:.4f}, std:{:.4f}'.format(np.mean(nrmse_), np.std(nrmse_)))
                





