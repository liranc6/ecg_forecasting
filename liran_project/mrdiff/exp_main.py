import os
import sys
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import random
import numpy as np
import yaml
from box import Box
from pprint import pprint
import wandb
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops_mrDiff as DataSet
from liran_project.utils.util import ecg_signal_difference

# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

# from mrDiff.exp.exp_main import Exp_Main
from mrDiff.data_process.etth_dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Wind, Dataset_Caiso, Dataset_Production, Dataset_Caiso_M, Dataset_Production_M
from mrDiff.data_process.financial_dataloader import DatasetH
from mrDiff.data_process.forecast_dataloader import ForecastDataset
from mrDiff.exp.exp_basic import Exp_Basic
from mrDiff.models_diffusion import DDPM
from mrDiff.utils.tools import EarlyStopping, adjust_learning_rate, visual
from mrDiff.utils.metrics import metric


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.set_models_using_meta = ["PDSB", "DDPM"]
        self.datasets = {}
        self.dataloaders = {}
        
    def _build_model(self):
        model_dict = {
            'DDPM': DDPM,
        }
        self.args.device = self.device
        model = model_dict[self.args.training.model_info.model].Model(self.args).float()

        if self.args.hardware.use_multi_gpu and self.args.hardware.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.optimization.learning_rate, weight_decay=self.args.optimization.weight_decay)
        return model_optim

    def _select_meta_optimizer(self):

        params = self.model.parameters()
        self.num_bridges = len(self.args.training.smoothing.smoothed_factors) + 1

        autol = AutoLambda(self.args, self.model, self.device, self.args.autol_init) 
        meta_weight_ls = np.zeros([self.args.training.iterations.train_epochs, self.num_bridges, self.args.num_vars], dtype=np.float32)
        meta_optimizer = optim.Adam([autol.meta_weights], lr=self.args.autol_lr)

        return autol, meta_weight_ls, meta_optimizer

    def read_data(self, flag):
        self._get_data(flag)
        
    def _get_data(self, flag):
        
        config_dict = self.args.config.to_dict()
        # split the windows to fixed size context and label windows
        fs = config_dict['data']['fs']
        context_window_size = config_dict['training']['sequence']['seq_len'] - config_dict['training']['sequence']['label_len']  # minutes * seconds * fs
        label_window_size = config_dict['training']['sequence']['label_len']  # minutes * seconds * fs
        window_size = context_window_size+label_window_size
        
        if flag == 'train' or flag == 'val':
            data_path = config['paths']['train_data']
            start_patiant = config['training']['patients']['start_patient']
            end_patiant = config['training']['patients']['end_patient']
        elif flag == 'test':
            data_path = config['paths']['test_data']
            start_patiant = config['testing']['patients']['start_patient']
            end_patiant = config['testing']['patients']['end_patient']

        dataset = DataSet(context_window_size,
                                label_window_size,
                                data_path,
                                start_patiant=start_patiant,
                                end_patiant=end_patiant,
                                data_with_RR=True,
                                return_with_RR=True,
                                # normalize_method = 'z_score',
                                )
        
        if flag == 'test':
            shuffle_flag = False 
            drop_last = False
            batch_size = self.args.optimization.test_batch_size
            freq=self.args.data.freq
        elif flag=='pred':
            shuffle_flag = False 
            drop_last = False 
            batch_size = 1
            freq=self.args.detail_freq
        else: # train or val
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.optimization.batch_size
            freq=self.args.data.freq

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=config['hardware']['num_workers'],
            drop_last=drop_last
        )
        
        self.datasets[flag] = dataset
        self.dataloaders[flag] = data_loader
        
        
        return dataset, data_loader
    
    def vali(self, vali_data, vali_loader, pretrain=False):
        
        total_loss = []
        self.model.eval()
        results = Metrics("val")
        
        # vali_loader_pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc='vali_loader_pbar', position=-1, leave=False)

        with torch.no_grad():
            for i, DATA in enumerate(vali_loader):

                if self.args.general.dataset in ['monash','lorenz']:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = DATA
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = DATA
                    
                batch_x_without_RR = batch_x[:, 0, :].unsqueeze(-1)
                batch_y_without_RR = batch_y[:, 0, :].unsqueeze(-1)

                batch_x_without_RR = batch_x_without_RR.float().to(self.device)
                batch_y_without_RR = batch_y_without_RR.float().to(self.device)

                batch_x_mark = None  # batch_x_mark.float().to(self.device)
                batch_y_mark = None  # batch_y_mark.float().to(self.device)
                
                if pretrain:
                    loss = self.model.pretrain_forward(batch_x_without_RR, batch_x_mark, batch_y_without_RR, batch_y_mark, pretrain_val=True)
                else:

                    if self.args.training.model_info.model in ["DDPM", "PDSB"]:

                        outputs_without_R_peaks = self.model.test_forward(batch_x_without_RR, batch_x_mark, batch_y_without_RR, batch_y_mark)

                        f_dim = -1 if self.args.general.features == 'MS' else 0
                        outputs_without_R_peaks = outputs_without_R_peaks[:, -self.args.training.sequence.pred_len:, f_dim:].permute(0, 2, 1)
                        batch_y = batch_y[:, -self.args.training.sequence.pred_len:, f_dim:].to(self.device)
                        loss = F.mse_loss(outputs_without_R_peaks.detach().cpu(), batch_y_without_RR.detach().cpu())

                    else:
                        loss = self.model.train_forward(batch_x_without_RR, batch_x_mark, batch_y_without_RR, batch_y_mark, train_val=True)

                results.append_ecg_signal_difference(batch_y.detach().cpu(), outputs_without_R_peaks.detach().cpu(), self.args.data.fs)
                results.append_loss(loss.detach().cpu())  # TODO check how loss looks like

                if self.args.training.model_info.model in ["DDPM"]:
                    if self.args.general.features == "M" or (self.args.general.dataset in ["caiso", "production"]): 
                        if i > 5:
                            break

        mean_results = results.calc_mean()
        self.model.train()
        return mean_results
    
    def train(self, setting):

        # train_data, train_loader = self._get_data(flag = 'train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag = 'test')
        
        _, train_loader = self.datasets['train'], self.dataloaders['train']
        vali_data, vali_loader = self.datasets['val'], self.dataloaders['val']
        # test_data, test_loader = self.datasets['test'], self.dataloaders['test']

        # Load pre-trained models if training mode is "TWO"
        if self.args.general.training_mode == "TWO":
            print('loading model')
            self.model.base_models.load_state_dict(torch.load(os.path.join(self.args.paths.checkpoints + setting, 'pretrain_checkpoint.pth')))

        path = os.path.join(self.args.paths.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)  # num_batches
        
        early_stopping = EarlyStopping(patience=self.args.optimization.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        
        # Create a learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                            steps_per_epoch=train_steps,
                            pct_start=self.args.optimization.pct_start,
                            epochs=self.args.training.iterations.train_epochs,
                            max_lr=self.args.optimization.learning_rate)

        
        update_stat_interval = 100  # update the statistics every 100 iterations
        
        train_epochs = self.args.training.iterations.train_epochs
        
        epochs_pbar = tqdm(range(train_epochs), total=train_epochs ,desc='epochs_pbar', position=0, leave=True)
        
        for epoch in epochs_pbar:
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            results = Metrics("train")
            
            train_loader_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='train_loader_pbar', position=1, leave=True)

            start_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in train_loader_pbar:
                
                batch_x_without_RR = batch_x[:, 0, :].unsqueeze(-1)
                batch_y_without_RR = batch_y[:, 0, :].unsqueeze(-1)
                    
                batch_x_without_RR = batch_x_without_RR.float().to(self.device)
                batch_y_without_RR = batch_y_without_RR.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                model_optim.zero_grad()
                loss = self.model.train_forward(batch_x_without_RR, None, batch_y_without_RR, None)  # used to be (batch_x, batch_x_mark, batch_y, batch_y_mark) but I think the marks are deprecated

                results.append_loss(loss.item())
                
                if (i + 1) % update_stat_interval == 0: # update the statistics every 100 iterations
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / update_stat_interval
                    left_time = speed * ((self.args.training.iterations.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                # elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                if self.args.optimization.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            if self.args.training.logging.sample:
                outputs = self.model.test_forward(batch_x_without_RR, None, batch_y_without_RR, None)
                outputs_without_R_peaks = outputs[:, -self.args.training.sequence.pred_len:, :].detach().permute(0, 2, 1).cpu() # permute(0, 2, 1).cpu()
                batch_y = batch_y[:, -self.args.training.sequence.pred_len:, :].detach().cpu() # permute(0, 2, 1).cpu()
            
                # pred = outputs.detach()  # outputs.detach().cpu().numpy()  # .squeeze()
                # true = batch_y[:, -self.args.training.sequence.pred_len:, :].detach()  # batch_y.detach().cpu().numpy()  # .squeeze()
                
                results.append_ecg_signal_difference(batch_y, outputs_without_R_peaks, self.args.data.fs)
                
                
                
            train_loss = results.calc_mean()
            
            
            vali_loss = self.vali(vali_data, vali_loader)
            
            log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "vali_loss": vali_loss
                }
            log.update(results.get_final())
            
            wandb.log(log)

            elapsed_time = time.time() - start_time
            epochs_pbar.set_postfix(time_elapsed=str(timedelta(seconds=int(elapsed_time))),
                                    epoch=epoch + 1,
                                    Steps=train_steps,
                                    Train_Loss=train_loss,
                                    Vali_Loss=vali_loss)

            #"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss)
            early_stopping(vali_loss['loss'], self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.optimization.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.paths.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(self.args.paths.checkpoints, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        results = Metrics("test")

        # self.model.eval()
        # with torch.no_grad():
        for i, (batch_x, batch_y
        , batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y
             = batch_y
            .float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # encoder - decoder
            start_time = time.time()
            
            batch_x_without_RR = batch_x[:, 0, :].unsqueeze(-1)
            batch_y_without_RR = batch_y
            [:, 0, :].unsqueeze(-1)

            outputs = self.model.test_forward(batch_x_without_RR, None, batch_y_without_RR, None)
            
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x_without_RR)[0]

            if i < 5:
                print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

            # if i == 1:
            #     raise Exception(">>>")

            outputs = outputs[:, -self.args.training.sequence.pred_len:, :]
            batch_y_without_RR = batch_y_without_RR[:, -self.args.training.sequence.pred_len:, :]
            outputs = outputs.detach().cpu().numpy()
            batch_y_without_RR = batch_y_without_RR.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y_without_RR  # batch_y.detach().cpu().numpy()  # .squeeze()

            # print(np.shape(pred), np.shape(true))
            # (32, 12, 1) (32, 12, 1)
            his = batch_x_without_RR.detach().cpu().numpy()

            if self.args.data.inverse:
                B, L, D = np.shape(pred)

                pred = rearrange(pred, 'b l d -> (b l) d')
                true = rearrange(true, 'b l d -> (b l) d')
                his = rearrange(his, 'b l d -> (b l) d')

                pred = test_data.inverse_transform(pred)
                true = test_data.inverse_transform(true)
                his = test_data.inverse_transform(his)
        
                pred = rearrange(pred, '(b l) d -> b l d', b=B, l=L)
                true = rearrange(true, '(b l) d -> b l d', b=B, l=L)
                his = rearrange(his, '(b l) d -> b l d', b=B)

            
            if i == 0:
                preds = pred
                trues = true
            else:
                preds = np.concatenate((preds, pred), axis=0)
                trues = np.concatenate((trues, true), axis=0)

            inputx.append(his)
            if i % 1 == 0 and i < 20:
                input = his # batch_x.detach().cpu().numpy()

                id_worst = self.args.training.identifiers.id_worst # -1 # -1

                history = input[0, -336:, id_worst]
                gt = true[0, :, id_worst]
                pd = pred[0, :, id_worst]
                visual(history, gt, pd, os.path.join(folder_path, str(i) + '.png'))
                
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        
        print(">>------------------>", np.shape(preds), np.shape(trues))

        id_worst = None

        if self.args.general.features == 'M' and self.args.training.analysis.vis_MTS_analysis:

            # print(np.shape(preds),np.shape(trues))
            N, B, L, D = np.shape(preds)
            VIS_P = preds.reshape((N*B, L, D))        
            VIS_T = trues.reshape((N*B, L, D))  

            res = np.mean((VIS_P - VIS_T) ** 2, axis=1)
            res = np.mean(res, axis=0)
            # print(">>>", np.shape(res))

            print("id_worst", np.argmax(res))
            id_worst = np.argmax(res)

            ind = np.argpartition(res, -5)[-5:]
            top5 = res[ind]
            print("top5", ind) # max

            plt.figure(figsize=(12,5))
            plt.bar(range(self.args.data.num_vars),res,align = "center",color = "steelblue",alpha = 0.6)
            plt.ylabel("MSE")
            plt.savefig(os.path.join(folder_path, 'MTS_errors.png'))
            
            plt.figure(figsize=(10,5))
            plt.hist(res, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.xlabel("mse")
            plt.ylabel("frequency")
            plt.savefig(os.path.join(folder_path, 'MTS_errors_hist.png'))

        # print(">>------------------>", np.shape(preds), np.shape(trues))

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = os.path.join(self.args.paths.checkpoints, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        
        metrics_keys = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr', 'nrmse']
        metrics_vals = metric(preds, trues)
        metrics_dict = {}

        for i in range(len(metrics_keys)):
            metrics_dict[metrics_keys[i]] = metrics_vals[i]

        results.append_metrics(metrics_dict)
        results.append_ecg_signal_difference(trues, preds, self.args.data.fs)
        
            
        test_loss = results.calc_mean()
        
        wandb.log(test_loss)

        return test_loss
    
    
class Metrics:
    def __init__(self, mode: str):
        self.metrics = defaultdict(lambda: 0)
        self.metrics.update({
            'mae': 0,
            'mse': 0,
            'rmse': 0,
            'mape': 0,
            'mspe': 0,
            'rse': 0,
            'corr': 0,
            'nrmse': 0,
        })
        self.tmp_metrics = defaultdict(lambda: [])
        
        self.mode = mode
        
    def calc_mean(self):
        for key, value in self.tmp_metrics.items():
            self.metrics[key] = np.mean(value)
            
        # Clear the tmp_metrics
        self.tmp_metrics = defaultdict(lambda: [])
            
        return self.metrics
            
    def append_ecg_signal_difference(self, true, pred, sampling_rate):
        diffs = ecg_signal_difference(true, pred, sampling_rate)
        for key, value in diffs.items():
            self.tmp_metrics[key].append(value)
        
        return diffs
    
    def append_loss(self, loss):
        self.tmp_metrics["loss"].append(loss)
        
    def append_metrics(self, metrics):
        for key, value in metrics.items():
            self.tmp_metrics[key].append(value)
    
    def print_final(self):
        pprint(self.get_final())
            
    def get_final(self):
        ret = {}
        for key, value in self.metrics.items():
            for key in self.metrics:
                ret[self.mode+"_"+key] = value
        return ret
        
    def __str__(self):
        return str(self.metrics)
    
    def __repr__(self):
        return str(self.metrics)