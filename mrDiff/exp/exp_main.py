import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from data_process.etth_dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Wind, Dataset_Caiso, Dataset_Production, Dataset_Caiso_M, Dataset_Production_M
from data_process.financial_dataloader import DatasetH
from data_process.forecast_dataloader import ForecastDataset

from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
from models_diffusion import DDPM


from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args
        self.set_models_using_meta = ["PDSB", "DDPM"]

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

    def _get_data(self, flag):
        
        args = self.args
        
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'electricity': DatasetH, 
            'solar_AL': DatasetH, 
            'exchange_rate': DatasetH, 
            'traffic': DatasetH, 
            'autoformer_ECL': Dataset_Custom, 
            'autoformer_traffic': Dataset_Custom, 
            'autoformer_weather': Dataset_Custom,
            'autoformer_exchange': Dataset_Custom,
            'autoformer_wind': Dataset_Wind, 
            'caiso': Dataset_Caiso, 
            'production': Dataset_Production, 
            'caiso_m': Dataset_Caiso_M, 
            'production_m': Dataset_Production_M, 
        }

        # autoformer_ECL, autoformer_traffic, autoformer_weather, autoformer_wind, autoformer_exchange, caiso, caiso_m, production, production_m

        Data = data_dict[self.args.general.dataset]
        timeenc = 0 if args.data.embed!='timeF' else 1
        if flag in ['test']:
            shuffle_flag = False; drop_last = False; batch_size = args.optimization.test_batch_size; freq=args.data.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.optimization.batch_size; freq=args.data.freq
        
        data_set = Data(
            args,
            root_path=args.paths.root_path,
            data_path=args.paths.data_path,
            flag=flag,
            size=[args.training.sequence.context_len, args.training.sequence.label_len, args.training.sequence.pred_len],
            features=args.general.features,
            target=args.data.target,
            inverse=args.data.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.data.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.hardware.num_workers,
        drop_last=drop_last)
        
        return data_set, data_loader
    
    def vali(self, vali_data, vali_loader, pretrain=False):
        
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, DATA in enumerate(vali_loader):

                if self.args.general.dataset in ['monash','lorenz']:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = DATA
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = DATA

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if pretrain:
                    loss = self.model.pretrain_forward(batch_x, batch_x_mark, batch_y, batch_y_mark, pretrain_val=True)
                else:

                    if self.args.training.model_info.model in ["DDPM", "PDSB"]:

                        outputs = self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)

                        f_dim = -1 if self.args.general.features == 'MS' else 0
                        outputs = outputs[:, -self.args.training.sequence.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.training.sequence.pred_len:, f_dim:].to(self.device)
                        loss = F.mse_loss(outputs.detach().cpu(), batch_y.detach().cpu())

                    else:
                        loss = self.model.train_forward(batch_x, batch_x_mark, batch_y, batch_y_mark, train_val=True)

                total_loss.append(loss.detach().cpu())

                if self.args.training.model_info.model in ["DDPM"]:
                    if self.args.general.features == "M" or (self.args.general.dataset in ["caiso", "production"]): 
                        if i > 5:
                            break

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):

        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag = 'test')

        # Load pre-trained models if training mode is "TWO"
        if self.args.general.training_mode == "TWO":
            print('loading model')
            self.model.base_models.load_state_dict(torch.load(os.path.join(self.args.paths.checkpoints + setting, 'pretrain_checkpoint.pth')))

        path = os.path.join(self.args.paths.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

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
        
        for epoch in range(self.args.training.iterations.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                start_time = time.time()

                model_optim.zero_grad()
                loss = self.model.train_forward(batch_x, None, batch_y, None)  # used to be (batch_x, batch_x_mark, batch_y, batch_y_mark) but I think the marks are deprecated

                train_loss.append(loss.item())
                
                if (i + 1) % update_stat_interval == 0: # update the statistics every 100 iterations
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / update_stat_interval
                    left_time = speed * ((self.args.training.iterations.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                end_time = time.time()
                # elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                if self.args.optimization.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            
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
            self.model.load_state_dict(torch.load(os.path.join(self.args.paths.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(self.args.paths.checkpoints, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # self.model.eval()
        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            
            # encoder - decoder
            start_time = time.time()

            outputs = self.model.test_forward(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            end_time = time.time()
            elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

            if i < 5:
                print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

            # if i == 1:
            #     raise Exception(">>>")

            outputs = outputs[:, -self.args.training.sequence.pred_len:, :]
            batch_y = batch_y[:, -self.args.training.sequence.pred_len:, :]
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            # print(np.shape(pred), np.shape(true))
            # (32, 12, 1) (32, 12, 1)
            his = batch_x.detach().cpu().numpy()

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

        mae, mse, rmse, mape, mspe, rse, corr, nrmse = metric(preds, trues)
        print('mae: {}, mse: {}, rmse: {}, mape: {}, mspe: {}, rse: {}, corr: {}, nrmse: {}'.format(mae, mse, rmse, mape, mspe, rse, corr, nrmse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae: {}, mse: {}, rmse: {}, mape: {}, mspe: {}, rse: {}, corr: {}, nrmse: {}'.format(mae, mse, rmse, mape, mspe, rse, corr, nrmse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, rse, corr, nrmse]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return mae, mse, rmse, mape, mspe, rse, corr, nrmse







