import yaml
import sys
from torch.utils.data import DataLoader, SubsetRandomSampler

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops_mrDiff as DataSet
from liran_project.utils.util import ecg_signal_difference, check_gpu_memory_usage
from liran_project.utils.common import *

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
        self.model_start_training_time = None
        
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
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.optimization.learning_rate, weight_decay=self.args.optimization.weight_decay)
        return self.model_optim

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
        
        config_dict = self.args.configs.to_dict()
        # split the windows to fixed size context and label windows
        fs = config_dict['data']['fs']
        context_window_size = config_dict['training']['sequence']['label_len']  # minutes * seconds * fs
        label_window_size = config_dict['training']['sequence']['pred_len']  # minutes * seconds * fs
        window_size = context_window_size+label_window_size
        
        if flag == 'train':
            data_path = config['paths']['train_data']
            start_patiant = config['training']['patients']['start_patient']
            end_patiant = config['training']['patients']['end_patient']
        elif flag == 'val':
            data_path = config['paths']['val_data']
            start_patiant = config['validation']['patients']['start_patient']
            end_patiant = config['validation']['patients']['end_patient']
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
                                normalize_method = self.args.data.norm_method,
                                )
        
        if flag == 'test':
            shuffle_flag = False 
            drop_last = False
            batch_size = self.args.optimization.test_batch_size
            freq=self.args.data.freq
            sampler = self._get_nth_sampler(dataset, n=8)
        elif flag=='pred':
            shuffle_flag = False 
            drop_last = False 
            batch_size = 1
            freq=self.args.detail_freq
            sampler = None
        elif flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.optimization.batch_size
            freq=self.args.data.freq
            sampler = None
        elif flag == 'val':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.optimization.batch_size
            freq=self
            sampler = self._get_nth_sampler(dataset, n=8)
        else:
            raise ValueError("Invalid flag")
        

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=config['hardware']['num_workers'],
            drop_last=drop_last,
            sampler=sampler
        )
        
        self.datasets[flag] = dataset
        self.dataloaders[flag] = data_loader
        
        
        return dataset, data_loader
    
    def _get_nth_sampler(self, dataset, n=1, seed=None):
        
        random.seed(seed)
        
        # Calculate indices of every nth batch
        indices = list(range(0, len(dataset), 1))
        random.shuffle(indices)
        
        # take every nth index
        indices = indices[::n]
        
        random.shuffle(indices)
        
        return SubsetRandomSampler(indices)

    def vali(self, vali_data, vali_loader, pretrain=False):
        
        total_loss = []
        self.model.eval()
        results = Metrics("val")
        
        # vali_loader_pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc='vali_loader_pbar', position=-1, leave=False)

        with torch.no_grad():
            vali_loader_pbar = tqdm(enumerate(vali_loader), total=len(vali_loader), desc='vali_loader_pbar', position=1, leave=True)
            for i, DATA in vali_loader_pbar:

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
                        batch_y = batch_y[:, :, -self.args.training.sequence.pred_len:].to(self.device)
                        loss = F.mse_loss(outputs_without_R_peaks.detach().cpu(), batch_y_without_RR.detach().cpu())

                    else:
                        loss = self.model.train_forward(batch_x_without_RR, batch_x_mark, batch_y_without_RR, batch_y_mark, train_val=True)

                results.append_ecg_signal_difference(batch_y.detach().cpu(), outputs_without_R_peaks.detach().cpu(), self.args.data.fs)
                results.append_loss(loss.detach().cpu())

                if self.args.training.model_info.model in ["DDPM"]:
                    if self.args.general.features == "M" or (self.args.general.dataset in ["caiso", "production"]): 
                        if i > 5:
                            break

        mean_results = results.calc_mean()
        self.model.train()
        return mean_results
    
    def train(self, setting):
            
        if self.args.resume_exp.resume and self.args.resume_exp.resume_configuration:
            resume_config = self.args.resume_exp
            checkpoint = torch.load(self.args.resume_exp.specific_chpt_path, map_location='cpu')
            self.args.update_config_from_dict(checkpoint["configuration_parameters"])
            # checkpoint["configuration_parameters"]
            checkpoint = None
            self.args.resume_exp = resume_config
            
        if "train" not in self.dataloaders.keys():
            self.read_data(flag='train')
        
        train_loader = self.dataloaders['train']
            
        if "val" not in self.dataloaders.keys():
            self.read_data(flag='val')

        vali_loader = self.dataloaders['val']

        # train_data, train_loader = self._get_data(flag = 'train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag = 'test')
        
        # test_data, test_loader = self.datasets['test'], self.dataloaders['test']
        train_steps = len(train_loader)  # num_batches

        # # Load pre-trained models if training mode is "TWO"
        # if self.args.general.training_mode == "TWO":
        #     print('loading model')
        #     self.model.base_models.load_state_dict(torch.load(os.path.join(self.args.paths.checkpoints + setting, 'pretrain_checkpoint.pth')))

        time_now = time.time()
            
        str_time_now = time.strftime("%d_%m_%Y_%H%M", time.localtime(time_now))
        self.model_start_training_time = str_time_now
        
        self.args.paths.checkpoints = os.path.join(self.args.paths.checkpoints, setting, self.model_start_training_time)
        tqdm.write(f"Saving model to {self.args.paths.checkpoints}")
        os.makedirs(self.args.paths.checkpoints, exist_ok=True)


        self.early_stopping = EarlyStopping(patience=self.args.optimization.patience, verbose=True)
        
        self.model_optim = self._select_optimizer()
        
        # Create a learning rate scheduler
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=self.model_optim,
                            steps_per_epoch=train_steps,
                            pct_start=self.args.optimization.pct_start,
                            epochs=self.args.training.iterations.train_epochs,
                            max_lr=self.args.optimization.learning_rate)
        
        resume_epoch = 0
        if self.args.resume_exp.resume:
            resume_epoch = self.load_checkpoint(self.args.resume_exp.specific_chpt_path)

        
        train_epochs = self.args.training.iterations.train_epochs
        
        update_stat_interval = 100  # update the statistics every 100 iterations
        
        epochs_pbar = tqdm(range(train_epochs), total=train_epochs ,desc='epochs_pbar', position=0, leave=True)
        
        save_prev_cpt = 1
        for epoch in epochs_pbar:
            while epoch < resume_epoch:
                epoch += 1
                epochs_pbar.update(1)
                epochs_pbar.set_postfix({"epoch": epoch + 1})
            
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            results = Metrics("train")
            
            train_loader_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='train_loader_pbar', position=1, leave=True)

            start_time = time.time()
            
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in train_loader_pbar:
                
                if epoch < 2 and batch_idx < 2 and self.args.hardware.print_gpu_memory_usage:
                    tqdm.write(  f"epoch: {epoch}, i: {batch_idx}"\
                            f"{check_gpu_memory_usage(self.device)}"
                            )
                
                batch_x_without_RR = batch_x[:, 0, :].unsqueeze(-1)
                batch_y_without_RR = batch_y[:, 0, :].unsqueeze(-1)
                    
                batch_x_without_RR = batch_x_without_RR.float().to(self.device)
                batch_y_without_RR = batch_y_without_RR.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                self.model_optim.zero_grad()
                if epoch < 2 and batch_idx < 2 and self.args.hardware.print_gpu_memory_usage:
                    loss = self.model.train_forward(batch_x_without_RR, None, batch_y_without_RR, None)
                else:
                    loss = self.model.train_forward(batch_x_without_RR, None, batch_y_without_RR, None)  # used to be (batch_x, batch_x_mark, batch_y, batch_y_mark) but I think the marks are deprecated

                results.append_loss(loss.item())
                
                if (batch_idx + 1) % update_stat_interval == 0: # update the statistics every 100 iterations
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / update_stat_interval
                    left_time = speed * ((self.args.training.iterations.train_epochs - epoch) * train_steps - batch_idx)
                    tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()

                loss.backward()
                self.model_optim.step()

                # elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x)[0]

                if self.args.optimization.lradj == 'TST':
                    adjust_learning_rate(self.model_optim, self.scheduler, epoch + 1, self.args, printout=False)
                    self.scheduler.step()

            tqdm.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # if self.args.training.logging.sample:
            #     outputs = self.model.test_forward(batch_x_without_RR, None, batch_y_without_RR, None)
            #     outputs_without_R_peaks = outputs[:, -self.args.training.sequence.pred_len:, :].detach().permute(0, 2, 1).cpu() # permute(0, 2, 1).cpu()
            #     batch_y = batch_y[:, -self.args.training.sequence.pred_len:, :].detach().cpu() # permute(0, 2, 1).cpu()
            
            #     # pred = outputs.detach()  # outputs.detach().cpu().numpy()  # .squeeze()
            #     # true = batch_y[:, -self.args.training.sequence.pred_len:, :].detach()  # batch_y.detach().cpu().numpy()  # .squeeze()
                
            #     results.append_ecg_signal_difference(batch_y, outputs_without_R_peaks, self.args.data.fs)
                
                
                
            train_loss = results.calc_mean()
            
            
            vali_loss = self.vali(None, vali_loader)
            
            log = {
                "epoch": epoch + 1,
                }
            
            for key, value in train_loss.items():
                if value != 0:
                    log["train_" + key] = value

            dict_vali_loss = {key: value for key, value in vali_loss.items() if value != 0}
            for key, value in dict_vali_loss.items():
                if value != 0:
                    log["vali_" + key] = value
            
            wandb.log(log)

            elapsed_time = time.time() - start_time
            epochs_pbar.set_postfix({"time_elapsed": str(timedelta(seconds=int(elapsed_time))),
                                    "epoch": epoch + 1, 
                                    "Steps": train_steps, 
                                    "Train_Loss": train_loss["loss"],
                                    "Vali_Loss": vali_loss["loss"]
                                    })
            
            #"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss)
            filenames_to_save = self.early_stopping(val_loss=vali_loss['loss'], model=self.model, epoch=epoch, dir_path=self.args.paths.checkpoints, metrics=dict_vali_loss)
            
            for filename_to_save in filenames_to_save:
                
                self.save_checkpoint(val_loss=vali_loss['loss'], 
                                    model=self.model,
                                    dir_path=self.args.paths.checkpoints,
                                    epoch=epoch,
                                    filename=f'{filename_to_save}',
                                    metrics=dict_vali_loss
                                )
                
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.optimization.lradj != 'TST':
                adjust_learning_rate(self.model_optim, self.scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()[0]))
                
            if epoch % self.args.training.logging.log_interval == 0:
                
                log_dir_path = os.path.join(self.args.paths.checkpoints, 'logs')
                                
                self.save_checkpoint(val_loss=vali_loss['loss'], 
                                        model=self.model,
                                        dir_path=log_dir_path,
                                        epoch=epoch,
                                        filename=f'{save_prev_cpt}_checkpoint.pth',
                                        metrics=dict(vali_loss)
                                    )
                
                save_prev_cpt = 1 - save_prev_cpt
                
                
                
                
        best_model_path = self.early_stopping.best_model_path
        print("Training finished")
        print(f"Best model path: ", best_model_path)
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return self.model


    def test(self, setting, time_path=None, test=0, visualize=False):

        if "test" not in self.dataloaders.keys():
            test_data, test_loader = self._get_data(flag='test')
        else:
            test_data, test_loader = self.datasets['test'], self.dataloaders['test']
        
        if time_path is None:
            if self.model_start_training_time is None:
                time_now = time.time()
                str_time_now = time.strftime("%d_%m_%Y_%H%M", time.localtime(time_now))
                self.model_start_training_time = str_time_now
            time_path = self.model_start_training_time
            assert time_path is not None, "time_path is None. Please provide a time_path from when the model was trained."
        if test:
            print('loading model')
            best_model_path = os.path.join(self.args.paths.checkpoints,'best_checkpoint.pth')
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        preds = []
        trues = []
        # inputx = []
        
        start_time = time.time()
        str_start_time = time.strftime("%d_%m_%Y_%H%M", time.localtime(start_time))
        
        folder_path = os.path.join(self.args.paths.checkpoints, setting, str_start_time)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # add the folder a note stating the path of the model used
        note_filename = os.path.join(folder_path, 'note.txt')
        note = f"Model used: {best_model_path}\n"\
                        f"Time: {str_start_time}\n"\
                        f"Setting: {setting}\n"\
        
        with open(note_filename, 'w') as f:
            f.write(note)
            
            
        results = Metrics("test")

        self.model.eval()
        with torch.no_grad():
            test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='test_pbar', position=0, leave=True)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_pbar:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # encoder - decoder           
                
                batch_x_without_RR = batch_x[:, 0, :].unsqueeze(-1)
                batch_y_without_RR = batch_y[:, 0, :].unsqueeze(-1)

                outputs = self.model.test_forward(batch_x_without_RR, None, batch_y_without_RR, None)

                # if i == 1:
                #     raise Exception(">>>")

                outputs = outputs.detach()[:, -self.args.training.sequence.pred_len:, :]
                batch_y_without_RR = batch_y_without_RR.detach()[:, -self.args.training.sequence.pred_len:, :]
                batch_y_with_RR = batch_y.detach()[:, :, -self.args.training.sequence.pred_len:]
                outputs = outputs.cpu().numpy()
                batch_y_without_RR = batch_y_without_RR.cpu().numpy()

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

                
                # if i == 0:
                #     preds = pred
                #     trues = true
                # else:
                #     preds = np.concatenate((preds, pred), axis=0)
                #     trues = np.concatenate((trues, true), axis=0)

                # inputx.append(his)
                if i % 1 == 0 and i < 20:
                    input = his # batch_x.detach().cpu().numpy()

                    id_worst = self.args.training.identifiers.id_worst # -1 # -1

                    history = input[0, -336:, id_worst]
                    gt = true[0, :, id_worst]
                    pd = pred[0, :, id_worst]
                    if visualize:
                        visual(history, gt, pd, os.path.join(folder_path, str(i) + '.png'))
                        
                # result saves
                metrics_keys = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'rse', 'corr', 'nrmse']
                metrics_vals = metric(pred, true)
                metrics_dict = {}

                for i in range(len(metrics_keys)):
                    metrics_dict[metrics_keys[i]] = metrics_vals[i]

                results.append_metrics(metrics_dict)
                
                
                if isinstance(true, np.ndarray):
                    true = torch.from_numpy(true).to(self.device).permute(0, 2, 1)
                    batch_y_second_channel = batch_y_with_RR[:, 1, :].unsqueeze(1)
                    true = torch.cat((true, batch_y_second_channel.to(self.device)), dim=1)
                else:
                    batch_y_second_channel = batch_y_with_RR[:, 1, :].unsqueeze(1)
                    true = torch.cat((true, batch_y_second_channel), dim=1).permute(0, 2, 1)
                    
                if isinstance(pred, np.ndarray):
                    pred = torch.from_numpy(pred).permute(0, 2, 1)
                    
                results.append_ecg_signal_difference(true, pred, self.args.data.fs)
                
                end_time = time.time()
                elapsed_time_ms = (end_time - start_time) * 1000 / np.shape(batch_x_without_RR)[0]

                if i < 5:
                    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
                
                test_pbar.set_postfix({"time_elapsed": f"{elapsed_time_ms:.2f} ms",
                                        "batch": i + 1,
                                        "Steps": len(test_loader)
                                        })
                    
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)
        
        print(">>------------------>", np.shape(preds), np.shape(trues))

        id_worst = None
        
        """
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

            if visualize:
                plt.figure(figsize=(12,5))
                plt.bar(range(self.args.data.num_vars),res,align = "center",color = "steelblue",alpha = 0.6)
                plt.ylabel("MSE")
                plt.savefig(os.path.join(folder_path, 'MTS_errors.png'))
            
                plt.figure(figsize=(10,5))
                plt.hist(res, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
                plt.xlabel("mse")
                plt.ylabel("frequency")
                plt.savefig(os.path.join(folder_path, 'MTS_errors_hist.png'))
        """
        # print(">>------------------>", np.shape(preds), np.shape(trues))

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        
            
        test_loss = results.calc_mean()
        log = {f"test_{key}": value for key, value in test_loss.items() if value != 0}
        
        # note += f"Test loss: {test_loss=}\n"
        wandb.log(log)

        return test_loss
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        
        # assuming at least those keys are present: model_state_dict, optimizer_state, epoch, loss_and_metrics, learning_rate_scheduler_state, configuration_parameters
        resume_config = self.args.resume_exp
        
        try:
            model_state_dict = checkpoint["model_state_dict"]
            self.model.load_state_dict(model_state_dict, strict=False)
            
            args_configs_box = self.args.configs
            
            if resume_config["resume_configuration"]:
                args_configs_box = checkpoint["configuration_parameters"]
                if 'resume' in args_configs_box.keys():  # for backward compatibility from previous versions
                    args_configs_box['resume_exp'] = args_configs_box['resume']
                    del args_configs_box['resume']
                
            if resume_config["resume_optimizer"]:
                self.model_optim.load_state_dict(checkpoint["optimizer_state"])
                print('Successfully loaded optimizer from checkpoint')
            if resume_config["resume_scheduler"]:
                self.scheduler.load_state_dict(checkpoint["learning_rate_scheduler_state"])
                print('Successfully loaded scheduler from checkpoint')
            if resume_config["resume_metrics"]:
                metrics = checkpoint["loss_and_metrics"]
                self.early_stopping.best_metrics.update(metrics)
                print('Successfully loaded metrics from checkpoint')
            
            self.args.update_config_from_dict(args_configs_box)
            self.args.resume_exp.was_resumed = True  
            
            print('Successfully loaded model from specific_chpt_path')
            
        except Exception as e:
            print(f"Exception: {e=}")
            raise Exception('specific_chpt_path not valid')
        
        return checkpoint["epoch"]
        
    
    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        
    def save_checkpoint(self, val_loss, model, dir_path, epoch=0, filename='checkpoint.pth', metrics={}):
        
        print(f"saving checkpoint to: {dir_path=}, {filename=}")
        # check if model has save_checkpoint() method
        if hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(dir_path=dir_path,
                                    filename=filename,
                                    epoch=epoch,
                                    model=model,
                                    val_loss=val_loss,
                                    metrics=metrics
                                    )
        else:
            
            os.makedirs(dir_path, exist_ok=True)
            filename = os.path.join(dir_path, filename)
            
            savings = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": self.model_optim.state_dict(),
                        "epoch": epoch,
                        "loss_and_metrics": metrics,
                        "learning_rate_scheduler_state": self.scheduler.state_dict(),
                        "configuration_parameters": self.args.configs.to_dict(),
                    }
            
            torch.save(savings, filename)
            
        print("checkpoint saved")
    
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