import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import defaultdict as defultdict
import os

plt.switch_backend('agg')

import pickle

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.optimization.learning_rate * (0.2 ** (epoch // 2))
    if args.optimization.lradj == 'type1':
        lr_adjust = {epoch: args.optimization.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.optimization.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.optimization.lradj == 'type3':
        lr_adjust = {epoch: args.optimization.learning_rate if epoch < 3 else args.optimization.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.optimization.lradj == 'constant':
        lr_adjust = {epoch: args.optimization.learning_rate}
    elif args.optimization.lradj == '3':
        lr_adjust = {epoch: args.optimization.learning_rate if epoch < 10 else args.optimization.learning_rate*0.1}
    elif args.optimization.lradj == '4':
        lr_adjust = {epoch: args.optimization.learning_rate if epoch < 15 else args.optimization.learning_rate*0.1}
    elif args.optimization.lradj == '5':
        lr_adjust = {epoch: args.optimization.learning_rate if epoch < 25 else args.optimization.learning_rate*0.1}
    elif args.optimization.lradj == '6':
        lr_adjust = {epoch: args.optimization.learning_rate if epoch < 5 else args.optimization.learning_rate*0.1}  
    elif args.optimization.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_metrics = defultdict(lambda: np.Inf)
        self.best_model_path = None

    def __call__(self, val_loss, model, dir_path, epoch, filename='best_checkpoint.pth', metrics={}, **kwargs):
        self.metrics = metrics
        score = val_loss
        
        filenames_to_save = []
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss = val_loss, model = model, dir_path = dir_path, epoch=epoch, filename = filename, metrics=metrics)
            self.best_model_path = os.path.join(dir_path, filename)
            filenames_to_save.append(f'{filename}')
        elif score >= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = filename
        else:
            self.best_score = score
            self.save_checkpoint(val_loss = val_loss, model = model, dir_path = dir_path, epoch=epoch, filename = filename, metrics=metrics)
            self.best_model_path = os.path.join(dir_path, filename)
            filenames_to_save.append(f'{filename}')
            self.counter = 0
        
        # assumption: we want to minimize the metrics
        for key, value in metrics.items():
            if value < self.best_metrics[key] - self.delta: 
                self.best_metrics[key] = value
                
                curr_filename = filename.split('.')[0]
                curr_filename = f'{curr_filename[:-4]}_{key}.pth'
                self.save_checkpoint(val_loss = val_loss, model = model, dir_path = dir_path, epoch=epoch, filename = curr_filename, metrics=metrics)
                filenames_to_save.append(f'{curr_filename}')
                
        return filenames_to_save

    def save_checkpoint(self, val_loss, model, dir_path, epoch=0, filename='checkpoint.pth', metrics={}):
        
        return
        # print(f"saving checkpoint to: {dir_path=}, {filename=}")
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
    
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
            filename = os.path.join(dir_path, filename)
            
            savings = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": model.optimizer.state_dict(),
                        "epoch": epoch,
                        "loss_and_metrics": metrics,
                        "learning_rate_scheduler_state": model.scheduler.state_dict(),
                        "configuration_parameters": model.args,
                    }
            
            torch.save(savings, filename)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    A class for standardizing data by removing the mean and scaling to unit variance.

    This class computes the mean and standard deviation of the data during fitting and applies the transformation to standardize the data. It also provides a method to inverse the transformation.

    Methods:
        fit(data): Computes the mean and standard deviation of the input data.
        transform(data): Standardizes the input data using the computed mean and standard deviation.
        inverse_transform(data): Reverts the standardized data back to its original scale.

    Args:
        data (numpy.ndarray or torch.Tensor): The input data to fit or transform.
    
    Returns:
        numpy.ndarray or torch.Tensor: The standardized or inverse transformed data.
    """

    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def visual(history, true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(8,5))
    ind_his = list(np.arange(0,len(history)))
    ind_out = list(np.arange(len(history), len(history)+len(true)))
    
    plt.plot(ind_his, history, '-', label='History', c='#000000', linewidth=1)
    plt.plot([ind_his[-1], ind_his[-1]+1], [history[-1], true[0]], '-', c='#000000', linewidth=1)
    
    plt.plot(ind_out, true, '-', label='GroundTruth', c='b', linewidth=1) # #999999
    
    if preds is not None:
        plt.plot(ind_out, preds, '-', label='Prediction', c='r', linewidth=1)  # #FFB733    

    plt.legend()
    plt.tight_layout()
    print(name)
    plt.savefig(name, bbox_inches='tight') 

    f = open(name[:-4]+'.pkl', "wb")
    pickle.dump(preds, f)
    f.close()

    f = open(name[:-4]+'_ground_truth.pkl', "wb")
    pickle.dump(true, f)
    f.close()

    f = open(name[:-4]+'_history.pkl', "wb")
    pickle.dump(history, f)
    f.close()



        