import numpy as np
import torch
import matplotlib.pyplot as plt
import time

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

    def __call__(self, val_loss, model, path, name='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, name='checkpoint.pth'):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + name)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
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



        