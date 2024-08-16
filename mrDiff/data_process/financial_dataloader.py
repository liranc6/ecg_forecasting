
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch

import numpy as np


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DatasetH(Dataset):

    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
         
        self.args = args
        self.P = args.training.sequence.seq_len
        self.h = args.training.sequence.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        file_name = os.path.join(args.datasets_dir,args.paths.data_path)
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        
        if features=='S':
            self.rawdat = np.expand_dims(self.rawdat[:,args.data.target], axis=-1)

        # print(np.shape(self.rawdat))  # (26304, 321)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        
        self.data_set = {}
        self.scale = np.ones(self.m)
        self.bias =  np.zeros(self.m)
        if self.args.training.sequence.pred_len > 90:
            train = 0.7
            valid = 0.1
            self.normalize = 4
        else:
            train = 0.6
            valid = 0.2
            self.normalize = 2

        self._normalized(self.normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
    
    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.data_set[0] = train_set
        self.data_set[1] = valid_set
        self.data_set[2] = test_set
    
    def __getitem__(self, index):
        
        idx_set = self.data_set[self.set_type] 
        end = idx_set[index] - self.h + 1
        start = end - self.P

        seq_x = self.dat[start:end, :]
        seq_y = self.dat[end:(idx_set[index]+1), :]
        seq_x_mark = np.zeros([np.shape(seq_x)[0], 1])
        seq_y_mark = np.zeros([np.shape(seq_y)[0], 1])

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_set[self.set_type] ) 

    def _normalized(self, normalize):

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            # normalized by the maximum value of entire matrix.
            self.dat = self.rawdat / np.max(self.rawdat)
        
        if (normalize == 2):
            # normlized by the maximum value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

        if (normalize == 3):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i]) #std
                self.bias[i] = np.mean(self.rawdat[:, i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
                
        if (normalize == 4):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:int(self.dat.shape[0]*0.7), i]) #std
                self.bias[i] = np.mean(self.rawdat[:int(self.dat.shape[0]*0.7), i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]

    def inverse_transform(self, data):
        
        normalize = self.normalize

        if (normalize == 0):
            return data

        if (normalize == 1):
            # normalized by the maximum value of entire matrix.
            # self.dat = self.rawdat / np.max(self.rawdat)
            return data * np.max(self.rawdat)
        
        if (normalize == 2):
            # normlized by the maximum value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                # self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
                data[:, i] = data[:, i]*np.max(np.abs(self.rawdat[:, i]))
            return data

        if (normalize == 3):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i]) #std
                self.bias[i] = np.mean(self.rawdat[:, i])
                # self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
                data[:, i] = data[:, i]*self.scale[i] + self.bias[i]
            return data
                
        if (normalize == 4):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:int(self.dat.shape[0]*0.7), i]) #std
                self.bias[i] = np.mean(self.rawdat[:int(self.dat.shape[0]*0.7), i])
                # self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
                data[:, i] = data[:, i]*self.scale[i] + self.bias[i]
            return data

            