

import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np


class ForecastDataset(Dataset):

    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
         
        self.args = args
        data_file = os.path.join(args.datasets_dir, args.paths.data_path)
        print('data file:',data_file)
        data = np.load(data_file,allow_pickle=True)
        data = data['data'][:,:,0]
        # (26208, 358)
        if features=='S':
            data = data[:,[args.data.target]]

        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 1 - train_ratio - valid_ratio
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]

        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]

        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')
        if len(test_data) == 0:
            raise Exception('Cannot organize enough test data')
        
        if self.args.data.normtype == 0: # we follow StemGNN and other related works for somewhat fair comparison (orz..), but we strongly suggest use self.args.data.normtype==2!!!
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_mean = np.mean(valid_data, axis=0)
            val_std = np.std(valid_data, axis=0)
            val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
            test_mean = np.mean(test_data, axis=0)
            test_std = np.std(test_data, axis=0)
            test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}
        elif self.args.data.normtype == 1:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            train_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            val_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            test_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        else:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            test_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        
        if flag == "train":
            self.data_set = sub_ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.data.norm_method, norm_statistics=train_normalize_statistic)
        if flag == "val":
            self.data_set = sub_ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                    normalize_method=args.data.norm_method, norm_statistics=val_normalize_statistic)
        if flag == "test":
            self.data_set = sub_ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                                    normalize_method=args.data.norm_method, norm_statistics=test_normalize_statistic)
        
    def __getitem__(self, index):

        seq_x, seq_y = self.data_set.__getitem__(index)

        seq_x_mark = np.zeros([np.shape(seq_x)[0], 1])
        seq_y_mark = np.zeros([np.shape(seq_y)[0], 1])

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):

        return self.data_set.__len__()

    def inverse_transform(self, data):

        return self.data_set.inverse_transform(data)


class sub_ForecastDataset(Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistics=None, interval=1):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistics = norm_statistics
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistics)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        target_data = self.data[hi:hi + self.horizon] #12:24
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

    def inverse_transform(self, data):

    	return de_normalized(data, self.normalize_method, self.norm_statistics)


def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data

