

import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            if self.args.training.identifiers.focus_variate>=0:
                df_data = df_raw.iloc[:, [self.args.training.identifiers.focus_variate]]
            else:
                self.target = 'OT'
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        """
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        """
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  
        seq_y = self.data_y[r_begin:r_end] 
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            if self.args.training.identifiers.focus_variate>-1:
                df_data = df_raw.iloc[:, [self.args.training.identifiers.focus_variate]]
            else:
                self.target = 'OT'
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        """
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        """
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        target='OT'

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove('date')

        df_raw = df_raw[['date']+cols+[target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            if self.args.training.identifiers.focus_variate>-1:
                df_data = df_raw.iloc[:, [self.args.training.identifiers.focus_variate]]
            else:
                self.target = 'OT'
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        """
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        """
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        # if self.args.model == "DDPM" and self.set_type == "train":
        #     return 500
        # else:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Wind(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = args.data.target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.dim_datetime_feats = np.shape(self.data_stamp)[-1]

    def __read_data__(self):

        target='wind_power'

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        print(cols)
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            if self.args.training.identifiers.focus_variate>-1:
                df_data = df_raw.iloc[:, [self.args.training.identifiers.focus_variate]]
            else:
                self.target = 'wind_power'
                df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        """
        from statsmodels.tsa.stattools import adfuller
 
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.timeenc = 0
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print("> {}-shape".format(self.flag), np.shape(self.data_x), np.shape(self.data_y))

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Caiso(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        root_path = args.datasets_dir

        set_split = {"last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
             "last12months":"2020-07-01 00", "last9months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.training.sequence.context_len
        self.label_len = args.training.sequence.label_len
        self.outsample_size = args.training.sequence.pred_len

        DATA_DIR = os.path.join(root_path, 'caiso_20130101_20210630.csv')
        data = pd.read_csv(DATA_DIR)

        data['Date'] = data['Date'].astype('datetime64')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        ids = np.arange(len(names))
        df_all = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_all = df_all.merge(current_df, on='Date', how='outer')

        # set index
        df_all = df_all.set_index('Date')
        values = df_all.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in df_all.index.tolist()])

        self.ids = ids

        # NORMALIZATION
        self.scaler = StandardScaler()
        self.values = values
        self.dates = dates

        print(">>> values", np.shape(self.values))
        # print(self.ids)
        # (10, 74472)
        # [0 1 2 3 4 5 6 7 8 9]
        """
        from statsmodels.tsa.stattools import adfuller
        data = self.values.T
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        val_cut_date = set_split["last18months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last9months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.indices = left_indices
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

            # self.scaler.fit(self.extracted_values)
            # self.extracted_values = self.scaler.transform(self.extracted_values)

        else:
            # self.extracted_values = self.values[:, right_indices]
            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)
            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)
            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        
        self.values = self.extracted_values

        self.dim_datetime_feats = -1
        

    def __getitem__(self, index):

        while True:
            insample = np.zeros((self.insample_size, 1))
            outsample = np.zeros((self.outsample_size+self.label_len, 1))
            
            sampled_index = np.random.randint(np.shape(self.extracted_values)[0])
            
            sampled_timeseries = self.extracted_values[sampled_index]

            cut_point = np.random.randint(low=self.insample_size, high=len(sampled_timeseries)-self.outsample_size, size=1)[0]

            insample_window = sampled_timeseries[cut_point - self.insample_size:cut_point]
            insample = np.expand_dims(insample_window, 1)

            outsample_window = sampled_timeseries[(cut_point-self.label_len):(cut_point+self.outsample_size)]
            outsample[:len(outsample_window)] = np.expand_dims(outsample_window, 1)

            if np.max(insample_window) != np.min(insample_window):
                break

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1])

    def __len__(self):

        if self.flag == 'train':
            return 5000
        else:
            return int(len(self.extracted_values)*((np.min(self.lens)+np.max(self.lens))/2))

    def inverse_transform(self, data):
        
        raise Exception("does not support inverse_transform")


class Dataset_Production(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        root_path = args.datasets_dir

        set_split = {"last12months":"2020-01-01 00","last9months":"2020-04-01 00",
               "last6months":"2020-07-01 00", "last3months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.training.sequence.context_len
        self.label_len = args.training.sequence.label_len
        self.outsample_size = args.training.sequence.pred_len

        DATA_PATH = os.path.join(root_path, 'production.csv')
        data = pd.read_csv(DATA_PATH, parse_dates=['Time'])
        data = data.set_index('Time')
        ids = np.arange(data.shape[1])
        values = data.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in data.index.tolist()])

        self.ids = ids
        self.values = values
        self.dates = dates

        # cut_date = set_split["last12months"]
        # date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        print(">>> values", np.shape(self.values))
        """
        from statsmodels.tsa.stattools import adfuller
        data = self.values.T
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(np.shape(data))
        print(self.args.dataset_name)
        result = adfuller(data[:,-1])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        """
        val_cut_date = set_split["last9months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last3months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
       
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        self.scaler = StandardScaler()

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

        else:
            # self.extracted_values = self.values[:, right_indices]

            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)

            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)

            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        self.values = self.extracted_values

        self.dim_datetime_feats = -1
        

    def __getitem__(self, index):

        while True:
            sampled_index = np.random.randint(len(self.extracted_values))
            sampled_timeseries = self.extracted_values[sampled_index]

            cut_point = np.random.randint(low=self.insample_size, high=len(sampled_timeseries)-self.outsample_size, size=1)[0]

            insample_window = sampled_timeseries[cut_point - self.insample_size:cut_point]
            insample = np.expand_dims(insample_window, 1)
            
            outsample_window = sampled_timeseries[(cut_point-self.label_len):(cut_point + self.outsample_size)]
            outsample = np.expand_dims(outsample_window, 1)
            
            # =================================================================================
            # global time steps
            idx = sampled_index
            timestamps1 = np.asarray(list(range(cut_point - self.insample_size, cut_point)))
            timestamps2 = np.asarray(list(range((cut_point-self.label_len), (cut_point + self.outsample_size))))

            if np.max(insample_window) != np.min(insample_window):
                break

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1])

    def __len__(self):

        if self.flag == 'train':
            return 5000
        else:
            return int(len(self.extracted_values)*((np.min(self.lens)+np.max(self.lens))/2))

    def inverse_transform(self, data):
        
        raise Exception("does not support inverse_transform")


class Dataset_Caiso_M(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        root_path = args.datasets_dir

        set_split = {"last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
             "last12months":"2020-07-01 00", "last9months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.training.sequence.context_len
        self.label_len = args.training.sequence.label_len
        self.outsample_size = args.training.sequence.pred_len

        DATA_DIR = os.path.join(root_path, 'caiso_20130101_20210630.csv')
        data = pd.read_csv(DATA_DIR)

        data['Date'] = data['Date'].astype('datetime64')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        ids = np.arange(len(names))
        df_all = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_all = df_all.merge(current_df, on='Date', how='outer')

        # set index
        df_all = df_all.set_index('Date')
        values = df_all.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in df_all.index.tolist()])

        self.ids = ids

        # NORMALIZATION
        self.scaler = StandardScaler()
        self.values = values
        self.dates = dates

        print(">>> values", np.shape(self.values))
        # print(self.ids)
        # (10, 74472)
        # [0 1 2 3 4 5 6 7 8 9]

        val_cut_date = set_split["last18months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last9months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.indices = left_indices
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

            # self.scaler.fit(self.extracted_values)
            # self.extracted_values = self.scaler.transform(self.extracted_values)

        else:
            # self.extracted_values = self.values[:, right_indices]
            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)
            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)
            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]

        self.values = self.extracted_values.T

        # print(">>>>>>>>>>>>>>>>>>>>>>", self.flag, np.shape(self.values))

        self.dim_datetime_feats = -1

        self.set_len = np.shape(self.values)[0]-self.outsample_size-self.insample_size
        

    def __getitem__(self, index):

        sampled_timeseries = self.values

        cut_point = index

        insample = sampled_timeseries[cut_point:(cut_point+self.insample_size)]
        outsample = sampled_timeseries[(cut_point+self.insample_size-self.label_len):(cut_point+self.insample_size+self.outsample_size)]

        # print(index, cut_point, self.set_len, np.shape(insample), np.shape(outsample))

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(cut_point,(cut_point+self.insample_size))))
        timestamps2 = np.asarray(list(range((cut_point+self.insample_size-self.label_len),(cut_point+self.insample_size+self.outsample_size))))

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1])

    def __len__(self):

        return self.set_len

    def inverse_transform(self, data):
        
        raise Exception("does not support inverse_transform")


class Dataset_Production_M(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        root_path = args.datasets_dir

        set_split = {"last12months":"2020-01-01 00","last9months":"2020-04-01 00",
               "last6months":"2020-07-01 00", "last3months":"2020-10-01 00"}

        self.flag = flag
        self.args = args

        self.insample_size = args.training.sequence.context_len
        self.label_len = args.training.sequence.label_len
        self.outsample_size = args.training.sequence.pred_len

        DATA_PATH = os.path.join(root_path, 'production.csv')
        data = pd.read_csv(DATA_PATH, parse_dates=['Time'])
        data = data.set_index('Time')
        ids = np.arange(data.shape[1])
        values = data.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in data.index.tolist()])

        self.ids = ids
        self.values = values
        self.dates = dates

        # cut_date = set_split["last12months"]
        # date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        print(">>> values", np.shape(self.values))

        val_cut_date = set_split["last9months"]
        val_date = datetime.strptime(val_cut_date, '%Y-%m-%d %H')
        test_cut_date = set_split["last3months"]
        test_date = datetime.strptime(test_cut_date, '%Y-%m-%d %H')

        left_indices = []
        mid_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
       
            if record_date < val_date:
                left_indices.append(i)
            else:
                if record_date < test_date:
                    mid_indices.append(i)
                else:
                    right_indices.append(i)

        self.scaler = StandardScaler()

        print("train/val/test:{}/{}/{}".format(len(left_indices), len(mid_indices), len(right_indices)))

        if flag == 'train':
            self.extracted_values = self.values[:, left_indices]
            self.extracted_dates = self.dates[left_indices]

            for i in range(len(self.extracted_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.extracted_values[i]), axis=1)
                self.scaler.fit(temp)
                self.extracted_values[i] = self.scaler.transform(temp).squeeze(1)

        else:
            # self.extracted_values = self.values[:, right_indices]

            # self.scaler.fit(self.values[:, left_indices])
            # self.extracted_values = self.scaler.transform(self.extracted_values)

            self.train_values = self.values[:, left_indices]
            if flag == 'val':
                self.indices = mid_indices
                self.test_values = self.values[:, mid_indices]
            else:
                self.indices = right_indices
                self.test_values = self.values[:, right_indices]

            for i in range(len(self.train_values)):
                self.scaler = StandardScaler()
                temp = np.expand_dims(np.asarray(self.train_values[i]), axis=1)
                temp_test = np.expand_dims(np.asarray(self.test_values[i]), axis=1)
                self.scaler.fit(temp)
                self.test_values[i] = self.scaler.transform(temp_test).squeeze(1)

            self.extracted_values = self.test_values
            self.extracted_dates = self.dates[right_indices]

        self.lens = [len(self.extracted_values[i]) for i in range(np.shape(self.extracted_values)[0])]
        self.values = self.extracted_values.T

        # print(">>>>>>>>>>>>>>>>>>>>>>", self.flag, np.shape(self.values))

        self.dim_datetime_feats = -1

        self.set_len = np.shape(self.values)[0]-self.outsample_size-self.insample_size


    def __getitem__(self, index):

        sampled_timeseries = self.values

        cut_point = index

        insample = sampled_timeseries[cut_point:(cut_point+self.insample_size)]
        outsample = sampled_timeseries[(cut_point+self.insample_size-self.label_len):(cut_point+self.insample_size+self.outsample_size)]

        # print(index, cut_point, self.set_len, np.shape(insample), np.shape(outsample))

        # =================================================================================
        # global time steps
        idx = 0
        timestamps1 = np.asarray(list(range(cut_point,(cut_point+self.insample_size))))
        timestamps2 = np.asarray(list(range((cut_point+self.insample_size-self.label_len),(cut_point+self.insample_size+self.outsample_size))))

        return insample, outsample, np.zeros([np.shape(insample)[0], 1]), np.zeros([np.shape(outsample)[0], 1])

    def __len__(self):

        return self.set_len

    def inverse_transform(self, data):
        
        raise Exception("does not support inverse_transform")
        