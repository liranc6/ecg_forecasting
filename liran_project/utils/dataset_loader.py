from pickle import FALSE
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import bisect
from collections import OrderedDict
import os
import sys
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/mrDiff/configs/config.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)



class SingleLeadECGDatasetCrops_SSSD(Dataset):
    def __init__(self, context_window_size, label_window_size, h5_filename, start_sample_from=0, data_with_RR=True, cache_size=5000, return_with_RR = False, start_patiant=0, end_patiant=-1):
    
        self.context_window_size = context_window_size
        self.label_window_size = label_window_size
        self.h5_file = h5py.File(h5_filename, 'r')
        self.group_keys = list(self.h5_file.keys())

        self.start_patiant = int(f'{start_patiant:05d}')
        self.end_patiant = int(f'{end_patiant:05d}')

        self.start_sample_from = start_sample_from
        
        if self.end_patiant == -1:
            self.end_patiant = int(self.group_keys[-1])

        assert self.start_patiant <= self.end_patiant, f"{self.start_patiant=} {self.end_patiant=}"
        
        self.keys = self.group_keys[self.start_patiant:self.end_patiant+1]
        datasets_sizes = []
        for key in self.group_keys:
            if int(key) < self.start_patiant:
                continue
            elif int(key) > self.end_patiant:
                break
            self.keys.append(key)
            print(f"{key=}")      
            item = self.h5_file[key]
            assert isinstance(item, h5py.Dataset)
            datasets_sizes.append(len(item))

        self.cumulative_sizes = np.cumsum(datasets_sizes)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.data_with_RR = data_with_RR
        self.return_with_RR = return_with_RR
        

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes.any() else 0

    @property
    def is_empty(self):
        return self.__len__() == 0

    def __getitem__(self, idx):
        if idx in self.cache.keys():
            x, y = self.cache[idx]
            if self.data_with_RR and not self.return_with_RR:
                # print("self.data_with_RR and not self.return_with_RR")
                x, y = x[0], y[0]
            else:
                pass
            return x, y


        # idx not in cache:

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)


        advance = self.start_sample_from #self.context_window_size + self.label_window_size
        # advance *= 2

        if dataset_idx == 0:
            start_idx = idx
        else:
            start_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        str_dataset_idx = str(self.keys[dataset_idx]) # Convert key to string

        end_idx = min(start_idx + self.cache_size, len(self.h5_file[str_dataset_idx])) 

        to_cache = self.h5_file[str_dataset_idx][start_idx: end_idx]

        if len(self.cache) + len(to_cache) >= self.cache_size:
            #clean the cache
            for i in range(len(to_cache)):
                self.cache.popitem(last=True)


        assert self.data_with_RR, f"{self.data_with_RR=}"
        if self.data_with_RR:
            # print("self.data_with_RR")
            for i in range(len(to_cache)):
                window = to_cache[i]
                signal_len = len(window[0])
                assert advance + self.context_window_size + self.label_window_size <= signal_len, f"{self.context_window_size=} + {self.label_window_size=} > {signal_len=}"
                x = window[:, advance: advance + self.context_window_size]
                y = window[:, advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)

        else:
            print("not self.data_with_RR")
            for i in range(len(to_cache)):
                window = to_cache[i]
                window = window[0]
                # print(f"{window.shape=}")
                assert self.context_window_size + self.label_window_size <= len(window), f"{self.context_window_size=} + {self.label_window_size=} > {len(window)=}"
                x = window[advance: advance + self.context_window_size]
                y = window[advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)

        x, y = self.cache[idx]
        if self.data_with_RR and not self.return_with_RR:
                x, y = x[0], y[0]
                # print("self.data_with_RR and not self.return_with_RR")
                # print(f"{x.shape=}")
                # print(f"{y.shape=}")
        else:
            pass
        return x, y


class SingleLeadECGDatasetCrops_mrDiff(Dataset):
    def __init__(self,context_window_size, label_window_size, h5_filename, start_sample_from=0, data_with_RR=True,
                cache_size=5000, return_with_RR = False, start_patiant=0, end_patiant=-1, normalize_method=None):

        self.context_window_size = context_window_size
        self.label_window_size = label_window_size
        self.h5_filename = h5_filename

        self.start_patiant = int(f'{start_patiant:05d}')
        self.end_patiant = int(f'{end_patiant:05d}')

        self.start_sample_from = start_sample_from
        
        with h5py.File(self.h5_filename, 'r') as h5_file:
            group_keys = list(h5_file.keys())
        
        if self.end_patiant == -1:
            self.end_patiant = int(group_keys[-1])

        assert self.start_patiant <= self.end_patiant, f"{self.start_patiant=} {self.end_patiant=}"
        
        self.keys = group_keys[self.start_patiant:self.end_patiant+1]
        datasets_sizes = []
        with h5py.File(self.h5_filename, 'r') as h5_file:
            for key in self.keys:
                item = h5_file[key]
                assert isinstance(item, h5py.Dataset)
                datasets_sizes.append(len(item))

        self.cumulative_sizes = np.cumsum(datasets_sizes)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.data_with_RR = data_with_RR
        self.return_with_RR = return_with_RR
        
        self.normalize_method = normalize_method
        self.norm_statistics = self._get_normalization_statistics(self.h5_filename)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes.any() else 0

    @property
    def is_empty(self):
        return self.__len__() == 0

    def __getitem__(self, idx):
        if idx in self.cache.keys():
            x, y = self.cache[idx]
            if self.data_with_RR and not self.return_with_RR:
                # print("self.data_with_RR and not self.return_with_RR")
                x, y = x[0], y[0]
            else:
                pass
            return x, y, 0, 0


        # idx not in cache:

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)


        advance = self.start_sample_from #self.context_window_size + self.label_window_size
        advance = int(advance)

        if dataset_idx == 0:
            start_idx = idx
        else:
            start_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        str_dataset_idx = str(self.keys[dataset_idx]) # Convert key to string
        
        with h5py.File(self.h5_filename, 'r') as h5_file:
            
            dataset = h5_file[str_dataset_idx]

            end_idx = min(start_idx + self.cache_size, len(dataset)) 

            to_cache = dataset[start_idx: end_idx]  # This is a list of windows, each window is a numpy array with shape (window_size) if no RR data,
                                                    # or (2, window_size) if there is RR data


        self._add_to_cache(to_cache, idx, advance)

        x, y = self.cache[idx]
        if self.data_with_RR and not self.return_with_RR:
                x, y = x[0], y[0]
                # print("self.data_with_RR and not self.return_with_RR")
                # print(f"{x.shape=}")
                # print(f"{y.shape=}")
        else:
            pass
        return x, y, 0, 0
    
    def _add_to_cache(self, to_cache, idx, advance):
        if len(self.cache) + len(to_cache) >= self.cache_size:
            #clean the cache
            for i in range(len(to_cache)):
                self.cache.popitem(last=True)

        if self.normalize_method:
            to_cache = normalized(to_cache, self.normalize_method, self.norm_statistics)
        
        if self.data_with_RR:
            for i in range(len(to_cache)):
                window = to_cache[i]
                signal_len = len(window[0])
                assert advance + self.context_window_size + self.label_window_size <= signal_len, f"{self.context_window_size=} + {self.label_window_size=} > {signal_len=}"
                x = window[:, advance: advance + self.context_window_size]
                y = window[:, advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)
        else:
            for i in range(len(to_cache)):
                window = to_cache[i]
                window = window[0]
                assert self.context_window_size + self.label_window_size <= len(window), f"{self.context_window_size=} + {self.label_window_size=} > {len(window)=}"
                x = window[advance: advance + self.context_window_size]
                y = window[advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)
            
    def _get_normalization_statistics(self, filename):
        mean = 0
        std = 0
        max_val = np.NINF
        min_val = np.Inf
        with h5py.File(filename, 'r') as h5_file:
            num_keys = len(self.keys)
            pbar_keys = tqdm(self.keys , total=num_keys)
            for i, key in enumerate(pbar_keys):
                pbar_keys.set_description(f"Reading {key}")
                if self.data_with_RR:
                    data = h5_file[key][()][:, 0, :]
                else:
                    data = h5_file[key][()]
                mean += np.mean(data, axis=0)
                std += np.std(data, axis=0)
                max_val = max(max_val, np.max(data))
                min_val = min(min_val, np.min(data))
                
                if i >10:
                    break
            
            scale = max_val - min_val + 1e-8
            mean /= num_keys
            std /= num_keys
            std = np.where(std == 0, 1, std)
            
        return {"max": max_val, "min": min_val, "scale": scale, "mean": mean, "std": std}
            
    def inverse_transform(self, data):
        
    	return de_normalized(data, self.normalize_method, self.norm_statistic)
    
    
def normalized(data, normalize_method, norm_statistics):
    if normalize_method == 'min_max':
        scale = norm_statistics['scale']
        data = (data - norm_statistics['min']) / scale
    elif normalize_method == 'z_score':
        mean = norm_statistics['mean']
        std = norm_statistics['std']
        data = (data - mean) / std
    return data

def de_normalized(data, normalize_method, norm_statistics):
    if normalize_method == 'min_max':
        scale = norm_statistics['scale']
        data = data * scale + norm_statistics['min']
    elif normalize_method == 'z_score':
        mean = norm_statistics['mean']
        std = norm_statistics['std']
        data = data * std + mean
    return data