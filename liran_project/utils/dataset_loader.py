from pickle import FALSE
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import bisect
from collections import OrderedDict



class SingleLeadECGDatasetCrops(Dataset):
    def __init__(self, context_window_size, label_window_size, h5_filename, data_with_RR=True, cache_size=5000, return_with_RR = False, start_patiant=0, end_patiant=-1):
               
        self.context_window_size = context_window_size
        self.label_window_size = label_window_size
        self.h5_file = h5py.File(h5_filename, 'r')
        self.group_keys = list(self.h5_file.keys())

        self.start_patiant = int(f'{start_patiant:05d}')
        self.end_patiant = int(f'{end_patiant:05d}')
        
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


        advance = 0 #self.context_window_size + self.label_window_size
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
                assert self.context_window_size + self.label_window_size <= signal_len, f"{self.context_window_size=} + {self.label_window_size=} > {signal_len=}"
                x = window[:, advance: advance + self.context_window_size]
                y = window[:, advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)

        else:
            print("not self.data_with_RR")
            for i in range(len(to_cache)):
                window = to_cache[i]
                window = window[0]
                print(f"{window.shape=}")
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
   