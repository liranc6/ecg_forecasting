from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import bisect
from collections import OrderedDict



class SingleLeadECGDatasetCrops(Dataset):
    def __init__(self, context_window_size, label_window_size, h5_filename, cache_size=5000):
        self.context_window_size = context_window_size
        self.label_window_size = label_window_size
        self.h5_file = h5py.File(h5_filename, 'r')
        self.keys = sorted(self.h5_file.keys())
        self.group_keys = sorted(self.h5_file.keys())
        datasets_sizes = []
        for key in self.keys:
            item = self.h5_file[key]
            assert isinstance(item, h5py.Dataset)
            datasets_sizes.append(len(item))

        self.cumulative_sizes = np.cumsum(datasets_sizes)
        self.cache = OrderedDict()
        self.cache_size = cache_size


    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes.any() else 0

    @property
    def is_empty(self):
        return self.__len__() == 0

    def __getitem__(self, idx):
        if idx in self.cache.keys():
            x, y = self.cache[idx]
            return x, y

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

        for i in range(len(to_cache)):
            window = to_cache[i]
            assert self.context_window_size + self.label_window_size <= len(window), "context_window_size+label_window_size > len(window)"
            x = window[advance: advance + self.context_window_size]
            y = window[advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
            self.cache[idx + i] = (x, y)

        x, y = self.cache[idx]
        return x, y
