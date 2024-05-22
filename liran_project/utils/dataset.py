from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import bisect



class SingleLeadECGDatasetCrops(Dataset):
    def __init__(self, context_window_size, label_window_size, h5_filename):
        self.context_window_size = context_window_size
        self.label_window_size = label_window_size
        self.h5_file = h5py.File(h5_filename, 'r')
        self.keys = sorted(self.h5_file.keys())
        self.group_keys = sorted(self.h5_file.keys())
        datasets_sizes = []
        for key in self.keys:
            item = self.h5_file[key]
            assert isinstance(item, h5py.Dataset)
            # print(f'{key=} : {len(item)=}')
            datasets_sizes.append(len(item))

        print(f'{datasets_sizes=}')
        self.cumulative_sizes = np.cumsum(datasets_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def is_empty(self):
        return self.__len__() == 0

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        key = str(self.keys[dataset_idx])  # Convert key to string
        window = self.h5_file[key][data_idx]
        x = window[:self.context_window_size]
        y = window[self.context_window_size:]
        assert self.context_window_size + self.label_window_size == len(
            window), "context_window_size+label_window_size != len(window)"
        return x, y