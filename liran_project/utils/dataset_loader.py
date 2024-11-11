import yaml
import sys
import threading

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/mrDiff/configs/config.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.utils.common import *

# Event to signal the stopwatch to stop
stop_event = threading.Event()

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
                self.cache.popitem(last=False)


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
        if self.normalize_method != "None":
            assert self.normalize_method in ['min_max', 'z_score'], f"{self.normalize_method=}"
            self.norm_statistics = self._get_normalization_statistics(self.h5_filename)
        else:
            self.norm_statistics = None

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
                                                    
            to_cache = to_cache[:self.cache_size]  # Ensure that the cache size is not exceeded
            
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
                if len(self.cache) > 0:
                    self.cache.popitem(last=False)
                else:
                    break

        if self.data_with_RR:
            to_cache[:, 0, :] = normalized(to_cache[:, 0, :], self.normalize_method, self.norm_statistics)
            for i in range(len(to_cache)):
                window = to_cache[i]
                signal_len = len(window[0])
                assert advance + self.context_window_size + self.label_window_size <= signal_len, f"{self.context_window_size=} + {self.label_window_size=} > {signal_len=}"
                x = window[:, advance: advance + self.context_window_size]
                y = window[:, advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)
        else:
            to_cache = normalized(to_cache, self.normalize_method, self.norm_statistics)
            for i in range(len(to_cache)):
                window = to_cache[i]
                window = window[0]
                assert self.context_window_size + self.label_window_size <= len(window), f"{self.context_window_size=} + {self.label_window_size=} > {len(window)=}"
                x = window[advance: advance + self.context_window_size]
                y = window[advance + self.context_window_size : advance + self.context_window_size + self.label_window_size]
                self.cache[idx + i] = (x, y)
            
    def _get_normalization_statistics(self, filename):
        mean = 0
        total_num_samples = 0
        max_val = np.NINF
        min_val = np.Inf
        welford = WelfordOnline()

        # Create and start the stopwatch thread
        # stop_event.clear()  # Clear the event before starting the thread
        # stopwatch_thread = threading.Thread(target=stopwatch, args=(f"creating_stats",))
        # stopwatch_thread.start()
        
        # try:
        with h5py.File(filename, 'r') as h5_file:
            num_keys = len(self.keys)
            early_stop = 50
            pbar_keys = tqdm(self.keys, total=min(num_keys, early_stop))
            
            start_time = time.time()
            pbar_keys.set_description(f"creating_stats")
            for idx, key in enumerate(pbar_keys):
                if idx > early_stop:
                    break
                data = h5_file[key][()][:, 0, :] if self.data_with_RR else h5_file[key][()]
                    
                curr_num_samples = data.shape[0]
                welford.add_points(data)
                total_num_samples += curr_num_samples

                max_val = max(max_val, np.max(data))
                min_val = min(min_val, np.min(data))
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Update the postfix
                pbar_keys.set_postfix({"time_elapsed": str(timedelta(seconds=int(elapsed_time))),
                                        })
        # finally:
        #     # Signal the stopwatch to stop and wait for the thread to finish
        #     stop_event.set()
        #     stopwatch_thread.join()

        # Assert that total_num_samples is greater than zero
        assert total_num_samples > 0, "Total number of samples is zero. Check the data."
        
        mean, std = welford.get_mean_and_stddev()

        # Avoid zero std dev
        std = np.where(std == 0, 1, std)  # Log a warning if std is 1
        stats = {'mean': mean, 'std': std, 'max': max_val, 'min': min_val}
        return stats     
            
def normalized(data, normalize_method, norm_statistics):
    """
    Normalize the given data using the specified normalization method.

    Parameters:
    data (array-like): The data to be normalized. shape=(B, 1, L)
    normalize_method (str): The normalization method to use. 
                            Options are 'min_max' for Min-Max normalization 
                            and 'z_score' for Z-score normalization.
    norm_statistics (dict): A dictionary containing the necessary statistics 
                            for normalization. For 'min_max', it should contain 
                            'min' and 'max'. For 'z_score', it should contain 
                            'mean' and 'std'.

    Returns:
    array-like: The normalized data.
    """
    if normalize_method == 'min_max':
        scale = norm_statistics['max'] - norm_statistics['min']
        data = (data - norm_statistics['min']) / scale
    elif normalize_method == 'z_score':
        mean = norm_statistics['mean']
        std = norm_statistics['std']
        data = (data - mean) / std
    return data

def de_normalized(data, normalize_method, norm_statistics):
    """
    De-normalizes the given data based on the specified normalization method and statistics.

    Parameters:
    data (numpy.ndarray or similar): The normalized data to be de-normalized. shape=(B, 1, L)
    normalize_method (str): The normalization method used. Supported methods are 'min_max' and 'z_score'.
    norm_statistics (dict): The statistics used for normalization. For 'min_max', it should contain 'min' and 'max'.
                            For 'z_score', it should contain 'mean' and 'std'.

    Returns:
    numpy.ndarray or similar: The de-normalized data.
    """
    if normalize_method == 'min_max':
        scale = norm_statistics['max'] - norm_statistics['min']
        data = data * scale + norm_statistics['min']
    elif normalize_method == 'z_score':
        mean = norm_statistics['mean']
        std = norm_statistics['std']
        data = data * std + mean
    return data


class WelfordOnline:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.variance = None

    def add_points(self, data):
        # Convert data to a NumPy array for easy manipulation
        data = np.asarray(data)
        k = data.shape[0]  # Number of new data points

        if k == 0:
            return  # No points to add
        
        # Initialize mean and variance if this is the first addition
        if self.mean is None:
            self.mean = np.zeros(data.shape[1])  # Assuming data has columns
            self.variance = np.zeros(data.shape[1])  # Assuming data has columns

        # Update the count
        self.n += k
        
        # Calculate the mean increment along axis=0
        mean_increment = np.mean(data, axis=0)

        # Update the overall mean
        old_mean = self.mean.copy()
        self.mean += (mean_increment - self.mean) * (k / self.n)

        # Update the variance for each dimension
        for i in range(data.shape[1]):
            self.variance[i] += np.sum((data[:, i] - old_mean[i]) * (data[:, i] - self.mean[i]))

    def get_variance(self):
        if self.n < 2:
            return float('nan')  # Not enough data for variance
        return self.variance / (self.n - 1)  # Sample variance

    def get_population_variance(self):
        if self.n == 0:
            return float('nan')  # Not enough data
        return self.variance / self.n  # Population variance
    
    def get_stddev(self):
        return np.sqrt(self.get_variance())
    
    def get_population_stddev(self):
        return np.sqrt(self.get_population_variance())
    
    def get_mean(self):
        return self.mean
    
    def get_mean_and_stddev(self):
        return self.mean, self.get_population_stddev()

    def get_mean(self):
        return self.mean  
    
    