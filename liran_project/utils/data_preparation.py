import itertools
import os
import wfdb
from tqdm import tqdm
import pandas as pd
import json
import os
import collections
import numpy as np
import glob
import h5py
import sys
from multiprocessing import Pool

server = "newton"
server_config_path = os.path.join("/home/liranc6/ecg_forecasting/liran_project/utils/server_config.json")

# set server configuration
with open(server_config_path) as f:
    server_config = json.load(f)
    server_config = server_config[server]
    project_path = server_config['project_path']
    my_data_dir = server_config['data_preprocess']['data_path']
    raw_data_dir = server_config['data_preprocess']['raw_data_dir']

sys.path.append(project_path)

from liran_project.utils.util import find_beat_indices


# ProjectPath = os.path.dirname(os.path.abspath(os.getcwd()))
# data_path = '/mnt/qnap/liranc6/data/'

def copy_and_count(args):
    input_h5_file, output_file, indices = args
    copy_datasets(input_h5_file, output_file, indices)
    # count_items(output_file)

def split_to_train_val_test(input_h5_file, output_directory, idx_start_val, idx_start_test, idx_end_test):
    # Paths to your source and destination files
    train_file = os.path.join(output_directory, "train", f'p0_to_p{idx_start_val-1}.h5')
    val_file = os.path.join(output_directory, "val", f'p{idx_start_val}_to_p{idx_start_test-1}.h5')
    test_file = os.path.join(output_directory, "test", f'p{idx_start_test}_to_p{idx_end_test}.h5')
    
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    # Indices for the datasets to be copied
    indices_1 = list(range(0, idx_start_val))    # 00000-00025
    indices_2 = list(range(idx_start_val-1, idx_start_test))   # 00026-00040
    indices_3 = list(range(idx_start_test, idx_end_test))   # 00040-00046

    tasks = [
        (input_h5_file, train_file, indices_1),
        (input_h5_file, val_file, indices_2),
        (input_h5_file, test_file, indices_3)
    ]

    with Pool() as pool:
        list(tqdm(pool.imap(copy_and_count, tasks), total=3, desc="Splitting datasets"))

    print("Datasets split into three HDF5 files successfully.")
    
# Function to copy datasets from source to destination HDF5 file
def copy_datasets(src_file, dest_file, dataset_indices):
    with h5py.File(src_file, 'r') as src, h5py.File(dest_file, 'w') as dest:
        for index in dataset_indices:
            dataset_name = f'/{index:05d}'
            if dataset_name in src:
                src.copy(dataset_name, dest)

def print_first_n_datasets_in_HDF5(hdf5_file, n=10):
    """
    Print the first n datasets in an HDF5 file.

    Parameters:
    - hdf5_file: The HDF5 file to read the data from.
    - n: The number of datasets to print.
    """
    with h5py.File(hdf5_file, 'r') as h5_file:
        datasets = []

        def visitor_func(name, node):
            if isinstance(node, h5py.Dataset):
                datasets.append(name)

        h5_file.visititems(visitor_func)
        for dataset in datasets[:n]:
            print(dataset)

def read_dataset_content(hdf5_file, dataset_name):
    """
    Read the content of a specific dataset from an HDF5 file.

    Parameters:
    - hdf5_file: The HDF5 file to read the data from.
    - dataset_name: The name of the dataset to read.
    """
    with h5py.File(hdf5_file, 'r') as h5_file:
        if dataset_name in h5_file:
            data = h5_file[dataset_name][:]
            return data
        else:
            print(f"Dataset {dataset_name} not found in the file.")
            return None

def compare_arrays(arr1, arr2):
    """
       Compare two arrays and return the differences.

       Parameters:
       - arr1: First array
       - arr2: Second array

       Returns:
       - 'equal' if arrays are identical, otherwise a DataFrame with differences.
       """
    arr1, arr2 = np.atleast_1d(arr1, arr2)

    # Find the differences
    diff_indices = np.where(arr1 != arr2)

    if diff_indices[0].size == 0:
        return 'equal'

    # Create a DataFrame to display the differences in a table
    df = pd.DataFrame({
        'Location': diff_indices[0],
        'arr1': arr1[diff_indices],
        'arr2': arr2[diff_indices]
    })

    # Return the DataFrame
    return df


def compare_files(original_filename, output_filename, timestamp):
    """
    Compare original and output ECG records and annotations.

    Parameters:
    - original_filename: Path to the original ECG record file.
    - output_filename: Path to the output ECG record file.
    - timestamp: Tuple representing the start and end time for comparison.
    """
    (start_time, end_time) = timestamp

    # Read the original and output records
    original_rec = wfdb.rdrecord(original_filename, sampfrom=start_time, sampto=end_time)
    output_rec = wfdb.rdrecord(output_filename)

    # Read the original and output annotations
    original_ann = wfdb.rdann(original_filename, 'atr', sampfrom=start_time, sampto=end_time, shift_samps=True)
    output_ann = wfdb.rdann(output_filename, 'atr')

    # Compare the records
    # assert np.array_equal(original_rec.p_signal, output_rec.p_signal), "The records are not the same"
    assert original_rec.fs == output_rec.fs, "The sampling frequencies are not the same"
    assert original_rec.units == output_rec.units, "The units are not the same"
    # assert original_rec.sig_name == output_rec.sig_name, "The signal names are not the same"
    assert original_rec.n_sig == output_rec.n_sig, "The number of channels are not the same"
    assert original_rec.sig_len == output_rec.sig_len, "The signal lengths are not the same"

    # Compare the annotations
    assert np.array_equal(original_ann.sample, output_ann.sample), "The annotations are not the same"
    assert original_ann.symbol == output_ann.symbol, "The annotation symbols are not the same"
    assert np.array_equal(original_ann.subtype, output_ann.subtype), "The annotation subtypes are not the same"
    assert np.array_equal(original_ann.chan, output_ann.chan), "The annotation channels are not the same"
    assert np.array_equal(original_ann.num, output_ann.num), "The number of annotations are not the same"
    assert original_ann.aux_note == output_ann.aux_note, "The auxiliary notes are not the same"


def create_subset_directory(filename):
    """
    Create a directory structure for storing subset files.

    Parameters:
    - filename: Path to the original ECG record file.

    Returns:
    - Directory path for the subset file.
    """
    subset_dir = os.path.join(my_data_dir,
                              'icentia11k-continuous-ecg_normal_sinus_subset'
                              )
    # Extract the patient ID and segment ID from the filename
    patient_id = int(filename.split('/')[-2][1:])

    # Create the directory path for the subset file
    write_dir = os.path.join(subset_dir, f'p{patient_id:05d}'[:3], f"p{patient_id:05d}")
    os.makedirs(write_dir, exist_ok=True)

    return write_dir


def create_new_subset_file(filename, timestamp):
    """
    Create a new subset file from a specified time range in the original file.

    Parameters:
    - filename: Path to the original ECG record file.
    - timestamp: Tuple representing the start and end time for the subset.

    Returns:
    - None
    """
    (start_time, end_time) = timestamp

    rec = wfdb.rdrecord(filename, sampfrom=start_time, sampto=end_time)
    ann = wfdb.rdann(filename, "atr", sampfrom=start_time, sampto=end_time, shift_samps=True)

    # Create a new directory to store the split files
    write_dir = create_subset_directory(filename)
    os.makedirs(write_dir, exist_ok=True)

    # Save the segment to a separate file
    output_filename = os.path.join(write_dir, f"{os.path.basename(filename)}_{start_time}_to_{end_time}")

    wfdb.wrsamp(record_name=os.path.basename(output_filename),
                write_dir=write_dir,
                fs=rec.fs,
                units=rec.units,
                sig_name=["ECG"],
                p_signal=rec.p_signal
                )

    # Write the annotations to a file
    wfdb.wrann(record_name=os.path.basename(output_filename),
               write_dir=write_dir,
               extension='atr',
               sample=ann.sample,
               symbol=ann.symbol,
               subtype=ann.subtype,
               chan=ann.chan,
               num=ann.num,
               aux_note=ann.aux_note
               )

    compare_files(filename, output_filename, timestamp)


def timestamps_of_rhythm_type_in_all_segments(filename, rhythm_type):
    """
    Find timestamps of a specific rhythm type in all segments of an ECG record.

    Parameters:
    - filename: Path to the ECG record file.
    - rhythm_type: Rhythm type to search for.

    Returns:
    - List of tuples representing start and end timestamps for the specified rhythm type.
    """
    r_type = ''
    if rhythm_type == "normal":
        r_type = '(N'
        # {"(N":0, "(AFIB":1, "(AFL":2, "(SVTA":3, "(VT":4 }

    # Read the annotations
    ann = wfdb.rdann(filename, 'atr')

    # Find the indices of '(N' and the corresponding ')'
    start_indices = [i for i, aux_note in enumerate(ann.aux_note) if aux_note == r_type]
    end_indices = [
        next(i for i, aux_note in enumerate(ann.aux_note[start_index:], start=start_index) if aux_note == ')')
        for start_index in start_indices
    ]

    # Find the timestamps of r_type and the corresponding ')'
    start_timestamps = [ann.sample[i] for i in start_indices]
    end_timestamps = [ann.sample[i] for i in end_indices]

    # Zip the start and end timestamps together to create tuples
    timestamps = list(zip(start_timestamps, end_timestamps))

    return timestamps


def extract_sinus_rhythms_to_new_subset(data_dir, min_window_size, start_patient_id=0, end_patient_id=10):
    """
        Iterate through patients and segments, extract NSR, and create new subset files.

        Parameters:
        - data_dir: Directory containing ECG data.
        - min_window_size: Minimum window size for NSR.

        Returns:
        - None
        """
    iterator = itertools.product(range(start_patient_id, end_patient_id), range(0, 50))
    num_of_new_files = 0
    for patient_id, segment_id in tqdm(iterator, desc='extract_sinus_rhythms_to_new_subset', total=(end_patient_id-start_patient_id+1) * 50):
        # print(f"patient_id: {patient_id}, segment_id: {segment_id}")
        filename = os.path.join(data_dir,
                                f'p{patient_id:05d}'[:3],
                                f'p{patient_id:05d}',
                                f'p{patient_id:05d}_s{segment_id:02d}')

        if not os.path.exists(f'{filename}.atr'):
            print(f"File {filename} not found.")
        else:
            timestamps = timestamps_of_rhythm_type_in_all_segments(filename, 'normal')
            for timestamp in timestamps:
                start_time, end_time = timestamp
                duration = end_time - start_time
                if duration >= min_window_size:  # Check if NSR is longer than 10 minutes (600 seconds)
                    num_of_new_files += 1
                    create_new_subset_file(filename, timestamp)

    print(f"Number of new files: {num_of_new_files}")


def extract_and_save_p_signal_to_HDF5(input_dir, output_file, with_R_beats=False):
    """
    Read ECG signals and save the p_signal data as NumPy arrays in an HDF5 file
    while preserving the directory hierarchy.

    Parameters:
    - input_dir: Directory containing ECG data.
    - output_file: The HDF5 file to save the p_signal data.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    num_files = sum([1 for _ in os.walk(input_dir)])
    # Open the HDF5 file in write mode
    with h5py.File(output_file, 'w') as h5_file:
        # Traverse the input directory to find data files
        for root, _, files in tqdm(os.walk(input_dir), desc="Processing files", total=num_files, unit="file"):
            for file in files:
                # Check if the file is a data file (ends with .dat)
                if file.endswith('.dat'):
                    record_name = os.path.splitext(file)[0]

                    # Get the relative path from the input directory
                    relative_path = os.path.relpath(root, input_dir)
                    dataset_name = os.path.join(relative_path, f"{record_name}_p_signal")

                    # Read the signal using wfdb
                    filename = os.path.join(root, record_name)
                    
                    signals, fields = wfdb.rdsamp(filename)

                    # Define beat types for annotation plotting
                    beat_types = ['N', 'Q', '+', 'V', 'S']

                    if with_R_beats:
                        # Read the annotations
                        ann = wfdb.rdann(filename, 'atr')
                        indices = [item for sublist in find_beat_indices(ann, beat_types).values() for item in sublist]
                        indices = np.array(indices) - 1
                        # create np array of the same size as the signal and fill it with zeros (default value), put 1 in the indices
                        # of the beats
                        beats = np.zeros(signals.shape[0])
                        beats[indices] = 1

                        # create np array with dims [2, len(signals)] to store the signal and the beats
                        data = np.vstack((signals[:, 0], beats))
                    else:
                        data = signals[:, 0]



                    # Save the p_signal data in the HDF5 file with the dataset name
                    h5_file.create_dataset(dataset_name, data=data)


def print_h5_hierarchy(file_path):
    """
    Print the directory hierarchy of an HDF5 file.

    Parameters:
    - file_path: The path to the HDF5 file.

    Returns:
    - None
    """
    with h5py.File(file_path, 'r') as h5_file:
        # Define a recursive function to print the hierarchy
        def print_group_hierarchy(group, indent=""):
            for name, item in group.items():
                if isinstance(item, h5py.Group):
                    print(f"{indent}Group: {name}")
                    print_group_hierarchy(item, indent + "  ")
                elif isinstance(item, h5py.Dataset):
                    print(f"{indent}Dataset: {item.name}")

        # Call the recursive function starting from the root group
        print_group_hierarchy(h5_file)


def split_and_save_data(input_h5_file, window_size, output_h5_file):
    """
    Split each dataset in the input HDF5 file into windows of the specified size
    and save the resulting windows into an output HDF5 file.

    :param input_h5_file: The input HDF5 file with datasets to split.
    :param window_size: The size of each window.
    :param output_h5_file: The output HDF5 file to save the split data.
    :return: None
    """
    def extract_integers(text):
        """
        Extract integers from the given text.

        :param text: The input text containing characters and integers.
        :return: A string containing only the integers found in the text.
        """
        return ''.join(filter(str.isdigit, str(text)))

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)

    #check if file is being written by another process
    import fcntl

    def is_file_locked(file_path):
        locked = None
        file_object = open(file_path, 'r')
        try:
            fcntl.flock(file_object, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = False
        except IOError:
            locked = True
        finally:
            file_object.close()
        return locked

    # Usage
    if is_file_locked(input_h5_file):
        print(f"The file {input_h5_file} is being written by another process.")
    else:
        print(f"The file {input_h5_file} is not locked.")

    print(f"{os.path.exists(input_h5_file)=}, {os.path.exists(output_h5_file)=}, its ok if output_h5_file doesnt exist yet")
    with h5py.File(input_h5_file, 'r') as input_file, h5py.File(output_h5_file, 'w') as output_file:
        total_leaf_iterations = 0
        for group_name, group_item in input_file.items():
            assert not isinstance(group_name, h5py.Group), "create only leaf groups"
            for subgroup_name, subgroup_item in tqdm(group_item.items(), desc="Processing Subgroups", unit="subgroup"):
                # print(f"subgroup_name: {subgroup_name}")
                assert not isinstance(subgroup_name, h5py.Group), "leaf groups"
                # dataset_data = []
                total_leaf_iterations += len(subgroup_item)

        progress_bar = tqdm(total=total_leaf_iterations, position=0, desc='Processing')
        for group_name, group_item in input_file.items():
            assert not isinstance(group_name, h5py.Group), "create only leaf groups"
            for subgroup_name, subgroup_item in group_item.items():
                # print(f"subgroup_name: {subgroup_name}")
                assert not isinstance(subgroup_name, h5py.Group), "leaf groups"
                dataset_data = []
                for dataset_name, dataset_item in subgroup_item.items():
                    # print(f"dataset_name: {dataset_name}")
                    assert not isinstance(dataset_name, h5py.Dataset)
                    # Split the dataset into windows
                    data = dataset_item[:] # data.shape = (2, len(signal))
                    num_windows = len(data[1]) // window_size
                    if num_windows > 1:
                        pass
                    
                    # window_data = np.array([data[:, i:i + window_size] for i in range(0, data.shape[1], window_size)])
                    window_data = np.array([data[:, i:i + window_size] for i in range(0, data.shape[1], window_size) if i + window_size <= data.shape[1]])
                    dataset_data.extend(window_data)
                    # Save each window as numpy array and add it to the output dataset
                    # for i in range(num_windows):
                    #     window_data = data[i * window_size: (i + 1) * window_size]
                        
                    #     dataset_data.append(window_data)

                    dataset_name = extract_integers(subgroup_name)
                    progress_bar.update(1)
                output_file.create_dataset(dataset_name, data=dataset_data)
    

def merge_datasets(input_h5_file, output_h5_file):
    """
    Merge all datasets in each group of the input HDF5 file into a single dataset,
    and save the resulting datasets into an output HDF5 file.

    :param input_h5_file: The input HDF5 file with datasets to merge.
    :param output_h5_file: The output HDF5 file to save the merged datasets.
    :return: None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)

    with (h5py.File(input_h5_file, 'r') as input_file, h5py.File(output_h5_file, 'w') as output_file):
        # Create the 'p00' group in the output file

        # Define a function to process groups and datasets
        def process_group(input_group):
            all_data = []
            for name, item in input_group.items():

                if isinstance(item, h5py.Dataset):
                    # Collect the dataset data into a list
                    all_data.append(item[:])

            # Save the concatenated data as a single dataset in the output file
            if all_data:
                output_dataset_name = input_group.name.split('/')[-1]
                output_dataset_name = extract_integers(output_dataset_name)
                output_group.create_dataset(output_dataset_name, data=all_data)

        def extract_integers(text):
            """
            Extract integers from the given text.

            :param text: The input text containing characters and integers.
            :return: A string containing only the integers found in the text.
            """
            return ''.join(filter(str.isdigit, str(text)))

        # Process each group in the 'p00' group of the input file
        for name, group in input_file.items():
            if isinstance(group, h5py.Group):
                new_name = extract_integers(name)
                output_group = output_file.create_group(new_name)
                for sub_name, sub_group in input_file[name].items():
                    if isinstance(sub_group, h5py.Group):
                        process_group(sub_group)


def count_items(file_path):
    def count_items_recursive(item):
        count = 0
        if isinstance(item, h5py.Group):
            group = item
            for name, item in group.items():
                print(f"Group: {name}")
                count += count_items_recursive(item)
        elif isinstance(item, h5py.Dataset):
            count += item.shape[0]
            print(f"Dataset: {item.name}, Count: {item.shape[0]}")
        return count

    total_count = 0
    with h5py.File(file_path, 'r') as f:
        for item_name, item in f.items():
            total_count += count_items_recursive(item)
    print(f"Total count: {total_count}")


def arrays_to_fixed_size_windows(input_h5_file, window_size, output_h5_file):
    base_name, extension = os.path.splitext(os.path.basename(output_h5_file))
    new_base_name = f"{base_name}_temp{extension}"
    temp_filename = os.path.join(os.path.dirname(output_h5_file), new_base_name)

    split_and_save_data(input_h5_file, window_size, temp_filename)
    merge_datasets(input_h5_file, output_h5_file)
    os.remove(temp_filename)

def extract_sinus_rhythms_to_p_signal_array(data_dir, output_file, min_window_size, start_patient_id=0, end_patient_id=10):
    """
        Iterate through patients and segments, extract NSR, and create new subset files.

        Parameters:
        - data_dir: Directory containing ECG data.
        - min_window_size: Minimum window size for NSR.

        Returns:
        - None
        """
    iterator = itertools.product(range(start_patient_id, end_patient_id), range(0, 50))
    num_of_new_files = 0
    # Define beat types for annotation plotting
    beat_types = ['N', 'Q', '+', 'V', 'S']
    with h5py.File(output_file, 'w') as h5_file:
        for patient_id, segment_id in tqdm(iterator, desc='extract_sinus_rhythms_to_new_subset', total=(end_patient_id-start_patient_id+1) * 50):
            add_to_dataset = np.array([])
            # print(f"patient_id: {patient_id}, segment_id: {segment_id}")
            filename = os.path.join(data_dir,
                                    f'p{patient_id:05d}'[:3],
                                    f'p{patient_id:05d}',
                                    f'p{patient_id:05d}_s{segment_id:02d}')

            if not os.path.exists(f'{filename}.atr'):
                pass
                # print(f"File {filename} not found.")
            else:
                dataset_name = f'/{patient_id:05d}'
                timestamps = timestamps_of_rhythm_type_in_all_segments(filename, 'normal')
                for timestamp in timestamps:
                    start_time, end_time = timestamp
                    duration = end_time - start_time
                    if duration >= min_window_size:  # Check if NSR is longer than 10 minutes (600 seconds)
                        num_of_new_files += 1
                        signals, _ = wfdb.rdsamp(filename, sampfrom=start_time, sampto=end_time)
                        ann = wfdb.rdann(filename, "atr", sampfrom=start_time, sampto=end_time, shift_samps=True)
                        indices = [item for sublist in find_beat_indices(ann, beat_types).values() for item in sublist]
                        indices = np.array(indices) - 1
                        # create np array of the same size as the signal and fill it with zeros (default value), put 1 in the indices
                        # of the beats
                        beats = np.zeros(signals.shape[0])
                        beats[indices] = 1
                        
                        # create np array with dims [2, len(signals)] to store the signal and the beats
                        data_sample = np.vstack((signals[:, 0], beats))
                        
                        # add to the patient dataset
                        add_to_dataset = np.concatenate((add_to_dataset, data_sample), axis=0)
                        
                        
            # check if dataset with the name of the patient_id already exists, if so, add the new data to the existing dataset, if not create a new dataset and add the data
            if dataset_name in h5_file:
                # add to the existing dataset
                h5_file[dataset_name].resize(h5_file[dataset_name].shape[0] + add_to_dataset.shape[0], axis=0)
                h5_file[dataset_name][-add_to_dataset.shape[0]:] = add_to_dataset
            else:
                # create a new dataset
                h5_file.create_dataset(dataset_name, data=add_to_dataset)

    print(f"Number of new files: {num_of_new_files}")

def split_signal_to_windows(signal, window_size):
    """
    Split a signal into windows of the specified size.

    Parameters:
    - signal: The input signal to split.
    - window_size: The size of each window.

    Returns:
    - windows: A list of windows.
    """
    windows = np.array([signal[:, i:i + window_size] for i in range(0, signal.shape[1]-window_size, window_size)])
    return windows
    

def process_segment(args):
    data_dir, patient_id, segment_id, min_window_size, beat_types = args
    add_to_dataset = np.array([])
    filename = os.path.join(data_dir,
                            f'p{patient_id:05d}'[:3],
                            f'p{patient_id:05d}',
                            f'p{patient_id:05d}_s{segment_id:02d}')

    if not os.path.exists(f'{filename}.atr'):
        # print(f"File {filename} not found.")
        return None, None
    else:
        dataset_name = f'/{patient_id:05d}'
        timestamps = timestamps_of_rhythm_type_in_all_segments(filename, 'normal')
        for timestamp in timestamps:
            start_time, end_time = timestamp
            duration = end_time - start_time
            if duration >= min_window_size:
                signals, _ = wfdb.rdsamp(filename, sampfrom=start_time, sampto=end_time)
                ann = wfdb.rdann(filename, "atr", sampfrom=start_time, sampto=end_time, shift_samps=True)
                indices = [item for sublist in find_beat_indices(ann, beat_types).values() for item in sublist]
                indices = np.array(indices) - 1
                beats = np.zeros(signals.shape[0])
                beats[indices] = 1
                data_sample = np.vstack((signals[:, 0], beats))
                windows = split_signal_to_windows(data_sample, min_window_size) # windows is a np.array of shape (n, 2, window_size)
                add_to_dataset = np.vstack((add_to_dataset, windows)) if add_to_dataset.size else windows
        return dataset_name, add_to_dataset

def parallel_extract_sinus_rhythms_to_p_signal_array(data_dir, output_file, min_window_size, start_patient_id=0, end_patient_id=10, val_perc = 0.1, test_perc = 0.1, TVT_output_dir=None):
    iterator = itertools.product(range(start_patient_id, end_patient_id+1), range(0, 150))
    beat_types = ['N', 'Q', '+', 'V', 'S']
    idx_end_test = end_patient_id
    idx_start_test = end_patient_id - int(test_perc * end_patient_id)
    idx_start_val = idx_start_test - int(val_perc * end_patient_id)
    
    # Paths to your source and destination files
    train_file = os.path.join(TVT_output_dir, "train", f'p0_to_p{idx_start_val-1}.h5')
    val_file = os.path.join(TVT_output_dir, "val", f'p{idx_start_val}_to_p{idx_start_test-1}.h5')
    test_file = os.path.join(TVT_output_dir, "test", f'p{idx_start_test}_to_p{idx_end_test}.h5')
    
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    

    with h5py.File(output_file, 'w') as h5_file, \
         h5py.File(train_file, 'w') as train_h5, \
         h5py.File(val_file, 'w') as val_h5, \
         h5py.File(test_file, 'w') as test_h5:
        with Pool() as pool:
            p_bar = tqdm(pool.imap(process_segment, [(data_dir, patient_id, segment_id, min_window_size, beat_types) for patient_id, segment_id in iterator]), 
                                desc='extract_sinus_rhythms_to_new_subset', total=(end_patient_id-start_patient_id) * 50)
            for dataset_name, add_to_dataset in p_bar:
                # if dataset_name is not None:
                #     tqdm.write(f"{dataset_name=}, {add_to_dataset.shape=}")
                if dataset_name and add_to_dataset.size > 0:
                    # if dataset_name in h5_file:
                    #     h5_file[dataset_name].resize(h5_file[dataset_name].shape[0] + add_to_dataset.shape[0], axis=0)
                    #     h5_file[dataset_name][-add_to_dataset.shape[0]:] = add_to_dataset
                    # else:
                    #     maxshape = (None,) + add_to_dataset.shape[1:]
                    #     h5_file.create_dataset(dataset_name, data=add_to_dataset, maxshape=maxshape, chunks=True)
                    
                    # Determine which file to write to based on patient_id
                    patient_id = int(dataset_name)  # int(dataset_name.split('_')[1])
                    if patient_id < idx_start_val:
                        target_h5 = train_h5
                    elif patient_id < idx_start_test:
                        target_h5 = val_h5
                    else:
                        target_h5 = test_h5

                    if dataset_name in target_h5:
                        target_h5[dataset_name].resize(target_h5[dataset_name].shape[0] + add_to_dataset.shape[0], axis=0)
                        target_h5[dataset_name][-add_to_dataset.shape[0]:] = add_to_dataset
                    else:
                        maxshape = (None,) + add_to_dataset.shape[1:]
                        target_h5.create_dataset(dataset_name, data=add_to_dataset, maxshape=maxshape, chunks=True)
                        
if __name__ == "__main__":
    # raw_data_dir = "/mnt/qnap/liranc6/data/icentia11k-continuous-ecg/static/published-projects/icentia11k-continuous-ecg/1.0"

    fs = 250 #sampling rate
    SECONDS_IN_MINUTE = 60
    # split the arrays to fixed size windows
    context_window_size = 10*SECONDS_IN_MINUTE*fs  # minutes * seconds * fs
    label_window_size = 5*SECONDS_IN_MINUTE*fs  # minutes * seconds * fs
    window_size = context_window_size+label_window_size
    total_num_minutes = window_size//fs//SECONDS_IN_MINUTE
    start_patient_id = 0
    end_patient_id = 500

    # creating subset of normal sinus rhythms (NSR) from the raw data
    min_window_size = window_size  # minutes * seconds * sampling rate
    # the reason for min_window_size is that I hope to forecast 1-5 minutes ahead.
    # and I dont know if a smaller window will give me enough context data
    # on top of that, I think I have enough data so I can fiter out the shorter NSR.'

    # extract_sinus_rhythms_to_new_subset(raw_data_dir, min_window_size, start_patient_id=46, end_patient_id=480)

    # after creating the subset, with 10 first patients I have more than 10 hours of NSR data
    # divided to 62 files of at least 10 minutes each.
    # I know its small but I dont need more for now. when I will, I will add more patients. (I used 10/11000 patients)
    # subset_data_dir = os.path.join(my_data_dir, 'icentia11k-continuous-ecg_normal_sinus_subset')

    # pSignal_npArray_data_dir_h5 = os.path.join(my_data_dir, "with_R_beats", 'icentia11k-continuous-ecg_normal_sinus_subset_npArrays.h5')

    # extract_and_save_p_signal_to_HDF5(subset_data_dir, pSignal_npArray_data_dir_h5, with_R_beats=True)

    split_pSignal_file = os.path.join(my_data_dir,
                                      "with_R_beats",
                                      'icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits',
                                      f'{total_num_minutes}_minutes_window_p{start_patient_id}_to_p{end_patient_id}.h5')
    val_perc = 0.1
    test_perc = 0.1
    
    parallel_extract_sinus_rhythms_to_p_signal_array(raw_data_dir,
                                                     split_pSignal_file,
                                                     min_window_size,
                                                     start_patient_id=start_patient_id,
                                                     end_patient_id=end_patient_id,
                                                     val_perc=val_perc,
                                                     test_perc=test_perc,
                                                     TVT_output_dir=os.path.join(my_data_dir,
                                                                    "with_R_beats",
                                                                    'icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits',
                                                                    f'{total_num_minutes}_minutes'
                                                                    ),
                                                                )

    # base_name, extension = os.path.splitext(os.path.basename(split_pSignal_file))
    # new_base_name = f"{base_name}_temp{extension}"
    # split_and_save_data(pSignal_npArray_data_dir_h5, window_size, split_pSignal_file)

    print("split_pSignal_file:")
    # print_first_n_datasets_in_HDF5(split_pSignal_file, 5)
    # print("temp_filename")
    # print_first_n_datasets_in_HDF5(temp_filename, 5)
    # merge_datasets(temp_filename, split_pSignal_file)
    # os.remove(temp_filename)
    # print_h5_hierarchy(split_pSignal_file)
    count_items(split_pSignal_file)

    idx_end_test = end_patient_id
    idx_start_test = end_patient_id - int(test_perc * end_patient_id)
    idx_start_val = idx_start_test - int(val_perc * end_patient_id)
    
    print("splitting to train, val, test")
    # split_to_train_val_test(input_h5_file=split_pSignal_file,
    #                         output_directory=os.path.join(my_data_dir,
    #                                                       "with_R_beats",
    #                                                       'icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits',
    #                                                       f'{total_num_minutes}_minutes'
    #                                                       ),
    #                         idx_start_val=idx_start_val,
    #                         idx_start_test=idx_start_test,
    #                         idx_end_test=idx_end_test
    #                         )

    print("file finished")


# at first I will try 9 minutes for sample and 1 minute for label.
# the idea is not to learn patient personal rythem but to learn the NSR rythem. afterwards, I hope to be able to
# adjust the model to the patient personal rythem. i.e. get good average results for all patients and then get good
# results for each patient.