import itertools
import os
import wfdb
from tqdm import tqdm
import pandas as pd
import json
import os
import collections
import numpy as np


ProjectPath = os.path.dirname(os.path.abspath(os.getcwd()))


def compare_arrays(arr1, arr2):
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
    (start_time, end_time) = timestamp

    # Read the original and output records
    original_rec = wfdb.rdrecord(original_filename, sampfrom=start_time, sampto=end_time)
    output_rec = wfdb.rdrecord(output_filename)

    # Read the original and output annotations
    original_ann = wfdb.rdann(original_filename, 'atr', sampfrom=start_time, sampto=end_time, shift_samps=True)
    output_ann = wfdb.rdann(output_filename, 'atr')

    # Compare the records
    assert original_rec.p_signal.all() == output_rec.p_signal.all(), "The records are not the same"
    assert original_rec.fs == output_rec.fs, "The sampling frequencies are not the same"
    assert original_rec.units == output_rec.units, "The units are not the same"
    # assert original_rec.sig_name == output_rec.sig_name, "The signal names are not the same"
    assert original_rec.n_sig == output_rec.n_sig, "The number of channels are not the same"
    assert original_rec.sig_len == output_rec.sig_len, "The signal lengths are not the same"

    # Compare the annotations
    assert original_ann.sample.all() == output_ann.sample.all(), "The annotations are not the same"
    assert original_ann.symbol == output_ann.symbol, "The annotation symbols are not the same"
    assert original_ann.subtype.all() == output_ann.subtype.all(), "The annotation subtypes are not the same"
    assert original_ann.chan.all() == output_ann.chan.all(), "The annotation channels are not the same"
    assert original_ann.num.all() == output_ann.num.all(), "The number of annotations are not the same"
    assert original_ann.aux_note == output_ann.aux_note, "The auxiliary notes are not the same"

    # is the following more efficient?
    # assert np.array_equal(original_ann.sample, output_ann.sample), "The annotations are not the same"
    # assert np.array_equal(original_ann.symbol, output_ann.symbol), "The annotation symbols are not the same"
    # assert np.array_equal(original_ann.subtype, output_ann.subtype), "The annotation subtypes are not the same"
    # assert np.array_equal(original_ann.chan, output_ann.chan), "The annotation channels are not the same"
    # assert np.array_equal(original_ann.num, output_ann.num), "The number of annotations are not the same"
    # assert np.array_equal(original_ann.aux_note, output_ann.aux_note), "The auxiliary notes are not the same"


def create_subset_directory(filename):
    subset_dir = '/home/liranc6/ecg/state-spaces/data/icentia11k-continuous-ecg_normal_sinus_subset'
    # Extract the patient ID and segment ID from the filename
    patient_id = int(filename.split('/')[-2][1:])

    # Create the directory path for the subset file
    write_dir = os.path.join(subset_dir, f'p{patient_id:05d}'[:3], f"p{patient_id:05d}")
    os.makedirs(write_dir, exist_ok=True)

    return write_dir


def create_new_subset_file(filename, timestamp):
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


def timestamps_of_normal_rhythms_in_all_segments(filename):
    # Read the annotations
    ann = wfdb.rdann(filename, 'atr')

    # Find the indices of '(N' and the corresponding ')'
    start_indices = [i for i, aux_note in enumerate(ann.aux_note) if aux_note == '(N']
    end_indices = [
        next(i for i, aux_note in enumerate(ann.aux_note[start_index:], start=start_index) if aux_note == ')')
        for start_index in start_indices
    ]

    # Find the timestamps of '(N' and the corresponding ')'
    start_timestamps = [ann.sample[i] for i in start_indices]
    end_timestamps = [ann.sample[i] for i in end_indices]

    # Zip the start and end timestamps together to create tuples
    timestamps = list(zip(start_timestamps, end_timestamps))

    return timestamps


def extract_sinus_rhythms_to_new_subset(data_dir, min_window_size):
    num_of_patients = 10
    iterator = itertools.product(range(0, num_of_patients+1), range(0, 50))
    num_of_new_files = 0
    for patient_id, segment_id in tqdm(iterator, total=num_of_patients * 50):
        print(f"patient_id: {patient_id}, segment_id: {segment_id}")
        filename = os.path.join(data_dir,
                                f'p{patient_id:05d}'[:3],
                                f'p{patient_id:05d}',
                                f'p{patient_id:05d}_s{segment_id:02d}')

        if not os.path.exists(f'{filename}.atr'):
            break
        else:
            timestamps = timestamps_of_normal_rhythms_in_all_segments(filename)
            for timestamp in timestamps:
                start_time, end_time = timestamp
                duration = end_time - start_time
                if duration >= min_window_size:  # Check if NSR is longer than 10 minutes (600 seconds)
                    num_of_new_files += 1
                    create_new_subset_file(filename, timestamp)

    print(f"Number of new files: {num_of_new_files}")


if __name__ == "__main__":
    raw_data_dir = "/home/liranc6/ecg/state-spaces/data/icentia11k-continuous-ecg"

    # creating subset of normal sinus rhythms (NSR) from the raw data

    min_window_size = 10*60*250  # minutes * seconds * sampling rate
    # the reason for min_window_size is that I hope to forecast 1-5 minutes ahead.
    # and I dont know if a smaller window will give me enough context data
    # on top of that, I think I have enough data so I can fiter out the shorter NSR.

    extract_sinus_rhythms_to_new_subset(raw_data_dir, min_window_size)

    # after creating the subset, with 10 first patients I have more than 10 hours of NSR data
    # divided to 62 files of at least 10 minutes each.
    # I know its small but I dont need more for now. when I will, I will add more patients. (I used 10/11000 patients)
    data_path = "/home/liranc6/ecg/state-spaces/data/icentia11k-continuous-ecg_normal_sinus_subset"

# at first I will try 9 minutes for sample and 1 minute for label.
# the idea is not to learn patient personal rythem but to learn the NSR rythem. afterwards, I hope to be able to
# adjust the model to the patient personal rythem. i.e. get good average results for all patients and then get good
# results for each patient.
