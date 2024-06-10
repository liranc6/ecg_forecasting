import os
import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import wfdb
import pandas as pd
from fastdtw import fastdtw
import neurokit2 as nk
from tqdm import tqdm

SAMPLING_RATE = 250
MIN_WINDOW_SIZE_FOR_NK_ECG_PROCESS = 10*SAMPLING_RATE

def plot_tensor(input_tensor, smoothed_tensor):
    """
    other option: plt.plot(range(len(input_tensor)), input_tensor, marker='o')
    """
    plt.figure(figsize=(12, 6))
    plt.plot(input_tensor, label='Original Tensor', marker='o')
    plt.plot(smoothed_tensor, label='Smoothed Tensor', marker='o')
    plt.title('Smoothed Tensor Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def smooth_batch(input_tensor, kernel_size=3, sigma=1):
    """
    Function to apply Gaussian smoothing to a batch of 1D tensors.
    exmaple:
    smooth_batch(a_batch, kernel_size=2*smooth_to_each_side+1, sigma=smooth_to_each_side)
    """
    # Check if the kernel size is odd and greater than zero
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be an odd number and greater than zero.")
    
    # Create a Gaussian kernel
    kernel_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (kernel_range / sigma)**2)
    kernel /= kernel.sum()
    

    padded_tensor = F.pad(input_tensor, (kernel_size // 2, kernel_size // 2), mode='constant', value=0)

    padded_tensor = padded_tensor.double()
    kernel = kernel.double()
    smoothed_tensor = F.conv2d(padded_tensor.unsqueeze(0).unsqueeze(0), kernel.view(1, -1).unsqueeze(0).unsqueeze(0), padding=(0, kernel_size // 2))

    return smoothed_tensor.squeeze(0).squeeze(0)

def smooth_tensor(input_tensor, kernel_size=3, sigma=1):
    # Check if the kernel size is odd and greater than zero
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kernel size must be an odd number and greater than zero.")
    
    # Create a Gaussian kernel
    kernel_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (kernel_range / sigma)**2)
    kernel /= kernel.sum()

    # Pad input tensor with zeros - kernel_size // 2 at the beginning and end
    padded_tensor = F.pad(input_tensor, (kernel_size // 2, kernel_size // 2), mode='constant', value=0)

    # Perform convolution
    smoothed_tensor = F.conv1d(padded_tensor.view(1, 1, -1), kernel.view(1, 1, -1), padding=0).view(-1)
    
    return smoothed_tensor

def prune_to_same_length(a, b, min_distance=50, max_iter = 4):
    for i in range(len(a)):
        if len(a) > len(b):
            if abs(b[0] - a[0]) > abs(b[0] - a[1]):
                assert b[0] - a[0] > min_distance, f"{b[0] - a[0]=}"
                a = a[1:]
            
            elif abs(a[-1] - b[-1]) > abs(b[-1] - a[-2]):
                assert a[-1] - b[-1] > min_distance, f"{a[-1] - b[-1]=}"
                a = a[:-1] 
        elif len(b) > len(a):
            if abs(b[0] - a[0]) > abs(b[0] - a[1]):
                assert abs(b[0] - a[0]) > min_distance, f"{b[0] - a[0]=}"
                b = b[1:]
            
            elif abs(a[-1] - b[-1]) > abs(b[-1] - a[-2]):
                assert abs(a[-1] - b[-1]) > min_distance, f"{a[-1] - b[-1]=}"
                b = b[:-1]
        else:
            break

    return [a, b]

def align_batch(binary_batch, binary_batch_pred, smooth_to_each_side=50):
    # assert bateches has only 1s and 0s
    assert torch.all((binary_batch == 0) | (binary_batch == 1)).item(), "binary_batch must contain only 1s and 0s. Found other values."
    assert torch.all((binary_batch_pred == 0) | (binary_batch_pred == 1)).item(), "binary_batch_pred must contain only 1s and 0s. Found other values."


    binary_batch_smoothed = smooth_batch(binary_batch, kernel_size=2*smooth_to_each_side+1, sigma=smooth_to_each_side)

    binary_batch_pred_modified = modify_a_pred_batch(binary_batch_smoothed, binary_batch_pred)

    binary_batch_pred_modified_indices = [torch.nonzero(row == 1).squeeze(1) for row in binary_batch_pred_modified]
    binary_batch_indices = [torch.nonzero(row == 1).squeeze(1) for row in binary_batch]

    # Preallocate tensor
    # pairs = torch.empty((len(binary_batch_indices), 2), dtype=torch.int64)

    # # Fill tensor
    # for i, (a, b) in enumerate(zip(binary_batch_indices, binary_batch_pred_modified_indices)):
    #     pairs[i] = torch.tensor(prune_to_same_length(a, b))

    pairs = torch.tensor([list(prune_to_same_length(a, b)) for a, b in zip(binary_batch_indices, binary_batch_pred_modified_indices)])

    # print(f"{pairs.shape=}")


    # pairs = torch.tensor([list(prune_to_same_length(a, b)) for a, b in zip(binary_batch_indices, binary_batch_pred_modified_indices)])

    return pairs[:, 0], pairs[:, 1]


def modify_a_pred_batch(a_smoothed_batch, a_pred_batch):
    batch_size = a_smoothed_batch.shape[0]
    modified_a_pred_batch = torch.zeros_like(a_pred_batch)

    for b in range(batch_size):
        a_smoothed = a_smoothed_batch[b]
        a_pred = a_pred_batch[b]
        intervals = []
        new_indices = []
        in_interval = False
        start = 0

        # create intervals
        for i in range(len(a_smoothed)):
            if a_smoothed[i] > 0 and not in_interval:
                start = i
                in_interval = True
            elif a_smoothed[i] == 0 and in_interval:
                intervals.append((start, i))
                in_interval = False

        # if the last element is in an interval
        if in_interval:
            intervals.append((start, len(a_smoothed)))

        for start, end in intervals:
            ones_indices = (start + torch.nonzero(a_pred[start:end] == 1)).squeeze().tolist()

            if isinstance(ones_indices, int):
                # ones_indices = [ones_indices]
                new_indices.append(ones_indices)
                continue

            if len(ones_indices) > 1:
                max_index = ones_indices[a_smoothed[ones_indices].argmax()]
                new_indices.append(max_index)

        modified_a_pred_batch[b, new_indices] = 1

    return modified_a_pred_batch

def indices_to_binary_tensor(indices, size_like):
    binary_tensor = np.zeros_like(size_like)
    binary_tensor[indices] = 1

    return binary_tensor

def get_intervals_around_ones(indices, tensor_len, smooth_to_each_side=50):
    """
    Get the intervals around the ones in a binary tensor.
    """

    intervals = {}

    for idx in indices:
        idx = idx.item()
        left = max(0, idx - smooth_to_each_side)
        right = min(tensor_len, idx + smooth_to_each_side)
        intervals[idx] = (
            left,
            right
            )
        
    return intervals

def align_indices(longer_list_of_indices, shorter_list_of_indices, tensor_len, smooth_to_each_side=50):
    """
    
    """

    assert len(longer_list_of_indices) > len(shorter_list_of_indices), "The longer list of indices must have more elements than the shorter list."

    def find_closest_index(ones_indices, target_idx):
        if len(ones_indices) == 1:
            return ones_indices[0]

        closest_idx = ones_indices[0]
        for idx in ones_indices:
            if abs(idx - target_idx) < abs(closest_idx - target_idx):
                closest_idx = idx

        return closest_idx.item()
    
    new_indices = []

    tensor_2 = torch.from_numpy(indices_to_binary_tensor(shorter_list_of_indices, torch.zeros(tensor_len)))

    if longer_list_of_indices.shape[0] > shorter_list_of_indices.shape[0]:
        intervals = get_intervals_around_ones(shorter_list_of_indices, tensor_len, smooth_to_each_side)

        for idx, [start, end] in intervals.items():
            tmp = torch.where(tensor_2[start:end] == 1)
            len_tmp = len(tmp)
            if len_tmp == 0:
                continue
            if len_tmp == 1:
                tmp = [tmp[0].item() + start]
            elif len_tmp == 2:
                tmp = (tmp[1].item() + start).to_list()
            else:
                assert False, f"{len(tmp)=}, {tmp=}"

            batch_2_ones_indices_in_interval = find_closest_index(tmp, idx)

            new_indices.append(batch_2_ones_indices_in_interval)

    
    assert len(new_indices) == len(shorter_list_of_indices), f"{len(new_indices)=}, {len(shorter_list_of_indices)=}"
    return torch.tensor(new_indices)

        












def ecg_signal_difference(ecg_batch, ecg_pred_batch, sampling_rate):
    """
    Compute the difference between multiple ECG signals.
    I created a function to it so I can use it as a level of abstraction, if I want to change the way I calculate the difference between two signals, I can do it here.

    other feuatures we can add:
    _ , y_waves_info = nk.ecg_delineate(y, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate, method="dwt")
        _, y_pred_waves_info = nk.ecg_delineate(y_pred, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate, method="dwt")
    wave_features = [
            "ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets",
            "ECG_Q_Peaks", "ECG_S_Peaks",
            "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets",
            "ECG_R_Onsets", "ECG_R_Offsets"
            ]

    """

    ecg_signals_batch = ecg_batch[:, 0, :]
    ecg_R_beats_batch = ecg_batch[:, 1, :]

    assert len(ecg_signals_batch) == len(ecg_pred_batch) and len(ecg_signals_batch) > 0, "Input lists must have the same length and contain at least one element."


    dtw_dist = dtw_distance_batch(ecg_signals_batch.cpu().numpy(), ecg_pred_batch.cpu().numpy())

    diffs = {"dtw_dist": dtw_dist}
    
    if len(ecg_signals_batch[-1]) < MIN_WINDOW_SIZE_FOR_NK_ECG_PROCESS:
        return diffs
        
    # Assuming ecg_batch_pred is a torch tensor with shape (batch_size, channels, sequence_length)
    # and sampling_rate is defined elsewhere.

    # Convert ecg_batch_pred to numpy array
    ecg_pred_batch_numpy = ecg_pred_batch.cpu().numpy()

    # Initialize new_ecg_batch_pred tensor
    new_ecg_pred_batch = torch.zeros((ecg_pred_batch.shape[0], 2, ecg_pred_batch.shape[1]))

    # Iterate over each prediction in the batch
    for i, ecg_pred in enumerate(ecg_pred_batch_numpy):
        ecg_pred = ecg_pred.squeeze()  # Remove singleton dimensions
        new_ecg_pred_batch[i][0] = torch.from_numpy(ecg_pred)  # Store the ECG prediction in the tensor


        # Process the ECG prediction with NeuroKit
        _, info = nk.ecg_process(ecg_pred, sampling_rate=sampling_rate)
        ecg_pred_R_beats_indices = info['ECG_R_Peaks']

        # Convert R beat indices to binary tensor
        binary_ecg_pred_R_beats = indices_to_binary_tensor(ecg_pred_R_beats_indices, ecg_pred)

        # Store the binary R beats in the tensor
        new_ecg_pred_batch[i][1] = torch.from_numpy(binary_ecg_pred_R_beats)
        

    ecg_pred_R_beats_batch = new_ecg_pred_batch[:, 1, :]

    ecg_R_beats_batch_indices = [torch.nonzero(row == 1).squeeze(1) for row in ecg_R_beats_batch]
    ecg_pred_R_beats_batch_indices = [torch.nonzero(row == 1).squeeze(1) for row in ecg_pred_R_beats_batch]

    ecg_len = ecg_signals_batch.shape[1]

    mean_extra_r_beats = 0

    for i, (y, y_pred) in enumerate(zip(ecg_R_beats_batch_indices, ecg_pred_R_beats_batch_indices)):
        if y_pred.shape != y.shape:
            # align the pred indices to the batch indices

            mean_extra_r_beats += abs(y.shape[0] - y_pred.shape[0])
            
            a, b = prune_to_same_length(y, y_pred, min_distance=50)

            if a.shape[0] > b.shape[0]:
                a = align_indices(a, b, ecg_len, smooth_to_each_side=50)
                # b, a = align_batch(ecg_pred_R_beats_batch[i].unsqueeze(0), ecg_R_beats_batch[i].unsqueeze(0))
            elif b.shape[0] > a.shape[0]:
                b = align_indices(b, a, ecg_len, smooth_to_each_side=50)
                # a, b = align_batch(ecg_R_beats_batch[i].unsqueeze(0), ecg_pred_R_beats_batch[i].unsqueeze(0))
                
            assert a.shape == b.shape, f"{a.shape=}, {b.shape=}"
            ecg_R_beats_batch_indices[i], ecg_pred_R_beats_batch_indices[i] = a , b

            # print(f"{ecg_R_beats_batch_indices[i].shape}, {ecg_pred_R_beats_batch_indices[i].shape}")
    
    mean_extra_r_beats /= len(ecg_R_beats_batch_indices)


    for i, (y, y_pred) in enumerate(zip(ecg_R_beats_batch_indices, ecg_pred_R_beats_batch_indices)):
        y, y_pred = indices_to_binary_tensor(y, torch.zeros(ecg_len)), indices_to_binary_tensor(y_pred, torch.zeros(ecg_len))
        ecg_R_beats_batch[i] = torch.from_numpy(y)
        ecg_pred_R_beats_batch[i] = torch.from_numpy(y_pred)

    

    mse_total = mse_distance_batch(ecg_R_beats_batch, ecg_pred_R_beats_batch)
    mae_total = mae_distance_batch(ecg_R_beats_batch, ecg_pred_R_beats_batch)

    diffs.update({"mse_total": mse_total, "mae_total": mae_total, "mean_extra_r_beats": mean_extra_r_beats})

    return diffs

def dtw_distance_batch(y_list, y_pred_list):
    """
    Dynamic Time Warping (DTW)
    DTW Distance is good for measuring ECG forecast accuracy because it can capture similarities between
    time-series signals with varying lengths or temporal distortions, which are common in ECG data due to irregular heartbeats or noise.
    """
    assert len(y_list) == len(y_pred_list) and len(y_list) > 0, "Input lists must have the same length and contain at least one element."
    assert isinstance(y_list, np.ndarray) and isinstance(y_pred_list, np.ndarray), "Both inputs must be numpy arrays."

    total_error = 0
    for y, y_pred in zip(y_list, y_pred_list):
        #to cpu because fastdtw does not support cuda tensors
        dtw_distance, _ = fastdtw(y.flatten(), y_pred.flatten())
        # dtw_distance = dtw.distance(a.cpu().numpy(), b.cpu().numpy())  # more accurate but much slower

        total_error += dtw_distance / len(y)

    return total_error / len(y_list)

def mse_distance_batch(y_batch, y_pred_batch):
    
    total_error = F.mse_loss(y_batch.float(), y_pred_batch.float()).item()

    return total_error
    # assert len(y_list) == len(y_pred_list) and len(y_list) > 0, "Input lists must have the same length and contain at least one element."
    # # assert isinstance(y_list, list) and isinstance(y_pred_list, list), "Both inputs must be lists."



    # total_error = 0
    # for y, y_pred in zip(y_list, y_pred_list):
    #     assert y.shape == y_pred.shape, "Each pair of tensors must have the same shape."
    #     assert isinstance(y, torch.Tensor) and isinstance(y_pred, torch.Tensor), "Both inputs in each pair must be tensors."

    #     total_error += F.mse_loss(y.float(), y_pred.float()).item()

    # return total_error / len(y_list)

def mae_distance_batch(y_list, y_pred_list):

    total_error = F.l1_loss(y_list.float(), y_pred_list.float()).item()
    return total_error
    # assert len(y_list) == len(y_pred_list) and len(y_list) > 0, "Input lists must have the same length and contain at least one element."
    # # assert isinstance(y_list, list) and isinstance(y_pred_list, list), "Both inputs must be lists."

    # total_error = 0
    # for y, y_pred in zip(y_list, y_pred_list):
    #     assert y.shape == y_pred.shape, "Each pair of tensors must have the same shape."
    #     assert isinstance(y, torch.Tensor) and isinstance(y_pred, torch.Tensor), "Both inputs in each pair must be tensors."

    #     total_error += torch.mean(torch.abs(y_pred.float() - y.float())).item()

    # return total_error / len(y_list)

def modify_z_and_omega(net, model_name, checkpoint, device):
    if model_name == "SSSDS4":
        net.residual_layer.residual_blocks[35].S42.s4_layer.kernel.kernel.z = checkpoint['model_state_dict']['residual_layer.residual_blocks.35.S42.s4_layer.kernel.kernel.z']
    elif model_name == "SSSDSA":
        for i in range(len(net.d_layers)):
            if f'd_layers.{i}.layer.kernel.kernel.z' in checkpoint['model_state_dict']:
                net.d_layers[i].layer.kernel.kernel.z = checkpoint['model_state_dict'][f'd_layers.{i}.layer.kernel.kernel.z'].to(device)
                net.d_layers[i].layer.kernel.kernel.omega = checkpoint['model_state_dict'][f'd_layers.{i}.layer.kernel.kernel.omega'].to(device)
        for i in range(len(net.c_layers)):
            if f'c_layers.{i}.layer.kernel.kernel.z' in checkpoint['model_state_dict']:
                net.c_layers[i].layer.kernel.kernel.z = checkpoint['model_state_dict'][f'c_layers.{i}.layer.kernel.kernel.z'].to(device)
                net.c_layers[i].layer.kernel.kernel.omega = checkpoint['model_state_dict'][f'c_layers.{i}.layer.kernel.kernel.omega'].to(device)
        for i in range(len(net.u_layers)):
            if f'u_layers.{i}.layer.kernel.kernel.z' in checkpoint['model_state_dict']:
                net.u_layers[i].layer.kernel.kernel.z = checkpoint['model_state_dict'][f'u_layers.{i}.layer.kernel.kernel.z'].to(device)
                net.u_layers[i].layer.kernel.kernel.omega = checkpoint['model_state_dict'][f'u_layers.{i}.layer.kernel.kernel.omega'].to(device)
            for j in range(len(net.u_layers[i])):
                if f'u_layers.{i}.{j}.layer.kernel.kernel.z' in checkpoint['model_state_dict']:
                    net.u_layers[i][j].layer.kernel.kernel.z = checkpoint['model_state_dict'][f'u_layers.{i}.{j}.layer.kernel.kernel.z'].to(device)
                    net.u_layers[i][j].layer.kernel.kernel.omega = checkpoint['model_state_dict'][f'u_layers.{i}.{j}.layer.kernel.kernel.omega'].to(device)

def find_beat_indices(ann, beat_types):
    """
    Find the indices of specified beat types in the annotations.
    
    Parameters:
    ann (wfdb.Annotation): The annotation object containing beat annotations.
    beat_types (list): List of beat types to find indices for (e.g., ['N', 'Q', '+', 'V', 'S']).
    
    Returns:
    dict: A dictionary with beat types as keys and lists of indices as values.
    """
    beat_indices = {beat: [] for beat in beat_types}
    for i, symbol in enumerate(ann.symbol):
        if symbol in beat_indices:
            beat_indices[symbol].append(ann.sample[i])
    return beat_indices

def print_signal(patient_id, segment_id, begins_at, ends_at, start, length, data_path):
    """
    Print and plot the ECG signal and annotations for a specified patient and segment.
    
    Parameters:
    patient_id (int): ID of the patient.
    segment_id (int): ID of the segment.
    begins_at (int): The starting sample index of the segment.
    ends_at (int): The ending sample index of the segment.
    start (int): The starting sample index for plotting.
    length (int): The length of the signal to plot (in samples).
    data_path (str): The base path to the data files.
    
    Returns:
    None
    """
    filename = os.path.join(data_path, f'p{patient_id:05d}'[:3], f'p{patient_id:05d}', 
                            f'p{patient_id:05d}_s{segment_id:02d}_{begins_at}_to_{ends_at}')
    
    # Read the ECG record and annotations
    rec = wfdb.rdrecord(filename, sampfrom=start, sampto=start + length)
    ann = wfdb.rdann(filename, "atr", sampfrom=start, sampto=start + length, shift_samps=True)
    
    # Create a time array in seconds
    time = np.arange(len(rec.p_signal)) / rec.fs
    
    # Create a new figure and two subplots, sharing both x axes
    fig, ax1 = plt.subplots(figsize=(15, 4))
    ax2 = ax1.twiny()
    
    # Plot the ECG signal on the first x-axis
    ax1.plot(time, rec.p_signal, label='ECG signal')
    # Label the y-axis
    ax1.set_ylabel('Amplitude')
    
    # Define beat types for annotation plotting
    beat_types = ['N', 'Q', '+', 'V', 'S']
    
    # Plot the annotations on the first x-axis
    for i in range(len(ann.sample)):
        if ann.symbol[i] in beat_types:
            ax1.plot(ann.sample[i] / rec.fs, rec.p_signal[ann.sample[i]], 'ro')
    
    # Set the limits of the primary x-axis
    ax1.set_xlim([0, time[-1]])
    
    # Set the limits of the secondary x-axis to match the primary x-axis
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(np.linspace(start, start + length, len(ax1.get_xticks())).astype(int))
    
    # Label the axes
    ax1.set_xlabel('Time (s)')
    ax2.set_xlabel('Sample Index')
    plt.title('ECG Signal')
    plt.show()
    
    print("Beat symbols locations:")
    indices = find_beat_indices(ann, beat_types)
    for beat_type, idx_list in indices.items():
        print(f"Beat type '{beat_type}': Indices {idx_list}")
    
    # Print the counts of beat symbols and auxiliary notes
    print("Beat symbols count:")
    print(pd.Series(ann.symbol).value_counts())
    
    print("Auxiliary notes count:")
    print(pd.Series(ann.aux_note).value_counts())
    
    

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    pbar = tqdm(range(T - 1, -1, -1), desc='Sampling', total=T ,leave=False, position=0)

    with torch.no_grad():
        for t in pbar:
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask
