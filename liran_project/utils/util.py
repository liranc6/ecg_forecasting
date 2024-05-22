import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import wfdb
import pandas as pd


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

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
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
