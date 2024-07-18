import os
import numpy as np
import torch
import random
from tqdm.notebook import tqdm
import re
import pandas as pd


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_epoch(path, num=-1, critirion='max'):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """
    assert os.path.exists(path), "path does not exist"
    assert os.path.isdir(path), "path is not a directory"
    assert num < 0, "num starts from end, end is -1"

    #print absulute path
    # print('abs path:', os.path.abspath(path))
    # assert os.path.abspath(path) == "/home/liranc6/ecg/ecg_forecasting/liran_project/results/mujoco/90/T200_beta00.0001_betaT0.02", "path is not valid"
    files = os.listdir(path)
    epoch = -1
    file_i = 1

    for filename in files[::-1]:
        if not os.path.isfile(os.path.join(path, filename)):
            continue
        assert len(filename) > 4
        if critirion == 'max' and filename[-4:] == '.pkl' \
            or critirion == 'best' and "best" in filename:
                
                parts = re.split('_|\.', filename)
                epoch = int(parts[-2])  # Convert the string to an integer
                if num+file_i==0:
                    return epoch
                file_i += 1

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

    
    The function first extracts the diffusion hyperparameters and checks that they have the correct lengths. It then initializes `x` with a tensor of standard Gaussian noise.

    The function then enters a loop that runs in reverse order from `T-1` to `0`. In each iteration of the loop, the function performs a step of the diffusion process.
    If the `only_generate_missing` flag is set, the function first fills in the missing data in `x` with the corresponding data from `cond`.
    The function then calculates the diffusion step and uses the model to predict the next state of `x`. The state of `x` is then updated based on the predicted
    state and the diffusion hyperparameters. If `t` is greater than 0, the function also adds a variance term to `x`.

    Finally, the function returns the generated tensor `x`.

    The `std_normal` function is a helper function that generates a tensor of standard Gaussian noise of a given size.
    This noise is used to initialize `x` and to add the variance term to `x` in each step of the diffusion process.
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
        for t in tqdm(range(T - 1, -1, -1)):
            if only_generate_missing == 1:
                # Blend x and cond based on mask, keeping x values where mask is 0 and cond values where mask is 1
                x = x * (1 - mask).float() + cond * mask.float()
            # Set the current diffusion step
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            # Predict epsilon_theta according to the current state
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict epsilon_theta according to epsilon_theta
            # update x_{t-1} to mu_theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                # If not the last step, add a noise term to x
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x

def t_uniform_sampling(T, B, sampling_params):
    """
    Generate random indices for uniform sampling of time steps.

    Args:
        T (int): Total number of time steps.
        B (int): Batch size.
        sampling_params (dict): Additional sampling parameters.

    Returns:
        torch.Tensor: Random indices for uniform sampling.

    """
    return torch.randint(T, size=(B, 1, 1)).cuda()

def t_importance_sampling(T, B, sampling_params):
    """
    This method samples t with probabilities proportional to the difficulty or importance of denoising at that step. 
    Performs importance sampling by sampling B indices based on the loss_per_step.

    Args:
        T (int): The total number of steps.
        loss_per_step (torch.Tensor): The loss per step.
        B (int): The number of samples to be drawn.

    Returns:
        torch.Tensor: A tensor of shape (B, 1, 1) containing the sampled indices.
    """
    loss_per_step = sampling_params.get('loss_per_step', None)
    assert loss_per_step is not None, "loss_per_step is required for importance sampling"
    probabilities = loss_per_step / loss_per_step.sum()
    return torch.multinomial(probabilities, num_samples=B, replacement=True).view(B, 1, 1).cuda()

def t_curriculum_learning(T, B, sampling_params):
    """
    Start with easier (later) time steps and gradually introduce earlier (harder) time steps as training progresses.
    Easier steps are the earlier time steps (smaller t), and the harder steps are the later time steps (larger t).

    Args:
        T (int): The maximum value for the random tensor.
        current_epoch (int): The current epoch of the training.
        total_epochs (int): The total number of epochs for the training.
        B (int): The batch size.

    Returns:
        torch.Tensor: A random tensor of shape (B, 1, 1) with values between start_t and T.
    """
    current_epoch = sampling_params.get('current_epoch', None)
    total_epochs = sampling_params.get('total_epochs', None)
    assert current_epoch is not None, "current_epoch is required for curriculum learning"
    assert total_epochs is not None, "total_epochs is required for curriculum learning"
    start_t = int((current_epoch / total_epochs) * T)
    return torch.randint(low=start_t, high=T, size=(B, 1, 1)).cuda()

def t_prioritized_sampling(T, B, sampling_params):
    """
    Perform prioritized sampling based on the loss values per step.

    Args:
        T (int): The total number of steps.
        loss_per_step (torch.Tensor): The loss values per step.
        B (int): The number of samples to be selected.

    Returns:
        torch.Tensor: The selected samples.

    """
    assert sampling_params is not None, "sampling_params is required for prioritized sampling"
    medians_loss_dict = sampling_params.get('medians_loss_dict', None)
    assert medians_loss_dict is not None, "loss_dict is required for prioritized sampling"
    t_samples = random.choices(list(medians_loss_dict.keys()), weights=list(medians_loss_dict.values()), k=B) #can also use: torch.multinomial
    t_samples = torch.tensor(t_samples).view(B, 1, 1).cuda()
    return t_samples
 
def t_adaptive_sampling(T, B, sampling_params):
    """
    Dynamically adjust the sampling strategy based on the model's current performance on different time steps..

    Args:
        T (int): The total number of samples.
        model_performance (torch.Tensor): The performance of the model for each sample.
        B (int): The number of samples to be selected.

    Returns:
        torch.Tensor: The selected samples.

    """
    model_performance = sampling_params.get('model_performance', None)
    assert model_performance is not None, "model_performance is required for adaptive sampling"
    difficulties = 1 / (model_performance + 1e-6)  # Avoid division by zero
    probabilities = difficulties / difficulties.sum()
    return torch.multinomial(probabilities, num_samples=B, replacement=True).view(B, 1, 1).cuda()


def t_scheduled_sampling(T, B, sampling_params):
    """
    Use a predetermined schedule to sample t values,
    potentially focusing on different ranges of t at different stages of training.

    Args:
        T (int): The upper bound for random integer generation.
        current_epoch (int): The current epoch.
        schedule (list): A list of (epoch_range, t_range) tuples defining the schedule.
        B (int): The batch size.

    Returns:
        torch.Tensor: A tensor of random integers.

    """
    current_epoch = sampling_params.get('current_epoch', None)
    schedule = sampling_params.get('schedule', None)
    assert current_epoch is not None, "current_epoch is required for scheduled sampling"
    assert schedule is not None, "schedule is required for scheduled sampling"

    for epoch_range, t_range in schedule:
        if epoch_range[0] <= current_epoch <= epoch_range[1]:
            return torch.randint(low=t_range[0], high=t_range[1], size=(B, 1, 1)).cuda()
    return torch.randint(T, size=(B, 1, 1)).cuda()

def t_gaussian_sampling(T, B, sampling_params):
    """
    Sample from a Gaussian distribution and clip the values to be within [0, T).

    Parameters:
    - T (int): The upper bound for the sampled values.
    - mean (float): The mean of the Gaussian distribution (default: 0).
    - std (float): The standard deviation of the Gaussian distribution (default: 1).
    - B (int): The batch size (default: 1).

    Returns:
    - samples (torch.Tensor): The sampled values, clamped to be within [0, T).
    """
    mean = sampling_params.get('mean', 0)
    std = sampling_params.get('std', 1)
    samples = torch.normal(mean, std, size=(B,)).clamp(0, T-1).long().view(B, 1, 1).cuda()
    return samples

def t_beta_sampling(T, B, sampling_params):
    """
    Sample from a Beta distribution and scale the samples to the range [0, T).

    Parameters:
    - T (int): The upper bound of the range.
    - alpha (float): The alpha parameter of the Beta distribution. Default is 1.
    - beta (float): The beta parameter of the Beta distribution. Default is 1.
    - B (int): The number of samples to generate. Default is 1.

    Returns:
    - samples (torch.Tensor): The generated samples, scaled to the range [0, T).
    """
    alpha = sampling_params.get('alpha', 1)
    beta = sampling_params.get('beta', 1)
    samples = torch.distributions.Beta(alpha, beta).sample((B,)) * (T-1)
    samples = samples.clamp(0, T-1).long().view(B, 1, 1).cuda()
    return samples

def t_half_cauchy_sampling(T, B, sampling_params):
    """
    Sample from a Half-Cauchy distribution.

    Args:
        T (int): The upper bound for the samples.
        scale (float, optional): The scale parameter of the Half-Cauchy distribution. Defaults to 1.
        B (int, optional): The number of samples to generate. Defaults to 1.

    Returns:
        torch.Tensor: The generated samples from the Half-Cauchy distribution.
    """
    scale = sampling_params.get('scale', 1)
    samples = torch.distributions.HalfCauchy(scale).sample((B,))
    samples = samples.clamp(0, T-1).long().view(B, 1, 1)
    if torch.cuda.is_available():
        samples = samples.cuda()
    return samples

def t_sampling_strategy(sampling_strategy, T, B, sampling_params):
    # Sample diffusion steps based on the chosen strategy
    if sampling_strategy == "uniform":
        diffusion_steps = t_uniform_sampling(T, B, sampling_params)
    elif sampling_strategy == "importance":
        diffusion_steps = t_importance_sampling(T, B, sampling_params)
    elif sampling_strategy == "curriculum":
        diffusion_steps = t_curriculum_learning(T, B, sampling_params)
    elif sampling_strategy == "prioritized":
        diffusion_steps = t_prioritized_sampling(T, B, sampling_params)
    elif sampling_strategy == "scheduled":
        diffusion_steps = t_scheduled_sampling(T, B, sampling_params)
    elif sampling_strategy == "adaptive":
        diffusion_steps = t_adaptive_sampling(T, B, sampling_params)
    elif sampling_strategy == "gaussian":
        diffusion_steps = t_gaussian_sampling(T, B, sampling_params)
    elif sampling_strategy == "beta":
        diffusion_steps = t_beta_sampling(T, B, sampling_params)
    elif sampling_strategy == "half_cauchy":
        diffusion_steps = t_half_cauchy_sampling(T, B, sampling_params)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    return diffusion_steps


def calculate_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1, sampling_strategy="uniform", **keywords):
    """
    Compute the loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    only_generate_missing (int):    flag indicating whether to compute the loss only for the masked elements (1)
                                    or for all elements (0)


    Returns:
    loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract diffusion hyperparameters
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    # Extract audio, condition, mask, and loss_mask from input tensor X
    audio = X[0].to(device)
    cond = X[1].to(device)
    mask = X[2].to(device)
    loss_mask = X[3].to(device)

    # Get the shape of the audio data
    B, C, L = audio.shape  # B is batchsize, C=num_channels, L is audio length

    # Extract sampling_params from keywords
    sampling_params = keywords.get('sampling_params', None)

    # Sample diffusion steps based on the chosen strategy
    diffusion_steps_t = t_sampling_strategy(sampling_strategy=sampling_strategy,
                                            T=T,
                                            B=B,
                                            sampling_params=sampling_params).cuda()

    # Generate standard normal distribution with the same shape as audio
    noise_distribution = std_normal(audio.shape)

    # If only_generate_missing is set to 1, modify z based on the mask
    # The result is a tensor where the values from audio are kept where mask is 1, and the values from z are kept where mask is 0.
    if only_generate_missing == 1:
        noise_distribution = audio * mask.float() + noise_distribution * (1 - mask).float()

    # Compute x_t from q(x_t|x_0) using the diffusion steps and Alpha_bar
    """
    The current_noisy_state tensor represents the state of the system at a certain diffusion step.
    It's a mix of the original data and some added noise, with the balance between the two controlled by the Alpha_bar parameter.

    torch.sqrt(Alpha_bar[diffusion_steps]) * audio: This represents the portion of the original data that is preserved in this diffusion step.
    torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z: This represents the new noise that is added in this diffusion step.

    current_noisy_state: The result is a tensor where the values are a mix of the scaled audio and z tensors.
    current_noisy_state is the tensor that represents the state of the system at a certain diffusion step. It's a mix of the original data
        and some added noise, with the balance between the two controlled by the Alpha_bar parameter.
    """
    current_noisy_state = torch.sqrt(Alpha_bar[diffusion_steps_t]) * audio + \
                    torch.sqrt(1 - Alpha_bar[diffusion_steps_t]) * noise_distribution

    # Predict epsilon_theta according to epsilon_theta using the current_noisy_state, cond, mask, and diffusion_steps. i.e. calling forward
    epsilon_theta = net(
        (current_noisy_state, cond, mask, diffusion_steps_t.view(B, 1),))

    # Compute the loss based on the value of only_generate_missing
    if only_generate_missing == 1:
        # If only_generate_missing is 1, compute the loss only for the masked elements
        # print(f"{epsilon_theta[loss_mask].shape=}")
        return loss_fn(epsilon_theta[loss_mask], noise_distribution[loss_mask]), diffusion_steps_t.view(-1).tolist()
    elif only_generate_missing == 0:
        # If only_generate_missing is 0, compute the loss for all elements
        return loss_fn(epsilon_theta, noise_distribution), diffusion_steps_t.view(-1).tolist()

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

    # Create a tensor of indices for the length of the mask
    length_index = torch.tensor(range(mask.shape[0]))

    # Split the length indices into k segments
    list_of_segments_index = torch.split(length_index, k)

    # Randomly select one of the segments
    s_nan = random.choice(list_of_segments_index)

    # For each channel in the mask, set the elements in the selected segment to 0
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    # Return the mask
    return mask

def get_mask_pred(sample, context_size, pred_size):
    """
    Applies masking to a given sample for forecasting.

    Args:
        sample (torch.Tensor): The input sample.
        context_size (int): The size of the context window.
        pred_size (int): The size of the prediction window.

    Returns:
        torch.Tensor: The masked sample.
    """

    assert sample.shape[1] == context_size + pred_size, f"{sample.size()=} != {context_size=} + {pred_size=}"

    # Initialize a mask of ones with the same shape as the sample
    mask = torch.ones(sample.shape)

    # Set the last label_size elements in each channel to 0
    mask[:, -pred_size:] = 0

    # Return the mask
    return mask