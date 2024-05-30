import os
import numpy as np
import torch
import random
from tqdm.notebook import tqdm


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

    for f in files[::-1]:
        if not os.path.isfile(os.path.join(path, f)):
            continue
        assert len(f) > 4
        if critirion == 'max' and f[-4:] == '.pkl':
            epoch = int(f[:-4])
            if num+file_i==0:
                return epoch
            file_i += 1
        elif critirion == 'best' and f[:5] == 'best_':
            epoch = f.split('_')[3]
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
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x

def calculate_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
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

    # Randomly sample diffusion steps from 1~T
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()

    # Generate standard normal distribution with the same shape as audio
    z = std_normal(audio.shape)

    # If only_generate_missing is set to 1, modify z based on the mask
    # The result is a tensor where the values from audio are kept where mask is 1, and the values from z are kept where mask is 0.
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()

    # Compute x_t from q(x_t|x_0) using the diffusion steps and Alpha_bar
    """
    The transformed_X tensor represents the state of the system at a certain diffusion step.
    It's a mix of the original data and some added noise, with the balance between the two controlled by the Alpha_bar parameter.

    torch.sqrt(Alpha_bar[diffusion_steps]) * audio: This represents the portion of the original data that is preserved in this diffusion step.
    torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z: This represents the new noise that is added in this diffusion step.

    transformed_X: The result is a tensor where the values are a mix of the scaled audio and z tensors.
    transformed_X is the tensor that represents the state of the system at a certain diffusion step. It's a mix of the original data
        and some added noise, with the balance between the two controlled by the Alpha_bar parameter.
    """
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + \
                    torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z

    # Predict \epsilon according to \epsilon_\theta using the transformed_X, cond, mask, and diffusion_steps. i.e. calling forward
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))

    # Compute the loss based on the value of only_generate_missing
    if only_generate_missing == 1:
        # If only_generate_missing is 1, compute the loss only for the masked elements
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        # If only_generate_missing is 0, compute the loss for all elements
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