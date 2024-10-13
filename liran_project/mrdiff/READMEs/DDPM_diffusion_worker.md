# DDPM_diffusion_worker.py Documentation

[link to file](../../../mrDiff/models_diffusion/DDPM_diffusion_worker.py)

# Diffusion_Worker Class Documentation

The `Diffusion_Worker` class is designed to handle diffusion-based processes using deep learning models. Below is a detailed breakdown and explanation with mathematical formulations:

## Initialization: `__init__` Method

The constructor initializes the diffusion model with several parameters, such as the diffusion steps, noise schedules, and model architecture (`u_net`).

### Diffusion Process

The class deals with a *diffusion process*, a form of probabilistic generative modeling where data is progressively corrupted by noise, and the goal is to reverse this process to generate samples.

Key attributes include:
- `diff_train_steps`: The number of diffusion steps during training ($N_{\text{train}}$).
- `diff_test_steps`: The number of diffusion steps during testing ($N_{\text{test}} = N_{\text{train}}$).
- `beta_{\text{start}}$, `beta_{\text{end}}`: Defines the noise schedule from $ \beta_{\text{start}} $ to $ \beta_{\text{end}} $.

### Noise Schedule

The `beta_schedule` is the method used to adjust noise levels over time. The noise level increases progressively over diffusion steps:
$$
\beta_t \in [\beta_{\text{start}}, \beta_{\text{end}}]
$$
Different types of schedules include:
- Linear: $ \beta_t = \text{linspace}(\beta_{\text{start}}, \beta_{\text{end}}, \text{steps}) $.
- Quadratic, Constant, Cosine, etc., which provide various ways of interpolating or modifying noise schedules.

## `set_new_noise_schedule` Method

This method defines the schedule of noise and computes parameters that define the diffusion process.

### Alphas and Beta Schedule

For a noise schedule defined by $ \beta_t $, the corresponding *alphas* are given by:
$$
\alpha_t = 1 - \beta_t
$$
The cumulative product of alphas, $ \prod_{i=1}^{t} \alpha_i $, is stored as `alphas_cumprod`. These values are essential for sampling from the noisy distribution.

### Posterior Distribution

During diffusion, the posterior $ q(x_{t-1} | x_t, x_0) $ is parameterized as:
$$
q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\mu_t(x_t, x_0), \sigma_t^2)
$$
Where:
- $ \mu_t $ is the posterior mean, defined using `posterior_mean_coef1` and `posterior_mean_coef2`.
- $ \sigma_t^2 $ is the posterior variance, stored as `posterior_variance`.

## Forward Diffusion Process: `q_sample`

The method `q_sample` generates a sample from the noisy distribution at a given timestep $ t $:
$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon
$$
Where $ \epsilon $ is Gaussian noise. This process gradually corrupts the original data $ x_0 $.

## Reverse Process and Sampling: `p_sample`

The reverse process, described in `p_sample`, generates samples from the posterior $ p(x_{t-1} | x_t) $ at each timestep. The reverse process aims to denoise the sample from $ x_T $ (fully corrupted) back to $ x_0 $ (original data).

## Loss Calculation: `get_loss`

The model supports two types of loss functions for training:
- **L1 loss**: 
$$
\mathcal{L}_{\text{L1}} = \left| \text{target} - \text{pred} \right|
$$
- **L2 loss (MSE)**:
$$
\mathcal{L}_{\text{L2}} = \left( \text{target} - \text{pred} \right)^2
$$
These losses are used to train the diffusion model by comparing the predicted noise (or denoised input) to the actual noise.

## Training Process: `train_forward`

During training, the model:
1. Samples a random timestep $ t $ and adds noise to the original data $ x_0 $.
2. Feeds the noisy data into the network (U-Net).
3. The model is trained to predict either the noise $ \epsilon $ or the original data $ x_0 $, depending on the parameterization.

### Parameterization

The model can be parameterized in two ways:
- `noise`: The model predicts the noise $ \epsilon $.
- $x_{\text{start}}$: The model predicts the original data $ x_0 $.

## Posterior Sampling: `q_posterior` and `p_mean_variance`

The methods `q_posterior` and `p_mean_variance` compute the posterior mean and variance at each timestep, which are used for sampling during inference.

## Final Sampling: `ddpm_sampling`

This method implements the **denoising diffusion probabilistic model (DDPM)** sampling process. It generates samples by iteratively denoising from random noise through the learned reverse process.

## Summary

- **Forward Diffusion**: The process gradually adds noise to the data.
- **Reverse Process**: The model is trained to denoise the noisy data and recover the original signal.
- **Loss Functions**: Both L1 and L2 losses are supported.
- **Noise Schedule**: Several noise schedules (linear, cosine, etc.) can be used to control how noise is added over time.
- **Sampling**: The model generates samples by reversing the diffusion process.