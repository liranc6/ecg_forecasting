# NoiseScheduleVP Class Overview

The `NoiseScheduleVP` class is a noise schedule wrapper used for defining forward Stochastic Differential Equations (SDEs) in Variance Preserving (VP) diffusion models, commonly applied in discrete and continuous diffusion processes. Here is a mathematical explanation of the main components of the class.

---

## Forward SDE for VP Diffusion Process

In diffusion models, the forward SDE is used to progressively add noise to data over time $t \in [0, T]$, transforming the data into noise. The goal of the diffusion process is to sample from a target distribution by solving a reverse-time SDE.

For a forward SDE in VP diffusion models, the conditional distribution of noisy data $x_t$ given original data $x_0$ follows a normal distribution:
$$
q_{t|0}(x_t | x_0) = \mathcal{N} \left( \alpha_t x_0, \sigma_t^2 I \right)
$$
where:
- $\alpha_t$ is a time-dependent coefficient controlling the scaling of $x_0$,
- $\sigma_t$ is the standard deviation of the noise at time $t$.

In this class, $\alpha_t$ and $\sigma_t$ are derived from the schedule and depend on whether the noise schedule is discrete or continuous.

## Discrete and Continuous Schedules

1. **Discrete Schedule**:
   - For discrete-time DPMs (Diffusion Probabilistic Models), the time steps $t$ are defined as:
     $$
     t_i = \frac{i+1}{N}, \quad i = 0, 1, \dots, N-1
     $$
     where $N$ is the total number of time steps.
   - The noise schedule is defined by either $\beta_n$ or $\hat{\alpha}_n$ (cumulative product of alphas), where:
     $$
     \log(\alpha_{t_n}) = 0.5 \log(\hat{\alpha}_n)
     $$
   - The forward SDE is solved from $t_N = 1$ to $t_0 = 10^{-3}$.

2. **Continuous Schedule**:
   - For continuous-time DPMs, two types of noise schedules are supported:
     - **Linear Schedule**: The noise increases linearly with time, controlled by parameters $\beta_0$ and $\beta_1$. The log of $\alpha_t$ is given by:
       $$
       \log(\alpha_t) = -0.25 t^2 (\beta_1 - \beta_0) - 0.5 t \beta_0
       $$
     - **Cosine Schedule**: The noise increases according to a cosine function, parameterized by $s$ and $\beta_{\text{max}}$. The log of $\alpha_t$ is:
       $$
       \log(\alpha_t) = \log \left( \cos \left( \frac{t + s}{1+s} \frac{\pi}{2} \right) \right) - \log \left( \cos \left( \frac{s}{1+s} \frac{\pi}{2} \right) \right)
       $$

## Key Variables

- **$\alpha_t$**: The scaling coefficient applied to the initial data $x_0$. It is computed as:
  $$
  \alpha_t = \exp(\log(\alpha_t))
  $$
  
- **$\sigma_t$**: The noise variance at time $t$, computed as:
  $$
  \sigma_t = \sqrt{1 - \exp(2 \log(\alpha_t))}
  $$
  
- **$\lambda_t$**: Defined as:
  $$
  \lambda_t = \log(\alpha_t) - \log(\sigma_t)
  $$
  It represents the half-log signal-to-noise ratio (SNR) and is an important quantity for noise scheduling in diffusion models.

## Inverse Function for $\lambda_t$

Given a value of $\lambda_t$, it is possible to compute the corresponding time $t$ using the inverse of the function. The exact formula depends on the schedule:
- **Linear Schedule**: The time is computed by solving a quadratic equation.
- **Cosine Schedule**: The time is computed using an arccos function.

## Discrete Interpolation

For discrete schedules, interpolation is applied to map between continuous time $t$ and the discrete time steps $t_i$. This allows the use of continuous-time formulations in discrete-time models.

---

## Summary of Mathematical Functions

1. **Log Mean Coefficient $\log(\alpha_t)$**:
   - Discrete: Interpolated using piecewise linear interpolation of precomputed log alphas.
   - Linear: $\log(\alpha_t) = -0.25 t^2 (\beta_1 - \beta_0) - 0.5 t \beta_0$.
   - Cosine: $\log(\alpha_t) = \log \left( \cos \left( \frac{t + s}{1+s} \frac{\pi}{2} \right) \right) - \text{offset}$.

2. **Alpha $\alpha_t$**: $\alpha_t = \exp(\log(\alpha_t))$.

3. **Standard Deviation $\sigma_t$**: $\sigma_t = \sqrt{1 - \exp(2 \log(\alpha_t))}$.

4. **Lambda $\lambda_t$**: $\lambda_t = \log(\alpha_t) - \log(\sigma_t)$.

5. **Inverse Lambda**: Computes the time $t$ from $\lambda_t$, with specific formulas depending on the noise schedule (linear or cosine).

This class is central to defining and handling the noise schedule in VP diffusion models, ensuring that the noise follows a pre-specified process for both discrete and continuous models.

# Noise Prediction Function

The `noise_pred_fn` function computes different predictions based on the input data $x$, continuous time $t$, and an optional conditioning variable $cond$.

### Key Components

1. **Input Parameters:**
        - **𝑥**: Input tensor (e.g., image or signal).
        - **𝑡**: Tensor of continuous time points for predictions.
        - **cond**: Optional conditioning tensor.

2. **Time Reshaping:** If **𝑡** is a single time point, it is expanded to match the batch size of **𝑥**.

**Model Input Preparation:** **𝑡_input** is generated from **𝑡** to match the model's expected input format.

**Model Prediction:** The model predicts based on **𝑥** and **𝑡_input**, optionally using **cond**.

## Model Types

### 1. Noise Prediction
Returns the model's output directly, predicting the noise component.

### 2. Starting Point Prediction
Uses the formula:

$$
x_0 = x - \alpha_t \cdot output
$$

to estimate the denoised signal $x_0$.

### 3. Velocity Prediction
Combines outputs using:

$$
v = \alpha_t \cdot output + \sigma_t \cdot x
$$

to estimate the signal's velocity.

### 4. Score Prediction
Computes the score function as:

$$
s(x) = -\sigma_t \cdot output
$$

indicating the gradient of the log probability density.

# model_wrapper Function Overview

Here’s a concise summary of the `model_wrapper` function:

## Purpose:
The `model_wrapper` function wraps a noise prediction model to work with continuous-time diffusion ODEs, providing flexibility for different types of diffusion models and guidance methods during sampling.

## Key Features:
- **Model Types Supported**:
  1. **Noise**: Predicts noise directly (e.g., DDPM).
  2. **$ x_{\text{start}} $**: Predicts the original data (e.g., $ x_0 $) at time 0.
  3. **$ v $**: Predicts velocity (used in "velocity parameterization" from DPMs).
  4. **Score**: Denoising score function (related to score matching).

- **Guidance Types Supported**:
  1. **Unconditional**: No conditioning, outputs based purely on noise prediction.
  2. **Classifier Guidance**: Uses an external classifier for guided sampling, adding a gradient from a classifier model to steer the process.
  3. **Classifier-Free Guidance**: Combines outputs from both a conditioned and unconditioned model to guide sampling (e.g., as seen in classifier-free diffusion guidance).

## Arguments:
- `model`: The diffusion model to wrap.
- `noise_schedule`: The noise schedule (e.g., `NoiseScheduleVP`).
- `model_type`: Type of model ("noise", "$ x_{\text{start}} $", "$ v $", or "score").
- `model_kwargs`: Additional keyword arguments for the model.
- `guidance_type`: The type of guidance during sampling ("uncond", "classifier", or "classifier-free").
- `condition`: Conditional input for guided sampling (used in "classifier" or "classifier-free").
- `unconditional_condition`: Unconditional input for classifier-free guidance.
- `guidance_scale`: Scale factor for the guidance strength.
- `classifier_fn`: Classifier function used for classifier guidance.
- `classifier_kwargs`: Additional arguments for the classifier function.

## Key Internal Functions:
- **`get_model_input_time`**: Converts continuous time ($ t_{\text{continuous}} $) to the model's input time, either for discrete or continuous-time DPMs.
- **`noise_pred_fn`**: Computes noise prediction based on the model type. Adjusts for the model's output (e.g., noise, $ x_{\text{start}} $, or velocity).
- **`cond_grad_fn`**: Computes the gradient from the classifier during classifier guidance.
- **`model_fn`**: Main function that returns the noise prediction for DPM-Solver. Handles guidance methods based on `guidance_type`.

## Output:
- Returns a noise prediction function `model_fn` that accepts the noisy data $ x $ and continuous time $ t_{\text{continuous}} $ for use in DPM-Solver.


# Class Overview: `DPM_Solver`

The `DPM_Solver` class implements various solvers for solving differential equations (DEs) that arise in diffusion models. Diffusion probabilistic models (DPMs) aim to denoise data through progressive refinement, modeling data distributions by learning a reverse process from noise to data. This solver addresses multiple steps in time discretization using both first-order and higher-order methods. The primary purpose of this class is to solve ordinary differential equations (ODEs) related to the diffusion process, approximating data at various time steps.

Key features include:
- First-order updates (similar to DDIM)
- Higher-order updates (second-order and third-order solvers)
- Flexibility in solver methods (e.g., `dpm_solver`, `taylor` methods)

### Method 1: `denoise_to_zero_fn`

This method applies denoising at the final step. Mathematically, this corresponds to solving the ODE from the given $s$ time to infinity using a first-order discretization scheme. It directly uses the `data_prediction_fn` to generate the result.

**Parameters:**
- `x`: The input tensor representing data at time $s$.
- `s`: The time tensor for which the model generates predictions.

**Returns:**
- The result of the `data_prediction_fn(x, s)`, effectively denoising the input $x$ at time $s$.

---

### Method 2: `dpm_solver_first_update`

This method implements a first-order solver, similar to DDIM, to update the input $x$ from time $s$ to time $t$.

#### Parameters:
- `x`: A PyTorch tensor representing the data at time $s$.
- `s`: A tensor representing the starting time, shaped as $(x.shape[0],)$.
- `t`: A tensor representing the ending time, also shaped as $(x.shape[0],)$.
- `model_s`: Optional. A tensor representing the model function evaluated at time $s$. If `None`, the method computes it using `self.model_fn(x, s)`.
- `return_intermediate`: Optional. If `True`, the method returns the intermediate model value at time $s$.

#### Mathematical Formulation:
Let:
- $\lambda_s = \text{marginal\_lambda}(s)$, $\lambda_t = \text{marginal\_lambda}(t)$
- $h = \lambda_t - \lambda_s$
- $\alpha_s = e^{\log\_alpha_s}, \alpha_t = e^{\log\_alpha_t}$
- $\sigma_s, \sigma_t$ are standard deviations at time $s$ and $t$.

The update depends on the prediction type (`self.predict_x0`):
1. **If `predict_x0 = True` (predict initial data $x_0$):**
   $$
   \phi_1 = \exp(-h) - 1
   $$
   $$
   x_t = \left( \frac{\sigma_t}{\sigma_s} \right) x - \alpha_t \phi_1 \cdot \text{model\_s}
   $$
2. **Otherwise:**
   $$
   \phi_1 = \exp(h) - 1
   $$
   $$
   x_t = \left( \frac{\exp(\log\_alpha_t - \log\_alpha_s)}{\exp(\log\_alpha_s)} \right) x - \sigma_t \phi_1 \cdot \text{model\_s}
   $$

#### Returns:
- `x_t`: A tensor representing the approximated solution at time $t$.
- Optionally, if `return_intermediate=True`, it also returns a dictionary with the intermediate model value.

---

### Method 3: `singlestep_dpm_solver_second_update`

This method implements a single-step second-order solver from time $s$ to time $t$. It allows for higher accuracy by using second-order updates, controlled by a parameter $r1$. It supports two types of solvers: `dpm_solver` and `taylor`.

#### Parameters:
- `x`: The input tensor at time $s$.
- `s`: The starting time tensor.
- `t`: The ending time tensor.
- `r1`: The hyperparameter for the second-order solver.
- `model_s`: Optional. The model function at time $s$. If `None`, it computes it using `self.model_fn(x, s)`.
- `return_intermediate`: Optional. If `True`, returns intermediate model values.
- `solver_type`: Either `'dpm_solver'` or `'taylor'`. Impacts performance.

#### Mathematical Formulation:
1. Compute the time steps and auxiliary time points:
   $$
   \lambda_{s+1} = \lambda_s + r1 \cdot h, \quad s1 = \text{inverse\_lambda}(\lambda_{s1})
   $$
   - $\sigma_{s1}, \log\_alpha_{s1} \quad \text{(marginals at time } s1\text{)}$
2. Update steps:
   **If `predict_x0 = True`:**
   $$
   x_{s1} = \left( \frac{\sigma_{s1}}{\sigma_s} \right) x - \alpha_{s1} \cdot \phi_{11} \cdot \text{model\_s}
   $$
   For `solver_type = dpm_solver`:
   $$
   x_t = \left( \frac{\sigma_t}{\sigma_s} \right) x - \alpha_t \phi_1 \cdot \text{model\_s} - \frac{0.5}{r1} \cdot \alpha_t \phi_1 \cdot (\text{model\_s1} - \text{model\_s})
   $$

#### Returns:
- `x_t`: Approximated solution at time $t$.
- Optionally, intermediate results if `return_intermediate=True`.

---

### Method 4: `singlestep_dpm_solver_third_update`

This method implements a third-order solver, offering even greater accuracy over the second-order method. It uses two auxiliary time points, controlled by $r1$ and $r2$.

#### Parameters:
- `x`: Input tensor at time $s$.
- `s`: Starting time tensor.
- `t`: Ending time tensor.
- `r1`, `r2`: Hyperparameters controlling the time step distribution.
- `model_s`, `model_s1`: Optional. Model values at times $s$ and $s1$.
- `return_intermediate`: Optional. Returns intermediate model values.
- `solver_type`: Either `dpm_solver` or `taylor`.

#### Mathematical Formulation:
This uses three time steps: $s$, $s1$, and $s2$:
1. Compute auxiliary points:
   $$
   \lambda_{s1} = \lambda_s + r1 \cdot h, \quad \lambda_{s2} = \lambda_s + r2 \cdot h
   $$
2. Compute the updates based on the auxiliary time points $s1$, $s2$.

#### Returns:
- `x_t`: The approximated solution at time $t$.
- Optionally, intermediate results if `return_intermediate=True`.

---

### Method 5: `multistep_dpm_solver_second_update`

This method implements a multistep second-order solver. It leverages the results from previous steps (`model_prev_list`) to update the current solution.

#### Parameters:
- `x`: Input tensor at the last time step.
- `model_prev_list`: List of model values from previous steps.
- `t_prev_list`: List of previous time steps.
- `t`: The current time step.
- `solver_type`: Solver type (either `dpm_solver` or `taylor`).

#### Mathematical Formulation:
1. Uses information from previous time steps (`model_prev_list`, `t_prev_list`) to update $x$ at time $t$ based on second-order ODE solvers.

#### Returns:
- `x_t`: The approximated solution at time $t$.


### Method 6: `multistep_dpm_solver_third_update`
- **Purpose**: Computes the third-order multistep update from a previous state to a desired time point using the DPM-Solver-3 method.
- **Parameters**:
  - `x`: A PyTorch tensor representing the initial state at the current time.
  - `model_prev_list`: A list of PyTorch tensors representing previously computed model states.
  - `t_prev_list`: A list of tensors indicating the previous time points corresponding to the model states.
  - `t`: A tensor representing the target time to which the solution is being computed.
  - `solver_type`: A string indicating the solver type (`'dpm_solver'` or `'taylor'`).
- **Returns**: A PyTorch tensor `x_t`, which is the approximated solution at time $t$.

**Mathematical Description**:  
The method uses various marginal quantities derived from a noise schedule, represented as $\lambda(t)$, $\sigma(t)$, and $\alpha(t)$, to compute the approximated state at time $t$ through a series of updates involving differences between previous states and their respective times. The computations can be outlined as follows:

1. Calculate the differences $h_1, h_0, h$ based on the marginal $\lambda$ values.
2. Compute the ratios $r_0$ and $r_1$ for these differences.
3. Determine $D1_0$, $D1_1$, and $D2$ as functions of the previous model states.
4. Depending on the `predict_x0` flag, compute the final state $x_t$ using:
   - For $\text{predict\_x0} = \text{True}$:
     $$
     x_t = \frac{\sigma_t}{\sigma_{prev_0}} x - \alpha_t \left( \exp(-h) - 1 \right) model_{prev_0} + \alpha_t \left( \frac{\exp(-h) - 1}{h} + 1 \right) D1 - \alpha_t \left( \frac{\exp(-h) - 1 + h}{h^2} - 0.5 \right) D2
     $$
   - For $\text{predict\_x0} = \text{False}$:
     $$
     x_t = \exp(\log(\alpha_t) - \log(\alpha_{prev_0})) x - \sigma_t \left( \exp(h) - 1 \right) model_{prev_0} - \sigma_t \left( \frac{\exp(h) - 1}{h} - 1 \right) D1 - \sigma_t \left( \frac{\exp(h) - 1 - h}{h^2} - 0.5 \right) D2
     $$

### Method 7: `singlestep_dpm_solver_update`
- **Purpose**: Performs a single-step update based on the specified order of the DPM solver.
- **Parameters**:
  - `x`: Initial state tensor.
  - `s`: Starting time tensor.
  - `t`: Ending time tensor.
  - `order`: An integer specifying the order of the DPM solver (1, 2, or 3).
  - `return_intermediate`: A boolean indicating whether to return intermediate model values.
  - `solver_type`: The type of solver to use.
  - `r1`, `r2`: Hyperparameters for higher-order solvers.
- **Returns**: A PyTorch tensor `x_t`, which is the approximated solution at time $t$.

**Mathematical Description**:  
This method invokes specific functions based on the order parameter:
- For order 1, it calls the first-order update.
- For orders 2 and 3, it calls the corresponding second or third-order update functions, which would involve more complex polynomial or exponential adjustments based on the defined models.

### Method 8: `multistep_dpm_solver_update`
- **Purpose**: Similar to `singlestep_dpm_solver_update`, but uses multistep updates for the specified order.
- **Parameters**: Same as `singlestep_dpm_solver_update`, with the addition of `model_prev_list` and `t_prev_list` to handle past states and times.
- **Returns**: A tensor `x_t` representing the approximated state at time $t$.

**Mathematical Description**:  
As with the singlestep method, this function will decide on the multistep update routine based on the order of the solver specified.

### Method 9: `dpm_solver_adaptive`
- **Purpose**: Implements an adaptive step-size solver, allowing for dynamic adjustment of the step size based on the progress of the solver.
- **Parameters**: 
  - `x`: Initial state tensor.
  - `order`: Integer defining the order of the solver (must be 2 or 3).
  - `t_T`, `t_0`: Starting and ending times for sampling.
  - `h_init`: Initial step size.
  - `atol`, `rtol`: Absolute and relative tolerances for the solver.
  - `theta`: Hyperparameter for adjusting the step size.
  - `t_err`: Error tolerance for time convergence.
  - `solver_type`: Type of solver to use.
- **Returns**: A tensor $x_0$, the estimated solution at the starting time $t_0$.

**Mathematical Description**:  
The adaptive method iteratively updates the state by:
1. Computing $x_{lower}$ and $x_{higher}$ through lower and higher order updates.
2. Evaluating the error $E$ between these two estimates normalized by a tolerance:
   $$
   E = \frac{\|x_{higher} - x_{lower}\|}{\delta}
   $$
3. Adjusting the step size $h$ based on the maximum error across dimensions, capped by the computed tolerance parameters.

### Method 10: `sample`
- **Purpose**: Computes samples at a specific endpoint using various sampling methods based on DPM-Solvers.
- **Parameters**: 
  - `x`: Initial state tensor at time $t_{start}$.
  - `steps`: Total number of function evaluations (NFE).
  - `t_{start}`, `t_{end}`: The starting and ending times for sampling.
  - `order`: The order of the DPM-Solver.
  - `skip_type`: Type of spacing for the time steps.
  - `method`: The specific sampling method to use.
  - `lower_order_final`: Whether to apply lower-order solvers in the final steps.
  - `denoise_to_zero`: Indicates if the final output should be denoised to zero.
- **Returns**: A tensor representing the generated sample.

**Mathematical Description**:  
The method combines solvers of varying orders based on the specified `method` parameter, effectively segmenting the sampling process into multiple sub-processes that utilize the defined number of function evaluations. It supports configurations to manage how the sampling progresses, including decisions to denoise fully or to adaptively adjust the order of solvers applied.

---
---
### Method: `interpolate_fn`

**Purpose**  
The `interpolate_fn` method performs piecewise linear interpolation given a set of keypoints defined by $x_p$ (x-coordinates) and $y_p$ (y-coordinates). It computes the value of the function $f(x)$ at points $x$ by utilizing linear segments defined between these keypoints, ensuring differentiability to facilitate backpropagation during training.

**Parameters**
- `x`: A PyTorch tensor of shape $[N, C]$, where:
  - $N$: Batch size.
  - $C$: Number of channels (typically $C = 1$ in the context of this class).
  
- `xp`: A PyTorch tensor of shape $[C, K]$, representing the x-coordinates of the keypoints, where $K$ is the number of keypoints.

- `yp`: A PyTorch tensor of shape $[C, K]$, representing the corresponding y-values of the keypoints.

**Returns**  
The method returns a tensor containing the interpolated function values $f(x)$, with the shape $[N, C]$.

**Mathematical Formulation**  
The interpolation process is based on piecewise linear segments defined by the keypoints $(x_{p_i}, y_{p_i})$. For a given input $x$, the interpolation can be mathematically expressed as follows:

1. **Sorting**: Create an augmented tensor that concatenates $x$ with the keypoints $x_p$:
   $$
   \text{all}_x = [x, x_p]
   $$

2. **Finding Indices**: Identify the index of the closest keypoint to each element in $x$:
   $$
   x_{\text{idx}} = \arg\min_i |x - x_{p_i}|
   $$

3. **Start and End Points**: Determine the indices of the starting and ending keypoints for the linear segment containing $x$:
   - If $x_{\text{idx}} = 0$ (indicating $x$ is less than the first keypoint), we set $start_{\text{idx}} = 1$ to avoid indexing errors.
   - If $x_{\text{idx}} = K$ (indicating $x$ exceeds the last keypoint), we set $start_{\text{idx}} = K - 2$.

4. **Linear Interpolation**: For each $x$, calculate the interpolated value using the formula:
   $$
   f(x) = start_y + \frac{(x - start_x)}{(end_x - start_x)} \cdot (end_y - start_y)
   $$
   where:
   - $start_y$ and $end_y$ are the y-values corresponding to the keypoints surrounding $x$.
   - $start_x$ and $end_x$ are the x-values of these keypoints.

**Summary**  
The `interpolate_fn` method in the `DPM_Solver` class provides a robust mechanism for differentiable piecewise linear interpolation, facilitating effective model training and inference by enabling evaluations at arbitrary points defined in relation to a set of keypoints. The careful handling of edge cases (e.g., values of $x$ beyond the keypoint range) ensures stability and reliability in practical applications.

