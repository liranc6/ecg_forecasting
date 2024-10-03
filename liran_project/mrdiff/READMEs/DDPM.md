# DDPM.py Documentation

[link to file](../../../mrDiff/models_diffusion/DDPM.py)

# `BaseMapping` class

### 1. **Linear Trend Layer**:

In this model, the trend component of the time series is modeled using linear layers. Suppose the input sequence is represented by:

- $\mathbf{x}_{\text{enc}} \in \mathbb{R}^{B \times L_{\text{enc}} \times C}$, where:
  - $B$ is the batch size.
  - $L_{\text{enc}}$ is the length of the input sequence (encoded context).
  - $C$ is the number of channels (variables) in the time series.

The model predicts a trend for the future, where the predicted trend sequence is of length $L_{\text{pred}}$. In the model, this is achieved by passing the input through a linear transformation.

#### Individual Linear Layers:
If `self.individual = True`, each channel $c$ in the input has its own linear transformation:
$$
\mathbf{y}^{(c)} = \mathbf{x}_{\text{enc}}^{(c)} \mathbf{W}_c + \mathbf{b}_c \quad \text{for} \; c = 1, 2, ..., C
$$
Where:
- $\mathbf{W}_c \in \mathbb{R}^{L_{\text{enc}} \times L_{\text{pred}}}$ is the weight matrix for the $c$-th channel.
- $\mathbf{b}_c \in \mathbb{R}^{L_{\text{pred}}}$ is the bias vector for the $c$-th channel.
- $\mathbf{x}_{\text{enc}}^{(c)} \in \mathbb{R}^{B \times L_{\text{enc}}}$ represents the sequence of the $c$-th variable in the batch.

#### Shared Linear Layer:
If `self.individual = False`, all channels share the same linear transformation:
$$
\mathbf{Y} = \mathbf{X}_{\text{enc}} \mathbf{W} + \mathbf{b}
$$
Where:
- $\mathbf{W} \in \mathbb{R}^{L_{\text{enc}} \times L_{\text{pred}}}$ is the shared weight matrix.
- $\mathbf{b} \in \mathbb{R}^{L_{\text{pred}}}$ is the shared bias vector.

### 2. **Permutation Operations**:

The input data is permuted to match the required input shape for the linear layers. Initially, the input is in the form $\mathbf{x}_{\text{enc}} \in \mathbb{R}^{B \times L_{\text{enc}} \times C}$. This is permuted to $\mathbf{x}_{\text{enc}}' \in \mathbb{R}^{B \times C \times L_{\text{enc}}}$, allowing the linear transformations to apply to the sequence length dimension $L_{\text{enc}}$.

After the linear transformation, the output $\mathbf{y}$ is permuted back to the original shape $\mathbf{y} \in \mathbb{R}^{B \times L_{\text{pred}} \times C}$.

### 3. **Normalization with RevIN**:

If window normalization is used (`use_window_normalization` is set to `True`), the input data is normalized using the RevIN method. RevIN normalizes each variable by subtracting the mean and dividing by the standard deviation. For a given channel $c$:

$$
\mathbf{x}_{\text{enc}}^{(c)} = \frac{\mathbf{x}_{\text{enc}}^{(c)} - \mu_c}{\sigma_c}
$$
Where:
- $\mu_c$ is the mean of the $c$-th channel over the sequence length $L_{\text{enc}}$.
- $\sigma_c$ is the standard deviation of the $c$-th channel over the sequence length $L_{\text{enc}}$.

Denormalization is performed in the reverse order:
$$
\mathbf{y}^{(c)} = \mathbf{y}^{(c)} \sigma_c + \mu_c
$$

### 4. **Loss Calculation**:

The loss function depends on the user configuration. Two possible loss functions are implemented:

#### a. **Mean Squared Error (MSE)**:
The MSE loss between the predicted output $\mathbf{y}$ and the ground truth $\mathbf{x}_{\text{dec}}$ is given by:
$$
\text{MSE}(\mathbf{y}, \mathbf{x}_{\text{dec}}) = \frac{1}{B \cdot L_{\text{pred}} \cdot C} \sum_{b=1}^{B} \sum_{t=1}^{L_{\text{pred}}} \sum_{c=1}^{C} \left( y_{b,t,c} - x_{\text{dec},b,t,c} \right)^2
$$
Where:
- $B$ is the batch size.
- $L_{\text{pred}}$ is the predicted sequence length.
- $C$ is the number of channels.

#### b. **Symmetric Mean Absolute Percentage Error (SMAPE)**:
The SMAPE loss is given by:
$$
\text{SMAPE}(\mathbf{y}, \mathbf{x}_{\text{dec}}) = \frac{100\%}{B \cdot L_{\text{pred}} \cdot C} \sum_{b=1}^{B} \sum_{t=1}^{L_{\text{pred}}} \sum_{c=1}^{C} \frac{|y_{b,t,c} - x_{\text{dec},b,t,c}|}{\frac{1}{2}(|y_{b,t,c}| + |x_{\text{dec},b,t,c}|) + \epsilon}
$$
Where $\epsilon$ is a small constant added for numerical stability.

### 5. **Test Forward Pass**:

In the test phase, the process is nearly identical to the training phase, except that no loss is computed. The predicted trends are simply returned after normalization (if enabled) and permutation to the original shape. The output for the test forward is:
$$
\mathbf{y}_{\text{test}} = \mathbf{X}_{\text{enc}} \mathbf{W} + \mathbf{b}
$$

### Full Workflow in Mathematical Terms:

Given input data $\mathbf{X}_{\text{enc}} \in \mathbb{R}^{B \times L_{\text{enc}} \times C}$:

1. **Normalize (if RevIN)**:
   $$
   \mathbf{X}_{\text{enc}} \leftarrow \frac{\mathbf{X}_{\text{enc}} - \mu}{\sigma}
   $$
2. **Permute** the data to shape $B \times C \times L_{\text{enc}}$.
3. **Apply Linear Trend Layer**:
   $$
   \mathbf{Y} = \mathbf{X}_{\text{enc}} \mathbf{W} + \mathbf{b}
   $$
4. **Permute back** to the shape $B \times L_{\text{pred}} \times C$.
5. **Denormalize** (if RevIN):
   $$
   \mathbf{Y} \leftarrow \mathbf{Y} \sigma + \mu
   $$
6. **Calculate loss** (if training):
   - Use either MSE or SMAPE as the loss function, depending on the configuration.

In testing, the same workflow applies, but the model directly returns the predicted values without computing loss.



# `series_decomp` class

### 1. **Series Decomposition**:

The purpose of this class is to decompose a time series $\mathbf{x}$ into two components:
- **Residual component**: The difference between the original time series and its trend (moving average).
- **Moving average (trend) component**: The smoothed version of the time series, calculated using a moving average filter.

Given:
- $\mathbf{x} \in \mathbb{R}^{B \times L \times C}$, where:
  - $B$ is the batch size.
  - $L$ is the length of the time series.
  - $C$ is the number of channels (variables) in the time series.

### 2. **Moving Average Calculation**:

The moving average (or trend) is computed by convolving the input time series $\mathbf{x}$ with a kernel of size $k$ (the `kernel_size`). Mathematically, this is represented as:

$$
\mathbf{m}_t = \frac{1}{k} \sum_{i=0}^{k-1} \mathbf{x}_{t-i}
$$
Where:
- $\mathbf{m}_t$ is the moving average at time step $t$.
- $k$ is the `kernel_size` (the size of the moving window).
- $\mathbf{x}_{t-i}$ represents the value of the time series at time step $t-i$.

In PyTorch, this operation is implemented as a convolution with a kernel of ones, scaled by $\frac{1}{k}$. This is handled by the `moving_avg` module, which applies the convolution over the input series.

### 3. **Residual Calculation**:

The residual component is the difference between the original time series $\mathbf{x}$ and the moving average $\mathbf{m}$:

$$
\mathbf{r}_t = \mathbf{x}_t - \mathbf{m}_t
$$
Where:
- $\mathbf{r}_t$ is the residual (or "detail") component at time step $t$.
- $\mathbf{x}_t$ is the value of the original time series at time step $t$.
- $\mathbf{m}_t$ is the moving average (trend) at time step $t$.

### Full Workflow in Mathematical Terms:

Given the input time series $\mathbf{x} \in \mathbb{R}^{B \times L \times C}$, the `forward` method performs the following steps:

1. **Moving Average Calculation**:
   The moving average (or trend) is computed by applying a convolution with a kernel size $k$ over the time series $\mathbf{x}$:

   $$
   \mathbf{m} = \text{Conv1D}(\mathbf{x}, \mathbf{w})
   $$
   Where:
   - $\mathbf{w}$ is a convolution kernel of ones scaled by $\frac{1}{k}$, i.e., $\mathbf{w} = \frac{1}{k} \mathbf{1}_{k}$.
   - The convolution is applied over the time dimension of $\mathbf{x}$.

2. **Residual Calculation**:
   The residual component is computed by subtracting the moving average $\mathbf{m}$ from the original time series $\mathbf{x}$:

   $$
   \mathbf{r} = \mathbf{x} - \mathbf{m}
   $$
   Where:
   - $\mathbf{r} \in \mathbb{R}^{B \times L \times C}$ is the residual component of the series.
   - $\mathbf{m} \in \mathbb{R}^{B \times L \times C}$ is the moving average (trend).

3. **Return the Decomposed Components**:
   The method returns both the residual $\mathbf{r}$ and the moving average $\mathbf{m}$ components:

   $$
   \text{output} = (\mathbf{r}, \mathbf{m})
   $$

### Summary of Operations:

1. **Input**: $\mathbf{x} \in \mathbb{R}^{B \times L \times C}$ (the original time series).
2. **Moving Average**:
   $$
   \mathbf{m}_t = \frac{1}{k} \sum_{i=0}^{k-1} \mathbf{x}_{t-i}
   $$
3. **Residual**:
   $$
   \mathbf{r}_t = \mathbf{x}_t - \mathbf{m}_t
   $$
4. **Output**: A tuple $(\mathbf{r}, \mathbf{m})$ containing the residual and moving average components.

This decomposition separates the input time series into its trend and residual components, useful for various time series analysis and forecasting tasks.


# `Model(nn.Module)` class


The class focuses on decomposing time series data, generating trends, and applying diffusion processes for forecasting. Below is an explanation with mathematical notations and descriptions that avoid using code.

---

### Model Initialization

The model takes a series of time points as inputs, aiming to predict future trends by applying decomposition techniques and a diffusion process.

Let the input sequence be represented as:

$$
X = \{x_1, x_2, \ldots, x_T\}, \quad X \in \mathbb{R}^{B \times N \times L}
$$

where \(B\) is the batch size, \(N\) is the number of variables (features), and \(L\) is the sequence length.

#### Sequence Lengths:
- \( \text{context\_len} \): Number of timesteps from the past used for prediction.
- \( \text{pred\_len} \): Number of timesteps predicted into the future.

The model has multiple "bridges" \(K\), which control the number of intermediate transformations applied to the input series, where each bridge works on a different smoothed version of the input.

### Decomposition and Trend Extraction

The model applies time-series decomposition to extract trends from the input sequence. For each time series \(X\), the decomposition step can be represented mathematically as:

$$
X = T + S + R
$$

where:
- \(T\) represents the trend component,
- \(S\) represents the seasonal component,
- \(R\) is the residual (noise).

In this model, the focus is on obtaining the trend components:

$$
X \longrightarrow T_i, \quad i = 1, 2, \ldots, K
$$

Here, \(T_i\) represents the trends obtained from \(K\) different levels of smoothing. Each smoothed trend corresponds to a different degree of factor-based smoothing (determined by \(\text{smoothed\_factors}\)).

### Multi-Trend Generation

The model iterates over the decomposition step to extract multiple trends from the input sequence. For each trend \(T_i\), the process is recursive:

$$
T_0 = X, \quad T_{i+1} = \text{decomp}(T_i)
$$

This decomposition is done for each bridge, producing a sequence of trends:

$$
T_1, T_2, \ldots, T_K
$$

### Diffusion Process

The model employs a diffusion process to generate new trends or samples from the existing trends. The diffusion process for each bridge \(i\) is applied as:

$$
X_{t-1} \longleftarrow X_t + \epsilon
$$

where \(X_t\) represents the trend at timestep \(t\), and \(\epsilon\) is a Gaussian noise term.

For the sampling process, reverse diffusion is applied to progressively reconstruct the sequence, starting from random noise:

$$
X_T \longrightarrow X_0
$$

This reverse diffusion incorporates both the linear guess (autoregression) and the conditional information from past trends:

$$
X_t = f_{\text{diff}}(X_t, \text{cond}(T_{\text{past}}))
$$

where \(\text{cond}(T_{\text{past}})\) refers to the conditions generated from past trends.

### Loss Computation

The loss is computed between the predicted future trends \(T_f\) and the actual future trends \(T_{\text{true}}\):

$$
\mathcal{L}(T_f, T_{\text{true}}) = \sum_{i=1}^K \mathcal{L}_i(T_{f,i}, T_{\text{true},i})
$$

For each bridge \(i\), the loss \(\mathcal{L}_i\) is defined based on the difference between the predicted and true trends at that level:

$$
\mathcal{L}_i = \| T_{f,i} - T_{\text{true},i} \|^2
$$

The total loss is the weighted sum of individual bridge losses.

### Summary of Key Processes
1. **Decomposition**: Break down the input sequence into multiple trends.
2. **Diffusion**: Apply noise and reverse-sample the trends.
3. **Autoregression**: Estimate the next sequence by linearly guessing based on past trends.
4. **Loss Minimization**: Minimize the difference between predicted and actual trends across all levels.

By combining these steps, the model aims to produce robust predictions for the future sequence based on both historical data and diffusion-based transformations.


