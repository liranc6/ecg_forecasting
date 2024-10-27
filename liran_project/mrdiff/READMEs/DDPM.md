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


## Function: train_forward

### Purpose
`train_forward` manages the training process by extracting trends from past and future data segments, generating autoregressive initial trends, and computing the total loss for the diffusion model. It can also monitor GPU memory usage at key stages if configured to do so.

### Process
- **GPU Memory Check (Optional)**: If `check_gpu_memory_usage` is set, it prints GPU memory usage.
- **Normalization**: If window normalization is enabled, input tensors `x_enc` and `x_dec` are normalized back to their original scale using `self.rev`.
  
#### Trend Computation:
1. **Future Trends**: The future data tensor `x_future` is extracted from `x_dec` and processed through `self.obtain_multi_trends`. A randomized future trend, `future_xT`, serves as a placeholder for additional trends.
2. **Past Trends**: Trends from `x_past` are computed similarly using the same method.
3. **Autoregressive Initial Trends**: `_compute_trends_and_guesses` is used to initialize trends for autoregressive modeling. This function returns a linear guess for the first bridge and a series of trends for later stages.

- **Loss Calculation**: The `_compute_total_loss` function computes the cumulative loss across all bridges, evaluating future trends against predictions.
- **Return**: The loss is averaged if `return_mean` is `True`, or it can be weighted by `meta_weights` if provided.

### Mathematical Formulation
Let:

- $ x_{\text{past}} $ represent the historical input,
- $ x_{\text{future}} $ represent the future input,
- `future_trends` and `past_trends` be trend lists.

For each time bridge $ i $, `_compute_total_loss` calculates:

$$
\texttt{loss}_i = \texttt{train\_forward}(X_0, X_1, \texttt{MASK}, \texttt{condition}, \texttt{ar\_init})
$$

where:

- $ X_0 $ is the current future trend `future_trends[i - 1]`,
- $ X_1 $ is the next future trend `future_trends[i]`,
- `MASK` is a binary mask tensor,
- `condition` contains concatenated relevant trends,
- `ar_init` is the autoregressive trend.

The final loss is computed as:

$$
\texttt{total\_loss} = \sum_{i=0}^{\text{num\_bridges}} \texttt{loss}_i
$$

or, if `return_mean` is `True`:

$$
\texttt{total\_loss} = \frac{1}{\text{num\_bridges}} \sum_{i=0}^{\text{num\_bridges}} \texttt{loss}_i
$$

---

## Auxiliary Function: _compute_total_loss

### Purpose
`_compute_total_loss` computes the cumulative loss for each bridge in the diffusion model by comparing predicted trends across stages.

### Process
- **Initialize Loss**: For the first bridge, it calculates loss using `x_future` and the first future trend.
- **Subsequent Bridges**: For each bridge $ i $, loss is computed using trends $ X_0 $ and $ X_1 $, along with relevant conditioning trends and masks.
- **Return**: The loss list for all bridges is returned to be averaged or summed based on `train_forward` requirements.

### Mathematical Formulation
For bridge $ i = 0 $, with $ X_1 = \text{future\_trends}[0] $ and $ X_0 = x_{\text{future}} $:

$$
\texttt{loss}_0 = \text{diffusion\_workers}[0].\texttt{train\_forward}(X_0, X_1, \texttt{MASK}, \texttt{condition})
$$

For subsequent bridges $ i > 0 $, where $ X_1 = \text{future\_trends}[i] $ and $ X_0 = \text{future\_trends}[i - 1] $:

$$
\texttt{loss}_i = \text{diffusion\_workers}[i].\texttt{train\_forward}(X_0, X_1, \texttt{MASK}, \texttt{condition})
$$

---

## Auxiliary Function: _compute_trends_and_guesses

### Purpose
This function computes initial trends and linear guesses required for autoregressive modeling in each bridge, essential for capturing temporal transitions in diffusion modeling.

### Process
- **First Bridge**: Initializes trends using `x_past` and `x_future`. If normalization is used, trends are reverted to their original scales.
- **Subsequent Bridges**: Trends for each bridge are computed using `future_trends[i-1]` and optionally `x_past` or `past_trends[i-1]`.
- **Linear Guess**: For each bridge $ i $, a linear trend is estimated using the base model, providing an approximation of the transition from past to future trends.

### Mathematical Formulation
For each bridge $ i $, the linear guess $ \texttt{linear\_guess}_i $ is computed as:

$$
\texttt{linear\_guess}_i = \text{base\_models}[i].\texttt{test\_forward}(X_{\text{past}}, x_{\text{mark\_enc}}, X_{\text{future}}, x_{\text{mark\_dec}})
$$

where $ X_{\text{past}} $ and $ X_{\text{future}} $ represent past and future trends.

### Summary of Key Processes
1. **Decomposition**: Break down the input sequence into multiple trends.
2. **Diffusion**: Apply noise and reverse-sample the trends.
3. **Autoregression**: Estimate the next sequence by linearly guessing based on past trends.
4. **Loss Minimization**: Minimize the difference between predicted and actual trends across all levels.

By combining these steps, the model aims to produce robust predictions for the future sequence based on both historical data and diffusion-based transformations.



# `moving_avg` Class


The `moving_avg` class implements a 1D moving average filter using PyTorch’s `AvgPool1d` function. Below is a breakdown of the functionality, with mathematical notations and explanations that omit code.

---

### Moving Average Filter

A moving average filter smooths a time series by averaging the values within a sliding window, which moves over the input sequence. The filter size (window length) and stride (step size) define how the averaging is performed.

Let $X = \{x_1, x_2, \dots, x_T\}$ represent the input time series, where $x_t$ is the value of the series at time $t$. The moving average operation computes the average over a window of size $k$ for each time step, shifting by a stride $s$ after each calculation.

For a window of size $k$ and stride $s$, the moving average at time $t$ is given by:

$$
\text{MA}(x_t) = \frac{1}{k} \sum_{i=0}^{k-1} x_{t-i}
$$

This results in a smoothed sequence of values, reducing the effect of short-term fluctuations.

### Padding at the Sequence Boundaries

One challenge with applying a moving average is the treatment of boundary values, especially near the beginning and end of the time series. To address this, the `moving_avg` class replicates values at the boundaries:

- **Front Padding**: The value at $x_1$ (the first element) is repeated $\frac{k-1}{2}$ times to create the padded values at the start of the sequence.
- **End Padding**: Similarly, the value at $x_T$ (the last element) is repeated $\frac{k-1}{2}$ times at the end.

Thus, for a kernel size $k$, the padded sequence $X_{\text{pad}}$ becomes:

$$
X_{\text{pad}} = \{\underbrace{x_1, x_1, \dots}_{\frac{k-1}{2}}, x_1, x_2, \dots, x_T, \underbrace{x_T, x_T, \dots}_{\frac{k-1}{2}}\}
$$

This padding ensures that the moving average filter can be applied across the entire original sequence without losing any boundary points.

### Applying the Moving Average

Once the sequence is padded, the moving average filter is applied by averaging values within each sliding window of size $k$. The stride $s$ controls how far the window moves after each averaging operation.

Mathematically, the smoothed value for the $t$-th time step after padding is:

$$
y_t = \frac{1}{k} \sum_{i=t}^{t+k-1} X_{\text{pad}, i}
$$

where $y_t$ represents the value at the $t$-th time step of the smoothed sequence.

The moving average is applied along the time dimension of the input, and the result is a smoothed version of the original sequence with a reduced level of short-term variation.

### Permutation of Dimensions

In some cases, time series data might be represented in a specific format, such as:
- Batch size $B$,
- Number of features $N$,
- Sequence length $L$.

To apply the 1D pooling operation correctly, the input is often permuted to match the expected dimensions for PyTorch’s `AvgPool1d`, which operates on the last dimension of the input. After applying the moving average, the sequence is permuted back to its original form.

---

### Summary of Key Processes

1. **Padding**: The input sequence is padded at both ends to account for the boundary effects caused by the moving average window.
2. **Moving Average Calculation**: A sliding window of size $k$ is applied, where each window is averaged to produce a smoothed value for the corresponding time step.
3. **Dimension Handling**: The sequence dimensions are rearranged to match the expected format for 1D pooling, and then returned to the original format after the moving average is applied.

The moving average filter smooths the input sequence, reducing the impact of noise or short-term fluctuations, and is particularly useful for trend extraction in time-series analysis.