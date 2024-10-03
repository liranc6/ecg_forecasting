# DDPM_CNNNet File Documentation
[Link to file](../../../mrDiff/models_diffusion/DDPM_CNNNet.py)

# My_DiffusionUnet_v0 Class Documentation

The `My_DiffusionUnet_v0` class implements a U-Net-based architecture for time-series data, used for diffusion processes. Below is an explanation of its components along with mathematical formulations.

## Initialization: `__init__` method

The model takes several key parameters:
- `args`: Configuration object holding training settings and hyperparameters.
- $num\_vars$: The number of variables in the input data.
- $seq\_len$: The length of the input time series.
- $pred\_len$: The length of the predicted time series.

### Time Embedding
The model embeds the timesteps using the `DiffusionEmbedding` layer:
$$
\text{DiffusionEmbedding}(t) \in \mathbb{R}^{\text{dim\_diff\_step}}
$$
where $t$ is the time-step index and $\text{dim\_diff\_step}$ is the embedding dimension.

### Input Projection
Depending on the configuration, the input data is either projected using a feature projection (`feature_projection`) or directly processed by the `InputConvNetwork`. If the feature projection is enabled, the input $x_t$ of size $B \times N \times H$ is first passed through:
$$
x_t' = \text{feature\_projection}(x_t)
$$
where $B$ is the batch size, $N$ is the number of variables, and $H$ is the sequence length.

## Forward Method: `forward`

The core logic of the U-Net architecture lies in the `forward` method, where the input $x_t$ (the time series data), timesteps, and optional conditioning (`cond`) are processed.

### Diffusion Embedding
The diffusion embedding $e_t$ for each timestep is computed and activated using the formula:
$$
e_t = \text{act}(\text{diffusion\_embedding}(t))
$$
where the activation function is defined as:
$$
\text{act}(x) = x \cdot \sigma(x)
$$
where $\sigma(x)$ is the sigmoid function.

The embedding is then reshaped and repeated along the sequence dimension to match the input size.

### Projection and Encoding
The input $x_t$ is projected:
$$
x_t' = \text{input\_projection}(x_t)
$$
and concatenated with the diffusion embedding:
$$
\text{out} = \text{enc\_conv}([e_t, x_t'])
$$
This yields an intermediate encoding for further processing.

### Conditional Projections
The model supports both individual and shared conditioning projections based on whether individual projections are enabled in the configuration. If individual projections are enabled, the output for each variable $i$ is computed as:
$$
\text{pred\_out}_i = \text{cond\_projection}_i(\text{cond}_i)
$$
otherwise, the shared projection is used:
$$
\text{pred\_out} = \text{cond\_projection}(\text{cond})
$$

### Future Mixup
If the model is configured to use future mixup, a masking matrix $\mathbf{M}$ is applied to combine the predicted output with ground truth values $y_{\text{clean}}$:
$$
\hat{y} = \mathbf{M} \cdot \text{pred\_out} + (1 - \mathbf{M}) \cdot y_{\text{clean}}
$$
where $\mathbf{M}$ is either randomly generated or based on a beta distribution depending on the configuration.

### Combining and Residual
The final output is constructed by combining the predicted output, the encoding, and possibly an autoregressive initialization or previous scale output:
$$
\text{out} = \text{combine\_conv}([\text{out}, \text{pred\_out}, \text{ar\_init}, \text{prev\_scale\_out}])
$$
If residual connections are enabled, the output is adjusted by adding the predicted output:
$$
\text{out} = \text{out} + \text{pred\_out}
$$

## Summary
The U-Net architecture processes time-series data through layers of convolution and embedding, generating predictions conditioned on both the input sequence and optional autoregressive outputs. It leverages diffusion processes to refine the predictions over time.

---

# InputConvNetwork Class Documentation

The `InputConvNetwork` class implements a convolutional network with flexible layers, used primarily for feature extraction in time-series data or similar sequential inputs. Below is an explanation of its components along with mathematical formulations.

## Initialization: `__init__` Method

The constructor of the class initializes the following parameters:
- `args`: A configuration object containing training parameters.
- $inp\_num\_channel$: The number of input channels $C_{\text{in}}$.
- $out\_num\_channel$: The number of output channels $C_{\text{out}}$.
- $num\_layers$: The number of convolutional layers, defaulting to 3.
- $ddpm\_channels\_conv$: Specifies the number of intermediate channels for the convolutional layers. If not provided, the value is taken from `args`.

### Convolution Parameters
The convolutional layers use:
- Kernel size: $3$
- Padding: $1$

This results in the output size of each convolution layer being the same as the input size, since the padding and kernel size are set to maintain the dimensions.

## Convolutional Layers

The network is built as a sequence of convolutional layers followed by batch normalization, activation (Leaky ReLU), and dropout.

### Single-Layer Case
If the number of layers $L = 1$, the network is reduced to a single convolutional layer:
$$
\mathbf{y} = \text{Conv1d}(\mathbf{x}, C_{\text{in}}, C_{\text{out}}, \text{kernel\_size}=3, \text{stride}=1, \text{padding}=1)
$$
where:
- $\mathbf{x} \in \mathbb{R}^{B \times C_{\text{in}} \times T}$
- $\mathbf{y} \in \mathbb{R}^{B \times C_{\text{out}} \times T}$

### Multi-Layer Case
If $L > 1$, the following layers are stacked:
- The first layer uses the input channels $C_{\text{in}}$, and subsequent layers use intermediate channels $C_{\text{hidden}}$, until the final layer outputs $C_{\text{out}}$.

For $L > 1$, the layers are defined as:
1. **First layer:**
$$
\mathbf{y}_1 = \text{Conv1d}(\mathbf{x}, C_{\text{in}}, C_{\text{hidden}}, \text{kernel\_size}=3, \text{stride}=1, \text{padding}=1)
$$
2. **Intermediate layers:**
Each intermediate layer consists of:
$$
\mathbf{y}_i = \text{Conv1d}(\mathbf{y}_{i-1}, C_{\text{hidden}}, C_{\text{hidden}}, \text{kernel\_size}=3, \text{stride}=1, \text{padding}=1)
$$
and is followed by:
$$
\mathbf{y}_i = \text{BatchNorm1d}(\mathbf{y}_i)
$$
$$
\mathbf{y}_i = \text{LeakyReLU}(0.1)(\mathbf{y}_i)
$$
$$
\mathbf{y}_i = \text{Dropout}(p=0.1)(\mathbf{y}_i)
$$

3. **Final layer:**
The final layer outputs:
$$
\mathbf{y}_L = \text{Conv1d}(\mathbf{y}_{L-1}, C_{\text{hidden}}, C_{\text{out}}, \text{kernel\_size}=3, \text{stride}=1, \text{padding}=1)
$$

## Forward Pass: `forward`

The forward method processes input $x$ through the layers of the network sequentially:
$$
\mathbf{y} = f(x)
$$
where $f(x)$ is the composition of convolution, batch normalization, activation (LeakyReLU), and dropout, repeated for each layer.

The final output $y$ has dimensions:
$$
\mathbf{y} \in \mathbb{R}^{B \times C_{\text{out}} \times T}
$$
where:
- $B$ is the batch size.
- $T$ is the length of the time series.

## Summary
The `InputConvNetwork` is a flexible convolutional network with configurable layers that process sequential data through multiple convolutional layers, allowing for efficient feature extraction. Each layer includes convolution, batch normalization, activation, and dropout to improve performance and generalization.

---

# Conv1dWithInitialization Class Documentation

The `Conv1dWithInitialization` class implements a 1D convolutional layer with a specific weight initialization method. Below is the explanation with mathematical formulation.

## Initialization: `__init__` Method

The constructor initializes a 1D convolutional layer using `torch.nn.Conv1d` and applies orthogonal initialization to the layer’s weights.

- **Convolutional Layer**: This layer is instantiated using the arguments passed through `kwargs`. The arguments are passed to the `torch.nn.Conv1d` layer, which accepts the following:
  - $in\_channels$ ($C_{\text{in}}$): The number of input channels.
 

 - $out\_channels$ ($C_{\text{out}}$): The number of output channels.
  - $kernel\_size$: The size of the convolution filter.
  - $stride$: The step size for the filter.
  - $padding$: Padding added to the input.

The 1D convolution operation is defined as:
$$
y(t) = \sum_{c=1}^{C_{\text{in}}} \sum_{k=0}^{K-1} w_{c,k} x_c(t - k) + b
$$
where:
- $y(t)$ is the output at time $t$.
- $x_c(t - k)$ is the input from channel $c$ at time $t - k$.
- $w_{c,k}$ is the weight at position $k$ for channel $c$.
- $b$ is the bias term.

### Orthogonal Initialization

The layer weights are initialized using orthogonal initialization. Orthogonal initialization is applied to ensure that the weight matrix is orthonormal, helping to preserve variance during backpropagation and improving training stability.

For a convolutional layer, the weight matrix $W$ is initialized such that:
$$
W^T W = I
$$
where $I$ is the identity matrix.

The initialization is done using:
$$
\text{torch.nn.init.orthogonal\_}(W, \text{gain}=1)
$$
where `gain=1` keeps the scale of the initialization unchanged.

## Forward Pass: `forward`

The forward method applies the convolution operation on the input data $x$. The input $x$ is processed by the convolutional layer using the weights initialized in the `__init__` method.

For an input $x \in \mathbb{R}^{B \times C_{\text{in}} \times T}$, where:
- $B$ is the batch size,
- $C_{\text{in}}$ is the number of input channels,
- $T$ is the length of the time series,

The output $y \in \mathbb{R}^{B \times C_{\text{out}} \times T_{\text{out}}}$ is obtained as:
$$
y = \text{Conv1d}(x)
$$
where $T_{\text{out}}$ is the output length, depending on the convolution settings (kernel size, stride, padding).

## Summary
The `Conv1dWithInitialization` class is a simple extension of the standard 1D convolution that incorporates orthogonal weight initialization. This initialization technique ensures that the weights are orthonormal, which can improve convergence during training. The forward method simply applies the initialized convolutional layer to the input data.
