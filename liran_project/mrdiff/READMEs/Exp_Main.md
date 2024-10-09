# README

## Class Overview

This class represents a neural network model with various attributes and components that contribute to the structure of the model. Below is a breakdown of the attributes and their descriptions.
This structured README only focuses on the class attributes and their descriptions, with each section foldable for easy navigation.

## Class Attributes

<details>
<summary><strong>1. Args</strong></summary>

The `args` object holds the configuration parameters for the model, including device information and other settings.

```python
args: <liran_project.mrdiff.src.parser.Args object at 0x7ff67c48bee0>
```

</details>

<details>
<summary><strong>2. Device</strong></summary>

Indicates the device where the model is deployed (e.g., CPU or GPU).

```python
device: 'cuda:0'
```

</details>

<details>
<summary><strong>3. Model</strong></summary>

Defines the overall structure of the model, including its components like base models, decomposition layers, and U-Nets.

```python
model: Model(
  (base_models): ModuleList(
    (0-3): 4 x BaseMapping(
      (Linear_Trend): ModuleList(
        (0): Linear(in_features=10, out_features=30, bias=True)
      )
      (rev): RevIN()
    )
  )
  (decompositions): ModuleList(
    (0): series_decomp(
      (moving_avg): moving_avg(
        (avg): AvgPool1d(kernel_size=(5,), stride=(1,), padding=(0,))
      )
    )
  )
  (u_nets): ModuleList(
    (0-2): 3 x My_DiffusionUnet_v0(
      (diffusion_embedding): DiffusionEmbedding(
        (projection1): Linear(in_features=256, out_features=256, bias=True)
      )
    )
  )
)
```

</details>

<details>
<summary><strong>4. Datasets</strong></summary>

Contains the datasets for training and validation.

```python
datasets: {
  'train': <SingleLeadECGDatasetCrops_mrDiff object>,
  'val': <SingleLeadECGDatasetCrops_mrDiff object>
}
```

</details>

<details>
<summary><strong>5. Dataloaders</strong></summary>

Provides the dataloaders for iterating through the datasets.

```python
dataloaders: {
  'train': <torch.utils.data.dataloader.DataLoader object>,
  'val': <torch.utils.data.dataloader.DataLoader object>
}
```

</details>

<details>
<summary><strong>6. Model Start Training Time</strong></summary>

Tracks when the model begins training.

```python
model_start_training_time: None
```

</details>

<details>
<summary><strong>7. Model Components</strong></summary>

Details about the components used in the model, such as `PDSB`, `DDPM`, etc.

```python
set_models_using_meta: ['PDSB', 'DDPM']
```

</details>

## Method: `print_attributes`

This method is used to display the current attributes and their values.

```python
def print_attributes(self):
    for attr, value in self.__dict__.items():
        print(f"{attr}: {value}")
```

Here's a section to include in your README that describes which components to save to allow for resuming training in case of interruptions:


## Saving Components for Resuming Training

When training a neural network, it is crucial to save specific components to enable the continuation of training in case of interruptions (e.g., power outages, crashes, or manual stops). The following components should be saved:

### 1. Model Weights
- **Description**: The model's learned weights contain the parameters that the network has optimized during training.
- **Save Method**: Use the `torch.save()` function to save the state dictionary of the model.
  
```python
torch.save(model.state_dict(), 'model_weights.pth')
```

### 2. Optimizer State
- **Description**: The optimizer's state includes information on how the optimizer updates the model weights, including momentum and learning rate schedules.
- **Save Method**: Save the state of the optimizer similarly to the model weights.
  
```python
torch.save(optimizer.state_dict(), 'optimizer_state.pth')
```

### 3. Training Epoch
- **Description**: Keep track of the epoch at which the training was interrupted. This allows you to resume from the exact point without starting over.
- **Save Method**: Save this information in a separate file or in the same checkpoint file as the model and optimizer states.
  
```python
epoch = 10  # Example: last completed epoch
torch.save(epoch, 'training_epoch.pth')
```

### 4. Loss and Metrics
- **Description**: Save the current values of loss and any relevant metrics (e.g., accuracy) to monitor performance after resuming.
- **Save Method**: Store this information in a file or a log for easy retrieval.
  
```python
loss = 0.5  # Example: last recorded loss
torch.save(loss, 'last_loss.pth')
```

### 5. Learning Rate Scheduler State
- **Description**: If you are using a learning rate scheduler, save its state to maintain the learning rate schedule across training sessions.
- **Save Method**: Use the same approach as above to save the scheduler state.
  
```python
torch.save(scheduler.state_dict(), 'scheduler_state.pth')
```

### 6. Configuration Parameters
- **Description**: Save the configuration parameters used during training, such as batch size, learning rate, and any other hyperparameters.
- **Save Method**: Store this information in a configuration file (e.g., JSON or YAML) for easy access when restarting.
  
```yaml
learning_rate: 0.001
batch_size: 32
num_epochs: 50
```

## Example Checkpoint Saving Function

Here’s a simple function that can be used to save all necessary components:

```python
def save_checkpoint(model, optimizer, epoch, loss, scheduler, config):
    torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(optimizer.state_dict(), 'optimizer_state.pth')
    torch.save(epoch, 'training_epoch.pth')
    torch.save(loss, 'last_loss.pth')
    torch.save(scheduler.state_dict(), 'scheduler_state.pth')
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
```

By saving these components, you can easily resume training from the last checkpoint without losing progress. Be sure to regularly save checkpoints during long training sessions to minimize data loss in case of an interruption.
```

---

This section details the components to save and provides example code snippets for saving each component. You can integrate it into your existing README to ensure that users understand how to manage interruptions during training effectively. Let me know if you'd like any adjustments or additional information!
