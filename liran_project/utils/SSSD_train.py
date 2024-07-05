import os
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import time
from datetime import timedelta
import h5py
from tqdm import tqdm
import numpy as np
import wandb
from collections import defaultdict

server = "newton"
server_config_path = os.path.join("/home/liranc6/ecg_forecasting/liran_project/utils/server_config.json"
                                  )

if server == "rambo":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# set server configuration
with open(server_config_path) as f:
    server_config = json.load(f)
    server_config = server_config[server]
    project_path = server_config['project_path']

sys.path.append(project_path)

from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops
from liran_project.utils.util import ecg_signal_difference

from SSSD_main.src.utils.util import find_epoch, print_size, calc_diffusion_hyperparams, calculate_loss, \
                                     get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_pred, sampling #, training_loss
from SSSD_main.src.imputers.SSSDSAImputer import SSSDSAImputer
from SSSD_main.src.imputers.SSSDS4Imputer import SSSDS4Imputer
# from SSSD_main.docs.instructions.PTB_XL.clinical_ts.timeseries_utils import NormalizeBatch


# Import your custom dataset class here
# from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops as CustomDataset

class NormalizeBatch(object):
    def __init__(self, input=True, channels=[]):
        self.channels = channels
        self.channels_keep = None
        self.input = input

    def __call__(self, sample):
        datax, labelx, static = sample
        data = datax if self.input else labelx

        batch_mean = torch.mean(data, dim=2, keepdim=True)
        batch_std = torch.std(data, dim=2, keepdim=True) + 1e-8

        if len(self.channels) > 0:
            if self.channels_keep is None:
                self.channels_keep = torch.tensor(list(set(range(data.shape[-1])) - set(self.channels)))

            batch_mean.index_fill_(0, self.channels_keep, 0)
            batch_std.index_fill_(0, self.channels_keep, 1)

        data = (data - batch_mean) / batch_std

        if self.input:
            return (data, labelx, static)
        else:
            return (datax, data, static)


def check_gpu_memory_usage(device_id=0):
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(device_id)
        gpu_name = device_properties.name
        total_memory = device_properties.total_memory / (1024 ** 3)  # Convert from bytes to GB
        reserved_memory = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # Convert from bytes to GB
        allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # Convert from bytes to GB
        free_memory = reserved_memory - allocated_memory  # Memory that is currently free

        print(f"GPU {device_id} ({gpu_name}) Memory Usage:")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Reserved Memory: {reserved_memory:.2f} GB")
        print(f"  Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  Free Memory: {free_memory:.2f} GB")

        return {
            'gpu_name': gpu_name,
            'total_memory_gb': total_memory,
            'reserved_memory_gb': reserved_memory,
            'allocated_memory_gb': allocated_memory,
            'free_memory_gb': free_memory
        }
    else:
        print("CUDA is not available.")
        return None
    
def train_new(output_directory,
          ckpt_iter, 
          n_iters, 
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          only_generate_missing,
          masking,
          missing_k,
          net,
          diffusion_config,
          diffusion_hyperparams,
          trainset_config,
          context_size,
          label_size,
          batch_size,
          train_dataset,
          val_dataset,
          wandb_mode,
          **kwargs):
    """
    Train Diffusion Models

    This function trains diffusion models using the given parameters.

    Parameters:
    output_directory (str):         Path to save model checkpoints.
    ckpt_iter (int or 'max'):       The pretrained checkpoint to be loaded. 
                                    If 'max' is selected, it automatically selects the maximum iteration.
    n_iters (int):                  Number of iterations to train.
    iters_per_ckpt (int):           Number of iterations to save checkpoint. 
                                    Default is 10k, for models with residual_channel=64 this number can be larger.
    iters_per_logging (int):        Number of iterations to save training log and compute validation loss. Default is 100.
    learning_rate (float):          Learning rate.
    use_model (int):                Model selection:
                                    0: DiffWave.
                                    1: SSSDSA.
                                    2: SSSDS4.
    only_generate_missing (int):    0: Apply diffusion to all samples.
                                    1: Only apply diffusion to missing portions of the signal.
    masking (str):                  Masking strategy:
                                    'mnr': Missing not at random.
                                    'bm': Blackout missing.
                                    'rm': Random missing.
    missing_k (int):                Number of missing time steps for each feature across the sample length.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # generate experiment (local) path
    window_info = "context{}_label{}".format(context_size, label_size)
    local_path = f"T{diffusion_config['T']}_beta0{diffusion_config['beta_0']}_betaT{diffusion_config['beta_T']}".replace('.','')

    current_time = time.strftime("%H_%M_%d_%m_%y")

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory,
                                    str(model_name),
                                    window_info,
                                    local_path,
                                    current_time)
    
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = net.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    def load_checkpoint(output_directory, ckpt_iter):
        assert ckpt_iter >= 0, "Invalid checkpoint iteration"
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            net.residual_layer.residual_blocks[35].S42.s4_layer.kernel.kernel.z = checkpoint['model_state_dict']['residual_layer.residual_blocks.35.S42.s4_layer.kernel.kernel.z']

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            return checkpoint["wandb_id"]
            
        except Exception as e:
            print(e)
            raise ValueError('Failed to load model at iteration {}'.format(ckpt_iter))

    print(f"{ckpt_iter=}")
            
    # load checkpoint
    critirion =  ckpt_iter
    wandb_id = None
    if critirion == 'max' or critirion == "best":
        print(f"{output_directory=}")
        ckpt_iter = find_epoch(output_directory, critirion=critirion)

    if ckpt_iter > 0:
        found_checkpoint = False
        last_file = -1
        last_valid_ckpt = ckpt_iter
        while not found_checkpoint:
            try:
                last_valid_ckpt = find_epoch(output_directory, num=last_file, critirion=critirion)
                if last_valid_ckpt < 0:
                    print('No valid checkpoint model found, start training from initialization.')

                elif last_valid_ckpt <= ckpt_iter:
                    ckpt_iter = last_valid_ckpt
                    wandb_id = load_checkpoint(output_directory, ckpt_iter)
                    found_checkpoint = True
                else:
                    last_file -= 1
            except Exception as e:
                print(e)
                last_file -= 1
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    if wandb_id:
        wandb.init(project="ecg_SSSD", id=wandb_id, resume="must", mode=wandb_mode, settings=wandb.Settings(code_dir="."))
        print(wandb.run.id)
    else:
        wandb.init(project=project_name, config=wandb_config, mode=wandb_mode, settings=wandb.Settings(code_dir="."))

    # Define the size of training and validation sets (e.g., 80% train, 20% validation)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size

    # Create indices for each split
    # indices = list(range(len(dataset)))
    # train_indices = indices[:train_size]
    # val_indices = indices[train_size:]

    # Create samplers for each split
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    start_time = time.time()
    
    # training
    epoch = ckpt_iter + 1
    pbar_outer = tqdm(total=n_iters, initial=ckpt_iter, position=0, leave=True)
    best_val_loss = float('inf')
    best_diffs = defaultdict(lambda: float("inf"))
    best_model_path = None

    normalizer = NormalizeBatch(input=True)

    while epoch < n_iters + 1:
        # torch.cuda.empty_cache()
        if epoch % 50 == 0:
            tqdm.write(f'epoch: {epoch}')
        for stage in ["train", "val"]:
            if stage == "train":
                net.train()
                train_losses = []
                pbar_inner = tqdm(enumerate(train_loader), total=len(train_loader), position=1, leave=True, dynamic_ncols=True)
            else: # validation
                net.eval()
                val_losses = []
                pbar_inner = tqdm(enumerate(val_loader), total=len(val_loader), position=1, leave=True, dynamic_ncols=True)
                total_diffs = defaultdict(lambda: 0)

            val_num_batches = float('inf')
            val_count_sampled_batches = 0
            for batch_i, batch in pbar_inner:
                
                assert context_size == batch[0].shape[-1], f"{context_size=} != {batch[0].shape[-1]=}"


                # Concatenate tensors along the last dimension
                concatenated_batch = torch.cat(batch, dim=-1)

                if stage == "train":
                    ecg_signals_batch = concatenated_batch
                    # ecg_signals_batch = normalizer((ecg_signals_batch.unsqueeze(1), None, None))[0].squeeze(1)
                elif stage == "val":
                    ecg_signals_batch = concatenated_batch[:, 0, :]
                    # ecg_signals_batch = normalizer((ecg_signals_batch.unsqueeze(1), None, None))[0].squeeze(1)

                    ecg_signals_labels = batch[1][:, 0, :]
                    # ecg_signals_labels = normalizer((ecg_signals_labels.unsqueeze(1), None, None))[0].squeeze(1)
                    ecg_R_beats_labels = batch[1][:, 1, :]
                    ecg_labels = torch.stack([ecg_signals_labels, ecg_R_beats_labels], dim=1)

                # Reshape to the desired shape
                ecg_signals_batch = ecg_signals_batch.unsqueeze(0)[:, :, :].permute(1, 0, 2).float().to(device)

                """
                copilot answer:
                In this code, masking is used to selectively ignore or pay attention to certain elements of the data during the training process.
                The mask is a tensor of the same shape as the input data, where each element of the mask corresponds to an element of the input data. 

                The type of mask applied depends on the `masking` variable, which can be 'rm', 'mnr', or 'bm'. Each of these values corresponds to a
                different masking strategy, implemented by the `get_mask_rm`, `get_mask_mnr`, and `get_mask_bm` functions respectively.

                Once the mask is created, it is permuted, repeated across the batch size, and converted to a float tensor on the GPU with `.float().cuda()`.
                The `loss_mask` is the logical negation of `mask`, converted to a boolean tensor with `.bool()`. 
                This means that wherever `mask` is True, `loss_mask` is False, and vice versa.

                The `mask` and `loss_mask` are then used in the `calculate_loss` function. While the exact usage depends on the implementation of
                `calculate_loss`, typically, elements of the input data where `mask` is True are ignored or treated differently during the computation
                of the loss. Conversely, elements where `loss_mask` is True are used normally. This allows the model to focus on certain parts of the
                data while ignoring others, which can be useful in many machine learning tasks.
                """


                transposed_mask = None
                if masking == 'rm':
                    transposed_mask = get_mask_rm(ecg_signals_batch[0], missing_k) # batch[0] is the first sample
                elif masking == 'mnr':
                    transposed_mask = get_mask_mnr(ecg_signals_batch[0], missing_k)
                elif masking == 'bm':
                    transposed_mask = get_mask_bm(ecg_signals_batch[0], missing_k)
                elif masking == 'pred':
                    transposed_mask = get_mask_pred(ecg_signals_batch[0], context_size, label_size)

                assert transposed_mask is not None, "Masking strategy not found"
                mask = transposed_mask #.permute(1, 0)
                mask = mask.repeat(ecg_signals_batch.size()[0], 1, 1).float().cuda()
                loss_mask = ~mask.bool()

                assert ecg_signals_batch.size() == mask.size() == loss_mask.size(), f'{ecg_signals_batch.size()=} {mask.size()=} {loss_mask.size()=}'

                ecg_signals_batch = ecg_signals_batch.to(device)
                mask = mask.to(device)
                loss_mask = loss_mask.to(device)
 
                X = ecg_signals_batch, ecg_signals_batch, mask, loss_mask #audio = X[0], cond = X[1], mask = X[2], loss_mask = X[3]

                if stage == "train":
                    # back-propagation
                    optimizer.zero_grad()
                    
                    train_loss = calculate_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                        only_generate_missing=only_generate_missing)
                    check_gpu_memory_usage()
                    
                    train_losses.append(train_loss.item())

                    train_loss.backward()
                    
                    optimizer.step()

                    # pbar_inner.set_description(f'Processing batch {i+1}')
                    pbar_inner.set_postfix({"training_loss": train_loss.item(),
                                                "iteration": epoch})
                    pbar_inner.update()

                else: # validation
                    with torch.no_grad():

                        if batch_i % 10 == 0: # take a few samples from each patient
                            val_loss = calculate_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                        only_generate_missing=only_generate_missing)
                            val_losses.append(val_loss.item())
                            pbar_inner.set_postfix({"validation_loss": val_loss.item(),
                                                    "iteration": epoch})
                            pbar_inner.update()

                            #calculate validation accuracy
                            print(f"generating ecg for validation batch {batch_i}")

                            if batch_i == 0 or epoch > 60:
                                val_count_sampled_batches += 1
                                generated_ecg = sampling(net,
                                                size=ecg_signals_batch.size(),
                                                diffusion_hyperparams=diffusion_hyperparams,
                                                cond=ecg_signals_batch,
                                                mask=mask,
                                                only_generate_missing=only_generate_missing
                                                )
                                # check_gpu_memory_usage()
                                print(f"generated_ecg shape: {generated_ecg.shape}")
                                
                                generated_ecg = generated_ecg[..., context_size:]

                                generated_ecg = generated_ecg.squeeze(1)

                                print("calculating accuracy")

                                curr_diffs = ecg_signal_difference(ecg_labels, generated_ecg, sampling_rate=trainset_config["sampling_rate"]) # return dtw_dist, mse_total, mae_total
                                for diff_name, val in curr_diffs.items():
                                    total_diffs[diff_name] += val
                

            if stage == "train":
                avg_train_loss = sum(train_losses) / len(train_losses)
                
            elif stage == "val":
                avg_val_loss = sum(val_losses) / len(val_losses)
                save_iter = False
                if avg_val_loss < best_val_loss:
                    save_iter = True
                for key, val in total_diffs.items():
                    if val < best_diffs[key]:
                        best_diffs[key] = val
                        save_iter = True
                        break
                if save_iter:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_directory, f'best_model:_iter:_{epoch}.pth')
                    save_model = {'model_state_dict': net.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'wandb_id': wandb.run.id}
                    save_model.update(total_diffs)
                    torch.save(save_model,
                                best_model_path)
                    tqdm.write(f'Validation loss improved at iteration {epoch}. Model saved at {best_model_path}')

                    
        elapsed_time = time.time() - start_time
        minuts_elapsed_time = int(elapsed_time)/60
        log = {key: val/val_count_sampled_batches for key, val in total_diffs.items()}
        log.update({"training_loss": avg_train_loss,
                   "validation_loss": avg_val_loss, 
                   "elapsed_time": minuts_elapsed_time , 
                   "iteration": epoch
                    })
        wandb.log(log)
        tqdm.write(f'Time elapsed: {str(timedelta(seconds=int(elapsed_time)))}')

        # save checkpoint
        if epoch > 0 and epoch % iters_per_ckpt == 0:
            checkpoint_name = '{}.pkl'.format(epoch)
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'wandb_id': wandb.run.id},
                    os.path.join(output_directory, checkpoint_name))
            tqdm.write(f'model at iteration {epoch} is saved')
            

        if epoch % iters_per_logging == 0:
                tqdm.write(f'iteration: {epoch} \tloss: {train_loss.item()}')
                wandb.save(best_model_path)
        epoch += 1
        pbar_outer.update()
    pbar_outer.close()
    total_time = time.time() - start_time
    print('Total time taken: ', str(timedelta(seconds=int(total_time))))

if __name__ == "__main__":

    torch.manual_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    train_config_path = os.path.join(project_path, 'SSSD_main', 'src','config','train_config.json')
    with open(train_config_path) as f:
        train_config = json.load(f)

    model_name = train_config["train_config"]["model_name"]
    train_config[f"output_directory_{server}"] = os.path.join(train_config["train_config"][f"output_directory_{server}"], model_name)

    model_config_path = os.path.join(train_config["train_config"][f"model_config_path_{server}"], f"config_{model_name}.json")

    with open(model_config_path) as f:
        model_config = json.load(f)

    # split the windows to fixed size context and label windows
    fs, context_num_minutes, context_num_secondes, label_window_num_minutes, label_window_num_secondes  = train_config["window_info"].values()

    context_window_size = (context_num_minutes*60 + context_num_secondes) * fs  # minutes * seconds * fs
    label_window_size = (label_window_num_minutes*60 + label_window_num_secondes) * fs  # minutes * seconds * fs
    window_size = context_window_size+label_window_size

    print(f"{context_num_secondes=}, {label_window_num_secondes=}")
    context_window_size -= (context_window_size%4) # patch bug fix for SSSDSA impputator forward, the input size should be divisible by 4
    label_window_size -= (label_window_size%4)


    idx_start_val = 33
    idx_start_test = 40
    idx_end_test = 46

    data_path = train_config["trainset_config"][f"data_path_{server}"]

    train_file = os.path.join(data_path,
                               "train",
                               f'p0_to_p{idx_start_val-1}.h5')
    
    val_file = os.path.join(data_path,
                                "val",
                                f'p{idx_start_val}_to_p{idx_start_test-1}.h5')
    test_file = os.path.join(data_path,
                                "test",
                                f'p{idx_start_test}_to_p{idx_end_test}.h5')

    # assert files exist
    assert os.path.isfile(train_file), f"{train_file=} does not exist"
    assert os.path.isfile(val_file), f"{val_file=} does not exist"
    assert os.path.isfile(test_file), f"{test_file=} does not exist"

    

    advance = 6*60*fs
    # Instantiate the class
    train_dataset = SingleLeadECGDatasetCrops(context_window_size, label_window_size, train_file, end_patiant=10)
    val_dataset = SingleLeadECGDatasetCrops(context_window_size, label_window_size, train_file, start_sample_from=advance, end_patiant=10, return_with_RR=True)

    assert len(train_dataset) > 0, f"{len(train_dataset)=} should be greater than 0"
    assert len(val_dataset) > 0, f"{len(val_dataset)=} should be greater than 0"


    batch_size = train_config["train_config"]["batch_size"]
    print(f"{batch_size=}")

    
    print(f"{batch_size * window_size=}")
    if model_name == "SSSDS4" and server == "rambo":
        max_batch_size = 31000
        assert max_batch_size >= batch_size * window_size, f"{max_batch_size=} should be greater than or equal to {batch_size * window_size=}"
    elif model_name == "SSSDSA":
        context_window_size -= (context_window_size%4) # patch bug fix for SSSDSA impputator forward, the input size should be divisible by 4
        label_window_size -= (label_window_size%4)

    

    wandb_config = {
                    "diffusion_config": train_config["diffusion_config"],
                    "train_config": train_config["train_config"],
                    "trainset_config": train_config["trainset_config"],
                    "iters_per_ckpt": train_config["train_config"]["iters_per_ckpt"],
                    "iters_per_logging": train_config["train_config"]["iters_per_logging"],
                    "n_iters": train_config["train_config"]["n_iters"],
                    "learning_rate": train_config["train_config"]["learning_rate"],
                    "model_name": model_name,
                    }
    
    # Initialize wandb with your project name and the relevant configuration
    project_name = f'ecg_SSSD_{model_name}'

         
    train_config[f"output_directory_{server}"] = os.path.join(train_config[f"output_directory_{server}"], train_config['train_config']['model_name'])
    trainset_config_SSSDS4 = {
        "test_data_path": train_config["trainset_config"]["test_data_path"],
        "segment_length": window_size,
        "sampling_rate": fs
    }

    if model_name == "SSSDS4":
        with open(model_config_path) as f:
            config_SSSDS4 = json.load(f)
        # Initialize your models and optimizers based on the chosen 'use_model'
        inner_model_config = config_SSSDS4['wavenet_config']
        net = SSSDS4Imputer(**inner_model_config).cuda()
    elif train_config['train_config']['model_name'] == "SSSDSA":
        with open(model_config_path) as f:
            config_SSSDSA = json.load(f)
        inner_model_config = config_SSSDSA['sashimi_config']
        net = SSSDSAImputer(**inner_model_config).cuda()


    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**train_config['diffusion_config'])

    train_new(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_directory=train_config["train_config"][f"output_directory_{server}"],
        ckpt_iter=train_config["train_config"]['ckpt_iter'],
        n_iters= train_config["train_config"]['n_iters'],
        iters_per_ckpt=train_config["train_config"]['iters_per_ckpt'],
        iters_per_logging=train_config["train_config"]['iters_per_logging'],
        learning_rate=train_config["train_config"]['learning_rate'],
        only_generate_missing=train_config["train_config"]['only_generate_missing'],
        masking= train_config["train_config"]['masking'],
        missing_k=train_config["train_config"]['missing_k'],
        net=net,
        diffusion_config=train_config['diffusion_config'],
        diffusion_hyperparams = diffusion_hyperparams,
        trainset_config = trainset_config_SSSDS4,
        context_size=context_window_size,
        label_size=label_window_size,
        batch_size=batch_size,
        wandb_mode=train_config["train_config"]["wandb_mode"]
        )

    wandb.finish()