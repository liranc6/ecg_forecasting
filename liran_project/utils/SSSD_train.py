"""
commands to run:
python SSSD_train.py --output_directory /path/to/save/model --config /path/to/config.json --ckpt_iter 0 --model SSSDS4Imputer --wandb_id your_wandb_id --world_size 4

example:
python SSSD_train.py --output_directory /home/liranc6/ecg/output_ecg_SSSD --config /home/liranc6/ecg/ecg_forecasting/SSSD_main/src/config/train_config.json --ckpt_iter 0 --model SSSDS4Imputer --world_size 4

"""




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
import argparse
import numpy as np
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp

wandb.login(key="25d3a6091764d64b1d2ed12613dc2444108c1cf7")

ProjectPath = "/home/liranc6/ecg/ecg_forecasting" #os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(ProjectPath)  # Add the parent directory to the sys.path

import liran_project.utils.dataset_loader as dataset_loader

from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops


sys.path.append('/home/liranc6/ecg/ecg_forecasting/SSSD_main')

from SSSD_main.src.utils.util import find_epoch, print_size, calc_diffusion_hyperparams, calculate_loss #, training_loss
from SSSD_main.src.utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_pred
import SSSD_main.src.utils.util as util
from SSSD_main.src.imputers.DiffWaveImputer import DiffWaveImputer
from SSSD_main.src.imputers.SSSDSAImputer import SSSDSAImputer
from SSSD_main.src.imputers.SSSDS4Imputer import SSSDS4Imputer

# Import your custom dataset class here
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops as CustomDataset


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4' 

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_new(rank, world_size, output_directory, ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging, learning_rate, only_generate_missing, masking, missing_k, model_name, net, diffusion_config, diffusion_hyperparams, wandb_config, train_config, context_size, label_size, batch_size):
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
    setup(rank, world_size)

    # Set device
    device = torch.device("cuda", rank)

    # Generate experiment (local) path
    window_info = "context{}_label{}".format(context_size, label_size)
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory,
                                    str(model_name),
                                    window_info,
                                    local_path)
    
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # Map diffusion hyperparameters to GPU
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # Predefine model
    net = net.cuda()

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    def load_checkpoint(output_directory, ckpt_iter):
        assert ckpt_iter >= 0, "Invalid checkpoint iteration"
        try:
            # Load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

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

            # Feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            return checkpoint["wandb_id"]
            
        except Exception as e:
            print(e)
            raise ValueError('Failed to load model at iteration {}'.format(ckpt_iter))

    print(f"{ckpt_iter=}")

    # Load checkpoint
    criterion = ckpt_iter
    wandb_id = None
    if criterion == 'max' or criterion == "best":
        print(f"{output_directory=}")
        ckpt_iter = find_epoch(output_directory, criterion=criterion)

    if ckpt_iter >= 0:
        found_checkpoint = False
        last_file = -1
        last_valid_ckpt = ckpt_iter
        while not found_checkpoint:
            try:
                last_valid_ckpt = find_epoch(output_directory, num=last_file, criterion=criterion)
                if last_valid_ckpt < 0:
                    print('No valid checkpoint model found, start training from initialization.')
                    break
                elif last_valid_ckpt <= ckpt_iter:
                    ckpt_iter = last_valid_ckpt
                    wandb_id = load_checkpoint(output_directory, ckpt_iter)
                    found_checkpoint = True
                else:
                    last_file -= 1
            except Exception as e:
                # print(e)
                last_file -= 1
                raise e
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # Initialize wandb with your project name and the relevant configuration
    wandb_project_name = f'ecg_SSSD_{model_name}'

    wandb_config["wandb_id"] = str(wandb_id)

    print(f"{wandb_id=}")

    if wandb_id:
        wandb.init(project=wandb_project_name, id=wandb_id, resume="must")
        print(wandb.run.id)

def main(rank, world_size, args):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_config_path = os.path.join(ProjectPath, 'SSSD_main', 'src','config','train_config.json')
    with open(train_config_path) as f:
        train_config = json.load(f)
    
    model_name = train_config["train_config"]["model_name"]
    train_config["output_directory"] = os.path.join(train_config["train_config"]["output_directory"], model_name)

    model_config_path = os.path.join(train_config["train_config"]["model_config_path"], f"config_{model_name}.json")

    with open(model_config_path) as f:
        model_config = json.load(f)

    # split the windows to fixed size context and label windows
    fs, context_num_minutes, context_num_secondes, label_window_num_minutes, label_window_num_secondes  = train_config["window_info"].values()

    context_window_size = (context_num_minutes*60 + context_num_secondes) * fs  # minutes * seconds * fs
    label_window_size = (label_window_num_minutes*60 + label_window_num_secondes) * fs  # minutes * seconds * fs
    window_size = context_window_size+label_window_size

    if model_name == "SSSDSA":
        context_window_size -= (context_window_size%4) # patch bug fix for SSSDSA impputator forward, the input size should be divisible by 4
        label_window_size -= (label_window_size%4)



    train_data_path = train_config["trainset_config"]["train_data_path"]
    # assert file exist
    assert os.path.isfile(train_data_path), f"{train_data_path=} does not exist"

    # Instantiate the class
    dataset = dataset_loader.SingleLeadECGDatasetCrops(context_window_size, label_window_size, train_data_path)
    batch_size = train_config["train_config"]["batch_size"]
    print(f"{batch_size=}")

    
    print(f"{batch_size * window_size=}")
    if model_name == "SSSDS4":
        max_batch_size = 31000
        assert max_batch_size >= batch_size * window_size, f"{max_batch_size=} should be greater than or equal to {batch_size * window_size=}"
    

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

         
    train_config["output_directory"] = os.path.join(train_config["output_directory"], train_config['train_config']['model_name'])
    trainset_config = {
        "train_data_path": train_config["trainset_config"]["train_data_path"],
        "test_data_path": train_config["trainset_config"]["test_data_path"],
        "segment_length": window_size,
        "sampling_rate": fs
    }

    with open(model_config_path) as f:
        config_SSSD_inner_model = json.load(f)
        
    if model_name == "SSSDS4":
        inner_model_config = config_SSSD_inner_model['wavenet_config']
    elif model_name == "SSSDSA":
        inner_model_config = config_SSSD_inner_model['sashimi_config']

    if model_name == "SSSDS4":
        net = SSSDS4Imputer(**inner_model_config).cuda()
    elif model_name == "SSSDSA":
        net = SSSDSAImputer(**inner_model_config).cuda()
    else:
        raise ValueError(f"model_name should be either 'SSSDS4' or 'SSSDSA', but got {model_name}")

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**train_config['diffusion_config'])


    # with open(args.config) as f:
    #     data = f.read()
    # config = json.loads(data)
    
    # net = eval(args.model)(**config["model_config_path"])
    # print_size(net)
    
    train_new(rank=rank,
              world_size=world_size, 
              output_directory=train_config["train_config"]["output_directory"],
              ckpt_iter=train_config["train_config"]['ckpt_iter'],
              n_iters= train_config["train_config"]['n_iters'],
              iters_per_ckpt=train_config["train_config"]['iters_per_ckpt'],
              iters_per_logging=train_config["train_config"]['iters_per_logging'],
              learning_rate=train_config["train_config"]['learning_rate'],
              only_generate_missing=train_config["train_config"]['only_generate_missing'],
              masking= train_config["train_config"]['masking'],
              missing_k=train_config["train_config"]['missing_k'],
              model_name = model_name,
              net=net,
              diffusion_config=train_config['diffusion_config'],
              diffusion_hyperparams = diffusion_hyperparams,
              train_config = train_config,
              context_size=context_window_size,
              label_size=label_window_size,
              batch_size=batch_size,
              wandb_config=wandb_config
              )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', default='/home/liranc6/ecg/output_ecg_SSSD', type=str, help='directory to save model')
    parser.add_argument('--config', default='SSSD_main/src/configs/ecg/default_ecg_config.json', type=str, help='JSON file for configuration')
    parser.add_argument('--ckpt_iter', default=0, type=int, help='checkpoint iteration')
    parser.add_argument('--model', default='SSSDS4Imputer', type=str, help='Model type: SSSDS4Imputer/SSSDSAImputer/DiffWaveImputer')
    parser.add_argument('--wandb_id', default=None, type=str, help='wandb id')
    parser.add_argument('--world_size', default=4, type=int, help='number of GPUs to use')
    args = parser.parse_args()
    
    world_size = args.world_size
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    cleanup()
