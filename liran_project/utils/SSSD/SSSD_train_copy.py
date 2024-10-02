import os
import json
import sys
import torch
from torch.utils.data import DataLoader
ProjectPath = "/home/liranc6/ecg/ecg_forecasting" #os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(ProjectPath)  # Add the parent directory to the sys.path

import liran_project.utils.dataset_loader as dataset_loader

import liran_project.train as liran_train
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops

import liran_project.train as src_train
import h5py
from tqdm import tqdm

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import sys

sys.path.append('/home/liranc6/ecg/ecg_forecasting/SSSD_main')

from SSSD_main.src.utils.util import find_max_epoch, print_size, calc_diffusion_hyperparams #, training_loss
from SSSD_main.src.utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_pred
import SSSD_main.src.utils.util as util
from SSSD_main.src.imputers.DiffWaveImputer import DiffWaveImputer
from SSSD_main.src.imputers.SSSDSAImputer import SSSDSAImputer
from SSSD_main.src.imputers.SSSDS4Imputer import SSSDS4Imputer

# Import your custom dataset class here
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops as CustomDataset


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

#working train, tries to use my castume dataset
#working for in_channels =65000 (which is a bit over 4.3 minuts in total, context+pred)
#I interapted bc it has 788 samples and I didnt want to wait, next task will be to check if I can run it fully to the end.
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

    # generate experiment (local) path
    window_info = "context{}_label{}".format(context_size, label_size)
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory,
                                    window_info,
                                     local_path)
    
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
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

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            
        except Exception as e:
            print(e)
            raise ValueError('Failed to load model at iteration {}'.format(ckpt_iter))



    print(f"{ckpt_iter=}")
            
    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        found_checkpoint = False
        last_file = -1
        last_valid_ckpt = ckpt_iter
        while not found_checkpoint:
            try:
                last_valid_ckpt = find_max_epoch(output_directory, last_file)
                if last_valid_ckpt < 0:
                    print('No valid checkpoint model found, start training from initialization.')
                    break
                elif last_valid_ckpt <= ckpt_iter:
                    ckpt_iter = last_valid_ckpt
                    load_checkpoint(output_directory, ckpt_iter)
                    found_checkpoint = True
                else:
                    last_file -= 1
            except Exception as e:
                print(e)
                last_file -= 1
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')


    # Specify the path to the H5 file
    file_path = "/mnt/qnap/liranc6/data/icentia11k-continuous-ecg_new_subsets/icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits/10minutes_window.h5"
    # Load data from the first dataset
    dataset = SingleLeadECGDatasetCrops(context_size, label_size, file_path)
    # Use DataLoader to handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # # Convert the NumPy array to a PyTorch tensor, make it a float tensor,
    # # and move it to the GPU if available
    # training_data = torch.from_numpy(first_dataset).float().cuda()
    # # print(f'{training_data.shape=}')
    # training_data = training_data[:200, :wavenet_config_SSSDS4["in_channels"]] # [:x, :y] take only the first x patients and y time steps to save memory
    # training_data.unsqueeze_(0) #split the data to batches of size 1
    # training_data.unsqueeze_(-1) #add a channel dimension
    
    # training_data = training_data.permute(0, 1, 3, 2) # the code expects the data to be in the shape of (batch_size, sequence_length, channels)
    
    # print(f'{training_data.shape=}')
    

    # training
    n_iter = ckpt_iter + 1
    pbar_outer = tqdm(total=n_iters, initial=ckpt_iter, position=0, leave=True)
    while n_iter < n_iters + 1:
        if n_iter % 50 == 0:
            tqdm.write(f'n_iter: {n_iter}')
        pbar_inner = tqdm(enumerate(dataloader), total=len(dataloader), position=1, leave=True, dynamic_ncols=True)
        for i, batch in pbar_inner:

            if i>50:
                break

            # Concatenate tensors along the last dimension
            concatenated_batch = torch.cat(batch, dim=-1)

            # Reshape to the desired shape
            batch = concatenated_batch.unsqueeze(0)[:, :, :].permute(1, 0, 2).float().to(device)

            # print(f"{batch=}\n"
            #     f"{len(batch)=}\n"
            #     f"{batch.size()=}")

            # if i % 10 == 0:
            #     print(f'{i=}')

            # TODO: what is the porpuse and use of the masking in here?
            """
            copilot answer:
            In this code, masking is used to selectively ignore or pay attention to certain elements of the data during the training process.
            The mask is a tensor of the same shape as the input data, where each element of the mask corresponds to an element of the input data. 

            The type of mask applied depends on the `masking` variable, which can be 'rm', 'mnr', or 'bm'. Each of these values corresponds to a
            different masking strategy, implemented by the `get_mask_rm`, `get_mask_mnr`, and `get_mask_bm` functions respectively.

            Once the mask is created, it is permuted, repeated across the batch size, and converted to a float tensor on the GPU with `.float().cuda()`.
            The `loss_mask` is the logical negation of `mask`, converted to a boolean tensor with `.bool()`. 
            This means that wherever `mask` is True, `loss_mask` is False, and vice versa.

            The `mask` and `loss_mask` are then used in the `training_loss` function. While the exact usage depends on the implementation of
            `training_loss`, typically, elements of the input data where `mask` is True are ignored or treated differently during the computation
            of the loss. Conversely, elements where `loss_mask` is True are used normally. This allows the model to focus on certain parts of the
            data while ignoring others, which can be useful in many machine learning tasks.
            """


            transposed_mask = None
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k) # batch[0] is the first sample
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)
            elif masking == 'pred':
                transposed_mask = get_mask_pred(batch[0], context_size, label_size)

            assert transposed_mask is not None, "Masking strategy not found"
            mask = transposed_mask #.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch #.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size(), f'{batch.size()=} {mask.size()=} {loss_mask.size()=}'
            
            # tqdm.write(f"{batch.size()=} == {mask.size()=} == {loss_mask.size()=}")
            # assert transposed_mask is not None, "Masking strategy not found"
            # mask = transposed_mask.permute(0, 2, 1)  # Changed this line
            # mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            # loss_mask = ~mask.bool()
            # batch = batch.permute(0, 2, 1)
            # 
            # assert batch.size() == mask.size() == loss_mask.size(), f'{batch.size()=} {mask.size()=} {loss_mask.size()=}'

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask #audio = X[0], cond = X[1], mask = X[2], loss_mask = X[3]
            
            loss = src_train.training_loss(net, nn.MSELoss(), X, diffusion_hyperparams,
                                only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(output_directory, checkpoint_name))
                tqdm.write(f'model at iteration {n_iter} is saved')
                
            # pbar_inner.set_description(f'Processing batch {i+1}')
            pbar_inner.set_postfix({"loss": loss.item(),
                                     "iteration": n_iter})
            pbar_inner.update()

        if n_iter % iters_per_logging == 0:
                tqdm.write(f'iteration: {n_iter} \tloss: {loss.item()}')
        n_iter += 1
        pbar_outer.update()
    pbar_outer.close()

if __name__ == "__main__":

    subset_data_dir = "/mnt/qnap/liranc6/data/icentia11k-continuous-ecg_normal_sinus_subset/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    # split the windows to fixed size context and label windows
    fs = 250
    context_num_minutes = 0
    context_num_secondes = 9
    label_window_num_minutes = 0
    label_window_num_secondes = 1

    context_window_size = (context_num_minutes*60 + context_num_secondes) * fs  # minutes * seconds * fs
    label_window_size = (label_window_num_minutes*60 + label_window_num_secondes) * fs  # minutes * seconds * fs
    window_size = context_window_size+label_window_size



    ten_minutes_window_file = '/mnt/qnap/liranc6/data/icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits/10minutes_window.h5'

    # Instantiate the class
    dataset = dataset_loader.SingleLeadECGDatasetCrops(context_window_size, label_window_size, ten_minutes_window_file)
    batch_size = 10
    print(f"{batch_size=}")

    max_batch_size = 25000
    assert max_batch_size >= batch_size * window_size, "max_batch_size should be greater than or equal to batch_size * window_size"
    print(batch_size * window_size)

    # Load the configuration files
    config_SSSDS4_path = os.path.join(ProjectPath, 'SSSD_main', 'src','config','config_SSSDS4.json') 
    config_SSSDSA_path = os.path.join(ProjectPath, 'SSSD_main', 'src','config','config_SSSDSA.json') 

    with open(config_SSSDS4_path) as f:
        config_SSSDS4 = json.load(f)

    with open(config_SSSDSA_path) as f:
        config_SSSDSA = json.load(f)

    # Parse necessary configurations for SSSDS4
    gen_config_SSSDS4 = config_SSSDS4['gen_config']
    train_config_SSSDS4 = config_SSSDS4['train_config']
    trainset_config_SSSDS4 = config_SSSDS4['trainset_config']
    diffusion_config_SSSDS4 = config_SSSDS4['diffusion_config']
    wavenet_config_SSSDS4 = config_SSSDS4['wavenet_config']

    # # Parse necessary configurations for SSSDSA
    # gen_config_SSSDSA = config_SSSDSA['gen_config']
    # train_config_SSSDSA = config_SSSDSA['train_config']
    # trainset_config_SSSDSA = config_SSSDSA['trainset_config']
    # diffusion_config_SSSDSA = config_SSSDSA['diffusion_config']
    # sashimi_config_SSSDSA = config_SSSDSA['sashimi_config']

    # Load your custom datasets
    # train_dataset_SSSDS4 = dataset
    # train_loader_SSSDS4 = DataLoader(train_dataset_SSSDS4, batch_size=batch_size, shuffle=True, num_workers=4)

    # train_dataset_SSSDSA = dataset
    # train_loader_SSSDSA = DataLoader(train_dataset_SSSDSA, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize your models and optimizers based on the chosen 'use_model'
    net_SSSDS4 = SSSDS4Imputer(**wavenet_config_SSSDS4).cuda()
    optimizer_SSSDS4 = torch.optim.Adam(net_SSSDS4.parameters(), lr=train_config_SSSDS4['learning_rate'])

    # net_SSSDSA = SSSDSAImputer(**sashimi_config_SSSDSA).cuda()
    # optimizer_SSSDSA = torch.optim.Adam(net_SSSDSA.parameters(), lr=train_config_SSSDSA['learning_rate'])

    # # Load checkpoints if available for both models
    # ckpt_path_SSSDS4 = os.path.join(train_config_SSSDS4["output_directory"], "T{}_beta0{}_betaT{}".format(
    #     diffusion_config_SSSDS4["T"], diffusion_config_SSSDS4["beta_0"], diffusion_config_SSSDS4["beta_T"]))
    # ckpt_path_SSSDSA = train_config_SSSDSA["output_directory"]

    # args = type('Arguments', (object,), {'ckpt_iter': 'max'})  # Mock argparse arguments
    # args.ckpt_iter = 'max'

    # model_path_SSSDS4 = os.path.join(ckpt_path_SSSDS4, '{}.pkl'.format(args.ckpt_iter))
    # model_path_SSSDSA = os.path.join(ckpt_path_SSSDSA, '{}.pkl'.format(args.ckpt_iter))

    # try:
    #     checkpoint_SSSDS4 = torch.load(model_path_SSSDS4, map_location='cpu')
    #     net_SSSDS4.load_state_dict(checkpoint_SSSDS4['model_state_dict'])
    #     optimizer_SSSDS4.load_state_dict(checkpoint_SSSDS4['optimizer_state_dict'])
    #     print('Successfully loaded SSSDS4 model at iteration {}'.format(args.ckpt_iter))
    # except:
    #     print('No valid SSSDS4 model found. Initializing from scratch.')
    # try:
    #     checkpoint_SSSDSA = torch.load(model_path_SSSDSA, map_location='cpu')
    #     net_SSSDSA.load_state_dict(checkpoint_SSSDSA['model_state_dict'])
    #     optimizer_SSSDSA.load_state_dict(checkpoint_SSSDSA['optimizer_state_dict'])
    #     print('Successfully loaded SSSDSA model at iteration {}'.format(args.ckpt_iter))
    # except:
    #     print('No valid SSSDSA model found. Initializing from scratch.')

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config_SSSDS4)
    output_directory = "/home/liranc6/ecg/ecg_forecasting/liran_project/results/icentia11k/SSSDS4_copy/"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        train_new(
            output_directory=output_directory,
            ckpt_iter=train_config_SSSDS4['ckpt_iter'],
            n_iters= 2,
            iters_per_ckpt=train_config_SSSDS4['iters_per_ckpt'],
            iters_per_logging=train_config_SSSDS4['iters_per_logging'],
            learning_rate=train_config_SSSDS4['learning_rate'],
            only_generate_missing=train_config_SSSDS4['only_generate_missing'],
            masking= train_config_SSSDS4['masking'],
            missing_k=train_config_SSSDS4['missing_k'],
            net=net_SSSDS4,
            diffusion_config=diffusion_config_SSSDS4,
            diffusion_hyperparams = diffusion_hyperparams,
            trainset_config = trainset_config_SSSDS4,
            context_size=context_window_size,
            label_size=label_window_size
            )
        
    prof.export_chrome_trace(os.path.join(output_directory, "trace.json"))



