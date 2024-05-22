import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import h5py
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

ProjectPath = "/home/liranc6/ecg/ecg_forecasting" #os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(ProjectPath)  # Add the parent directory to the sys.path

import liran_project.utils.dataset_loader as dataset_loader

from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops

sys.path.append('/home/liranc6/ecg/ecg_forecasting/SSSD_main')

from SSSD_main.src.utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from SSSD_main.src.utils.util import get_mask_mnr, get_mask_bm, get_mask_rm, get_mask_pred

import SSSD_main.src.utils.util as util
from SSSD_main.src.imputers.DiffWaveImputer import DiffWaveImputer
from SSSD_main.src.imputers.SSSDSAImputer import SSSDSAImputer
from SSSD_main.src.imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from statistics import mean


def generate(output_directory,
            ckpt_path,
            data_path,
            ckpt_iter,
            use_model,
            masking,
            missing_k,
            only_generate_missing,
            context_size,
            label_size,
            batch_size,
            **kwargs):
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
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
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)
    
    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, 
                             window_info,
                             local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(model_path, map_location='cpu')
    try:
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except Exception as e:
        print(f"{e=}")
        raise Exception('No valid model found')

     # Specify the path to the H5 file
    file_path = data_path
    # Load data from the first dataset
    dataset = SingleLeadECGDatasetCrops(context_size, label_size, file_path)
    # Use DataLoader to handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    print('Data loaded')

    all_mse = []

    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True, dynamic_ncols=True):

        if i > 0:
            break
        
        original_batch = torch.cat(batch, dim=-1).unsqueeze(0)[:, :, :].permute(1,0,2).float()
        # create a zero tensore with the same size as batch[1]
        to_pred = torch.zeros(batch[1].size())

        #concat batch[0] and pred
        batch = torch.cat((batch[0], to_pred), dim=1).unsqueeze(0)[:, :, :].float().to(device)

        if masking == 'mnr':
            mask_T = get_mask_mnr(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'bm':
            mask_T = get_mask_bm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == 'rm':
            mask_T = get_mask_rm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()

        ##pred mask
        elif masking == 'pred':
            mask = get_mask_pred(batch[0], context_size=context_size, pred_size=label_size)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda() #?
            
        batch = batch.permute(1,0,2)
        mask = mask.permute(1,0,2)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        generated_audio = sampling(net, 
                                   size=batch.size(),
                                   diffusion_hyperparams=diffusion_hyperparams,
                                   cond=batch,
                                   mask=mask,
                                   only_generate_missing=only_generate_missing)

        end.record()
        torch.cuda.synchronize()

        print('generated utterances of random_digit at iteration {} in {} seconds'.format(ckpt_iter,
                                                                                            int(start.elapsed_time(
                                                                                            end) / 1000)))

        
        batch = original_batch
        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy() 
        
        
        outfile = f'imputation{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, generated_audio)

        outfile = f'original{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, batch)

        outfile = f'mask{i}.npy'
        new_out = os.path.join(ckpt_path, outfile)
        np.save(new_out, mask)

        print('saved generated samples at iteration %s' % ckpt_iter)
        
        mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        all_mse.append(mse)
    
    print('Total MSE:', mean(all_mse))

def visualize_prediction(actual, predicted):
    plt.figure(figsize=(10, 5))  # Create a new figure
    plt.plot(actual, color='red', label='Actual')  # Plot the actual data in blue
    plt.plot(predicted, color='blue', label='Predicted')  # Plot the predicted data in red
    plt.title('Actual vs Predicted')  # Set the title of the plot
    plt.xlabel('Time')  # Set the label for the x-axis
    plt.ylabel('Value')  # Set the label for the y-axis
    plt.legend()  # Show the legend
    plt.show()  # Display the plot

def autoegressive_predictor(net, data_path, use_model, input, singel_window_size, num_window_to_pred, diffusion_hyperparams, only_generate_missing):
    window_info = "context{}_label{}".format(len(input), singel_window_size)
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
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, 
                             window_info,
                             local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    checkpoint = torch.load(model_path, map_location='cpu')
    try:
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except Exception as e:
        print(f"{e=}")
        raise Exception('No valid model found')
    
    input = torch.from_numpy(input).float().cuda()


    print('Data loaded')
    

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    config_file = "/home/liranc6/ecg/ecg_forecasting/SSSD_main/src/config/config_SSSDS4.json"
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)

    gen_config = config['gen_config']

    train_config =  config["train_config"]  # training parameters

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']


    # split the windows to fixed size context and label windows
    fs = 250
    context_num_minutes = 0
    context_num_secondes = 7
    label_window_num_minutes = 0
    label_window_num_secondes = 4

    context_window_size = (context_num_minutes*60 + context_num_secondes) * fs  # minutes * seconds * fs
    label_window_size = (label_window_num_minutes*60 + label_window_num_secondes) * fs  # minutes * seconds * fs
    window_size = context_window_size+label_window_size

    ten_minutes_window_file = '/mnt/qnap/liranc6/data/icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits/10minutes_window.h5'

    # Instantiate the class
    dataset = dataset_loader.SingleLeadECGDatasetCrops(context_window_size, label_window_size, ten_minutes_window_file)
    batch_size = 10
    print(f"{batch_size=}")

    max_batch_size = 27500
    assert max_batch_size >= batch_size * window_size, "max_batch_size should be greater than or equal to batch_size * window_size"
    print(batch_size * window_size)


    generate(**gen_config,
            ckpt_iter="55",
            use_model=train_config["use_model"],
            data_path= "/mnt/qnap/liranc6/data/icentia11k-continuous-ecg_new_subsets/icentia11k-continuous-ecg_normal_sinus_subset_npArrays_splits/10minutes_window.h5",
            masking=train_config["masking"],
            missing_k=train_config["missing_k"],
            only_generate_missing=train_config["only_generate_missing"],
            context_size=context_window_size,
            label_size=label_window_size,
            batch_size=batch_size
                )
