import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import random
import numpy as np
import yaml
from box import Box
from pprint import pprint
import wandb
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops_mrDiff as DataSet
from liran_project.utils.util import ecg_signal_difference, check_gpu_memory_usage
from liran_project.mrdiff.exp_main import Exp_Main

# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

# from mrDiff.exp.exp_main import Exp_Main
from mrDiff.data_process.etth_dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Wind, Dataset_Caiso, Dataset_Production, Dataset_Caiso_M, Dataset_Production_M
from mrDiff.data_process.financial_dataloader import DatasetH
from mrDiff.data_process.forecast_dataloader import ForecastDataset
from mrDiff.exp.exp_basic import Exp_Basic
from mrDiff.models_diffusion import DDPM
from mrDiff.utils.tools import EarlyStopping, adjust_learning_rate, visual
from mrDiff.utils.metrics import metric
from liran_project.mrdiff.src.parser import Args


def main():
    args = parse_args(CONFIG_FILENAME)
    
    pprint(vars(args))
    
    # Convert Box object to dictionary
    config_dict = args.config.to_dict()

    # Access the configuration values using dictionary syntax
    random_seed = config_dict['general']['random_seed']
    tag = config_dict['general']['tag']
    dataset = config_dict['general']['dataset']
    features = config_dict['general']['features']

    learning_rate = config_dict['optimization']['learning_rate']
    batch_size = config_dict['optimization']['batch_size']

    context_len = config_dict['training']['sequence']['context_len']
    label_len = config_dict['training']['sequence']['label_len']
    model = config_dict['training']['model_info']['model']
    pred_len = config_dict['training']['sequence']['pred_len']
    iterations = config_dict['training']['iterations']['itr']

    inverse = config_dict['data']['inverse']
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    
    # wandb
    wandb_init_config ={
            "mode": args.wandb.mode,
            "project": args.wandb.project,
            "save_code": args.wandb.save_code,
        }
    
    if args.wandb.resume != "None":
        wandb_init_config.update({
                                "id": args.wandb.resume,
                                "resume": args.wandb.resume
                                })
        
        if args.wandb.resume_from != "None":
            wandb_init_config["config"] = args.wandb.resume_from
            
        run = wandb.init(**wandb_init_config)
        print(f"Resuming wandb run id: {wandb.run.id}")
        
        def log_config_diffs(old_config, new_config, step):
            diffs = {}
            for key in new_config:
                if key not in old_config or old_config[key] != new_config[key]:
                    diffs[key] = {'old': old_config.get(key), 'new': new_config[key]}
        
            if diffs:
                note = f"Config changes at step {step}:\n"
                for key, value in diffs.items():
                    note += f"{key}: {value['old']} -> {value['new']}\n"
                wandb.run.notes = (wandb.run.notes or "") + note + "\n\nAdditional information added later:\n"
        
        old_config = wandb.config.copy()
        wandb.config.update(args)
        new_config = wandb.config.copy()
        log_config_diffs(old_config, new_config, step="update_args")
                
    else:
        wandb.init(**wandb_init_config, config=args)
        print(f"New wandb run id: {wandb.run.id}")
        
    fix_seed = random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    for iteration in range(iterations):
        # setting record of experiments

        # random seed
        # fix_seed = iteration if iterations > 1 else random_seed

        setting = f"{model}_{dataset}_ft{features}_sl{context_len}_ll{label_len}_pl{pred_len}_lr{learning_rate}_bs{batch_size}_inv{inverse}_itr{iteration}"
        
        if tag is not None:
            setting += f"_{tag}"

        exp = Exp_Main(args)
        
        exp.read_data('train')
        exp.read_data('val')
        exp.read_data('test')

        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
        
    

if __name__ == "__main__":
    main()