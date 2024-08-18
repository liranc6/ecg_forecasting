CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/mrDiff/configs/config.yml'

# Cell 1: Import necessary libraries
import os
import sys
import torch
from torch.utils.data import DataLoader
import time
import random
import yaml
# Cell 1: Import necessary libraries
import os
import sys
import torch
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import yaml
from box import Box
import pprint

# Add the parent directory to the sys.path
ProjectPath = os.path.abspath(os.getcwd())
sys.path.append(ProjectPath)

# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

from mrDiff.exp.exp_main import Exp_Main

# Add the parent directory to the sys.path
ProjectPath = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))

from mrDiff.exp.exp_main import Exp_Main

from liran_project.mrdiff.main_default import Args

import argparse

# Create a mock parser
mock_parser = argparse.ArgumentParser(description='Mock Parser for Testing')

# Add arguments with values different from their defaults
mock_parser.add_argument('--random_seed', type=int, default=2023, help='random seed')
mock_parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
mock_parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset')
mock_parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
mock_parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')

# Mock arguments
cmd_args = mock_parser.parse_args([
    '--random_seed', '35',
    '--train_epochs', '300',
    '--dataset', 'ETTh2',
    '--learning_rate', '0.01',
    '--batch_size', '128'
])

# Print the parsed arguments
print(cmd_args)

args = Args(CONFIG_FILENAME, cmd_args)

print('Args in experiment:')
pprint.pprint(vars(args))