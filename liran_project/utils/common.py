TQDM_TYP = "terminal"

import os
import sys
import time
import wandb

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
from datetime import timedelta
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from pickle import FALSE
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
import bisect
from collections import OrderedDict
import threading

if TQDM_TYP == "terminal":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
    
CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.utils.util import check_gpu_memory_usage

