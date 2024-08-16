import os
import sys
import torch
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import yaml

# Add the parent directory to the sys.path
ProjectPath = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))

# Define default arguments
class Args:
    def __init__(self, config_filename):
        # assert config is YAML file
        assert config_filename.endswith('.yml')
        self.config_filename = config_filename
        
        # read config file
        self.config = self.read_config()
        
    def read_config(self):
        with open(self.config_filename, 'r') as file:
            config = yaml.safe_load(file)
        return config
    