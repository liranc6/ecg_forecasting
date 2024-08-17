import argparse
import os
import torch
import random
import numpy as np
import yaml
from box import Box
import sys
from pprint import pprint

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/mrDiff/configs/config.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

from mrDiff.exp.exp_main import Exp_Main

from liran_project.mrdiff.src.parser import parse_args

def main():
    args = parse_args()
    # Now you can use args as needed
    pprint(vars(args))

if __name__ == "__main__":
    main()