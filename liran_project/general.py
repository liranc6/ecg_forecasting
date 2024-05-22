import itertools
import os
import wfdb
from tqdm import tqdm
import pandas as pd
import json
import os
import collections
import numpy as np
import glob
import h5py
import sys
import torch
import pickle
sys.path.append('..')
import utils.data_preparation as data_preparation
import time
import wandb

ProjectPath = "/home/liranc6/ecg/ecg_forecasting"

import os
import wandb

def upload_logs_to_project(directory_path, project_name, run_id):
    # Initialize W&B with the project name where you want to upload the log files
    wandb.init(project=project_name, id=run_id, resume="allow")
    
    # Log files from the specified run directory
    log_files_path = os.path.join(directory_path, "files")
    log_files = os.listdir(log_files_path)
    for log_file in log_files:
        # Load and log the contents of each log file to the W&B run
        with open(os.path.join(log_files_path, log_file), 'r') as file:
            for line in file:
                # Split the line using the colon (:) delimiter
                line_parts = line.strip().split(':')
                # Check if the line has the expected number of parts
                if len(line_parts) == 2:
                    metric_name, metric_value = line_parts
                    # Log the metric to the W&B run
                    wandb.log({metric_name: float(metric_value)})
                else:
                    # Handle the case where the line doesn't contain the expected number of values
                    # For example, you can skip the line or log a warning
                    print(f"Warning: Invalid line format in {log_file}: {line}")

    # Finish the W&B run
    wandb.finish()


def main():
    directory_path = "/home/liranc6/ecg/ecg_forecasting/wandb/run-20240317_184130-2j5yadob"
    project_name = "ecg_SSSD_SSSDSA"
    run_id = "2j5yadob"
    
    upload_logs_to_project(directory_path, project_name, run_id)

if __name__ == "__main__":
    main()
