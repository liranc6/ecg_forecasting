import yaml
import sys

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.utils.common import *
from liran_project.mrdiff.src.parser import parse_args

class Forecaster:
    def __init__(self, cpt_file):
        self.args = parse_args(CONFIG_FILENAME)
        self._read_cpt_file(cpt_file)
    
    
    def _read_cpt_file(self, cpt_file):
        assert os.path.exists(cpt_file), f"Checkpoint file {cpt_file} does not exist"
        
        cpt_data = torch.load(cpt_file, map_location='cpu')
        self.args.update_config_from_dict(cpt_data['configuration_parameters'])

    def forecast(self, data):
        return self.model.forecast(data)