import yaml
import sys
import subprocess

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.dataset_loader import SingleLeadECGDatasetCrops_mrDiff as DataSet
from liran_project.utils.util import ecg_signal_difference, check_gpu_memory_usage, Debbuger
from liran_project.mrdiff.exp_main import Exp_Main
from liran_project.mrdiff.src.parser import Args
from liran_project.utils.common import *


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


def main():
    args = parse_args(CONFIG_FILENAME)
    
    pprint(vars(args))
    
    # Convert Box object to dictionary
    config_dict = args.configs.to_dict()

    # Access the configuration values using dictionary syntax
    random_seed = config_dict['general']['random_seed']
    tag = config_dict['general']['tag']
    dataset = config_dict['general']['dataset']
    features = config_dict['general']['features']

    learning_rate = config_dict['optimization']['learning_rate']
    batch_size = config_dict['optimization']['batch_size']

    context_len = config_dict['training']['sequence']['label_len']
    label_len = config_dict['training']['sequence']['pred_len']
    model = config_dict['training']['model_info']['model']
    pred_len = config_dict['training']['sequence']['pred_len']
    iterations = config_dict['training']['iterations']['itr']

    inverse = config_dict['data']['inverse']
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True
    
    if args.wandb.resume in ["allow", "must", "auto"] and args.resume_exp.resume is not True:
        raise ValueError("The 'resume' argument must be set to True in order to resume a wandb run.")
    
    # wandb
    wandb_project_name = args.wandb.project
    wandb_id = args.wandb.id if args.wandb.id != "None" else None
    wandb_mode = args.wandb.mode if args.wandb.mode != "None" else "online"
    wandb_resume = args.wandb.resume if args.wandb.resume != "None" else None
    
    # if args.wandb.resume != "None":
    #     wandb_id = args.wandb.id
    #     wandb_mode = args.wandb.mode
    #     wandb_resume = args.wandb.resume
    #     # wandb.init(project=project_name, id=wandb_id, resume="must", mode=wandb_mode)
    #     # wandb_init_config.update({
    #     #                         "id": args.wandb.resume,
    #     #                         "resume": args.wandb.resume
    #     #                         })
        
    if args.wandb.resume_from != "None":
        wandb_init_config ={
            "mode": args.wandb.mode,
            "project": args.wandb.project,
            "save_code": args.wandb.save_code,
        }
        
        wandb_init_config = {
                                "fork_from": args.wandb.resume_from
                            }

        wandb.init(resume_from = args.wandb.resume_from)

        # Extract the expected step from args.wandb.resume_from
        resume_from_parts = args.wandb.resume_from.split("?_step=")
        expected_step = int(resume_from_parts[1])  # Extract the step part and convert to integer

        # Assert that the run started from the specified step
        assert wandb.run.step == expected_step, f"Expected step {expected_step}, but got {wandb.run.step}"
                
        print(f"Resuming wandb run id: {wandb.run.id}")   
    else:
        wandb.init(project=wandb_project_name,
                   id = wandb_id,
                   mode = wandb_mode,
                   resume = wandb_resume,
                   config=args)
        print(f"New wandb run id: {wandb.run.id}")
    
    if args.wandb.save_code:
        wandb.save("main_default.py")
        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Construct the relative path
        relative_path = os.path.join(current_dir, "exp_main.py")

        # Save the file using the relative path
        wandb.save(relative_path)
    
        # Get the current Git commit ID
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')

        # Log the current Git commit ID
        wandb.config.update({"git_commit_id": commit_id}, allow_val_change=True)
        
    if args.wandb.resume != "None" or args.wandb.resume_from != "None":
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
        
        old_config = dict(wandb.config)
        wandb.config.update(args, allow_val_change=True)
        new_config = dict(wandb.config)
        log_config_diffs(old_config, new_config, step="update_args")
        
        
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
        
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>')
        with Debbuger(debug=args.debug):
            exp.train(setting)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        
        torch.cuda.empty_cache()
        
    

if __name__ == "__main__":
    main()