import yaml
import sys
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.util import Debbuger
from liran_project.mrdiff.exp_main import Exp_Main, ExpMainLightning
from liran_project.utils.common import *


# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.best_metrics = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                if metric_name.startswith('val_'):
                    if metric_name not in self.best_metrics or metric_value < self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        print(f"Improved {metric_name}: {metric_value:.4f}")
                        if metric_name in ["val_loss", "val_mean_extra_r_beats", "val_dtw_dist"]:
                            save_path = os.path.join(self.dirpath, f"{metric_name}.ckpt")
                            print(f"Saving checkpoint for {metric_name} to file {save_path}")
                            self.save_checkpoint(trainer, pl_module, save_path)

    def save_checkpoint(self, trainer, pl_module, save_path):
        # Save the checkpoint to the specified path
        trainer.save_checkpoint(save_path)


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.best_metrics = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                if metric_name.startswith('val_'):
                    if metric_name not in self.best_metrics or metric_value < self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        print(f"Improved {metric_name}: {metric_value:.4f}")
            super(CustomEarlyStopping, self).on_validation_end(trainer, pl_module)       
        
                                
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
        wandb_logger = WandbLogger(project=wandb_project_name, id = wandb_id,)
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
        wandb.config.update({"git_commit_id": commit_id})
        print(f"Git commit ID: {commit_id}")
        
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
        
        
    pl.seed_everything(random_seed)
    
    for iteration in range(iterations):
        # setting record of experiments

        # random seed
        # fix_seed = iteration if iterations > 1 else random_seed

        setting = f"{model}_{dataset}_ft{features}_sl{context_len}_ll{label_len}_pl{pred_len}_lr{learning_rate}_bs{batch_size}_inv{inverse}_itr{iteration}"
        
        if tag is not None:
            setting += f"_{tag}"

        time_now = time.time()
            
        str_time_now = time.strftime("%d_%m_%Y_%H%M", time.localtime(time_now))
        model_start_training_time = str_time_now
        args.paths.checkpoints = os.path.join(args.paths.checkpoints, setting, model_start_training_time)
        exp = ExpMainLightning(args)
        
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>')
        with Debbuger(debug=args.debug):
            trainer = Trainer(
                devices=2,
                accelerator="cuda",
                max_epochs=args.training.iterations.train_epochs,
                num_sanity_val_steps=0,  # Disable sanity check
                # use_distributed_sampler = True,
                callbacks=[
                    CustomModelCheckpoint(
                        dirpath=args.paths.checkpoints,
                        filename='{epoch}-{val_loss:.2f}',
                        save_top_k=1,
                        monitor='val_loss',
                        mode='min',
                        start_epoch=2
                    ),
                    CustomEarlyStopping(
                        monitor='val_loss',
                        patience=args.optimization.patience,
                        mode='min',
                        start_epoch=2,
                    ),
                    ModelSummary(max_depth=3)
                ],
                strategy=FSDPStrategy(
                                        sharding_strategy="FULL_SHARD",
                                        # cpu_offload=True,
                                      ), #DDPStrategy(find_unused_parameters=True)
            )
            trainer.fit(exp)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        trainer.test(exp)
        
        torch.cuda.empty_cache()
        
    

if __name__ == "__main__":
    main()
    
    
    