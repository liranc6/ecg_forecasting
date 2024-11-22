import yaml
import sys
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.tuner import Tuner

import time

CONFIG_FILENAME = '/home/liranc6/ecg_forecasting/liran_project/mrdiff/src/config_ecg.yml'

assert CONFIG_FILENAME.endswith('.yml')

with open(CONFIG_FILENAME, 'r') as file:
    config = yaml.safe_load(file)

# Add the parent directory to the sys.path
ProjectPath = config['project_path']
sys.path.append(ProjectPath)

from liran_project.mrdiff.src.parser import parse_args
from liran_project.utils.util import MemorySnapshot
from liran_project.mrdiff.exp_main import Exp_Main, ExpMainLightning
from liran_project.utils.common import *


# Add the directory containing the exp module to the sys.path
exp_module_path = os.path.join(ProjectPath, 'mrDiff')
sys.path.append(exp_module_path)

class DelayedModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint that delays checkpointing until after a specified start_epoch.
    """
    def __init__(self, start_epoch: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)
        else:
            if self.verbose:
                print(f"Skipping checkpointing for epoch {trainer.current_epoch} (start_epoch={self.start_epoch})")
    
    def save_checkpoint(self, trainer, pl_module):
        checkpoint = super().save_checkpoint(trainer, pl_module)
        wandb_id = trainer.wandb_logger.experiment.id
        checkpoint['wandb_id'] = wandb_id
        return checkpoint
                        
def multi_checkpoints(dirpath, monitor: list = 'vali_loss', start_epoch=10,):
    
    checkpoints = [
        DelayedModelCheckpoint(
            start_epoch=start_epoch,
            
            dirpath=dirpath,
            filename="chpt",
            auto_insert_metric_name=True,
            
            save_top_k=1,
            monitor=m,
            mode='min',
            
            save_weights_only=True,
            
            save_last=False,
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        )
        for m in monitor
    ]   
    
    return checkpoints

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.best_metrics = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            # metrics = trainer.callback_metrics
            # for metric_name, metric_value in metrics.items():
            #     if metric_name.startswith('val_'):
            #         if metric_name not in self.best_metrics or metric_value < self.best_metrics[metric_name]:
            #             self.best_metrics[metric_name] = metric_value
            #             print(f"Improved {metric_name}: {metric_value:.4f}")
            super(CustomEarlyStopping, self).on_validation_end(trainer, pl_module)       
 
class SecPerIterProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        current_time = time.time()
        elapsed = current_time - self.start_time
        sec_per_iter = elapsed / (batch_idx + 1)
        self.train_progress_bar.set_postfix({'sec/it': f"{sec_per_iter:.2f}"})
               
@rank_zero_only
def init_wandb_logger(args):
    wandb_project_name = args.wandb.project
    wandb_id = args.wandb.id if args.wandb.id != "None" else None
    wandb_mode = args.wandb.mode if args.wandb.mode != "None" else "online"
    wandb_resume = args.wandb.resume if args.wandb.resume != "None" else None

    wandb_logger = WandbLogger(project=wandb_project_name, config=args,
                               mode=wandb_mode, resume=wandb_resume, id=wandb_id,
                               log_model=True
                               )

    if args.wandb.resume_from != "None":
        wandb_init_config = {
            "mode": args.wandb.mode,
            "project": args.wandb.project,
            "save_code": args.wandb.save_code,
            "resume": args.wandb.resume_from
        }

        wandb_logger.experiment(**wandb_init_config)

        # Extract the expected step from args.wandb.resume_from
        resume_from_parts = args.wandb.resume_from.split("?_step=")
        expected_step = int(resume_from_parts[1])  # Extract the step part and convert to integer

        # Assert that the run started from the specified step
        assert wandb_logger.experiment.step == expected_step, f"Expected step {expected_step}, but got {wandb_logger.experiment.step}"
                
        print(f"Resuming wandb run id: {wandb_logger.experiment.id}")   
    else:
        pass

    if args.wandb.save_code:
        wandb_logger.experiment.save("main_default.py")
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, "exp_main.py")
        wandb_logger.experiment.save(relative_path)
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        wandb_logger.experiment.config.update({"git_commit_id": commit_id})
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
                wandb_logger.experiment.notes = (wandb_logger.experiment.notes or "") + note + "\n\nAdditional information added later:\n"
        
        old_config = dict(wandb_logger.experiment.config)
        wandb_logger.log_hyperparams(args)
        new_config = dict(wandb_logger.experiment.config)
        log_config_diffs(old_config, new_config, step="update_args")

    return wandb_logger
                                
                                
def main():
    args = parse_args(CONFIG_FILENAME)
    pprint(vars(args))
    config_dict = args.configs.to_dict()
    random_seed = config_dict['general']['random_seed']
    pl.seed_everything(random_seed)

    wandb_logger = init_wandb_logger(args)

    model = config_dict['training']['model_info']['model']
    dataset = config_dict['general']['dataset']
    features = config_dict['general']['features']
    context_len = config_dict['training']['sequence']['label_len']
    label_len = config_dict['training']['sequence']['pred_len']
    pred_len = config_dict['training']['sequence']['pred_len']
    learning_rate = config_dict['optimization']['learning_rate']
    batch_size = config_dict['optimization']['batch_size']
    inverse = config_dict['data']['inverse']
    iterations = config_dict['training']['iterations']['itr']
    tag = config_dict['general']['tag']
    
    # if 'SLURM_JOB_ID' in os.environ:
    #     cluster_environment = SLURMEnvironment()
        
    strategy = config_dict['pytorch_lightning']['strategy']
    use_distributed_sampler = True
    if strategy == "DDP":
        strategy = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    elif strategy == "FSDP":
        strategy = FSDPStrategy(sharding_strategy="FULL_SHARD",
                                )
        use_distributed_sampler = False
    elif strategy == "DeepSpeed":
        strategy=DeepSpeedStrategy(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
        
    num_GPUs = torch.cuda.device_count()

    for iteration in range(iterations):
        setting = f"{model}_{dataset}_ft{features}_sl{context_len}_ll{label_len}_pl{pred_len}_lr{learning_rate}_bs{batch_size}_inv{inverse}_itr{iteration}"
        if tag is not None:
            setting += f"_{tag}"

        time_now = time.time()
        str_time_now = time.strftime("%d_%m_%Y_%H%M", time.localtime(time_now))
        model_start_training_time = str_time_now
        args.paths.checkpoints = os.path.join(args.paths.checkpoints, setting, model_start_training_time)
        checkpoint_path = args.resume_exp.resume_path if args.resume_exp.resume else None
        exp = ExpMainLightning.load_model(checkpoint_path, args) if checkpoint_path else ExpMainLightning(args)

        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>')
        with MemorySnapshot(debug=args.hardware.memory_snapshot):
            trainer = Trainer(
                devices=torch.cuda.device_count(),
                accelerator="cuda",
                max_epochs=args.training.iterations.train_epochs,
                num_sanity_val_steps=0,
                logger=wandb_logger,
                callbacks=[
                    *multi_checkpoints(dirpath=args.paths.checkpoints,
                                      monitor=['vali_loss', 'vali_extra_r_beats_mean', 'vali_dtw_distance_mean'],
                                      start_epoch=10,
                                      ),
                    SecPerIterProgressBar(),
                    ModelSummary(max_depth=3)
                ],
                strategy= strategy,
                use_distributed_sampler = use_distributed_sampler,
                enable_progress_bar=True,
                # plugins=cluster_environment,
                # fast_dev_run=True,
            )
            # tuner = Tuner(trainer)
            # tuner.scale_batch_size(model, mode="binsearch")
            
            trainer.fit(exp, ckpt_path=args.resume_exp.resume_path if args.resume_exp.resume else None)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        trainer.test(exp)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()