import torch
import numpy as np
import yaml
import argparse
import random
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os

from source.models.model_neat import BaseModel
from source.models.protran_model import ProTranModel
from source.callbacks import *


# ============= Run ============= #
if __name__== '__main__':
    
    ########### Setup params, data, etc ###########
    # read params
    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('--yaml_file', type=str, default='./example_cfgs/protran_electricity.yaml', help='Path to the yaml config file')

    parser.add_argument('--log_dir', type=str, default=None, help='Directory for logging')

    opt = parser.parse_args()
    opt = vars(opt) # get a dict of the arguments

    yaml_file_path = opt["yaml_file"]
    with open(yaml_file_path) as f:
        var = yaml.safe_load(f)

    # ============= Set and Create Log Dirs ============= #
    if opt["log_dir"] is not None:
        LOGDIR = opt["log_dir"]
    else:
        LOGDIR = var['log_dir']
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    # ============= Activate CUDA ============= #
    arg_cuda = int(var['gpu']>0) and torch.cuda.is_available()
    device = torch.device("cuda" if arg_cuda else "cpu")

    if str(device) == "cuda":
        print('cuda activated')
    print(f"Device is {device}")

    # ============= Set seeds ============= #
    seed = var['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ============= Create Model ============= #
    print(f'Config is: var{var}')

    print('model inputs read')
    if var['model']['type'] == 'foretran':
        model_try = BaseModel(**var)
    elif var['model']['type'] == 'protran':
        model_try = ProTranModel(**var)
    else:
        raise ValueError("model type must be 'foretran' or 'protran'")
    print('model is ready')

    if str(device) =="cuda":
        accelerator="gpu"
    else:
        accelerator="cpu"

    # ============= Create Trainer ============= #
    timer_callback = TrainingTimer()
    checkpoint_callback = ModelCheckpoint(every_n_epochs=var["train"]["ckpt_period"], save_top_k=-1, monitor=None)
    plot_pred_callback = plotPredictions_mean_var_plotly_dims(samples=var['testing']['samples'], reconstruct=var['model']['reconstruct'])
    plot_metrics_callback = plotMetric_timeseries_plotly(samples=var['testing']['samples'], reconstruct=var['model']['reconstruct'], reduction='mean')
    calculate_crps_callback = CRPSCallback(samples=100)
    
    callbacks = [
        checkpoint_callback, 
        timer_callback, 
        plot_pred_callback,
        plot_metrics_callback,
        calculate_crps_callback,
        InferenceTimerCallback(num_trials = 100),
        GradCheck()
    ]
    
    trainer = pl.Trainer(
        max_epochs=var['optim']['max_epoch'],
        callbacks=callbacks,
        accelerator=accelerator,
        devices=var['gpu'],
        logger=pl.loggers.CSVLogger(LOGDIR), # change to any logger that you want, we used Wandb
        log_every_n_steps=1
    )

    # fit the model
    trainer.fit(model_try)
    print("Training is done.")

    # test the model
    with torch.no_grad():
        trainer.test(model=model_try)
        print("Testing is done.")
