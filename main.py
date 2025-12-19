import os
import torch
import numpy as np
import random
import argparse

import warnings
warnings.filterwarnings("ignore")

from engine.solver import Trainer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader, build_dataloader_cond

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    - seed (int): The seed value.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional steps for CuDNN backend
    os.environ['PYTHONHASHSEED'] = str(seed)

# Example usage:
set_seed(2023)

#class Args_Example:
#    def __init__(self) -> None:
#        self.config_path = './Config/etth.yaml'
#        self.save_dir = './forecasting_exp'
#        self.gpu = 0
#        os.makedirs(self.save_dir, exist_ok=True)

class Args_Example:
    def __init__(self, config_path, save_dir, gpu):
        self.config_path = config_path
        self.save_dir = save_dir
        self.gpu = gpu
        os.makedirs(self.save_dir, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp',
                        help='Directory to save experiment results.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #args =  Args_Example()
    args_parsed = parse_arguments()
    args = Args_Example(args_parsed.config_path, args_parsed.save_dir, args_parsed.gpu)
    seq_len = 96
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = instantiate_from_config(configs['model']).to(device)
    #model.use_ff = False
    model.fast_sampling = True
    #configs['solver']['max_epochs']=100
    dataloader_info = build_dataloader(configs, args)
    dataloader = dataloader_info['dataloader']
    trainer = Trainer(config=configs, args=args, model=model, dataloader={'dataloader':dataloader})
    trainer.train()
    args.mode = 'predict'
    args.pred_len = seq_len
    test_dataloader_info = build_dataloader_cond(configs, args)
    test_scaled = test_dataloader_info['dataset'].samples
    scaler = test_dataloader_info['dataset'].scaler
    seq_length, feat_num = seq_len*2, test_scaled.shape[-1]
    pred_length = seq_len
    real = test_scaled
    test_dataset = test_dataloader_info['dataset']
    test_dataloader = test_dataloader_info['dataloader']
    sample, real_ = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])
    mask = test_dataset.masking
    mse = mean_squared_error(sample.reshape(-1), real_.reshape(-1))
    mae = mean_absolute_error(sample.reshape(-1), real_.reshape(-1))
    print(mse,mae)

