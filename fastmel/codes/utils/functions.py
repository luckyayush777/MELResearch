import argparse
from omegaconf import OmegaConf
import os

def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True)
    _args = parser.parse_args()
    
    config = OmegaConf.load(_args.config)
    
    config_name = os.path.splitext(os.path.basename(_args.config))[0]
    
    return config, config_name
