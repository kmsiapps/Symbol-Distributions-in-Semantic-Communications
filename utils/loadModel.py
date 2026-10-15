import os
import yaml
import torch
from models import *

__all__ = ['loadModel']

def loadModel(exp, device='cuda'):
    with open(f"logs/{exp}/config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
        
    model_name = config_dict["model"]
    model_params = config_dict["model_params"]
    for fname in os.listdir(f"logs/{exp}"):
        if "best" in fname:
            break
    model_ckpt = f"logs/{exp}/{fname}"
    
    model_params['CIFAR'] = 'cifar' in exp
    model = globals()[model_name](**model_params).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    
    if model_name in ['NTSCC_plus']:
        remove_list = []
        for key in ckpt['state_dict']:
            if 'mask' in key and 'ntc' not in key:
                # print(key)
                remove_list.append(key)
        for key in remove_list:
            del ckpt['state_dict'][key]
    
    model.load_state_dict(ckpt["state_dict"], strict=False)
    
    return model
