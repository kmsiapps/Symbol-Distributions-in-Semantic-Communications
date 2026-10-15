#%%

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.semantic_layers.density_estimation import *
from models import *

from utils.dataset import loadSingleDataset
from utils.loadModel import *
from utils.loadImage import *


exp = 'NTSCC_Lo72_base'
num_symbols = 16384 * 2
pdf_len = 200

# Load Kodak dataset
valid_dataset = loadSingleDataset(
    type='Kodak',
    root_dir='/datasets/Kodak'
)
loader = DataLoader(valid_dataset,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)
torch.set_printoptions(linewidth=200)

with torch.no_grad():
    model = loadModel(exp)
    rate_choice = model.rate_choice
    
    for img in loader:
        img = img.to('cuda')
        out = model(img)
        y = out['y']
        dB, C, H, W = y.shape
        y = y.reshape(2, dB//2, C, H, W).permute(1, 0, 3, 4, 2).contiguous().reshape(dB//2, 2*H*W, C)
        spatial_size = 2*H*W
        
        for img_idx in range(dB//2):
            img_symbols = []
            for ch_idx in range(spatial_size):
                ch_len = rate_choice[out['indexes'][img_idx*spatial_size+ch_idx]]
                print(ch_len)
                sym = y[img_idx, ch_idx, :ch_len]
                img_symbols.append(sym)
            img_symbol = torch.cat(img_symbols, dim=0)
            print(img_symbol.shape)
        
        break
            