#%%
# RD curve benchmark
# Based on CompressAI and following open-source models:
# (TCM) https://github.com/jmliu206/LIC_TCM
# (M2T) https://github.com/wsxtyrdd/Masked-Transformer-For-Image-Compression/tree/master
# (ELIC) https://github.com/VincentChandelier/ELiC-ReImplemetation

import math
import torch
from torchvision import transforms
import numpy as np
import json
import os
import tqdm
import png
from utils.dataset import indexedImageDataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_attn, load_state_dict)
from models import *

from utils.dataset import loadSingleDataset

#%%

from utils.dataset import indexedImageDataset

dataset = indexedImageDataset(
            root_dir='/datasets/Kodak',
            fname_format="kodim{:02}.png",
            num_images=24,
            transform=transforms.ToTensor()
        )

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

networks = {}
outputs = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
# Create json files from CompressAI pretrained models

metric = 'mse'  # only pre-trained model for mse are available for now
quality = 1     # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)

models = [
    'bmshj2018_factorized',
    'bmshj2018_hyperprior',
    'mbt2018',
    'cheng2020_attn'
]

for model in models:
    networks[model] = []
    for quality in range(1, 9):
        try:
            networks[model].append(globals()[model](quality=quality, pretrained=True).eval().to(device))
        except:
            pass

#%%
# Create json files from custom models
model_params = {
  'N': 192,
  'M': 320,
  'num_slices': 5,
  'small': False
}

# ELIC =====================

ckpt_dir = './trained_ckpt/ELIC/'
ckpts = os.listdir(ckpt_dir)

networks['elic'] = []
for c in ckpts:
    m = ELIC(**model_params)
    state_dict = load_state_dict(torch.load(ckpt_dir + c))
    m = m.from_state_dict(state_dict).eval().to(device)
    networks['elic'].append(m)
#%%

# TCM =====================

model_params = {
    'N': 64,
    'M': 320,
    'config': [2, 2, 2, 2, 2, 2],
    'head_dim': [8, 16, 32, 32, 16, 8],
    'drop_path_rate': 0.0,
}

ckpt_dir = './trained_ckpt/TCM_small/'
ckpts = os.listdir(ckpt_dir)

networks['tcm'] = []
for c in ckpts:
    m = TCM(**model_params)
    checkpoint = torch.load(ckpt_dir + c, map_location=device)
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    m.load_state_dict(dictory)
    m = m.eval().to(device)
    networks['tcm'].append(m)
    
#%%

with torch.no_grad():
    for name, nets in networks.items():
        psnr_list = []
        ms_ssim_val_list = []
        bit_rate_list = []
        y_anchor_bits_list = []
        y_non_anchor_bits_list = []
        z_bits_list = []
        y_groups_bits_list = []
        
        for net in nets:
            psnr = 0.
            ms_ssim_val = 0.
            bit_rate = 0.
            count = 0
            
            y_anchor_bits = 0
            y_non_anchor_bits = 0
            z_bits = 0
            y_groups_bits = torch.zeros((5,)).to(device)
            
            for x in dataset:
                x = x.unsqueeze(0).to(device)
                
                rv = net(x)
                rv['x_hat'].clamp_(0, 1)
                
                psnr += compute_psnr(x, rv["x_hat"])
                ms_ssim_val += compute_msssim(x, rv["x_hat"])
                bit_rate += compute_bpp(rv)
                
                k = rv["likelihoods"]['y'].clone()
                
                y_anchor_bits += torch.sum(k[:, :, 0::2, 0::2]) + torch.sum(k[:, :, 1::2, 1::2])
                y_non_anchor_bits += torch.sum(k[:, :, 0::2, 1::2]) + torch.sum(k[:, :, 1::2, 0::2])
                z_bits += torch.sum(rv["likelihoods"]['z'])
                y_groups_bits += torch.FloatTensor(
                    [
                        torch.sum(k[:, :16, :, :]),
                        torch.sum(k[:, 16:32, :, :]),
                        torch.sum(k[:, 32:64, :, :]),
                        torch.sum(k[:, 64:128, :, :]),
                        torch.sum(k[:, 128:, :, :]),
                    ]
                ).to(device)
                
                count += 1
            
            psnr /= count
            ms_ssim_val /= count
            bit_rate /= count
            
            y_anchor_bits /= count
            y_non_anchor_bits /= count
            z_bits /= count
            y_groups_bits /= count
            
            psnr_list.append(psnr)
            ms_ssim_val_list.append(ms_ssim_val)
            bit_rate_list.append(bit_rate)
            
            y_anchor_bits_list.append(float(y_anchor_bits))
            y_non_anchor_bits_list.append(float(y_non_anchor_bits))
            z_bits_list.append(float(z_bits))
            y_groups_bits_list.append(list(y_groups_bits.cpu()))
            
        outputs.append({
            'name': name,
            'description': '',
            'results': {
                'psnr-rgb': psnr_list,
                'ms-ssim-rgb': ms_ssim_val_list,
                'bpp': bit_rate_list,
                'y_anchor_bits': y_anchor_bits_list,
                'y_non_anchor_bits': y_non_anchor_bits_list,
                'z_bits': z_bits_list,
                'y_groups_bits': y_groups_bits_list
            }
        })
#%%

save_dir = './results/kodak_reproduction'

for o in outputs:
    with open(f'{save_dir}/{o["name"]}.json', 'w') as f:
        json.dump(o, f)

#%%

load_dir = './results/kodak'

outputs = []

files = filter(lambda x: x[-5:] == '.json', os.listdir(load_dir))
for filename in files:
    with open(f'{load_dir}/{filename}') as f:
        d = json.load(f)
    outputs.append(d)

#%%
# BPP R-D Curve

conventional = ['VTM', 'JPEG', 'BPG']

MARKER_SIZE = 4
XLIM = [0.0, 1.6]

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
# plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
for o in outputs:
    m = o['results']
    
    if o['name'] in conventional:
        marker = '--'
    else:
        marker = 'o-'
    
    if 'bpp' not in m:
        continue
    
    sort_indexes = sorted(range(0, len(m['bpp'])), key=lambda i:m['bpp'][i])
    for key in m:
        m[key] = [m[key][i] for i in sort_indexes]
    
    if 'psnr-rgb' in m:
        axes[0].plot(m['bpp'], m['psnr-rgb'], marker, label=o['name'], markersize=MARKER_SIZE)
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('Kodak - PSNR (RGB)')
    axes[0].set_xlim(XLIM)
    axes[0].set_ylim([23., 41.5])

    if 'ms-ssim-db-rgb' in m:
        axes[1].plot(m['bpp'], m['ms-ssim-db-rgb'], marker, label=o['name'], markersize=MARKER_SIZE)
    elif 'ms-ssim-rgb' in m:
        axes[1].plot(m['bpp'], -10*np.log10(1-np.array(m['ms-ssim-rgb'])), marker, label=o['name'], markersize=MARKER_SIZE)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('Kodak - MS-SSIM (log)')
    axes[1].set_xlim(XLIM)
    axes[1].set_ylim([8., 22.])

fig.tight_layout()
plt.show()

#%%
# Bandwidth Ratio R-D Curve

conventional = ['VTM', 'JPEG', 'BPG']

MARKER_SIZE = 4
XLIM = [0, 0.60]
SNR = 10
target = 'awgn_measured' # or 'awgn_capacity'

bps_awgn_ldpc_qam = {
    # (SNR_dB): (bits/symbol)
    'awgn_measured': {
        # Best result for every combination of
        # (3072, 6144), (3072, 4608), (1536, 4608), (5632, 6144) code rate
        # BPSK, (4, 16, 64, 256)-QAM modulations
        # Tested on Sionna
        0: 0.666,
        5: 1.333,
        10: 2.666,
        15: 4.0,
        20: 5.5,
        25: 7.333
    },
    'awgn_capacity': {
        # log_2 (1 + SNR) (bits/symbol)
        0: 1,
        5: 2.05,
        10: 3.45,
        15: 5.2,
        20: 6.65,
        25: 8.3
    },
}

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
# plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')

for o in outputs:
    m = o['results']
    if o['name'] in conventional:
        marker = '--'
    elif 'cbr' in m:
        marker = 'x-'
    else:
        marker = 'o-'
    
    if 'bpp' in m:
        spp = np.array(m['bpp']) / bps_awgn_ldpc_qam[target][SNR]
        label = o['name'] + '_LDPC'
    elif 'cbr' in m:
        spp = np.array(m['cbr']) * 3
        label = o['name']

    sort_indexes = sorted(range(0, len(spp)), key=lambda i:spp[i])
    for key in m:
        m[key] = [m[key][i] for i in sort_indexes]
    spp = [spp[i] for i in sort_indexes]

    if 'psnr-rgb' in m:
        axes[0].plot(spp, m['psnr-rgb'], marker, label=label, markersize=MARKER_SIZE)
    axes[0].legend(loc='lower right')
    axes[0].grid(True)
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Symbol-rate [Symbols/Pixel]')
    axes[0].title.set_text(f'Kodak - PSNR (RGB), AWGN {SNR} dB')
    axes[0].set_xlim(XLIM)
    axes[0].set_ylim([23., 41.5])

    if 'ms-ssim-db-rgb' in m:
        axes[1].plot(spp, m['ms-ssim-db-rgb'], marker, label=label, markersize=MARKER_SIZE)
    elif 'ms-ssim-rgb' in m:
        axes[1].plot(spp, -10*np.log10(1-np.array(m['ms-ssim-rgb'])), marker, label=label, markersize=MARKER_SIZE)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Symbol-rate [Symbols/Pixel]')
    axes[1].title.set_text(f'Kodak - MS-SSIM (log), AWGN {SNR} dB')
    axes[1].set_xlim(XLIM)
    axes[1].set_ylim([8., 22.])

fig.tight_layout()
plt.show()
# %%
from models.NTSCC_PP.net.ntscc import NTSCC_plus

class config:
    channel_type = 'awgn'
    SNR = 10
    device = torch.device("cuda:0")

    eta = 0.2
    multiple_rate = [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320]
    logger = False
    
ckpt_dir = './trained_ckpt/NTSCC++/'
ckpts = os.listdir(ckpt_dir)

networks = {}
networks['ntscc+'] = []
for c in ckpts:
    m = NTSCC_plus(config)
    
    try:
        pretrained = torch.load(ckpt_dir + c)['state_dict']
    except:
        pretrained = torch.load(ckpt_dir + c)
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        if'mask' not in key:
            result_dict[result_key] = weight
    m.load_state_dict(result_dict, strict=False)
    del result_dict, pretrained
    
    m = m.eval().to(device)
    networks['ntscc+'].append(m)

NUM_REPETITIONS = 10

outputs = []
with torch.no_grad():
    for idx, items in enumerate(networks.items()):
        name, nets = items
        
        psnr_list = []
        ms_ssim_val_list = []
        bit_rate_list = []
        
        t = tqdm(total=len(nets) * len(dataset) * NUM_REPETITIONS,
                 desc=f'{idx+1}/{len(networks)}: ')
        
        for net in nets:
            psnr = 0.
            ms_ssim_val = 0.
            bit_rate = 0.
            count = 0
            for x in dataset:
                x = x.clone().unsqueeze(0).to(device)
                for i in range(NUM_REPETITIONS):
                    n, c, h, w = x.size()
                    
                    results = net(x)
                    sideinfo = np.array(results['indexes'].reshape(h // 16, w // 16).cpu().numpy(), dtype=np.uint8)
                    png.from_array(sideinfo, 'L').save("output.png")
                    bits_num = os.path.getsize('output.png') * 8
                    
                    cbr_k = bits_num / 8 / (h * w)
                    
                    psnr += compute_psnr(x, results["x_hat"])
                    ms_ssim_val += compute_msssim(x, results["x_hat"])
                    bit_rate += results["k"] / (n*c*h*w) # Actually cbr
                    
                    count += 1
                    t.update(1)
                
            psnr /= count
            ms_ssim_val /= count
            bit_rate /= count
            
            psnr_list.append(psnr)
            ms_ssim_val_list.append(ms_ssim_val)
            bit_rate_list.append(float(bit_rate))
            
        outputs.append({
            'name': name,
            'description': '',
            'results': {
                'psnr-rgb': psnr_list,
                'ms-ssim-rgb': ms_ssim_val_list,
                'cbr': bit_rate_list
            }
        })
        
        t.close()
        
#%%
from models.SemELIC.model import SemELIC

ckpts = ['./logs/SemELIC_10dB/best_ckpt_ep757.pth.tar']
networks = {}
networks['semElic'] = []

model_params = {
    'N': 192,
    'M': 320,
    'small': False,
    'snrdB': 10, 
    'num_symbols': 16384
}

for c in ckpts:
    m = SemELIC(**model_params)
    m.load_state_dict(torch.load(c)['state_dict'])
    
    m = m.eval().to(device)
    networks['semElic'].append(m)

NUM_REPETITIONS = 10

with torch.no_grad():
    for idx, items in enumerate(networks.items()):
        name, nets = items
        
        psnr_list = []
        ms_ssim_val_list = []
        bit_rate_list = []
        
        t = tqdm(total=len(nets) * len(dataset) * NUM_REPETITIONS,
                 desc=f'{idx+1}/{len(networks)}: ')
        
        for net in nets:
            psnr = 0.
            ms_ssim_val = 0.
            bit_rate = 0.
            count = 0
            for x in dataset:
                x = x.clone().unsqueeze(0).to(device)
                for i in range(NUM_REPETITIONS):
                    n, c, h, w = x.size()
                    
                    results = net(x)
                    
                    psnr += compute_psnr(x, results["x_hat"])
                    ms_ssim_val += compute_msssim(x, results["x_hat"])
                    
                    count += 1
                    t.update(1)
            
            psnr /= count
            ms_ssim_val /= count
            bit_rate /= count
            
            psnr_list.append(psnr)
            ms_ssim_val_list.append(ms_ssim_val)
            bit_rate_list.append(float(bit_rate))
            
        outputs.append({
            'name': name,
            'description': '',
            'results': {
                'psnr-rgb': psnr_list,
                'ms-ssim-rgb': ms_ssim_val_list,
                'cbr': [0.25 / 3]
            }
        })
        
        t.close()

# %%
save_dir = './results/kodak_reproduction'

for o in outputs:
    with open(f'{save_dir}/{o["name"]}.json', 'w') as f:
        json.dump(o, f)
# %%
