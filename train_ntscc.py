import math
import os
import random
import sys
import time
import logging
from datetime import datetime
from pynvml.smi import nvidia_smi
import png
import shutil

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb

from utils.dataset import loadDatasets
from models.semantic_layers.density_estimation import GaussianDensityEstimation, GaussianKernalDensityEstimation



def logger_configuration(exp, phase, method='NTC', save_log=True):
    logger = logging.getLogger("NTSCC")
    workdir = f'{config.log_dir}/{exp}'
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/logger.log'
    samples = workdir + '/samples'
    models = workdir + '/models'
    weights = workdir + '/weights'
    if save_log:
        makedirs(workdir)
        makedirs(samples)
        makedirs(models)
        makedirs(weights)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, is_best, base_dir, epoch, filename="checkpoint.pth.tar"):
    with open(f"{base_dir}/cktp.txt", 'w') as f:
        f.write(f"Best Checkpoint : Epoch {epoch}")
        
    if is_best:
        torch.save(state, base_dir + "/checkpoint_best_loss.pth.tar")
    else:
        torch.save(state, base_dir + "/weights/" + filename)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def load_weights(net, model_path, load_featenc=True):
    try:
        pretrained = torch.load(model_path)['state_dict']
    except:
        pretrained = torch.load(model_path)
    result_dict = {}
    for key, weight in pretrained.items():
        result_key = key
        if load_featenc:
            if 'mask' not in key:
                result_dict[result_key] = weight
        else:
            # if 'attn_mask' not in key and 'rate_adaption.mask' not in key\
            #         and 'fe' not in key and 'fd' not in key:
            result_dict[result_key] = weight
    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained


def load_checkpoint(logger, device, net, optimizer_G, aux_optimizer, model_path):
    logger.info("Loading " + str(model_path))
    pre_dict = torch.load(model_path, map_location=device)

    result_dict = {}
    for key, weight in pre_dict["state_dict"].items():
        result_key = key
        if 'mask' not in key:
            result_dict[result_key] = weight
    net.load_state_dict(result_dict, strict=False)

    # optimizer_G.load_state_dict(pre_dict["optimizer"])
    aux_optimizer.load_state_dict(pre_dict["aux_optimizer"])
    # lr_scheduler.load_state_dict(pre_dict["lr_scheduler"])

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, num_pixels=None):
        N, C, H, W = target.size()
        out = {}
        if num_pixels is None:
            num_pixels = N * H * W
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["cbr_y"] = output["k"] / (N * C * H * W)
        out["mse_loss_ntc"] = self.mse(output["x_hat_ntc"], target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["bpp_loss"] = out["bpp_y"] + out["bpp_z"]
        out["cbr"] = out["cbr_y"]

        out["ntscc_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss"] + out["mse_loss_ntc"]) + out["bpp_loss"]
        return out


class RateDistortionDensityLoss(nn.Module):
    def __init__(self, rate_choice, lmbda=1e-2, K=1e-2, loss_type='MSE'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.gaussian_density = GaussianDensityEstimation()
        self.kernel_denstity = GaussianKernalDensityEstimation(chunk_size=512, kernel_size=None)
        self.lmbda = lmbda
        self.K = K
        self.loss_type = loss_type
        self.rate_choice = rate_choice

    def forward(self, output, target, num_pixels=None):
        N, C, H, W = target.size()
        out = {}
        
        # NTSCC loss
        if num_pixels is None:
            num_pixels = N * H * W
        out["bpp_y"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["bpp_z"] = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        out["cbr_y"] = output["k"] / (N * C * H * W)
        out["mse_loss_ntc"] = self.mse(output["x_hat_ntc"], target)
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["bpp_loss"] = out["bpp_y"] + out["bpp_z"]
        out["cbr"] = out["cbr_y"]
        
        # density loss
        y = output['y']
        dB, C, H, W = y.shape
        y = y.reshape(2, dB//2, C, H, W).permute(1, 0, 3, 4, 2).contiguous().reshape(dB//2, 2*H*W, C)
        spatial_size = 2*H*W
        ch_symbols = []
        for img_idx in range(dB//2):
            for ch_idx in range(spatial_size):
                ch_len = self.rate_choice[output['indexes'][img_idx*spatial_size+ch_idx]]
                sym = y[img_idx, ch_idx, :ch_len]
                ch_symbols.append(sym)
        symbols = torch.cat(ch_symbols, dim=0).unsqueeze(0)
        
        out["gce"] = self.gaussian_density.entropy(symbols)
        out["entropy"] = self.kernel_denstity.entropy(symbols)
        out["kldiv"] = out["gce"] - out["entropy"]
        norm_S = symbols / torch.mean(symbols**2, dim=-1, keepdims=True)**0.5
        out["norm_gce"] = self.gaussian_density.entropy(norm_S)
        out["norm_ent"] = self.kernel_denstity.entropy(norm_S)
        out["norm_kl"] = out["norm_gce"] - out["norm_ent"]
        out["gce"] = torch.mean(out["gce"])
        out["entropy"] = torch.mean(out["entropy"])
        out["kldiv"] = torch.mean(out["kldiv"])
        out["norm_gce"] = torch.mean(out["norm_gce"])
        out["norm_ent"] = torch.mean(out["norm_ent"])
        out["norm_kl"] = torch.mean(out["norm_kl"])
        
        # Full loss
        if self.loss_type == 'MSE':
            out["ntscc_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss"] + out["mse_loss_ntc"]) + out["bpp_loss"]
        if self.loss_type == 'KLdv':
            out["ntscc_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"] + self.K * out['kldiv']
            out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss"] + out["mse_loss_ntc"]) + out["bpp_loss"] + self.K * out['kldiv']
        return out

def train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                    lr_scheduler, device, logger, writer):
    elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys, kldiv, norm_kl = [AverageMeter() for _ in range(10)]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys, kldiv, norm_kl]
    step = 0
    # for batch_idx, input_image in enumerate(train_loader):
    for input_image in train_loader:
        net.train()
        input_image = input_image.to(device)
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()
        step += 1
        start_time = time.time()

        results = net(input_image)
        out_criterion = criterion(results, input_image)
        out_criterion["loss"].backward()

        if config.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip_max_norm)
        optimizer_G.step()

        aux_loss = net.ntc.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)

        losses.update(out_criterion["loss"].item())
        bppys.update(out_criterion["bpp_y"].item())
        bppzs.update(out_criterion["bpp_z"].item())

        psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
        psnr_ntcs.update(psnr_ntc.item())

        losses.update(out_criterion["loss"].item())
        psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
        psnrs.update(psnr.item())
        cbrs.update(out_criterion["cbr"].item())
        cbr_ys.update(out_criterion["cbr_y"])
        
        kldiv.update(out_criterion["kldiv"].item())
        norm_kl.update(out_criterion["norm_kl"].item())

        if (step % config.print_every) == 0:
            process = step / (train_loader.__len__()) * 100.0
            log_info = [
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'KL_div {kldiv.val:.4f} ({kldiv.avg:.4f})',
                f'norm_KL {norm_kl.val:.4f} ({norm_kl.avg:.4f})',
                f'CBR {cbrs.val:.3f} ({cbrs.avg:.4f})',
                f'CBR_y {cbr_ys.val:.3f} ({cbr_ys.avg:.4f})',
                f'PSNR_NTC {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}'
            ]
            log = (' | '.join(log_info))
            logger.info(log)
    lr_scheduler.step()
            
    writer.add_scalar('PSNR', psnrs.avg, epoch)
    writer.add_scalar('KL_div', kldiv.avg, epoch)
    writer.add_scalar('norm_KL', norm_kl.avg, epoch)
    writer.add_scalar('CBR', cbrs.avg, epoch)
    writer.add_scalar('CBR_y', cbr_ys.avg, epoch)
    writer.add_scalar('PSNR_NTC', psnr_ntcs.avg, epoch)
    writer.add_scalar('Bpp_y', bppys.avg, epoch)
    writer.add_scalar('Bpp_z', bppzs.avg, epoch)
    writer.add_scalar('losses', losses.avg, epoch)
    for i in metrics:
        i.clear()


def test(net, criterion, test_loader, device, logger, writer, epoch):
    with torch.no_grad():
        net.eval()
        elapsed, losses, bst_losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys, spp, kldiv, norm_kl = [AverageMeter() for _ in range(12)]
        for batch_idx, input_image in enumerate(test_loader):
        # for input_image in test_loader:
            input_image = input_image.to(device)

            _, _, h, w = input_image.shape

            ph = 128
            pw = 64

            new_h = (h + ph - 1) // ph * ph
            new_w = (w + pw - 1) // pw * pw
            num_pixels = new_h * new_w
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            input_image_pad = F.pad(
                input_image,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            start_time = time.time()
            results = net(input_image_pad)

            # side information should be reliably transmitted to the receiver
            sideinfo = np.array(results['indexes'].reshape(new_h // 16, new_w // 16).cpu().numpy(), dtype=np.uint8)
            png.from_array(sideinfo, 'L').save("temp/output.png")
            try:
                # lossless compression the side information k using FLIF (https://github.com/FLIF-hub/FLIF)
                os.system('flif --overwrite -e temp/output.png temp/output.flif')
                bits_num = os.path.getsize('temp/output.flif') * 8
            except:
                # revert to PNG format
                bits_num = os.path.getsize('temp/output.png') * 8

            #  2/3 rate LDPC + 16QAM for AWGN SNR=10dB
            cbr_k = bits_num / 8 / (new_h * new_w)
            # or ideal capacity-achieving channel code
            # cbr_k = bits_num / np.log2(1 + 10 ** (net.channel.chan_param / 10)) / (new_h * new_w * 3)

            results["x_hat"] = F.pad(
                results["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )
            results["x_hat_ntc"] = F.pad(
                results["x_hat_ntc"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
            )
            out_criterion = criterion(results, input_image, num_pixels=num_pixels)

            elapsed.update(time.time() - start_time)

            losses.update(out_criterion["loss"].item())
            bppys.update(out_criterion["bpp_y"].item())
            bppzs.update(out_criterion["bpp_z"].item())

            psnr_ntc = 10 * (torch.log(1. / out_criterion["mse_loss_ntc"]) / np.log(10))
            psnr_ntcs.update(psnr_ntc.item())

            losses.update(out_criterion["loss"].item())
            bst_losses.update(out_criterion["ntscc_loss"].item())

            psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
            psnrs.update(psnr.item())
            cbr_ys.update(out_criterion["cbr_y"])
            cbr = out_criterion["cbr_y"].item() + cbr_k
            cbrs.update(cbr)
            spp.update(cbr * 3)
            kldiv.update(out_criterion["kldiv"].item())
            norm_kl.update(out_criterion["norm_kl"].item())

        log = (' | '.join([
            f'Step [{(batch_idx + 1)}/{test_loader.__len__()}]',
            f'Loss {losses.val:.3f} ({losses.avg:.3f})',
            f'Time {elapsed.avg:.2f}',
            f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
            f'SPP {spp.val:.2f} ({spp.avg:.2f})',
            f'KL_div {kldiv.val:.4f} ({kldiv.avg:.4f})',
            f'norm_KL {norm_kl.val:.4f} ({norm_kl.avg:.4f})',
            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
            f'CBR_y {cbr_ys.val:.4f} ({cbr_ys.avg:.4f})',
            f'PSNR_NTC {psnr_ntcs.val:.2f} ({psnr_ntcs.avg:.2f})',
            f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
            f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})'
        ]))
        logger.info(log)
    
        writer.add_scalar('PSNR', psnrs.avg, epoch)
        writer.add_scalar('SPP', spp.avg, epoch)
        writer.add_scalar('KL_div', kldiv.avg, epoch)
        writer.add_scalar('norm_KL', norm_kl.avg, epoch)
        writer.add_scalar('CBR', cbrs.avg, epoch)
        writer.add_scalar('CBR_y', cbr_ys.avg, epoch)
        writer.add_scalar('PSNR_NTC', psnr_ntcs.avg, epoch)
        writer.add_scalar('Bpp_y', bppys.avg, epoch)
        writer.add_scalar('Bpp_z', bppzs.avg, epoch)
        writer.add_scalar('losses', losses.avg, epoch)

    return bst_losses.avg


# def online_adaptation_test(net, criterion, test_loader, device, logger, step_num=20):
#     elapsed, losses, psnrs, bppys, bppzs, psnr_ntcs, cbrs, cbr_ys = [AverageMeter() for _ in range(8)]
#     for batch_idx, input_image in enumerate(test_loader):
#         input_image = input_image.to(device)
#         start_time = time.time()
#         with torch.no_grad():
#             y = net.ntc.g_a(input_image)
#         params = [
#             {'params': net.f_e0.parameters(), 'lr': 1e-4},
#             {'params': net.trans_sep_enc.parameters(), 'lr': 1e-4},
#             {'params': net.trans_ctx_enc.parameters(), 'lr': 1e-4},
#             {'params': torch.nn.Parameter(y, requires_grad=True), 'lr': 5e-3}
#         ]
#         optimizer = torch.optim.Adam(params)
#         for i in range(step_num):


def parse_args(argv):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./config/config_ntscc.yaml',
                        help='Path to config file to replace defaults.')
    parser.add_argument('--experiment_name', type=str, default='experiment names')
    parser.add_argument('--seed', type=int, default=1024,
                        help='Random seed.')
    parser.add_argument('--gpu-ids', action='append', default=['0'], help='GPU id to use.')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only (and do not run training).')

    # logging
    parser.add_argument('--log-dir', type=str, help='logs/')
    parser.add_argument('--exp-name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        help='Experiment name, unique id for trainers, logs.')
    parser.add_argument('--wandb', action="store_true",
                        help='Use wandb for logging.')
    parser.add_argument('--print-every', type=int, default=30,
                        help='Frequency of logging.')

    # dataset
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--train-dataset-root', type=str)
    parser.add_argument('--valid-dataset', type=str)
    parser.add_argument('--valid-dataset-root', type=str)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--use-cache', action='store_true', default=False)
    
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--valid-repetitions', type=int, default=1)
    parser.add_argument('--use_mini_dataset', action='store_true', default=False)

    # optimization
    parser.add_argument('--train-lambda', type=float, default=0.18,
                        help='Rate distortion trade-off hyper-parameter.')
    parser.add_argument('--train-K', type=float, default=0.01)
    parser.add_argument('--loss-type', type=str, default='MSE')
    parser.add_argument('--optimizer-type', type=str, default='adam',
                        help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                             'and fused optimizers from `apex`, if apex is installed.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Base optimizer learning rate.')
    parser.add_argument('--aux-lr', type=float, default=0.001,
                        help='Auxiliary optimizer learning rate for factorized entropy model.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Eps value for numerical stability.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay, applied only to decoder weights.')
    parser.add_argument('--clip-max-norm', type=float, default=1.0,
                        help='Gradient clipping for stable training.')

    # trainer
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run the training.')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint model")
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights.')
    parser.add_argument('--pretrained-ntc', type=str, default=None,
                        help='Path to pretrained ntc model weights.')
    parser.add_argument('--save', action="store_true",
                        help="Save the model at every epoch (no overwrite).")
    parser.add_argument('--save-every', type=int, default=10000,
                        help='Frequency of saving the model.')
    parser.add_argument('--test-every', type=int, default=5000,
                        help='Frequency of running validation.')

    # network
    parser.add_argument('--model', type=str, default='hyperprior',
                        help='Network architecture.')
    parser.add_argument('--input_resolution', type=int, default=(16, 16),
                        help='Input resolution of the network.')
    args = parser.parse_args(argv)
    return args


def main(argv):
    global config
    config = parse_args(argv)
    
    nvsmi = nvidia_smi.getInstance()
    res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.used")["gpu"]
    for i in config.gpu_ids:
        i = int(i)
        mem_used = res_util_list[i]['fb_memory_usage']['used']
        gpu_util = res_util_list[i]['utilization']['gpu_util']
        if mem_used < 16000 and gpu_util == 0:
            config.gpu = i
            torch.cuda.set_device(f"cuda:{config.gpu}")
            print(f"Using GPU No. {config.gpu}")
            break
    
    rate_choice = [1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320]
    model_params = {
        'eta': 0.2,
        'channel_type': 'awgn',
        'SNR': 10,
        'multiple_rate': rate_choice
    }
    
    import torch.optim as optim
    from models import NTSCC_plus, NTSCC_ELIC
    
    model_architectures = {
        "NTSCC_plus": NTSCC_plus,
        'NTSCC_ELIC': NTSCC_ELIC
    }

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    device = "cuda"
    config.device = device
    job_type = 'test' if config.test_only else 'train'
    exp_name = "[{}_SNR={}_lmbda={}] ".format(config.model, model_params['SNR'], config.train_lambda) + config.exp_name
    print(exp_name)
    exp_name = config.experiment_name
    global workdir
    workdir, logger = logger_configuration(exp_name, job_type,
                                           method=model_architectures[config.model], save_log=(not config.test_only))
    # Copy config
    shutil.copyfile(config.config, os.path.join(workdir, 'config.yaml'))
    config.logger = logger
    logger.info(config.__dict__)

    net = model_architectures[config.model](**model_params).to(device)
    net = net.cuda(f"cuda:{config.gpu}")
    # criterion = RateDistortionLoss(lmbda=config.train_lambda)
    criterion = RateDistortionDensityLoss(rate_choice=rate_choice, lmbda=config.train_lambda, K=config.train_K, loss_type=config.loss_type)
    
    train_writer = SummaryWriter(os.path.join(workdir, "train"))
    valid_writer = SummaryWriter(os.path.join(workdir, "valid"))

    if config.test_only:
        if config.pretrained is not None:
            load_weights(net, config.pretrained)
        else:
            raise ValueError("Please specify the checkpoint path for testing.")
        # test_loader = get_test_loader(config.eval_dataset_path)
        test(net, criterion, test_loader, device, logger)

    else:
        if config.wandb:
            wandb_init_kwargs = {
                'project': 'NTSCC',
                'name': exp_name,
                'save_code': True,
                'job_type': job_type
            }
            wandb.init(**wandb_init_kwargs)

        train_loader, test_loader = loadDatasets(
            train_data_type=config.train_dataset,
            train_root_dir=config.train_dataset_root,
            train_batch_size=config.train_batch_size,
            valid_data_type=config.valid_dataset,
            valid_root_dir=config.valid_dataset_root,
            num_cpus=config.num_workers,
            mini=config.use_mini_dataset,
            pin_memory=True,
            distributed=False
        )

        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
        tot_epoch = config.epochs

        if config.pretrained is not None:
            pre_dict = torch.load(config.pretrained, map_location=device)
            result_dict = {}
            # for key, weight in pre_dict["state_dict"].items():
            for key, weight in pre_dict.items():
                result_key = key
                if 'mask' not in key:
                    result_dict[result_key] = weight
            net.load_state_dict(result_dict, strict=False)
        elif config.pretrained_ntc is not None:
            net.load_pretrained_ntc(config.pretrained_ntc)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[600000], gamma=0.1)

        start_epoch = 0
        best_loss = float("inf")
        for epoch in range(start_epoch, tot_epoch):

            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            
            train_one_epoch(epoch, net, criterion, train_loader, test_loader, optimizer_G, aux_optimizer,
                            lr_scheduler, device, logger, train_writer)
            
            loss = test(net, criterion, test_loader, device, logger, valid_writer, epoch)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        
            if is_best:
                save_checkpoint(
                    {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer_G.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    workdir,
                    epoch
                )
                        
            if (epoch + 1) % config.save_every == 0:
                save_checkpoint(
                    {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer_G.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    False,
                    workdir,
                    epoch,
                    filename='EP{}.pth.tar'.format(epoch)
                )


if __name__ == '__main__':
    main(sys.argv[1:])