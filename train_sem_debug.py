import argparse
import math
import random
import sys
import os
import yaml
import shutil
from tqdm import tqdm
from pynvml.smi import nvidia_smi
from multiprocessing import Manager

import torch
import torch.nn as nn
import torch.profiler as profiler
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

from models import *
from utils.dataset import loadDatasets
from criterions.density_loss import DensityLoss

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class progressMeter:
    def __init__(self, distributed, is_master, desc, writer, total_data_len, epoch, use_pbar):
        self.states = torch.zeros(1, device='cuda', dtype=torch.float32)
        self.distributed = distributed
        self.states_name_idx = {'batch_size':0}
        self.num_states = 0
        if distributed and not is_master:
            self.master = False
            return
        self.master = True
        self.states_master = torch.zeros(1, device='cuda', dtype=torch.float32)
        
        self.writer = writer
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}: '
        if use_pbar:
            self.pbar = tqdm(total=total_data_len, ncols=70, file=sys.stdout, desc=self.desc)
        
    def update(self, states):
        assert 'batch_size' in states.keys(); "Batch size should be provided in states!"
        self.states[0] = states.pop('batch_size')
        for name, val in states.items():
            if name not in self.states_name_idx.keys():
                self.num_states += 1
                self.states_name_idx[name] = self.num_states
                self.states = torch.concat((self.states, torch.zeros((1), device='cuda')))
            idx = self.states_name_idx[name]
            if val.shape != ():
                val = torch.mean(val)
            self.states[idx] = val.item() * self.states[0]
        del states
        
        if self.distributed:
            dist.reduce(self.states, 0, dist.ReduceOp.SUM)
            
        if not self.distributed or self.master:
            more = len(self.states) - len(self.states_master)
            if more > 0:
                self.states_master = torch.concat((self.states_master, torch.zeros((more), device='cuda')))
            self.states_master += self.states
    
    def verbose_states(self):
        if self.master:
            avg_states = self.states_master / self.states_master[0]
            if self.use_pbar:
                print('\r', end='')
                self.pbar.update(int(self.states[0]))
            else:
                print(self.desc, end='')
            out = ''
            for name, idx in self.states_name_idx.items():
                if name != 'batch_size':
                    out += f" | {name}: {avg_states[idx]:.3f}"
            print(out, end='')
            if not self.use_pbar:
                print("")
    
    def write_summary(self):
        if self.master:
            avg_states = self.states_master / self.states_master[0]
            for name, idx in self.states_name_idx.items():
                if name != 'batch_size':
                    self.writer.add_scalar(name, avg_states[idx], self.epoch)
            if self.use_pbar:
                self.pbar.close()
            return avg_states[1] # Average Loss
        return 0
    
            
def train_one_epoch(model, epoch, dataloader, criterion, optimizer,
                    writer, clip_max_norm, gpu_id, distributed, is_master):
    model.train()
    total_images = len(dataloader.dataset)
    progress = progressMeter(distributed, is_master, 'train', writer, total_images, epoch, use_pbar=True)
    
    for d in dataloader:
        d = d.to(gpu_id)
        optimizer.zero_grad()

        out_net = model(d)

        loss_dict = criterion(out_net, d)
        loss_dict['loss'].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['MSE'])
        loss_dict['batch_size'] = len(d)
        progress.update(loss_dict)
        progress.verbose_states()
        
    progress.write_summary()
    

def test_epoch(model, epoch, repeats, dataloader, criterion, writer, gpu_id, distributed, is_master):
    model.eval()
    total_images = len(dataloader.dataset) * repeats
    progress = progressMeter(distributed, is_master, 'valid', writer, total_images, epoch, use_pbar=False)

    with torch.no_grad():
        for _ in range(repeats):
            for d in dataloader:
                d = d.to(gpu_id)
                out_net = model(d)
                loss_dict = criterion(out_net, d)
                
                loss_dict['PSNR'] = 10 * torch.log10(1/loss_dict['MSE'])
                loss_dict['batch_size'] = len(d)
                progress.update(loss_dict)
            
    progress.verbose_states()
    avg_loss = progress.write_summary()

    return avg_loss


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("yaml", type=str, help='config yaml file location')
    args = parser.parse_args(argv)
    
    with open(args.yaml) as f:
        args_dict = yaml.safe_load(f)
    
    class Args:
        def __init__(self, dict):
            self.dict = dict
        
        def __getattr__(self, member):
            return self.dict[member] if member in self.dict else None
        
        def __str__(self):
            return '\n'.join([f'{k}: {v}' for k, v in self.dict.items()])
    
    config = Args(args_dict)
    config.yaml = args.yaml

    return config


def init_distributed(args):
    # Check distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False 
        return
    args.distributed = True

    # Multi process setup
    torch.cuda.set_device(args.gpu)
    print(f'distributed init (rank {args.rank})', flush=True)
    dist.init_process_group(backend=args.dist_backend)
    dist.barrier()


def main(argv):
    # Argument parsing
    args = parse_args(argv)
    
    # Distributed Learning / Single GPU Learning
    init_distributed(args)
    if args.distributed:
        # args.learning_rate *= len(args.gpu_ids)
        args.world_size = len(args.gpu_ids)
        local_gpu_id = args.gpu
    else:
        local_gpu_id = None
        nvsmi = nvidia_smi.getInstance()
        res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.used")["gpu"]
        for i in args.gpu_ids:
            i = int(i)
            mem_used = res_util_list[i]['fb_memory_usage']['used']
            gpu_util = res_util_list[i]['utilization']['gpu_util']
            if mem_used < 1000 and gpu_util == 0:
                local_gpu_id = i
                args.rank = 0
                args.gpu = i
                torch.cuda.set_device(local_gpu_id)
                print(f"Using GPU No. {local_gpu_id}")
                break
        if local_gpu_id is None:
            print("No GPUs Available!")
            return
    
    EXPERIMENT_NAME = args.experiment_name
    
    # Make Directory
    save_path = os.path.join(args.log_dir, EXPERIMENT_NAME)
    if args.rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        weight_path = os.path.join(save_path, "weights")
        if not os.path.exists(weight_path):
            os.makedirs(weight_path, exist_ok=True)
    
        # Copy config
        shutil.copyfile(args.yaml, os.path.join(save_path, 'config.yaml'))
        
    # Reproduction
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    # Dataset
    if args.use_cache:
        manager = Manager()
        img_cache = manager.dict()
    else:
        img_cache = None
    train_dataloader, valid_dataloader = loadDatasets(
        train_data_type=  args.train_dataset,
        train_root_dir=   args.train_dataset_root,
        train_batch_size= args.train_batch_size,
        valid_data_type=  args.valid_dataset,
        valid_root_dir=   args.valid_dataset_root,
        num_cpus=         args.num_workers,
        mini=             args.use_mini_dataset,
        pin_memory=       False,
        distributed=      args.distributed,
        cache=            img_cache
    )
    args.valid_repetitions = 1 if args.valid_repetitions is None else args.valid_repetitions
    
    # Model
    model_params = args.model_params
    model = globals()[args.model](**model_params)
    model = model.cuda(local_gpu_id)
    if args.rank == 0:
        print(summary(model, torch.zeros((1, 3, 256, 256), device='cuda')))
    if args.distributed:
        model = DDP(module=model, device_ids=[local_gpu_id])
    
    # Training utils (criterion, optimizer, scheduler)
    criterion = DensityLoss(lmbda=args.lmbda, loss_type=args.loss_type, kernel_size=args.kernel_size, \
        decay_lmbda=args.decay_lmbda, factorized=args.factorized)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestone_epoch, gamma=0.1, last_epoch=-1)

    # Load Checkpoints
    start_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        checkpoint = torch.load(args.checkpoint, 
                                map_location=torch.device('cuda:{}'.format(local_gpu_id)))
        if args.distributed:
            if 'module' in list(checkpoint["state_dict"].keys())[0]:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.module.load_state_dict(checkpoint["state_dict"])
        else:
            if 'module' in list(checkpoint["state_dict"].keys())[0]:
                new_state_dict = {}
                for k in checkpoint["state_dict"]:
                    new_state_dict[k.replace('module.', '')] = checkpoint["state_dict"][k]
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint["state_dict"])
            
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("Loaded", args.checkpoint)
    
    # Summary writer      
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    valid_writer = SummaryWriter(os.path.join(save_path, "valid"))
    
    
    best_loss = float("inf")
    old_lr = 'None'
    # Start training!
    
    model(torch.zeros((1, 3, 256, 256)).cuda())
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,  # Wait for 1 step before starting profiling
            warmup=0,  # Warmup for 1 step
            active=1,  # Profile for 1 steps
            repeat=1  # Repeat the cycle
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(save_path),  # Save to TensorBoard
        record_shapes=True,  # Record shapes of the inputs and outputs
        profile_memory=True,  # Profile memory usage
        with_stack=True  # Record stack information
    ) as profiler:
        
        for epoch in range(start_epoch, args.epochs):
            # Print learning rate update
            if args.rank == 0:
                now_lr = optimizer.param_groups[0]['lr']
                if old_lr != now_lr:
                    print(f"Learning rate change: {old_lr} -> {now_lr}")
                    old_lr = now_lr
            
            if args.distributed:
                train_dataloader.batch_sampler.sampler.set_epoch(epoch)

            train_one_epoch(
                model=         model,
                epoch=         epoch,
                dataloader=    train_dataloader,
                criterion=     criterion,
                optimizer=     optimizer,
                writer=        train_writer,
                clip_max_norm= args.clip_max_norm,
                gpu_id=        local_gpu_id,
                distributed=   args.distributed,
                is_master=     (args.rank==0)
            )
            
            loss = test_epoch(
                model=         model,
                epoch=         epoch,
                repeats=       args.valid_repetitions,
                dataloader=    valid_dataloader, 
                criterion=     criterion,
                writer=        valid_writer,
                gpu_id=        local_gpu_id,
                distributed=   args.distributed,
                is_master=     (args.rank==0)
            )
            lr_scheduler.step()
            profiler.step() 

            # # Check best
            # if args.rank == 0:
            #     is_best = loss < best_loss
            #     best_loss = min(loss, best_loss)
            
            #     # Save CKPT
            #     if args.save_model:
            #         state = {
            #                 "epoch": epoch,
            #                 "state_dict": model.state_dict(),
            #                 "loss": loss,
            #                 "optimizer": optimizer.state_dict(),
            #                 "lr_scheduler": lr_scheduler.state_dict(),
            #         }
            #         # Save latest
            #         torch.save(state, os.path.join(save_path, f"latest_ckpt_ep{epoch}.pth.tar"))
            #         for fname in os.listdir(save_path):
            #             if 'latest_ckpt' in fname and fname != f"latest_ckpt_ep{epoch}.pth.tar":
            #                 os.remove(os.path.join(save_path, fname))
            #         # Save every 50
            #         if (epoch+1) % 50 == 0:
            #             torch.save(state, os.path.join(save_path, 'weights', f"Epoch_{epoch}.pth.tar"))
            #         # Save best
            #         if is_best:
            #             torch.save(state, os.path.join(save_path, f"best_ckpt_ep{epoch}.pth.tar"))
            #             for fname in os.listdir(save_path):
            #                 if 'best_ckpt' in fname and fname != f"best_ckpt_ep{epoch}.pth.tar":
            #                     os.remove(os.path.join(save_path, fname))


if __name__ == "__main__":
    main(sys.argv[1:])