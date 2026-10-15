import torch
import torch.nn as nn
from models.semantic_layers.density_estimation import GaussianDensityEstimation, GaussianKernalDensityEstimation

class lambdaMilestone:
    def __init__(self, initial_value, decay_epoch, decay_factor):
        self.lmbda = initial_value
        self.epoch_count = 0
        self.decay_epoch = decay_epoch
        self.decay_factor = decay_factor
    
    def get_lambda(self):
        if self.epoch_count == self.decay_epoch:
            self.lmbda *= self.decay_factor
            self.epoch_count += 1
        else:
            self.epoch_count = 0
        return torch.tensor(self.lmbda)


class DensityLoss(nn.Module):
    """Custom density loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, loss_type='mse', num_symbols=16384, kernel_size=None, decay_lmbda=False, factorized=True):
        super().__init__()
        self.mse = nn.MSELoss()
        decay_factor = 0.32 if decay_lmbda else 1
        self.lmbda_scheduler = lambdaMilestone(lmbda, 100, decay_factor)
        self.num_symbols = num_symbols
        self.loss_type = loss_type
        if loss_type == 'mse':
            print("The loss type is MSE. Argument Lambda will be set to 1.")
            self.lmbda = 1
        self.gaussian_density = GaussianDensityEstimation(self.num_symbols)
        self.kernel_denstity = GaussianKernalDensityEstimation(chunk_size=512, kernel_size=kernel_size)
        self.factorized = factorized
            
    def forward(self, output, target):
        out = {}
        out["MSE"] = torch.mean((output["x_hat"] - target)**2, dim=(1, 2, 3))
        
        y = output['y']
        symbols = torch.concat([y.real, y.imag], dim=-1)
        if self.factorized:
            symbols = symbols.permute(1, 0).contiguous()
        
        out["gce"] = self.gaussian_density.entropy(symbols)
        out["entropy"] = self.kernel_denstity.entropy(symbols)
        out["kldiv"] = out["gce"] - out["entropy"]
        
        norm_S = symbols / torch.mean(symbols**2, dim=-1, keepdims=True)**0.5
        out["norm_gce"] = self.gaussian_density.entropy(norm_S)
        out["norm_ent"] = self.kernel_denstity.entropy(norm_S)
        out["norm_kl"] = out["norm_gce"] - out["norm_ent"]
    
        if self.factorized:
            out["gce"] = torch.mean(out["gce"])
            out["entropy"] = torch.mean(out["entropy"])
            out["kldiv"] = torch.mean(out["kldiv"])
            out["norm_gce"] = torch.mean(out["norm_gce"])
            out["norm_ent"] = torch.mean(out["norm_ent"])
            out["norm_kl"] = torch.mean(out["norm_kl"])
            
        if self.loss_type == 'mse':
            out["loss"] = torch.mean(out["MSE"])
        else:
            if self.loss_type == 'KL_div':
                density_loss = out["kldiv"]
            elif self.loss_type == 'norm_KL':
                density_loss = out["norm_kl"]
            else:
                assert "Not supported loss_type!"
            
            lmbda = self.lmbda_scheduler.get_lambda()
            if self.lmbda_scheduler.decay_factor != 1:
                out["lmbda"] = lmbda
            
            if self.factorized:
                mean_mse = torch.mean(out["MSE"])
                out["loss"] = mean_mse + lmbda * density_loss
            else:
                loss = out["MSE"] + lmbda * density_loss
                out["loss"] = torch.mean(loss)

        return out
    