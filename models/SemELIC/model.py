"""
A simple test algorithm to rewrite the network
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from compressai.models.utils import conv, deconv

from models.SemELIC.block import AttentionBlock, ResidualBlock, conv1x1
from models.semantic_layers.channel import AWGN

class SemELIC(nn.Module):
    def __init__(self, N=192, M=320, small=False, snrdB=10, num_symbols=16384, **kwargs):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
        
        if small:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResidualBlock(N),
                conv(N, N),
                ResidualBlock(N),
                conv(N, N),
                ResidualBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResidualBlock(N),
                deconv(N, N),
                ResidualBlock(N),
                deconv(N, N),
                ResidualBlock(N),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                conv(N, N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                AttentionBlock(N),
                conv(N, N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                conv(N, M),
                AttentionBlock(M),
            )

            self.g_s = nn.Sequential(
                AttentionBlock(M),
                deconv(M, N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                deconv(N, N),
                AttentionBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                deconv(N, N),
                ResidualBlock(N),
                ResidualBlock(N),
                ResidualBlock(N),
                deconv(N, 3),
            )
        
        # Assume H, W = 256, 256
        if 'CIFAR' in kwargs and kwargs['CIFAR'] == True:
            HW = 32
        else:
            HW = 256
        H_s, H_w = HW//self.downsampling_factor, HW//self.downsampling_factor
        proj_dim = num_symbols // (H_s * H_w) * 2
        self.enc_proj = conv1x1(M, proj_dim) 
        self.dec_proj = conv1x1(proj_dim, M)
        self.channel = AWGN(snrdB=snrdB)


    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        y = self.g_a(x)
        y = self.enc_proj(y)
        y = y.permute(0, 2, 3, 1).contiguous() # BCHW to BHWC
        B, H, W, C = y.shape
        y = y.reshape(B, -1, 2)
        y = torch.complex(y[..., 0], y[..., 1])
        
        y_hat = self.channel(y)
        
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = y_hat.reshape(B, H, W, C)
        y_hat = y_hat.permute(0, 3, 1, 2).contiguous() # BHWC to BCHW
        y_hat = self.dec_proj(y_hat)
        x_hat = self.g_s(y_hat)

        # return x_hat
        return {
            "x_hat": x_hat, "y":y
        }

