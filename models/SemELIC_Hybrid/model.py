"""
A simple test algorithm to rewrite the network
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import trunc_normal_
from models.ELIC.ELICUtilis.layers import (
    AttentionBlock,
    conv3x3,
)

from compressai.models.google import CompressionModel
from compressai.models.utils import conv, deconv

from models.semantic_layers.channel import AWGN

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

class ELIC_Hybrid(CompressionModel):

    def __init__(self, N=192, M=320, num_symbols=16384, snrdB=10, small=False, **kwargs):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        self.num_symbols = num_symbols

        """
             N: channel number of main network
             M: channnel number of latent space
             
        """
        if small:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResidualBottleneckBlock(N),
                conv(N, N),
                ResidualBottleneckBlock(N),
                conv(N, N),
                ResidualBottleneckBlock(N),
                conv(N, M),
            )

            self.g_s = nn.Sequential(
                deconv(M, N),
                ResidualBottleneckBlock(N),
                deconv(N, N),
                ResidualBottleneckBlock(N),
                deconv(N, N),
                ResidualBottleneckBlock(N),
                deconv(N, 3),
            )
        else:
            self.g_a = nn.Sequential(
                conv(3, N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                conv(N, N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                AttentionBlock(N),
                conv(N, N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                conv(N, M),
                AttentionBlock(M),
            )

            self.g_s = nn.Sequential(
                AttentionBlock(M),
                deconv(M, N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                deconv(N, N),
                AttentionBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                deconv(N, N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                ResidualBottleneckBlock(N),
                deconv(N, 3),
            )

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N*3//2),
            nn.ReLU(inplace=True),
            conv3x3(N*3//2, M),
        )
        
        self.ntc_enc = nn.Sequential(
            conv1x1(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, M)
        )
        
        num_symbol_channels = self.num_symbols // 16 // 16 * 2
        self.jscc_enc = nn.Sequential(
            conv1x1(M, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, num_symbol_channels)
        )
        
        self.channel = AWGN(snrdB=snrdB)
        
        self.hybrid_dec = conv1x1(num_symbol_channels + M, M)


    def tensor_to_symbols(self, z):
        z = z.clone()
        z = z.permute(0, 2, 3, 1).contiguous() # BCHW to BHWC
        B, H, W, C = z.shape
        z = z.reshape(B, -1, 2)
        z = torch.complex(z[..., 0], z[..., 1])
        return z

    def symbols_to_tensor(self, y_hat, B, C, H, W):
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = y_hat.reshape(B, H, W, C)
        y_hat = y_hat.permute(0, 3, 1, 2).contiguous() # BHWC to BCHW
        
        return y_hat


    def forward(self, x):
        y = self.g_a(x)
        
        y_jscc = self.jscc_enc(y)
        
        B, C, H, W = y_jscc.shape
        y_jscc = self.tensor_to_symbols(y_jscc)
        y_jscc_hat = self.channel(y_jscc)
        y_jscc_hat = self.symbols_to_tensor(y_jscc_hat, B, C, H, W)
        

        y_ntc = self.ntc_enc(y)
        z = self.h_a(y_ntc)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_ntc_hat = self.h_s(z_hat)
        
        y_hat = self.hybrid_dec(torch.concat([y_jscc_hat, y_ntc_hat], dim=1))
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"z": z_likelihoods},
        }
