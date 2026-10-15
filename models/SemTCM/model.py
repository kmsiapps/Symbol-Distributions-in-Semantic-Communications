from compressai.layers import (
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    conv1x1
)

import torch.nn as nn
import torch

from models.SemTCM.block import ConvTransBlock
from models.semantic_layers.channel import AWGN

class SemTCM(nn.Module):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128,  M=320, snrdB=10, num_symbols=16384, **kwargs):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        dim = N
        self.M = M
        self.snrdB = snrdB
        self.num_symbols = num_symbols
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)
        
        # Assume H, W = 256, 256
        H_s, H_w = 256//self.downsampling_factor, 256//self.downsampling_factor
        proj_dim = num_symbols // (H_s * H_w)
        self.enc_proj = conv1x1(M, proj_dim) 
        self.dec_proj = conv1x1(proj_dim, M)
        self.channel = AWGN(snrdB=snrdB)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4
    
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
