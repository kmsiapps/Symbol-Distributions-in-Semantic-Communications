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
    CheckboardMaskedConv2d,
)
from compressai.models.utils import conv, deconv

from models.semantic_layers.channel import AWGN
from models.NTSCC_PP.layer.layers import CrossAttentionLayer as CAL


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_val_dim, input_query_dim, output_dim, **kwargs):
        super().__init__()
        self.cal = CAL(
            **kwargs
        )
        dim = kwargs['dim']
        self.val_input_proj = conv1x1(input_val_dim, dim)
        self.val_output_proj = conv1x1(dim, output_dim)
        self.query_input_proj = conv1x1(input_query_dim, dim)
    
    def forward(self, v, q):
        v = self.val_input_proj(v)
        q = self.query_input_proj(q)
        v = self.cal(v, q)
        v = self.val_output_proj(v)
        
        return v


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


class SemELIC_Ctx(nn.Module):
    def __init__(self, N=192, M=320, num_slices=5, small=False, snrdB=10, num_symbols=16384, num_side_symbols=1024, snrdB_sim=10, **kwargs):
        super().__init__()
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices

        """
             N: channel number of main network
             M: channnel number of latent space
             
             num_symbols: # of symbols (main channel)
             num_side_symbols: # of symbols (hyperprior)
             
             snrdB_sim: snrdB of simulated channels (i.e., used for z_hat = h_s(channel(z)))
             
        """
        self.groups = [0, 16, 16, 32, 64, 192] #support depth
        self.anchor_non_anchor_ratio = 1 # num_anchor_symbols/num_non_anchor_symbols
        
        self.channel = AWGN(snrdB=snrdB)
        self.sim_channel = AWGN(snrdB=snrdB_sim)
        
        # Assume H, W = 256, 256
        H_s, H_w = 256//self.downsampling_factor, 256//self.downsampling_factor
        
        num_channels = num_symbols // (H_s * H_w) * 2
        num_anchor_channels = math.floor(num_symbols // (H_s * H_w) * 2 * (self.anchor_non_anchor_ratio / (self.anchor_non_anchor_ratio + 1)))
        num_non_anchor_channels = num_channels - num_anchor_channels
        
        num_anchor_channels *= 2
        num_non_anchor_channels *= 2
        # NOTE: num_anchor_channels and num_non_anchor_channels are multiplied by 2, as they are masked out in checkerboard pattern (see forward)
        
        multiplier = num_anchor_channels / sum(self.groups)
        self.proj_dim_anchor = [round(self.groups[i+1] * multiplier) for i in range(num_slices-1)]
        self.proj_dim_anchor.append(num_anchor_channels - sum(self.proj_dim_anchor))
        
        multiplier = num_non_anchor_channels / sum(self.groups)
        self.proj_dim_non_anchor = [round(self.groups[i+1] * multiplier) for i in range(num_slices-1)]
        self.proj_dim_non_anchor.append(num_non_anchor_channels - sum(self.proj_dim_non_anchor))
        
        depth = 1
        num_heads = 4
        window_size = 4
        dim = 128

        self.enc_anchor = \
        nn.ModuleList(
            # CrossAttentionLayer(
            #     input_val_dim=self.groups[i+1],
            #     input_query_dim=self.groups[i+1]*2,
            #     output_dim=self.proj_dim_anchor[i],
            #     dim=dim,
            #     input_resolution=(H_s, H_w),
            #     depth=depth,
            #     num_heads=num_heads,
            #     window_size=window_size    
            # ) for i in range(num_slices)
            conv1x1(self.groups[i+1]*3, self.proj_dim_anchor[i]) for i in range(num_slices)
        ) # TODO: maybe better architecture than naive 1x1 conv
        # TODO: set up ratio for enc_anchor and dec_anchor, and change proj_dim (enc_anchor + enc_non_anchor should produce num_symbols!)
        # TODO: see checkerboard paper for possible ratio?
        
        self.dec_anchor = nn.ModuleList(
            # CrossAttentionLayer(
            #     input_val_dim=self.proj_dim_anchor[i],
            #     input_query_dim=self.groups[i+1]*2,
            #     output_dim=self.groups[i+1],
            #     dim=dim,
            #     input_resolution=(H_s, H_w),
            #     depth=depth,
            #     num_heads=num_heads,
            #     window_size=window_size    
            # ) for i in range(num_slices)
            conv1x1(self.proj_dim_anchor[i] + self.groups[i+1]*2, self.groups[i+1]) for i in range(num_slices)
        )
        
        self.enc_non_anchor = nn.ModuleList(
            # CrossAttentionLayer(
            #     input_val_dim=self.groups[i+1],
            #     input_query_dim=self.groups[i+1]*2,
            #     output_dim=self.proj_dim_non_anchor[i],
            #     dim=dim,
            #     input_resolution=(H_s, H_w),
            #     depth=depth,
            #     num_heads=num_heads,
            #     window_size=window_size    
            # ) for i in range(num_slices)
            conv1x1(self.groups[i+1]*3, self.proj_dim_non_anchor[i]) for i in range(num_slices)
        )
        
        self.dec_non_anchor = nn.ModuleList(
            # CrossAttentionLayer(
            #     input_val_dim=self.proj_dim_non_anchor[i],
            #     input_query_dim=self.groups[i+1]*2,
            #     output_dim=self.groups[i+1],
            #     dim=dim,
            #     input_resolution=(H_s, H_w),
            #     depth=depth,
            #     num_heads=num_heads,
            #     window_size=window_size    
            # ) for i in range(num_slices)
            conv1x1(self.proj_dim_non_anchor[i] + self.groups[i+1]*2, self.groups[i+1]) for i in range(num_slices)
        )
        
        H_s, H_w = 256//self.downsampling_factor//4, 256//self.downsampling_factor//4
        # For z, 4x additional downsampling is done
        side_proj_dim = num_side_symbols // (H_s * H_w) * 2
        
        self.enc_side_proj = conv1x1(N, side_proj_dim)
        self.dec_side_proj = conv1x1(side_proj_dim, N)
        
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
            conv3x3(N*3//2, 2*M),
        )

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 224, stride=1,
                     kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1]*2, stride=1, kernel_size=5),
            ) for i in range(1,  num_slices)
        ) ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py

        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
            self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1
            ) for i in range(num_slices)
        )## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py

        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                        i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1]*2),
            ) for i in range(num_slices)
        ) ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数


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
        B, C, H, W = y.size() ## The shape of y to generate the mask

        z = self.h_a(y)
        z = self.enc_side_proj(z)
        
        B, C, H, W = z.shape
    
        # Encoder
        z = self.tensor_to_symbols(z)
        z_hat_tx = self.sim_channel(z)
        z_hat_tx = self.symbols_to_tensor(z_hat_tx, B, C, H, W)
        z_hat_tx = self.dec_side_proj(z_hat_tx)
        latent_meanscale_tx = self.h_s(z_hat_tx)
        
        # Decoder
        z_hat_rx = self.channel(z)
        z_hat_rx = self.symbols_to_tensor(z_hat_rx, B, C, H, W)
        z_hat_rx = self.dec_side_proj(z_hat_rx)
        latent_meanscale_rx = self.h_s(z_hat_rx)
        # # TODO: for test, using _tx (Fixed, but retaining here for a note)
        latent_meanscale_rx = latent_meanscale_tx
        
        # Encoder
        anchor_tx = torch.zeros_like(y).to(x.device)
        non_anchor_tx = torch.zeros_like(y).to(x.device)
        
        anchor_tx[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor_tx[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor_tx[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor_tx[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        
        anchor_split = torch.split(anchor_tx, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor_tx, self.groups[1:], 1)
        
        B, C, H, W = y.size()
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_slices = torch.split(y, self.groups[1:], 1)

        y_hat_slices_tx = []
        y_hat_slices_rx = []
        s_non_anchors = []
        s_anchors = []
        
        for slice_index, y_slice in enumerate(y_slices):
            # Channel-conditional coding
            if slice_index == 0:
                support_slices_tx = []
                support_slices_rx = []
            elif slice_index == 1:
                support_slices_tx = y_hat_slices_tx[0]
                support_slices_rx = y_hat_slices_rx[0]
                support_slices_meanscale_tx = self.cc_transforms[slice_index-1](support_slices_tx)
                support_slices_meanscale_rx = self.cc_transforms[slice_index-1](support_slices_rx)
            else:
                support_slices_tx = torch.concat([y_hat_slices_tx[0], y_hat_slices_tx[slice_index-1]], dim=1)
                support_slices_rx = torch.concat([y_hat_slices_rx[0], y_hat_slices_rx[slice_index-1]], dim=1)
                # Shouldn't it be torch.concat(y_hat_slices)? TODO
                support_slices_meanscale_tx = self.cc_transforms[slice_index-1](support_slices_tx)
                support_slices_meanscale_rx = self.cc_transforms[slice_index-1](support_slices_rx)
            
            ##support mean and scale
            # Encoder 
            support_tx = latent_meanscale_tx if slice_index == 0 else torch.concat(
                [support_slices_meanscale_tx, latent_meanscale_tx], dim=1)
            
            # Decoder
            support_rx = latent_meanscale_rx if slice_index == 0 else torch.concat(
                [support_slices_meanscale_rx, latent_meanscale_rx], dim=1)
            
            ### checkboard process 1
            ### Decode anchor only using channel information
            # Encoder
            meanscale_anchor_tx = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support_tx], dim=1))
            y_anchor = anchor_split[slice_index]
            s_anchor_masked = self.enc_anchor[slice_index](torch.concat([y_anchor, meanscale_anchor_tx], dim=1))
            # s_anchor_masked = self.enc_anchor[slice_index](y_anchor, meanscale_anchor_tx)
            
            # Assume W is evenly divisible by 2
            B, C, H, W = s_anchor_masked.shape
            s_anchor = torch.zeros((B, C, H, W//2)).to(s_anchor_masked.device)
            s_anchor[:, :, 0::2, :] = s_anchor_masked[:, :, 0::2, 0::2]
            s_anchor[:, :, 1::2, :] = s_anchor_masked[:, :, 1::2, 1::2]
            
            B, C, H, W = s_anchor.shape
            s_anchor = self.tensor_to_symbols(s_anchor)
            s_anchors.append(s_anchor)
            
            s_hat_anchor_tx = self.sim_channel(s_anchor)
            s_hat_anchor_tx = self.symbols_to_tensor(s_hat_anchor_tx, B, C, H, W)
            
            s_hat_anchor_masked_tx = torch.zeros((B, C, H, W*2)).to(s_anchor_masked.device)
            s_hat_anchor_masked_tx[:, :, 0::2, 0::2] = s_hat_anchor_tx[:, :, 0::2, :]
            s_hat_anchor_masked_tx[:, :, 1::2, 1::2] = s_hat_anchor_tx[:, :, 1::2, :]
            
            y_hat_anchor_tx = self.dec_anchor[slice_index](torch.concat([s_hat_anchor_masked_tx, meanscale_anchor_tx], dim=1))
            # y_hat_anchor_tx = self.dec_anchor[slice_index](s_hat_anchor_masked_tx, meanscale_anchor_tx)
            y_hat_anchor_tx[:, :, 0::2, 1::2] = 0
            y_hat_anchor_tx[:, :, 1::2, 0::2] = 0
            
            # Decoder
            meanscale_anchor_rx = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support_rx], dim=1))
            s_hat_anchor_rx = self.channel(s_anchor)
            s_hat_anchor_rx = self.symbols_to_tensor(s_hat_anchor_rx, B, C, H, W)
            
            s_hat_anchor_masked_rx = torch.zeros((B, C, H, W*2)).to(s_anchor_masked.device)
            s_hat_anchor_masked_rx[:, :, 0::2, 0::2] = s_hat_anchor_rx[:, :, 0::2, :]
            s_hat_anchor_masked_rx[:, :, 1::2, 1::2] = s_hat_anchor_rx[:, :, 1::2, :]
            
            # y_hat_anchor_rx = self.dec_anchor[slice_index](torch.concat([s_hat_anchor_masked_rx, meanscale_anchor_rx], dim=1))
            # y_hat_anchor_rx = self.dec_anchor[slice_index](s_hat_anchor_masked_rx, meanscale_anchor_rx)
            y_hat_anchor_rx = self.dec_anchor[slice_index](torch.concat([s_hat_anchor_masked_tx, meanscale_anchor_tx], dim=1))
            # TODO: this is rx, but for test, using tx (Fixed, but retaining here for a note)
            y_hat_anchor_rx[:, :, 0::2, 1::2] = 0
            y_hat_anchor_rx[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            # Encoder
            masked_context_tx = self.context_prediction[slice_index](y_hat_anchor_tx)
            meanscale_non_anchor_tx = self.ParamAggregation[slice_index](
                torch.concat([masked_context_tx, support_tx], dim=1))
            
            # Decoder
            masked_context_rx = self.context_prediction[slice_index](y_hat_anchor_rx)
            meanscale_non_anchor_rx = self.ParamAggregation[slice_index](
                torch.concat([masked_context_rx, support_rx], dim=1))
            
            # Encoder
            y_non_anchor = non_anchor_split[slice_index]
            s_non_anchor_masked = self.enc_non_anchor[slice_index](torch.concat([y_non_anchor, meanscale_non_anchor_tx], dim=1))
            # s_non_anchor_masked = self.enc_non_anchor[slice_index](y_non_anchor, meanscale_non_anchor_tx)
            
            # Assume W is evenly divisible by 2
            B, C, H, W = s_non_anchor_masked.shape
            s_non_anchor = torch.zeros((B, C, H, W//2)).to(s_non_anchor_masked.device)
            s_non_anchor[:, :, 0::2, :] = s_non_anchor_masked[:, :, 0::2, 1::2]
            s_non_anchor[:, :, 1::2, :] = s_non_anchor_masked[:, :, 1::2, 0::2]
            
            B, C, H, W = s_non_anchor.shape
            s_non_anchor = self.tensor_to_symbols(s_non_anchor)
            s_non_anchors.append(s_non_anchor)
            
            s_hat_non_anchor_tx = self.sim_channel(s_non_anchor)
            s_hat_non_anchor_tx = self.symbols_to_tensor(s_hat_non_anchor_tx, B, C, H, W)
            
            s_hat_non_anchor_masked_tx = torch.zeros((B, C, H, W*2)).to(s_non_anchor_masked.device)
            s_hat_non_anchor_masked_tx[:, :, 0::2, 1::2] = s_hat_non_anchor_tx[:, :, 0::2, :]
            s_hat_non_anchor_masked_tx[:, :, 1::2, 0::2] = s_hat_non_anchor_tx[:, :, 1::2, :]
            
            y_hat_non_anchor_tx = self.dec_non_anchor[slice_index](torch.concat([s_hat_non_anchor_masked_tx, meanscale_non_anchor_tx], dim=1))
            # y_hat_non_anchor_tx = self.dec_non_anchor[slice_index](s_hat_non_anchor_masked_tx, meanscale_non_anchor_tx)
            y_hat_non_anchor_tx[:, :, 0::2, 0::2] = 0
            y_hat_non_anchor_tx[:, :, 1::2, 1::2] = 0
            
            # Decoder
            s_hat_non_anchor_rx = self.channel(s_non_anchor)
            s_hat_non_anchor_rx = self.symbols_to_tensor(s_hat_non_anchor_rx, B, C, H, W)
            
            s_hat_non_anchor_masked_rx = torch.zeros((B, C, H, W*2)).to(s_non_anchor_masked.device)
            s_hat_non_anchor_masked_rx[:, :, 0::2, 1::2] = s_hat_non_anchor_rx[:, :, 0::2, :]
            s_hat_non_anchor_masked_rx[:, :, 1::2, 0::2] = s_hat_non_anchor_rx[:, :, 1::2, :]
            
            # y_hat_non_anchor_rx = self.dec_non_anchor[slice_index](torch.concat([s_hat_non_anchor_masked_rx, meanscale_non_anchor_rx], dim=1))
            # y_hat_non_anchor_rx = self.dec_non_anchor[slice_index](s_hat_non_anchor_masked_rx, meanscale_non_anchor_rx)
            y_hat_non_anchor_rx = self.dec_non_anchor[slice_index](torch.concat([s_hat_non_anchor_masked_tx, meanscale_non_anchor_tx], dim=1))
            # TODO: this is _rx, but for test, using _tx (Fixed, but retaining here for a note)
            y_hat_non_anchor_rx[:, :, 0::2, 0::2] = 0
            y_hat_non_anchor_rx[:, :, 1::2, 1::2] = 0
            
            # Encoder
            y_hat_slice_tx = y_hat_anchor_tx + y_hat_non_anchor_tx
            y_hat_slices_tx.append(y_hat_slice_tx)
            
            # Decoder
            y_hat_slice_rx = y_hat_anchor_rx + y_hat_non_anchor_rx
            y_hat_slices_rx.append(y_hat_slice_rx)
            
            
        y_hat = torch.cat(y_hat_slices_rx, dim=1)
        x_hat = self.g_s(y_hat)
        
        # symbols = torch.cat([z] + s_anchors + s_non_anchors, axis=1)
        symbols = z

        return {
            "x_hat": x_hat,
            "y": symbols, # TODO: make it real symbols to fix density loss (currently placeholder)
            
            "z": z,
            "s_anchors": s_anchors,
            "s_non_anchors": s_non_anchors,
        }
