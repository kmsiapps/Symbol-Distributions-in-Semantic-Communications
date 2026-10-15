import torch
import torch.nn as nn
    
class AWGN(nn.Module):
    def __init__(self, snrdB):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # dB to normal scale
        
    def forward(self, x):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"
        assert len(x.shape) == 2, "input shape should be (B, num_symbols)"
        
        # Noise Generation
        sig_power = torch.mean(torch.abs(x)**2)
        no = torch.sqrt(sig_power/self.snr)
        n = no * torch.randn(x.shape, dtype=torch.complex64, device='cuda')
        
        # Apply
        y = x + n
    
        return y
    
    