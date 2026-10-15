import numpy as np
import torch
import torch.nn as nn

__all__ = ['GaussianDensityEstimation', 'GaussianKernalDensityEstimation', 'VLCPowerDensityEstimation']

class GaussianDensityEstimation(nn.Module):
    def __init__(self, num_symbols=512):
        super().__init__()
        self.eps = 1e-30
        self.sigma_eps = 1e-5
        
    def prob(self, x, mu=0, sigma=1):
        '''
        Calculates probability density value of 'x' in Gaussian distribution
        '''
        # sigma = torch.where(sigma < self.sigma_eps, self.sigma_eps, sigma)
        # print(x.shape, '!')
        # if torch.is_tensor(mu):
        #     print(mu.shape, '!!')
        # if torch.is_tensor(sigma):
            # print(sigma.shape, '!!!')
        exponent = -1/2 * ((x - mu)/sigma)**2
        # if torch.is_tensor(exponent):
        #     print(exponent.shape, '!!!!')
        pdf = 1 / (sigma * (2 * np.pi)**0.5) * torch.exp(exponent)
        return pdf
    
    def entropy(self, x, mu=0, sigma=1):
        '''
        Calculates Gaussian cross entropy of x
        '''
        pdf = self.prob(x, mu, sigma)
        return -torch.mean(torch.log(pdf + self.eps), dim=-1)


class GaussianKernalDensityEstimation(nn.Module):
    def __init__(self, chunk_size, kernel_size=None):
        super().__init__()
        self.eps = 1e-30      
        self.h = kernel_size
        self.chunk_size = chunk_size
        self.normal = GaussianDensityEstimation(chunk_size)
        
    def chunk_prob(self, x, x_point=None):
        if x_point is None:
            x_point = x
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, axis=0)
        if len(x_point.shape) == 1:
            x_point = torch.unsqueeze(x_point, axis=0)
        
        num_x = x.shape[-1]
        if self.h is None:
            h = 1.06 * num_x**(-0.2)
        else:
            h = self.h
        x_point = x_point.unsqueeze(dim=-1).repeat_interleave(num_x, dim=-1)
        x_point = x_point.permute(1, 0, 2).contiguous()
            
        kde_pdf = torch.mean(self.normal.prob(x_point, mu=x, sigma=h), dim=-1)
        kde_pdf = kde_pdf.permute(1, 0).contiguous()
        return kde_pdf
    
    def prob(self, x, x_point=None):
        B, num_symbols = x.shape
        num_chunks = 0
        if num_symbols > self.chunk_size * 2:
            if num_symbols % self.chunk_size != 0:
                num_chunks = num_symbols // self.chunk_size - 1
                x_res = x[:,self.chunk_size*num_chunks:]
                x = x[:,:self.chunk_size*num_chunks]
            else:
                num_chunks = num_symbols // self.chunk_size
                x_res = None
            x = x.reshape(B*num_chunks, self.chunk_size)
            chunked = True
        else:
            chunked = False
        
        if x_point is None: # point based probability
            if chunked:
                pdf = self.chunk_prob(x, None).reshape(B, self.chunk_size*num_chunks)
                if x_res is not None:
                    pdf_res = self.chunk_prob(x_res, None)
                    pdf = torch.concat([pdf, pdf_res], dim=-1)
            else:
                pdf = self.chunk_prob(x, None)
        else: # requires distribution
            x_len = len(x_point)
            if chunked:
                pdf = self.chunk_prob(x, x_point).reshape(B, num_chunks, x_len)
                if x_res is not None:
                    pdf_res = self.chunk_prob(x_res, x_point).unsqueeze(1)
                    pdf = torch.concat([pdf, pdf_res], dim=1)
            else:
                pdf = self.chunk_prob(x, x_point).unsqueeze(1)
            pdf = pdf.mean(dim=1, keepdim=False)
        return pdf

    def entropy(self, x):
        pdf = self.prob(x)
        return -torch.mean(torch.log(pdf + self.eps), dim=-1)
    
    
class VLCPowerDensityEstimation(nn.Module):
    def __init__(self, SNRdB=10):
        super().__init__()
        self.n = 10 ** (- SNRdB / 20)
        
    def prob(self, x):
        '''
        Calculates probability density value of 'x' in VLC distribution
        '''
        # pdf = 1 / (np.pi**2 * x**2 + 1)
        pdf = 1 / (np.pi * self.n) / (1 + x**2/self.n**2)
        return pdf
    
    # def entropy(self, x):
    #     '''
    #     Calculates VLC cross entropy of x
    #     '''
    #     pdf = self.prob(x)
    #     return -torch.mean(torch.log(pdf + self.eps))
    
