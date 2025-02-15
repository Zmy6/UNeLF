import torch
import torch.nn as nn
import numpy as np


class scale(nn.Module):
    def __init__(self, H, W, req_grad, init_scale=None):
        super(scale, self).__init__()
        self.H = H
        self.W = W

        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32), requires_grad=req_grad)  
    
    def forward(self, i=None): 

        sc = self.scale
        return sc