import torch
import torch.nn as nn
import numpy as np


class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  
        self.order = order  

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                    coe_y = torch.tensor(np.sqrt(init_focal / float(H)), requires_grad=False).float()
                elif self.order == 1:
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                    coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

    def forward(self, i=None):
        if self.fx_only:
            if self.order == 2:
                fxfy = torch.stack([self.fx ** 2 * self.W, self.fx ** 2 * self.W])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
        else:
            if self.order == 2:
                fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
            else:
                fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
        return fxfy
