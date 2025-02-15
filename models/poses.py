import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, my_devices, row, init_c2w=None):

        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.row = row
        self.bx = nn.Parameter(torch.tensor([0.001], dtype=torch.float32), requires_grad=learn_t)  # (1, )
        self.by = nn.Parameter(torch.tensor([0.001], dtype=torch.float32), requires_grad=learn_t)  # (1, )
        self.device = my_devices

    def forward(self, cam_id):
        x = self.bx*torch.tensor(cam_id//self.row - self.row//2).to(device=self.device) 
        y = self.by*torch.tensor(cam_id%self.row - self.row//2).to(device=self.device)        
        z = torch.tensor([0.0]).to(device=self.device)
        t = torch.cat([x, y, z], dim = 0)
        c2w = make_c2w(t)  # (4, 4)
            
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w