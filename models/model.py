# # Copyright (C) 2023 OPPO. All rights reserved.

# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# import torch
# from torch.nn.parameter import Parameter
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# from typing import List, Optional, Union
# from collections import OrderedDict

# class NeuLFmm(nn.Module):
#     def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
#         super(NeuLFmm, self).__init__()

#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skips

#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
#         # self.views_linears = nn.ModuleList([nn.Linear(3 + W, W//2)])
#         self.feature_linear = nn.Linear(W, W)
#         self.normal_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
#         self.normal_act = nn.Tanh()
#         self.normal_linears2 = nn.Linear(W//2, 3)
#         self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
#         self.pts_linears_rgb = nn.ModuleList(
#         [nn.Linear(input_ch+3, W+3)] + [nn.Linear(W+3, W+3) if i not in self.skips else nn.Linear(W+3 + input_ch+3, W+3) for i in range(D-1)])
#         self.feature_linear_rgb = nn.Linear(W+3, W)
#         self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
#         self.rgb_linear = nn.Linear(W//2, 3)
#         self.rgb_act = nn.Sigmoid()

#     def forward(self,x_input):
#         # positional embedding
#         x = x_input
#         x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
#         o1 = torch.sin(x)
#         o2 = torch.cos(x)
#         input_pts = torch.cat([o1, o2], dim=-1)

#         h = input_pts
#         # normal embedding
#         for i, l in enumerate(self.pts_linears):
#             h = self.pts_linears[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts, h], -1)

#         feature = self.feature_linear(h)
#         h = feature

#         # normal
#         for i, l in enumerate(self.normal_linears):
#             h_normal = self.normal_linears[i](h)
#             h_normal = F.relu(h_normal)

#         normal = self.normal_act( self.normal_linears2(h_normal))
#         input_pts_na = torch.cat([input_pts,normal],dim=-1)
#         h = input_pts_na
        
#         # neural light field
#         for i, l in enumerate(self.pts_linears_rgb):
#             h = self.pts_linears_rgb[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([input_pts_na, h], -1)
#         # if self.use_viewdirs:
#         feature = self.feature_linear_rgb(h)
#         h = feature
    
#         for i, l in enumerate(self.views_linears):
#             h = self.views_linears[i](h)
#             h = F.relu(h)

#         rgb = self.rgb_linear(h)
#         rgb = self.rgb_act(rgb)

#         return rgb

# class NeuLFmm_won(nn.Module):
#     def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
#         super(NeuLFmm_won, self).__init__()

#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skips

#         self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
#         self.pts_linears_rgb = nn.ModuleList(
#         [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
#         self.feature_linear_rgb = nn.Linear(W, W)
#         self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
#         self.rgb_linear = nn.Linear(W//2, 3) 
#         self.rgb_act = nn.Sigmoid()

#     def forward(self,x_input):
#         # positional embedding
#         x = x_input
#         x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
#         o1 = torch.sin(x)
#         o2 = torch.cos(x)
#         input_pts = torch.cat([o1, o2], dim=-1)
#         h = input_pts
#         feature = input_pts
    
#         # neural light field
#         for i, l in enumerate(self.pts_linears_rgb):
#             h = self.pts_linears_rgb[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([feature, h], -1)
#         # if self.use_viewdirs:
#         feature = self.feature_linear_rgb(h)
#         h = feature
    
#         for i, l in enumerate(self.views_linears):
#             h = self.views_linears[i](h)
#             h = F.relu(h)

#         rgb = self.rgb_linear(h)
#         rgb = self.rgb_act(rgb)

#         return rgb
# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Union
from collections import OrderedDict

class LeakyReLU(nn.Module):
    def __init__(
        self,
        a
    ):
        super().__init__()


        self.a = a

        self.act = nn.LeakyReLU(self.a, True)

    def forward(self, x):
        return self.act(x)

class WindowedPE(nn.Module):
    def __init__(
        self, in_channels, n_freqs = 0, cur_iter = 0, wait_iters = 0, max_freq_iter = 0, freq_multiplier = 2.0
    ):
        super().__init__()

        self.n_freqs = n_freqs
        self.cur_iter = cur_iter
        self.wait_iters = wait_iters
        self.max_freq_iter = float(max_freq_iter)
        self.exclude_identity = False

        self.funcs = [torch.sin, torch.cos]
        self.freq_multiplier = 2.0
        self.freq_bands = self.freq_multiplier ** torch.linspace(1, self.n_freqs, self.n_freqs)

        self.in_channels = in_channels
        if self.exclude_identity:
            self.out_channels = in_channels * (len(self.funcs) * self.n_freqs)
        else:
            self.out_channels = in_channels * (len(self.funcs) * self.n_freqs + 1)

        self.dummy_layer = nn.Linear(1, 1)

    def weight(self, j):
        if self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter < self.wait_iters:
            return 0.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0

        cur_iter = (self.cur_iter - self.wait_iters)
        alpha = (cur_iter / self.max_freq_iter) * self.n_freqs
        return (1.0 - np.cos(np.pi * np.clip(alpha - j, 0.0, 1.0))) / 2

    def forward(self, x):
        out = []

        if not self.exclude_identity:
            out += [x]

        for j, freq in enumerate(self.freq_bands):
            for func in self.funcs:
                out += [self.weight(j) * func(freq * x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i

class NeuLFmmPE(nn.Module):
    def __init__(self,D=8,W=256,input_ch=102,skips=[4]):
        super(NeuLFmmPE, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # self.views_linears = nn.ModuleList([nn.Linear(3 + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.normal_linears = nn.ModuleList([nn.Linear(W, W//2)])
        # self.normal_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.normal_act = nn.Tanh()
        self.normal_linears2 = nn.Linear(W//2, 3)
        # self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
        self.pts_linears_rgb = nn.ModuleList(
        [nn.Linear(input_ch+3, W+3)] + [nn.Linear(W+3, W+3) if i not in self.skips else nn.Linear(W+3 + input_ch+3, W+3) for i in range(D-1)])
        self.feature_linear_rgb = nn.Linear(W+3, W)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.views_linears = nn.ModuleList([nn.Linear(W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)
        self.rgb_act = nn.Sigmoid()
        # self.pe = WindowedPE(in_channels = 6, max_freq_iter=0)
        self.pe = WindowedPE(in_channels = 6, n_freqs = 8, max_freq_iter = 6000)

    def forward(self,x_input):
        # positional embedding
       
        # print(x_input.shape) torch.Size([1024, 6])
    
        # x = x_input
        # x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
        # o1 = torch.sin(x)
        # o2 = torch.cos(x)
        # input_pts = torch.cat([o1, o2], dim=-1) # torch.Size([1024, 256])

        input_pts = self.pe(x_input) # torch.Size([1024, 102])
        h = input_pts
        # normal embedding
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        feature = self.feature_linear(h)
        h = feature

        # normal
        for i, l in enumerate(self.normal_linears):
            h_normal = self.normal_linears[i](h)
            h_normal = F.relu(h_normal)

        normal = self.normal_act( self.normal_linears2(h_normal))
        input_pts_na = torch.cat([input_pts,normal],dim=-1)
        h = input_pts_na
        
        # neural light field
        for i, l in enumerate(self.pts_linears_rgb):
            h = self.pts_linears_rgb[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_na, h], -1)
        # if self.use_viewdirs:
        feature = self.feature_linear_rgb(h)
        h = feature
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)

        return rgb

class NeuLFmm(nn.Module):
    def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
        super(NeuLFmm, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # self.views_linears = nn.ModuleList([nn.Linear(3 + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.normal_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.normal_act = nn.Tanh()
        self.normal_linears2 = nn.Linear(W//2, 3)
        self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
        self.pts_linears_rgb = nn.ModuleList(
        [nn.Linear(input_ch+3, W+3)] + [nn.Linear(W+3, W+3) if i not in self.skips else nn.Linear(W+3 + input_ch+3, W+3) for i in range(D-1)])
        self.feature_linear_rgb = nn.Linear(W+3, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)
        self.rgb_act = nn.Sigmoid()

    def forward(self,x_input):
        # positional embedding
        x = x_input
        x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
        o1 = torch.sin(x)
        o2 = torch.cos(x)
        input_pts = torch.cat([o1, o2], dim=-1)

        h = input_pts
        # normal embedding
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        feature = self.feature_linear(h)
        h = feature

        # normal
        for i, l in enumerate(self.normal_linears):
            h_normal = self.normal_linears[i](h)
            h_normal = F.relu(h_normal)

        normal = self.normal_act( self.normal_linears2(h_normal))
        input_pts_na = torch.cat([input_pts,normal],dim=-1)
        h = input_pts_na
        
        # neural light field
        for i, l in enumerate(self.pts_linears_rgb):
            h = self.pts_linears_rgb[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts_na, h], -1)
        # if self.use_viewdirs:
        feature = self.feature_linear_rgb(h)
        h = feature
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)

        return rgb

class NeuLFmm_won(nn.Module):
    def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
        super(NeuLFmm_won, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
        self.pts_linears_rgb = nn.ModuleList(
        [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.feature_linear_rgb = nn.Linear(W, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3) 
        self.rgb_act = nn.Sigmoid()

    def forward(self,x_input):
        # positional embedding
        x = x_input
        x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
        o1 = torch.sin(x)
        o2 = torch.cos(x)
        input_pts = torch.cat([o1, o2], dim=-1)
        h = input_pts
        feature = input_pts
    
        # neural light field
        for i, l in enumerate(self.pts_linears_rgb):
            h = self.pts_linears_rgb[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([feature, h], -1)
        # if self.use_viewdirs:
        feature = self.feature_linear_rgb(h)
        h = feature
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)

        return rgb

class UNeLF(nn.Module):
    def __init__(self,D=8,W=256,input_ch=256,skips=[4]):
        super(UNeLF, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # self.views_linears = nn.ModuleList([nn.Linear(3 + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        # self.normal_linears = nn.ModuleList([nn.Linear(input_ch+3, W//2)])
        self.normal_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.normal_act = nn.Tanh()
        self.normal_linears2 = nn.Linear(W//2, 3)
        # self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 3)), requires_grad=False)
        self.b_normalAlbedo = Parameter(torch.normal(mean=0.0, std=16, size=(int(input_ch/2), 6)), requires_grad=False)
        self.pts_linears_rgb = nn.ModuleList(
        [nn.Linear(input_ch+3, W+3)] + [nn.Linear(W+3, W+3) if i not in self.skips else nn.Linear(W+3 + input_ch+3, W+3) for i in range(D-1)])
        self.feature_linear_rgb = nn.Linear(W+3, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)
        self.rgb_act = nn.Sigmoid()
        self.layer_activation = LeakyReLU(0.25)


    def forward(self,x_input):
        # positional embedding
        # x = x_input[:, 0:3]
        # x_dir = x_input[:, 3:6]
        x = x_input
        x = (2.0 * np.pi * x) @ self.b_normalAlbedo.T
        o1 = torch.sin(x)
        o2 = torch.cos(x)
        input_pts = torch.cat([o1, o2], dim=-1)
        h = input_pts
        # normal embedding
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.layer_activation(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        feature = self.feature_linear(h)
        # h = torch.cat([feature, x_dir], -1)
        h = feature
        # print(h.shape)
        # normal
        for i, l in enumerate(self.normal_linears):
            h_normal = self.normal_linears[i](h)
            h_normal = self.layer_activation(h_normal)

        normal = self.normal_act( self.normal_linears2(h_normal))
        input_pts_na = torch.cat([input_pts,normal],dim=-1)
        h = input_pts_na
        
        # neural light field
        for i, l in enumerate(self.pts_linears_rgb):
            h = self.pts_linears_rgb[i](h)
            h = self.layer_activation(h)
            if i in self.skips:
                h = torch.cat([input_pts_na, h], -1)
        # if self.use_viewdirs:
        feature = self.feature_linear_rgb(h)
        h = feature
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = self.layer_activation(h)

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)

        return rgb