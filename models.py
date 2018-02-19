from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from ProxProp import ProxPropLinear, ProxPropConv2d

class ProxPropConvNet(nn.Module):
    def __init__(self, input_size, num_classes, tau_prox, optimization_mode='prox_cg1'):
        super(ProxPropConvNet, self).__init__()
        img_dim = input_size[0]
        self.layers = nn.Sequential(
            ProxPropConv2d(img_dim, 16, kernel_size=5, tau_prox=tau_prox, padding=2, optimization_mode=optimization_mode),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ProxPropConv2d(16, 20, kernel_size=5, tau_prox=tau_prox, padding=2, optimization_mode=optimization_mode),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ProxPropConv2d(20, 20, kernel_size=5, tau_prox=tau_prox, padding=2, optimization_mode=optimization_mode),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(input_size[1]*input_size[2]//16 * 20 , 10)

    def forward(self, x):
        x = self.layers(x)
        return self.final_fc(x.view(x.size(0), -1))


class ProxPropMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, tau_prox=1., optimization_mode='prox_cg1'):
        super(ProxPropMLP, self).__init__()
        input_size_flat = reduce(mul, input_size, 1)
        self.layers = []
        self.layers.append(ProxPropLinear(input_size_flat, hidden_sizes[0], tau_prox=tau_prox, optimization_mode=optimization_mode))
        for k, _ in enumerate(hidden_sizes[:-1]):
            self.layers.append(ProxPropLinear(hidden_sizes[k], hidden_sizes[k+1], tau_prox=tau_prox, optimization_mode=optimization_mode))
        self.layers = nn.ModuleList(self.layers)
        self.final_fc = nn.Linear(hidden_sizes[-1], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.final_fc(x)
        return x

