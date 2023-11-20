import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu, softmax


class SmallNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        self.convMid = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(256, n_class, kernel_size=5, padding=2)
        self.softmaxAct = nn.Softmax2d()

    def forward(self, x):
        x1 = relu(self.conv1(x))
        x2 = self.pool(x1)
        x3 = relu(self.convMid(x2))
        x4 = self.upconv(x3)

        x5 = torch.cat([x4, x1], dim=1)
        out = self.softmaxAct(relu(self.conv2(x5)))
        return out  


import SmallRun

SmallRun.run(SmallNet)