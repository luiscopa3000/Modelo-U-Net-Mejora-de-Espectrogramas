import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------
#   Discriminator (modular)
# ---------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_features=32):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, 4, 2, 1),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.35)  # Dropout estratégico
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 1),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.35)  # Capa más profunda -> dropout más alto
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 1),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.35)  
        )
        self.final = nn.Sequential(
            nn.Conv2d(base_features * 8, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return self.final(x4)





