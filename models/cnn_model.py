"""
Simple CNN Baseline
"""

import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        return self.network(x)
