"""
InceptionV3
"""

import torch.nn as nn

from torchvision import models


class InceptionModel(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        output = self.model(x)

        if hasattr(output, "logits"):
            return output.logits

        return output
