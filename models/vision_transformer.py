"""
---------------------------------------------------------
Vision Transformer Model
Brain Tumor Classification
---------------------------------------------------------
"""

import torch
import torch.nn as nn
import timm


class VisionTransformerModel(nn.Module):

    def __init__(
        self,
        num_classes=4,
        model_name="vit_base_patch16_224",
        pretrained=True
    ):

        super(VisionTransformerModel, self).__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def build_model():

    model = VisionTransformerModel(
        num_classes=4,
        pretrained=True
    )

    return model


if __name__ == "__main__":

    model = build_model()

    print(model)

    sample = torch.randn(2, 3, 224, 224)

    output = model(sample)

    print("Output Shape :", output.shape)
