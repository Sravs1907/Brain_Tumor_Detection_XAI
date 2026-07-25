"""
---------------------------------------------------------
SHAP Explanation
---------------------------------------------------------
"""

import shap

import torch

import numpy as np

from PIL import Image

from torchvision import transforms

from models.vision_transformer import VisionTransformerModel


DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

])


model = VisionTransformerModel(num_classes=4)

model.load_state_dict(

    torch.load(

        "../outputs/models/best_vit_model.pth",

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()


def explain(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    background = torch.zeros_like(tensor)

    explainer = shap.DeepExplainer(

        model,

        background

    )

    shap_values = explainer.shap_values(

        tensor

    )

    shap.image_plot(

        shap_values,

        tensor.cpu().numpy()

    )

    print("SHAP Explanation Generated")


if __name__=="__main__":

    explain("sample.jpg")
