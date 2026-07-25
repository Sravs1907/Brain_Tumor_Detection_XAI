"""
---------------------------------------------------------
Grad-CAM for Vision Transformer
---------------------------------------------------------
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.vision_transformer import VisionTransformerModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAMGenerator:

    def __init__(self):

        self.model = VisionTransformerModel(num_classes=4)

        self.model.load_state_dict(

            torch.load(

                "../outputs/models/best_vit_model.pth",

                map_location=DEVICE

            )

        )

        self.model.to(DEVICE)

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Resize((224,224)),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[0.485,0.456,0.406],

                std=[0.229,0.224,0.225]

            )

        ])

    def generate(self,image_path):

        image = Image.open(image_path).convert("RGB")

        rgb_image = np.array(image)/255.0

        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        target_layer = self.model.model.blocks[-1].norm1

        cam = GradCAM(

            model=self.model,

            target_layers=[target_layer]

        )

        grayscale_cam = cam(

            input_tensor=input_tensor

        )[0]

        visualization = show_cam_on_image(

            rgb_image,

            grayscale_cam,

            use_rgb=True

        )

        os.makedirs(

            "../outputs/heatmaps",

            exist_ok=True

        )

        output_path = "../outputs/heatmaps/gradcam_result.jpg"

        cv2.imwrite(

            output_path,

            cv2.cvtColor(

                visualization,

                cv2.COLOR_RGB2BGR

            )

        )

        print("Grad-CAM saved to:", output_path)


if __name__=="__main__":

    GradCAMGenerator().generate("sample.jpg")
