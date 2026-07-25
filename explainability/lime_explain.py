"""
---------------------------------------------------------
LIME Explanation
---------------------------------------------------------
"""

import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from lime import lime_image

from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt

from models.vision_transformer import VisionTransformerModel


DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)


model = VisionTransformerModel(num_classes=4)

model.load_state_dict(

    torch.load(

        "../outputs/models/best_vit_model.pth",

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


def batch_predict(images):

    model.eval()

    batch = torch.stack([

        transform(Image.fromarray(img))

        for img in images

    ])

    batch = batch.to(DEVICE)

    with torch.no_grad():

        outputs = model(batch)

        probabilities = torch.softmax(outputs,dim=1)

    return probabilities.cpu().numpy()


def explain(image_path):

    image = np.array(

        Image.open(image_path).convert("RGB")

    )

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(

        image,

        batch_predict,

        top_labels=1,

        hide_color=0,

        num_samples=1000

    )

    temp, mask = explanation.get_image_and_mask(

        explanation.top_labels[0],

        positive_only=True,

        num_features=10,

        hide_rest=False

    )

    plt.imshow(

        mark_boundaries(temp,mask)

    )

    plt.axis("off")

    plt.savefig(

        "../outputs/heatmaps/lime_result.png",

        bbox_inches="tight"

    )

    print("LIME Result Saved")
