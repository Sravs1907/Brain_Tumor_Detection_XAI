"""
---------------------------------------------------------
Prediction Script
---------------------------------------------------------
"""

import torch

from torchvision import transforms

from PIL import Image

from models.vision_transformer import VisionTransformerModel

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)

classes = [

    "Glioma",

    "Meningioma",

    "No Tumor",

    "Pituitary"

]

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

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


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs,dim=1)

        confidence,prediction = torch.max(

            probabilities,

            dim=1

        )

    print()

    print("="*50)

    print("Prediction")

    print("="*50)

    print(f"Class : {classes[prediction.item()]}")

    print(f"Confidence : {confidence.item()*100:.2f}%")

    print("="*50)


if __name__=="__main__":

    predict("sample.jpg")
