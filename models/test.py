"""
---------------------------------------------------------
Model Testing Script
---------------------------------------------------------
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from vision_transformer import VisionTransformerModel

# ---------------- Configuration ---------------- #

TEST_DIR = "../dataset/test"

IMAGE_SIZE = 224

BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Image Transform ---------------- #

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ---------------- Dataset ---------------- #

test_dataset = datasets.ImageFolder(

    TEST_DIR,

    transform=transform

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

# ---------------- Load Model ---------------- #

model = VisionTransformerModel(num_classes=4)

model.load_state_dict(

    torch.load(

        "../outputs/models/best_vit_model.pth",

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()

# ---------------- Testing ---------------- #

correct = 0

total = 0

predictions = []

ground_truth = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

        predictions.extend(predicted.cpu().numpy())

        ground_truth.extend(labels.cpu().numpy())

accuracy = 100 * correct / total

print("="*50)

print(f"Test Accuracy : {accuracy:.2f}%")

print("="*50)

print("\nClassification Report\n")

print(

    classification_report(

        ground_truth,

        predictions,

        target_names=test_dataset.classes

    )

)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        ground_truth,

        predictions

    )

)
