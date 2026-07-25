"""
---------------------------------------------------------
Training Script
Brain Tumor Detection using Vision Transformer
---------------------------------------------------------
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vision_transformer import VisionTransformerModel

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

TRAIN_DIR = "../dataset/train"
VALID_DIR = "../dataset/valid"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -------------------------------------------------------
# Image Transformations
# -------------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

validation_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

train_dataset = datasets.ImageFolder(

    TRAIN_DIR,

    transform=train_transform

)

validation_dataset = datasets.ImageFolder(

    VALID_DIR,

    transform=validation_transform

)

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False

)

# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = VisionTransformerModel(

    num_classes=4

)

model = model.to(DEVICE)

# -------------------------------------------------------
# Loss
# -------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

scheduler = optim.lr_scheduler.StepLR(

    optimizer,

    step_size=5,

    gamma=0.5

)

# -------------------------------------------------------
# Validation Function
# -------------------------------------------------------

def validate():

    model.eval()

    correct = 0

    total = 0

    validation_loss = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted==labels).sum().item()

    accuracy = 100 * correct / total

    validation_loss /= len(validation_loader)

    return validation_loss, accuracy

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

best_accuracy = 0

os.makedirs("../outputs/models", exist_ok=True)

print("="*60)

print("Training Started")

print("="*60)

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)

        correct += (predicted==labels).sum().item()

    scheduler.step()

    train_accuracy = 100 * correct / total

    train_loss = running_loss / len(train_loader)

    validation_loss, validation_accuracy = validate()

    print()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    print("-"*40)

    print(f"Training Loss      : {train_loss:.4f}")

    print(f"Training Accuracy : {train_accuracy:.2f}%")

    print(f"Validation Loss   : {validation_loss:.4f}")

    print(f"Validation Accuracy : {validation_accuracy:.2f}%")

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        torch.save(

            model.state_dict(),

            "../outputs/models/best_vit_model.pth"

        )

        print("Best model saved.")

print()

print("="*60)

print("Training Completed")

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

print("="*60)
