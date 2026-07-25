"""
---------------------------------------------------------
Project Configuration
---------------------------------------------------------
"""

IMAGE_SIZE = 224

NUM_CLASSES = 4

BATCH_SIZE = 16

LEARNING_RATE = 0.0001

EPOCHS = 20

MODEL_PATH = "../outputs/models/best_vit_model.pth"

TRAIN_DIR = "../dataset/train"

VALID_DIR = "../dataset/valid"

TEST_DIR = "../dataset/test"

CLASS_NAMES = [

    "Glioma",

    "Meningioma",

    "No Tumor",

    "Pituitary"

]
