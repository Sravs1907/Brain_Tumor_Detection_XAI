"""
------------------------------------------------------
Data Augmentation Module
------------------------------------------------------
"""

import albumentations as A


def get_train_augmentation():

    transform = A.Compose(

        [

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.3),

            A.Rotate(limit=20, p=0.5),

            A.RandomBrightnessContrast(p=0.4),

            A.GaussianBlur(p=0.2),

            A.ShiftScaleRotate(

                shift_limit=0.05,

                scale_limit=0.05,

                rotate_limit=10,

                p=0.5

            ),

        ]

    )

    return transform


def get_validation_augmentation():

    transform = A.Compose([])

    return transform
