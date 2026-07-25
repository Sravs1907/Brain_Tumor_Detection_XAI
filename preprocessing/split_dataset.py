"""
------------------------------------------------------
Dataset Split Utility
------------------------------------------------------
"""

import os

import random

import shutil


random.seed(42)


def split_dataset(

    source_dir,

    train_dir,

    valid_dir,

    test_dir,

    train_ratio=0.8,

    valid_ratio=0.1

):

    classes = os.listdir(source_dir)

    for cls in classes:

        cls_path = os.path.join(source_dir, cls)

        images = os.listdir(cls_path)

        random.shuffle(images)

        total = len(images)

        train_end = int(total * train_ratio)

        valid_end = int(total * (train_ratio + valid_ratio))

        splits = {

            train_dir: images[:train_end],

            valid_dir: images[train_end:valid_end],

            test_dir: images[valid_end:]

        }

        for split_folder, image_list in splits.items():

            os.makedirs(

                os.path.join(split_folder, cls),

                exist_ok=True

            )

            for img in image_list:

                shutil.copy(

                    os.path.join(cls_path, img),

                    os.path.join(split_folder, cls, img)

                )


if __name__ == "__main__":

    print("Dataset Split Utility")
