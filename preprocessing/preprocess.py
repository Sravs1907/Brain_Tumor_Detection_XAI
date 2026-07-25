"""
---------------------------------------------------------
Brain Tumor Detection using Vision Transformer
Preprocessing Module

Author : Your Name
---------------------------------------------------------
"""

import os
import cv2
import numpy as np


class ImagePreprocessor:
    """
    Image preprocessing class.
    Responsible for:
        1. Reading MRI images
        2. Resizing
        3. Normalization
    """

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def load_image(self, image_path):
        """
        Reads an image from disk.

        Parameters:
            image_path (str)

        Returns:
            image (numpy array)
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Unable to load image: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def resize_image(self, image):
        """
        Resize image to Vision Transformer input size.
        """

        return cv2.resize(image, self.image_size)

    def normalize_image(self, image):
        """
        Normalize image pixel values between 0 and 1.
        """

        image = image.astype(np.float32)

        image = image / 255.0

        return image

    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline.
        """

        image = self.load_image(image_path)

        image = self.resize_image(image)

        image = self.normalize_image(image)

        return image


if __name__ == "__main__":

    sample_path = "sample.jpg"

    processor = ImagePreprocessor()

    try:

        image = processor.preprocess(sample_path)

        print("Image Shape :", image.shape)

        print("Image Type :", image.dtype)

        print("Minimum Pixel :", image.min())

        print("Maximum Pixel :", image.max())

    except Exception as e:

        print(e)
