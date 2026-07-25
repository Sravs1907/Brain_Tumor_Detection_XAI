"""
---------------------------------------------------------
Visualization Helper
---------------------------------------------------------
"""

import matplotlib.pyplot as plt
import cv2


def display(image_path,title="Image"):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


if __name__=="__main__":

    display("../outputs/heatmaps/gradcam_result.jpg")
