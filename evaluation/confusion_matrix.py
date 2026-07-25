"""
---------------------------------------------------------
Confusion Matrix
---------------------------------------------------------
"""

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(

        y_true,

        y_pred,

        class_names

):

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=class_names

    )

    disp.plot(

        cmap="Blues"

    )

    plt.title(

        "Confusion Matrix"

    )

    plt.savefig(

        "../outputs/graphs/confusion_matrix.png"

    )

    plt.show()
