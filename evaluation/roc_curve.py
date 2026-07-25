"""
---------------------------------------------------------
ROC Curve
---------------------------------------------------------
"""

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.preprocessing import label_binarize

import numpy as np


def plot_roc(

        y_true,

        y_score,

        num_classes=4

):

    y_true = label_binarize(

        y_true,

        classes=[0,1,2,3]

    )

    plt.figure(figsize=(7,6))

    for i in range(num_classes):

        fpr,tpr,_ = roc_curve(

            y_true[:,i],

            y_score[:,i]

        )

        roc_auc = auc(

            fpr,

            tpr

        )

        plt.plot(

            fpr,

            tpr,

            label=f"Class {i} AUC={roc_auc:.2f}"

        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(

        "../outputs/graphs/roc_curve.png"

    )

    plt.show()
