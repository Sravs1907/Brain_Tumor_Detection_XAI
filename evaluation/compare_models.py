"""
---------------------------------------------------------
Model Comparison
---------------------------------------------------------
"""

import pandas as pd

import matplotlib.pyplot as plt


def compare_models():

    results = {

        "Model":[

            "CNN",

            "InceptionV3",

            "DenseNet121",

            "Vision Transformer"

        ],

        "Accuracy":[

            0.93,

            0.91,

            0.93,

            0.98

        ]

    }

    dataframe = pd.DataFrame(

        results

    )

    print(dataframe)

    plt.figure(figsize=(8,5))

    plt.bar(

        dataframe["Model"],

        dataframe["Accuracy"]

    )

    plt.ylabel("Accuracy")

    plt.title(

        "Model Comparison"

    )

    plt.savefig(

        "../outputs/graphs/model_comparison.png"

    )

    plt.show()


if __name__=="__main__":

    compare_models()
