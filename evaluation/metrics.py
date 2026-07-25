"""
---------------------------------------------------------
Evaluation Metrics
---------------------------------------------------------
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("="*50)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("="*50)

    return {

        "Accuracy":accuracy,

        "Precision":precision,

        "Recall":recall,

        "F1":f1

    }
