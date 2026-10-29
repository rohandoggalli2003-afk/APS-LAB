# evaluation_plotting.py
# Confusion matrices, evaluation metrics, and threshold analysis.

import pandas as pd
from IPython.display import display

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test, probabilities):
    actual_malignant = (y_test.values == 0).astype(int)

    for threshold in [0.30, 0.50, 0.70]:
        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant,
            predicted_malignant
        )

        print(f"\nThreshold={threshold}:")
        print(cm)

    y_pred = model.predict(X_test)

    print(
        "sklearn Accuracy: ",
        accuracy_score(y_test, y_pred)
    )

    print(
        "sklearn Precision: ",
        precision_score(y_test, y_pred)
    )

    print(
        "sklearn Recall: ",
        recall_score(y_test, y_pred)
    )

    print(
        "sklearn F1: ",
        f1_score(y_test, y_pred)
    )

    thresholds = [0.1, 0.30, 0.50, 0.70, 0.90]
    metrics_data = []

    for threshold in thresholds:
        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predicted_malignant,
            labels=[0, 1]
        )

        TN, FP, FN, TP = cm.ravel()

        metrics_data.append({
            'Threshold': threshold,
            'Accuracy': accuracy_score(
                y_test,
                predicted_malignant
            ),
            'Precision': precision_score(
                y_test,
                predicted_malignant
            ),
            'Recall': recall_score(
                y_test,
                predicted_malignant
            ),
            'F1-score': f1_score(
                y_test,
                predicted_malignant
            ),
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'TP': TP
        })

    threshold_metrics = pd.DataFrame(metrics_data)
    display(threshold_metrics.round(3))

    return threshold_metrics