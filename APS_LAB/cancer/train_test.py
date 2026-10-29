# train_test.py
# Creating training/testing samples and training the Logistic Regression model.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression


def train_and_test_model(X, y):
    # create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("training Size: ", len(y_train))
    print("testing Size: ", len(y_test))

    print("\nTraining Proportions: ")
    print(y_train.value_counts(normalize=True).sort_index())

    print("\nTesting Proportions: ")
    print(y_test.value_counts(normalize=True).sort_index())

    # train A LOGISTIC regression Model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    # obtain predicted probabilities
    probabilities = model.predict_proba(X_test)

    print(probabilities[:5])

    print(probabilities)

    results = pd.DataFrame({
        "actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    print(results.head(10))

    results["Actual_label"] = results["actual_class"].map({
        1: "malignant",
        0: "benign"
    })

    print(
        results[
            ["Actual_label", "P_malignant", "P_benign"]
        ].head(10)
    )

    # use a threshold to genearte pridictions
    threshold = 0.50

    results["Predicted_malignant"] = (
        results["P_malignant"] > threshold
    ).astype(int)

    results["Predicted_label"] = results["Predicted_malignant"].map({
        1: "malignant",
        0: "benign"
    })

    print(
        results[
            ["Actual_label", "Predicted_label", "P_malignant", "P_benign"]
        ].head(10)
    )

    # compare different thresholds
    for thresholds in [0.30, 0.50, 0.70]:
        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold={threshold}:"
            f"predicted malignant cases = {predictions.sum()}"
        )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities,
        results
    )