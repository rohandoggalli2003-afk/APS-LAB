"""
Assignment: Applying Probability and Statistics to Evaluate a Machine Learning Model
Dataset: Breast Cancer Wisconsin Diagnostic dataset (scikit-learn)
Models: Logistic Regression (baseline) vs Random Forest (comparison model)

Reproducibility:
    pip install numpy pandas scikit-learn scipy
    python probability_statistics_model_comparison.py

The script uses one stratified 75/25 train/test split. Both models are evaluated on
the exact same held-out test set. The 95% Wilson interval quantifies uncertainty in
each model's accuracy. An exact two-sided McNemar test (binomial form) compares
paired correctness outcomes on the same test examples.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.25
ALPHA = 0.05


def wilson_confidence_interval(successes: int, total: int, alpha: float = 0.05):
    """Return a two-sided Wilson score confidence interval for a binomial proportion."""
    # For a 95% CI, z = 1.959963984540054. Avoid an extra dependency for this constant.
    # This implementation supports the assignment's default alpha=0.05.
    if alpha != 0.05:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))
    else:
        z = 1.959963984540054

    if total <= 0:
        raise ValueError("total must be greater than zero")
    p_hat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p_hat + z2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt((p_hat * (1 - p_hat) / total) + (z2 / (4 * total * total)))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def main():
    # 1. Load dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target  # 0 = malignant, 1 = benign in this dataset

    print("=" * 78)
    print("DATASET OVERVIEW")
    print("=" * 78)
    print(f"Dataset: Breast Cancer Wisconsin Diagnostic")
    print(f"Number of rows: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print("Target mapping: 0 = malignant, 1 = benign")
    print("Class counts:")
    print(y.map({0: "malignant", 1: "benign"}).value_counts().to_string())

    # 2. Sampling: stratified random hold-out split; same test rows for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTraining rows: {len(X_train)}")
    print(f"Held-out test rows: {len(X_test)}")
    print("Sampling method: stratified random train/test split (75%/25%)")
    print(f"Random seed: {RANDOM_STATE}")

    # 3. Define models. Scaling is fitted only on training data through the pipeline.
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    predictions = {}
    result_rows = []
    correct_counts = {}

    # 4. Train and evaluate each model on the same held-out test set
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        correct = int(np.sum(y_pred == y_test))
        correct_counts[name] = correct
        ci_low, ci_high = wilson_confidence_interval(correct, len(y_test), ALPHA)

        # The positive class for precision/recall/F1 is malignant (label 0).
        result_rows.append(
            {
                "Model": name,
                "Correct predictions": correct,
                "Test samples": len(y_test),
                "Accuracy": accuracy,
                "Accuracy 95% CI lower": ci_low,
                "Accuracy 95% CI upper": ci_high,
                "Malignant precision": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
                "Malignant recall/sensitivity": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
                "Malignant F1-score": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            }
        )

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Correct predictions: {correct}/{len(y_test)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"95% Wilson CI for accuracy: [{ci_low:.4f}, {ci_high:.4f}]")
        print("Confusion matrix (rows=true, columns=predicted; labels [malignant, benign]):")
        # sklearn's sorted label order is [0=malignant, 1=benign]
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        print(f"Malignant precision: {result_rows[-1]['Malignant precision']:.4f}")
        print(f"Malignant recall/sensitivity: {result_rows[-1]['Malignant recall/sensitivity']:.4f}")
        print(f"Malignant F1-score: {result_rows[-1]['Malignant F1-score']:.4f}")

    results = pd.DataFrame(result_rows)
    print("\n" + "=" * 78)
    print("COMPARISON TABLE")
    print("=" * 78)
    print(results.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Accuracy 95% CI lower": "{:.4f}".format,
        "Accuracy 95% CI upper": "{:.4f}".format,
        "Malignant precision": "{:.4f}".format,
        "Malignant recall/sensitivity": "{:.4f}".format,
        "Malignant F1-score": "{:.4f}".format,
    }))

    # 5. Paired hypothesis test: exact McNemar test using discordant pairs.
    # a_only: Logistic Regression correct while Random Forest is wrong.
    # b_only: Random Forest correct while Logistic Regression is wrong.
    lr_correct = predictions["Logistic Regression"] == y_test.to_numpy()
    rf_correct = predictions["Random Forest"] == y_test.to_numpy()
    lr_only = int(np.sum(lr_correct & ~rf_correct))
    rf_only = int(np.sum(~lr_correct & rf_correct))
    discordant = lr_only + rf_only

    # Under H0 of equal error probability, each discordant pair is equally likely
    # to favour either model. Exact binomial test is the exact McNemar test.
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                k=min(lr_only, rf_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    print("\n" + "=" * 78)
    print("PAIRED HYPOTHESIS TEST: EXACT McNEMAR TEST")
    print("=" * 78)
    print("H0: The two models have the same probability of being correct on a test case.")
    print("H1: The models differ in their probability of being correct on a test case.")
    print(f"Correct only for Logistic Regression: {lr_only}")
    print(f"Correct only for Random Forest: {rf_only}")
    print(f"Total discordant test cases: {discordant}")
    print(f"Two-sided exact p-value: {p_value:.4f}")
    print(f"Significance level alpha: {ALPHA:.2f}")
    if p_value < ALPHA:
        print("Decision: Reject H0. The difference is statistically significant at 5%.")
    else:
        print("Decision: Fail to reject H0. The difference is not statistically significant at 5%.")
        print("Interpretation: The observed accuracy difference could plausibly reflect test-sample variation.")

    # Save a machine-readable results table alongside the script.
    results.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved results table to: model_comparison_results.csv")
    print("\nImportant limitation: this is one fixed hold-out split. Results may change with")
    print("another split; use repeated stratified cross-validation for a more robust follow-up.")


if __name__ == "__main__":
    main()
