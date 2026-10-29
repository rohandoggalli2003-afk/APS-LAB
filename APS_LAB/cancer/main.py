# main.py
# Main program: calls each operation one by one.

from data_analysis import load_and_analyze_data
from train_test import train_and_test_model
from evaluation_plotting import evaluate_model


def main():

    # 1. DATA INPUT + LOADING + ANALYSIS
    print("\n========== DATA INPUT AND ANALYSIS ==========")

    data, X, y, class_distribution = load_and_analyze_data()

    # 2. TRAINING + TESTING
    print("\n========== TRAINING AND TESTING ==========")

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        probabilities,
        results
    ) = train_and_test_model(X, y)

    # 3. EVALUATION + PLOTTING/THRESHOLD ANALYSIS
    print("\n========== EVALUATION AND THRESHOLD ANALYSIS ==========")

    threshold_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        probabilities
    )

    print("\n========== PROGRAM COMPLETED ==========")


if __name__ == "__main__":
    main()