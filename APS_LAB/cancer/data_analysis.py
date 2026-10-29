# data_analysis.py
# Loading the Breast Cancer dataset and performing data analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer


def load_and_analyze_data():
    # load dataset
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)

    y = pd.Series((data.target == 0).astype(int), name="malignant")

    print(y.value_counts())

    print("Feature matrix shape: ", X.shape)
    print("Target shape: ", y.shape)
    print("Class names: ", data.target_names)

    # Examine the class Distribution

    class_counts = y.value_counts().sort_index()
    class_distribution = pd.DataFrame({
        "class": data.target_names,
        "count": class_counts.values,
        "Probability": class_counts.values / len(y)
    })
    print(class_distribution)

    class_distribution.plot(
        x="class",
        y="count",
        kind="bar",
        legend=False,
        color=["brown", "black"]
    )
    plt.ylabel("Number of observations")
    plt.title("Class distribution")
    plt.xticks(rotation=0)
    plt.show()

    class_distribution.plot(
        x="class",
        y="count",
        kind="pie",
        legend=False,
        color=["tomato", "steelblue"]
    )
    plt.ylabel("Number of observations")
    plt.title("Class distribution")
    plt.xticks(rotation=0)
    plt.show()

    return data, X, y, class_distribution