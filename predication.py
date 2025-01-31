import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from util import load_csv, load_json


def perform_prediction(df: pd.DataFrame) -> None:
    """
    Perform prediction using Decision Tree Classifier, evaluate the model,
    and visualize important metrics such as confusion matrix, ROC curve, and Decision Tree.

    Args:
    - df (pd.DataFrame): DataFrame containing the features and target variable.

    Returns:
    - None
    """
    # Ensure target column exists in the DataFrame
    if "Academic Performance (%)" not in df.columns:
        raise ValueError(
            "'Academic Performance (%)' column is missing in the DataFrame."
        )

    # Define features (X) and target (y)
    X = df.drop(
        columns=["Academic Performance (%)", "Cluster"], errors="ignore"
    )  # Features
    y = df["Academic Performance (%)"]  # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    stderr = np.sqrt(mean_squared_error(y_test, y_pred))

    # Display metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Standard Error: {stderr:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=X.columns.tolist(),
        filled=True,
        max_depth=2,
        class_names=[str(cls) for cls in clf.classes_],
    )
    plt.title("Decision Tree Classifier Visualization")
    plt.show()

    # ROC Curve and AUC
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot(
            [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()
    except ValueError:
        print(
            "ROC curve cannot be computed due to absence of probabilities for the positive class."
        )


if __name__ == "__main__":
    file: dict[str, str] = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    perform_prediction(load_csv(file["output"]))
