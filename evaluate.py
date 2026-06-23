import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt


def evaluate_model(
    model,
    X_test,
    y_test
):

    model.eval()

    with torch.no_grad():

        probabilities = model(X_test)

        predictions = (
            probabilities >= 0.5
        ).float()

    accuracy = (
        predictions.eq(y_test)
        .sum()
        .item()
        / len(y_test)
    )

    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(
        y_test.numpy(),
        predictions.numpy()
    )

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(
        classification_report(
            y_test.numpy(),
            predictions.numpy()
        )
    )

    fpr, tpr, _ = roc_curve(
        y_test.numpy(),
        probabilities.numpy()
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.2f}"
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()

    return accuracy
