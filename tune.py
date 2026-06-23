import torch
import torch.nn as nn
import torch.optim as optim

from model import LogisticRegressionModel


def tune_learning_rate(
    X_train,
    y_train,
    X_test,
    y_test,
    input_dim
):

    learning_rates = [
        0.01,
        0.05,
        0.1
    ]

    criterion = nn.BCELoss()

    best_lr = None
    best_acc = 0

    for lr in learning_rates:

        model = LogisticRegressionModel(
            input_dim
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr
        )

        for _ in range(100):

            optimizer.zero_grad()

            outputs = model(X_train)

            loss = criterion(
                outputs,
                y_train
            )

            loss.backward()

            optimizer.step()

        with torch.no_grad():

            preds = (
                model(X_test) >= 0.5
            ).float()

        acc = (
            preds.eq(y_test)
            .sum()
            .item()
            / len(y_test)
        )

        print(
            f"LR={lr}  Accuracy={acc:.4f}"
        )

        if acc > best_acc:

            best_acc = acc
            best_lr = lr

    return best_lr, best_acc
