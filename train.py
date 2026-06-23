import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train):

    criterion = nn.BCELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        weight_decay=0.01
    )

    epochs = 1000

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(
            outputs,
            y_train.view(-1,1)
        )

        loss.backward()

        optimizer.step()

    return model
