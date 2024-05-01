import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim

def cross_validate_tf(data, labels, model, num_folds=5, epochs=10, batch_size=8):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_accuracy = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(data, labels)):
        print(f'Fold {fold+1}/{num_folds}')

        x_train, x_val = data[train_indices], data[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]

        model = model()

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        _, accuracy = model.evaluate(x_val, y_val, verbose=1)
        fold_accuracy.append(accuracy)

        print(f'Validation accuracy: {accuracy}')

    print(f'Average validation accuracy: {np.mean(fold_accuracy)}')

def cross_validate_torch(model, dataloader, num_folds=5, epochs=10):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_accuracy = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataloader.dataset)):
        print(f'Fold {fold+1}/{num_folds}')

        train_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices)
        )
        val_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_indices)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            fold_accuracy.append(accuracy)

        print(f'Validation accuracy: {accuracy}')

    print(f'Average validation accuracy: {np.mean(fold_accuracy)}')
