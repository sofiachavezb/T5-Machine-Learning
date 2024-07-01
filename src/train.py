from typing import Union
import os
import dotenv

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from .SGD import SGD
from .dataset_loader import DatasetLoader
from .linear_layer_model import LinearLayerModel

logsoftmax = torch.nn.LogSoftmax(dim=1)
seed = int(os.getenv("SEED"))

def train(model: torch.nn.Module, optimizer: Union[SGD, Optimizer], batch_size: int, epochs: int):
    """
    Train the given model using the specified optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (Union[SGD, torch.optim.Optimizer]): The optimizer to use.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs to train for.

    Returns:
        None
    """
    # Load the dataset
    dataset_loader = DatasetLoader(batch_size, seed)
    train_batches, test_batches = dataset_loader.get_datasets()

    # Convert the data to PyTorch DataLoader
    train_loader = DataLoader(TensorDataset(
        torch.tensor([x for xs, ys in train_batches for x in xs], dtype=torch.float32),
        torch.tensor([y for xs, ys in train_batches for y in ys], dtype=torch.long)
    ), batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(TensorDataset(
        torch.tensor([x for xs, ys in test_batches for x in xs], dtype=torch.float32),
        torch.tensor([y for xs, ys in test_batches for y in ys], dtype=torch.long)
    ), batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xs, ys in train_loader:
            optimizer.zero_grad()
            output = model(xs)
            loss = criterion(output, ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")


    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xs, ys in test_loader:
            output = model(xs)
            preds = model.predict(output)
            all_preds.extend(preds.numpy())
            all_labels.extend(ys.numpy())
    return losses, all_preds, all_labels