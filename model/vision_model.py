#!/usr/bin/env python3
"""
cnn_federated.py

This file defines two CNN models:
  - CNN_FMNIST for Fashion MNIST (28x28, single-channel)
  - CNN_CIFAR for CIFAR-10 (32x32, 3-channel)

It also provides functions to:
  - Load the FMNIST and CIFAR-10 datasets with appropriate transforms.
  - Train a local model and compute the gradient update (difference between
    trained model and initial model).
  - Flatten the gradient into a single vector (useful for aggregation).
  - Save and load model states (to support federated rounds).

Usage:
    from cnn_federated import (
         CNN_FMNIST, CNN_CIFAR,
         get_fmnist_dataloaders, get_cifar_dataloaders,
         local_training_and_get_gradient,
         save_model, load_model
    )

    # For Fashion MNIST:
    train_loader, test_loader = get_fmnist_dataloaders(batch_size=64)
    model = CNN_FMNIST(num_classes=10)
    model.to(device)
    flat_update, data_size = local_training_and_get_gradient(
                                  model, train_loader, device=device,
                                  local_epochs=1, lr=0.01)
    save_model(model, "saved_models/fmnist_round1.pt")

    # For CIFAR-10:
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)
    model = CNN_CIFAR(num_classes=10)
    model.to(device)
    flat_update, data_size = local_training_and_get_gradient(
                                  model, train_loader, device=device,
                                  local_epochs=1, lr=0.01)
    save_model(model, "saved_models/cifar_round1.pt")
"""

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class TextCNN(nn.Module):
    """
    A Text Convolutional Neural Network (TextCNN) model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        embed_dim (int): The dimension of the word embeddings.
        num_filters (int): The number of filters (output channels) for each convolution kernel size.
        filter_sizes (List[int]): A list of kernel heights (e.g., [3, 4, 5] corresponding to 3-grams, 4-grams, 5-grams).
        num_class (int): The number of output classes for classification.
        dropout (float): The dropout probability applied to the concatenated pooled features.
        padding_idx (int): The index of the padding token in the vocabulary, used by the embedding layer.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_filters: int,
                 filter_sizes: List[int],
                 num_class: int,
                 dropout: float = 0.5,
                 padding_idx: int = 1,  # Default assumes <pad> is often index 1, but should be passed correctly
                 concise=True,
                 ):
        super(TextCNN, self).__init__()

        # --- Validation ---
        if not vocab_size > 0:
            raise ValueError("vocab_size must be greater than 0")
        if not embed_dim > 0:
            raise ValueError("embed_dim must be greater than 0")
        if not num_class > 0:
            raise ValueError("num_class must be greater than 0")
        if not filter_sizes:
            raise ValueError("filter_sizes list cannot be empty")
        if not all(fs > 0 for fs in filter_sizes):
            raise ValueError("All filter_sizes must be positive integers")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Create convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # Input channels (1 for text)
                out_channels=num_filters,  # Number of filters
                kernel_size=(fs, embed_dim)  # Kernel: (height, width) = (filter_size, embed_dim)
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer: input size is total number of filters across all kernel sizes
        total_output_filters = num_filters * len(filter_sizes)
        self.fc = nn.Linear(total_output_filters, num_class)
        if not concise:
            print(f"Initialized TextCNN:")
            print(f"  Vocab Size: {vocab_size}")
            print(f"  Embed Dim: {embed_dim}")
            print(f"  Num Filters per Size: {num_filters}")
            print(f"  Filter Sizes: {filter_sizes}")
            print(f"  Total Output Filters: {total_output_filters}")
            print(f"  Num Classes: {num_class}")
            print(f"  Dropout: {dropout}")
            print(f"  Padding Index: {padding_idx}")

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = []
        for i, conv in enumerate(self.convs):
            try:
                conv_out = F.relu(conv(embedded))
                conved.append(conv_out)
            except Exception as e:
                print(f"!!!!! ERROR applying conv {i} !!!!!")
                print(f"Conv layer details: {conv}")  # Print the layer config
                print(f"Input shape was: {embedded.shape}")
                raise e  # Re-raise the exception
        # 1. Embedding
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embed_dim)

        # 2. Prepare for Conv2d
        # Add a channel dimension: (batch_size, 1, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)

        # 3. Convolution + Activation + Pooling for each filter size
        # Apply each conv layer to the embedded input
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i] shape: (batch_size, num_filters, seq_len - filter_sizes[i] + 1, 1)

        # Apply max-pooling over the sequence length dimension
        # Pool kernel size should cover the entire height of the convolved output
        pooled = [F.max_pool2d(conv_out, (conv_out.shape[2], 1)).squeeze(3).squeeze(2)
                  for conv_out in conved]
        # pooled[i] shape: (batch_size, num_filters)

        # 4. Concatenate Pooled Features & Apply Dropout
        # Concatenate along the feature dimension (num_filters)
        cat = torch.cat(pooled, dim=1)
        # cat shape: (batch_size, num_filters * len(filter_sizes))

        dropped = self.dropout(cat)

        # 5. Fully Connected Layer (Output)
        logits = self.fc(dropped)
        # logits shape: (batch_size, num_class)

        return logits


# ---------------------------
# Model Definitions
# ---------------------------

class CNN_FMNIST(nn.Module):
    """
    A simple CNN for Fashion MNIST.
    Input: 1 x 28 x 28, Output: 10 classes.
    Architecture: [Conv -> ReLU -> MaxPool] x 2, then FC layers.
    """

    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_CIFAR(nn.Module):
    """
    A CNN for CIFAR-10.
    Input: 3 x 32 x 32, Output: 10 classes.
    Architecture: 3 Conv layers (with BatchNorm and ReLU) and 2 FC layers.
    """

    def __init__(self, num_classes=10):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, 32, 32)
        x = self.pool(x)  # (batch, 64, 16, 16)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, 16, 16)
        x = self.pool(x)  # (batch, 128, 8, 8)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 256, 8, 8)
        x = self.pool(x)  # (batch, 256, 4, 4)
        x = x.view(x.size(0), -1)  # (batch, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------
# Federated Training Functions
# ---------------------------

def train_local_model(model: nn.Module,
                      train_loader: DataLoader,
                      criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: torch.device,
                      epochs: int = 1) -> nn.Module:
    """
    Train the model on the given DataLoader for a specified number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    return model


def compute_gradient_update(initial_model: nn.Module,
                            trained_model: nn.Module) -> list:
    """
    Compute the gradient update as the difference between the trained model's parameters
    and the initial model's parameters. Returns a list of tensors.
    """
    grad_update = []
    for init_param, trained_param in zip(initial_model.parameters(), trained_model.parameters()):
        grad_update.append(trained_param.detach().cpu() - init_param.detach().cpu())
    return grad_update


def flatten_gradients(grad_list: list) -> np.ndarray:
    """
    Flatten a list of gradient tensors into a single 1D numpy array.
    """
    flat_grad = torch.cat([g.view(-1) for g in grad_list])
    return flat_grad.numpy()


def local_training_and_get_gradient(model: nn.Module,
                                    train_loader: DataLoader,
                                    device: torch.device,
                                    local_epochs: int = 1, batch_size=64,
                                    lr: float = 0.01) -> tuple:
    """
    Perform local training on a copy of the model using the provided DataLoader.
    Returns:
      - flat_update: flattened numpy array representing the update (trained - initial)
      - data_size: number of samples used in training.
    """
    local_model = copy.deepcopy(model)
    local_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=lr)

    # Save initial parameters
    initial_model = copy.deepcopy(local_model)

    # Train locally
    local_model = train_local_model(local_model, train_loader, criterion, optimizer, device, epochs=local_epochs)

    # Compute gradient update (i.e., parameter difference)
    grad_update = compute_gradient_update(initial_model, local_model)

    # Flatten the gradient list into a single vector
    flat_update = flatten_gradients(grad_update)

    data_size = len(train_loader.dataset)
    return flat_update, data_size


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc(x)
        return x

# ---------------------------
# Example Main (for testing)
# ---------------------------

# if __name__ == "__main__":
#     # For testing, you can switch between FMNIST and CIFAR.
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Train CNN on FMNIST or CIFAR for FL simulation")
#     parser.add_argument("--dataset", type=str, default="FMNIST", choices=["FMNIST", "CIFAR"])
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--epochs", type=int, default=1)
#     parser.add_argument("--lr", type=float, default=0.01)
#     args = parser.parse_args()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     if args.dataset.upper() == "FMNIST":
#         train_loader, test_loader = get_fmnist_dataloaders(batch_size=args.batch_size)
#         model = CNN_FMNIST(num_classes=10)
#     else:
#         train_loader, test_loader = get_cifar_dataloaders(batch_size=args.batch_size)
#         model = CNN_CIFAR(num_classes=10)
#
#     model.to(device)
#     print(f"Training {args.dataset} model on {device}...")
#
#     # Train locally and get gradient update
#     flat_update, data_size = local_training_and_get_gradient(model, train_loader,
#                                                              device=device,
#                                                              local_epochs=args.epochs,
#                                                              lr=args.lr)
#     print("Gradient update shape:", flat_update.shape)
#     print("Local dataset size:", data_size)
#
#     # Save the model for this round (e.g., round1)
#     save_model(model, f"saved_models/{args.dataset.lower()}_model_round1.pt")
