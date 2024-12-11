import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import pandas as pd
import polars as pl

from sklearn.metrics import mean_squared_error

from Models.MiniAttention import FullTransformer
from Models.SimpleNN import SimpleNN
from Utillity.LossFunctions import r2_score
from dataHandler import create_dataloaders, load_data, split_data, load_data_symbol


def train_model(model, loader, optimizer, loss_function, device):
    model.train()  # Set model to training mode
    total_loss = 0

    moving_loss = 0
    moving_period = 100

    all_probs = []
    all_targets = []
    all_weights = []

    # Total number of iterations
    total_iterations = len(loader)
    print(f"Total iterations: {total_iterations}")

    # Initialize iteration counter
    iteration = 1

    # Iterate over batches
    for X_batch, Y_batch, weights_batch in loader:
        iteration += 1

        # Move data to specified device (CPU or GPU)
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        weights_batch = weights_batch.to(device)

        # Reset gradient to 0
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_batch)

        # Compute weighted loss
        loss_per_sample = loss_function(outputs, Y_batch)
        weighted_loss = loss_per_sample * weights_batch.unsqueeze(1)
        loss = weighted_loss.mean()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()
        moving_loss += loss.item()

        # Log information every 100 iterations
        if iteration % moving_period == 0 or iteration == total_iterations:
            print(f"Iteration {iteration}/{total_iterations}, Average_Loss: {moving_loss/moving_period:.4f}")
            moving_loss = 0

        # Collect data for metrics
        all_probs.append(outputs[:, 6].detach().cpu())
        all_targets.append(Y_batch[:, 6].cpu())
        all_weights.append(weights_batch.cpu())

    # Compute overall metrics
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()
    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)

    avg_loss = total_loss / total_iterations
    return avg_loss, mse, r2

def evaluate_model(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    all_weights = []
    with torch.no_grad():
        for X_batch, Y_batch, weights_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)

            all_probs.append(outputs[:, 6].cpu())
            all_targets.append(Y_batch[:, 6].cpu())
            all_weights.append(weights_batch.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()

    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)

    return mse, r2

def main():
    model = SimpleNN(79, 64, 9, 5, use_noise=False, dropout_prob=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    X, Y, weights = load_data_symbol(0)

    train_X, train_Y, train_weights, val_X, val_Y, val_weights, test_X, test_Y, test_weights = (
        split_data(X, Y, weights, 0.8, 0.1, 0.1))

    train_loader, val_loader, test_loader = create_dataloaders(train_X, train_Y, train_weights, val_X, val_Y,
                                                               val_weights, test_X, test_Y, test_weights, shuffle=False, batch_size=4096)

    print("Data Loaded!")

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_function = nn.MSELoss(reduction='none')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs = 20
    best = float('-inf')
    degraded = 0
    best_model = model

    for epoch in range(epochs):
        train_loss, train_mse, train_r2 = train_model(model, train_loader, optimizer, loss_function, device)
        val_mse, val_r2 = evaluate_model(model, val_loader, device)

        print(f'epoch {epoch} train loss {train_loss:.4f}, train_r2 {train_r2:.4f}, train_mse {train_mse:.4f}, val_mse {val_mse:.4f}, val_r2 {val_r2:.4f}')
        if val_r2 > best:
            best = val_r2
            best_model = copy.deepcopy(model)
            model.save()
            degraded = 0
        else:
            degraded += 1
        if degraded > 5:
            break
    model = best_model

    test_mse, test_r2 = evaluate_model(model, test_loader, device)
    print(f'test R2 score is {test_r2}')

    train_model(model, train_loader, optimizer, loss_function, device)


if __name__ == '__main__':
    model = FullTransformer(dim_in=79, dim_attn=64, attention_depth=3, mlp_depth=2, dim_out=9, heads=8, rotary_emb=True, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    main()
