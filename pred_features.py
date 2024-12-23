import copy

import torch
from torch import optim, nn

from Models.SimpleNN import SimpleNN


def train_model(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0

    moving_loss = 0
    moving_period = 100

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

        # Reset gradient to 0
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)

        # Compute loss
        loss_per_sample = loss_function(outputs.squeeze(), Y_batch)
        loss = loss_per_sample.mean()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()
        moving_loss += loss.item()

        # Log information every 100 iterations
        if iteration % moving_period == 0 or iteration == total_iterations:
            print(f"Iteration {iteration}/{total_iterations}, Average_Loss: {moving_loss / moving_period:.4f}")
            moving_loss = 0

    avg_loss = total_loss / total_iterations
    return avg_loss

def evaluate_model(model, loader, loss_function, device):
    model.eval()

    total_iterations = len(loader)
    print(f"Total iterations: {total_iterations}")

    total_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch, weights_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)

            # Compute loss
            loss = loss_function(outputs.squeeze(), Y_batch).mean()
            # Accumulate total loss
            total_loss += loss.item()

    mean_loss = total_loss / total_iterations

    return mean_loss

def main():
    model = SimpleNN(87, 64, 1, 5, use_noise=False, dropout_prob=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    XY, time, weights = load_data_xy_symbol(0)
    train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights = split_data_train_eval(XY, time, weights, 0.9, 0.1)

    train_loader, eval_loader = create_feature_dataloaders(train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights, 256, True, 1)

    print("Data Loaded!")

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_function = nn.MSELoss(reduction='none')

    epochs = 20

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, loss_function, device)
        val_loss = evaluate_model(model, eval_loader, loss_function, device)

        print(f'epoch {epoch} train loss {train_loss:.4f}, val_loss {val_loss:.4f}')


if __name__ == '__main__':
    main()
