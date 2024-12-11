import os

import torch
import torch.nn as nn

from Models.Noise import GaussianNoise


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_noise=False, dropout_prob=0.0):
        """
        Parameters:
        - input_size: Number of input features.
        - hidden_dim: Dimension of the hidden layers.
        - output_size: Number of output features.
        - use_noise: Whether to apply Gaussian noise to the input.
        - dropout_prob: Dropout probability. Set to 0.0 to disable dropout.
        - num_hidden_layers: Number of fully connected hidden layers.
        """
        super(SimpleNN, self).__init__()

        self.use_noise = use_noise
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        # Apply Gaussian noise (if defined elsewhere)
        if use_noise:
            self.noise = GaussianNoise(std=0.1)

        # Define layers dynamically using nn.Sequential
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())

        # Add additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_size))

        # Combine into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor of shape (batch_size, input_size).

        Returns:
        - Output tensor of shape (batch_size, output_size).
        """
        # Optionally add noise to the input
        if self.use_noise:
            x = self.noise(x)

        # Pass the input through the sequential layers
        out = self.network(x)

        # return out.squeeze()
        return out

    def save(self, path=".\\savedModels\\simple_nn.pt"):
        """
        Save the model state dictionary to a file.

        Parameters:
        - path: The file path where the model will be saved. Default is '..\\savedModels\\simple_nn.pt'.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, *args, **kwargs):
        """
        Load the model state dictionary from a file and create a new model instance.

        Parameters:
        - path: The file path from which to load the model.
        - *args, **kwargs: Additional arguments for the model's constructor.

        Returns:
        - An instance of SimpleNN with the loaded weights.
        """
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return model
