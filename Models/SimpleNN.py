import os

import torch
import torch.nn as nn

from Models.Noise import GaussianNoise


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, noise=0.0, dropout_prob=0.0, batch_norm=True):
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

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.dropout_prob = dropout_prob

        # Define layers dynamically using nn.Sequential
        layers = []

        if batch_norm:
            layers.append(nn.BatchNorm1d(self.input_size, momentum=0.001))

        layers.append(GaussianNoise(std=noise))

        if batch_norm:
            layers.append(nn.BatchNorm1d(self.input_size, momentum=0.001))

        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_dim))
        layers.append(nn.ReLU())

        # Add additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_size))

        # Combine into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential layers
        out = self.network(x)

        # return out.squeeze()
        return out

    def save(self, experiment_id, path=".\\savedModels\\"):
        # Construct the full file path with the experiment ID
        file_name = f"simple_nn_id_{experiment_id}.pt"
        full_path = os.path.join(path, file_name)

        # Save the model's state dictionary to the constructed path
        torch.save(self.state_dict(), full_path)


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, noise=0.0, dropout_prob=0.0, batch_norm=True):
        """
        Parameters:
        - input_size: Number of input features.
        - hidden_dim: Dimension of the hidden layers.
        - output_size: Number of output features.
        - use_noise: Whether to apply Gaussian noise to the input.
        - dropout_prob: Dropout probability. Set to 0.0 to disable dropout.
        - num_hidden_layers: Number of fully connected hidden layers.
        """
        super(SimpleRNN, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.dropout_prob = dropout_prob


        self.batch_norm_1 = nn.BatchNorm1d(self.input_size)

        self.noise = GaussianNoise(std=noise)

        self.batch_norm_2 = nn.BatchNorm1d(self.input_size)

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=num_layers, nonlinearity="relu", dropout=self.dropout_prob, batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):

        out = x.transpose(1, 2)

        out = self.batch_norm_1(out)
        out = self.noise(out)
        out = self.batch_norm_2(out)

        out = out.transpose(1, 2)

        out, _ = self.rnn(out)

        out = self.output_layer(out[:, -1])

        return out

    def save(self, experiment_id, path=".\\savedModels\\"):
        # Construct the full file path with the experiment ID
        file_name = f"simple_nn_id_{experiment_id}.pt"
        full_path = os.path.join(path, file_name)

        # Save the model's state dictionary to the constructed path
        torch.save(self.state_dict(), full_path)
