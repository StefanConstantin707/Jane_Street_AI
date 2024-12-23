import os

import torch
import torch.nn as nn

from Models.Noise import GaussianNoise


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, noise=0.0, dropout_prob=0.0):
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

        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        self.input_size = input_size
        self.output_size = output_size

        # Define layers dynamically using nn.Sequential
        layers = []

        layers.append(nn.BatchNorm1d(self.input_size))

        # Apply Gaussian noise (if defined elsewhere)
        if noise > 0.0:
            self.noise_layer = GaussianNoise(std=noise)

        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_dim))
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

        if self.training:
            x = self.noise_layer(x)

        # Pass the input through the sequential layers
        out = self.network(x)

        # return out.squeeze()
        return out

    def save(self, experiment_id, path=".\\savedModels\\"):
        """
        Save the model state dictionary to a file.

        Parameters:
        - experiment_id: The ID of the experiment used to create a unique file name.
        - path: The directory path where the model will be saved. Default is '.\\savedModels\\'.
        """
        # Construct the full file path with the experiment ID
        file_name = f"simple_nn_id_{experiment_id}.pt"
        full_path = os.path.join(path, file_name)

        # Save the model's state dictionary to the constructed path
        torch.save(self.state_dict(), full_path)

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

class SymbolEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SymbolEmbedding, self).__init__()

        self.embedding = nn.Embedding(100, embedding_dim)

    def forward(self, x):
        # x: N, in_dim
        # symbol_id: N, 1
        symbol_id = x[:, -1].to(torch.int32)

        # N -> N, embedding_dim
        emb = self.embedding(symbol_id)

        # N, (in_dim - 1) ++ N, embedding_dim -> N, (in_dim + embedding_dim - 1)
        x = torch.cat((x[:, :-1], emb), dim=1)

        return x

class SymbolAndTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim_symbol, embedding_dim_time):
        super(SymbolAndTimeEmbedding, self).__init__()

        self.embedding_dim_symbol = embedding_dim_symbol
        self.embedding_dim_time = embedding_dim_time
        if embedding_dim_symbol > 0:
            self.embedding_s = nn.Embedding(100, embedding_dim_symbol)
        else:
            self.embedding_s = nn.Identity()

        if embedding_dim_time > 0:
            self.embedding_t = nn.Embedding(968, embedding_dim_time)
        else:
            self.embedding_t = nn.Identity()

    def forward(self, x):
        # x: N, in_dim
        # symbol_id: N, 1
        # time_id: N, 1
        symbol_id = x[:, -2].to(torch.int32)
        time_id = x[:, -1].to(torch.int32)

        # N -> N, embedding_dim
        emb_symbol = self.embedding_s(symbol_id)
        emb_time = self.embedding_t(time_id)

        if self.embedding_dim_symbol > 0 and self.embedding_dim_time > 0:
            # N, (in_dim - 2) ++ N, embedding_dim_symbol ++ N, embedding_dim_time -> N, (in_dim + embedding_dim_symbol + embedding_dim_time - 2)
            x = torch.cat((x[:, :-2], emb_symbol, emb_time), dim=1)
        elif self.embedding_dim_symbol > 0:
            x = torch.cat((x[:, :-2], emb_symbol), dim=1)
        elif self.embedding_dim_time > 0:
            x = torch.cat((x[:, :-2], emb_time), dim=1)
        else:
            return x[:, :-2]

        return x

class SimpleNNEmbed(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, symbol_emb_dim, time_emb_dim, noise=0.0, dropout_prob=0.0):
        """
        Parameters:
        - input_size: Number of input features.
        - hidden_dim: Dimension of the hidden layers.
        - output_size: Number of output features.
        - use_noise: Whether to apply Gaussian noise to the input.
        - dropout_prob: Dropout probability. Set to 0.0 to disable dropout.
        - num_hidden_layers: Number of fully connected hidden layers.
        """
        super(SimpleNNEmbed, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.symbol_emb_dim = symbol_emb_dim
        self.time_emb_dim = time_emb_dim

        self.nn_input_size = input_size + self.symbol_emb_dim + self.time_emb_dim - 2

        # Define layers dynamically using nn.Sequential
        layers = []
        layers.append(SymbolAndTimeEmbedding(self.symbol_emb_dim, self.time_emb_dim))
        layers.append(nn.BatchNorm1d(self.nn_input_size))
        # Apply Gaussian noise (if defined elsewhere)
        if noise > 0.0:
            layers.append(GaussianNoise(std=noise))
        # Input layer
        layers.append(nn.Linear(self.nn_input_size, hidden_dim))
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
        # Pass the input through the sequential layers
        out = self.network(x)

        if self.training is not True:
            out = torch.mean(out, dim=1)

        # return out.squeeze()
        return out

    def save(self, experiment_id, path=".\\savedModels\\"):
        """
        Save the model state dictionary to a file.

        Parameters:
        - experiment_id: The ID of the experiment used to create a unique file name.
        - path: The directory path where the model will be saved. Default is '.\\savedModels\\'.
        """
        # Construct the full file path with the experiment ID
        file_name = f"simple_nn_id_{experiment_id}.pt"
        full_path = os.path.join(path, file_name)

        # Save the model's state dictionary to the constructed path
        torch.save(self.state_dict(), full_path)

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

class SimpleNNSampling(SimpleNN):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, noise=0.0, dropout_prob=0.0, nu_rows_sampled=1):
        super(SimpleNNSampling, self).__init__(input_size, hidden_dim, output_size, num_layers, noise, dropout_prob)
        self.nu_rows_sampled = nu_rows_sampled

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor of shape (batch_size, input_size).

        Returns:
        - Output tensor of shape (batch_size, output_size).
        """

        if self.training:
            x = self.noise_layer(x)
        else:
            x = x.view(-1, self.input_size)

        # Pass the input through the sequential layers
        out = self.network(x)

        if self.training is not True:
            out = out.view(-1, self.nu_rows_sampled, self.output_size)
            out = torch.mean(out, dim=1)

        return out
