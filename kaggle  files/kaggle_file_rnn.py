import math
import os

import numba
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.stats import linregress
from torch.utils.data import Dataset


def r2_score(y_pred, y_true, weights):
    numerator = torch.sum(weights * (y_true - y_pred)**2)
    denominator = torch.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator
    return r2_score

def weighted_mse(y_pred, y_true, weights):
    loss_fct = nn.MSELoss(reduction='none')

    unweighted_loss = loss_fct(y_pred, y_true)

    weighted_loss = weights * unweighted_loss

    return weighted_loss.mean()


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


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


class GeneralDataset(Dataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = None, dual_loading: bool = False):
        self.data_type = data_type
        if self.data_type != 'train' and self.data_type != 'eval':
            raise ValueError('Type must be either train or eval')

        self.in_size = in_size
        self.out_size = out_size
        self.single_symbol = single_symbol
        self.sort_symbols = sort_symbols
        self.collect_data_at_loading = collect_data_at_loading
        self.device = device
        self.nu_days = end_date - start_date

        if dual_loading:
            start_date_1 = start_date
            end_date_1 = (end_date - start_date) // 2
            start_date_2 = end_date_1
            end_date_2 = end_date
            train_data_1 = self._load_partial_data(path, start_date_1, end_date_1)
            train_data_1 = self._extract_train_data(train_data_1).to(device)

            train_data_2 = self._load_partial_data(path, start_date_2, end_date_2)
            train_data_2 = self._extract_train_data(train_data_2).to(device)
            self.data = torch.cat((train_data_1, train_data_2), dim=0)
        else:
            train_data = self._load_partial_data(path, start_date, end_date)
            self.data = self._extract_train_data(train_data)

        if normalize_features:
            self._normalize_features()

        self.nu_rows = self.data.shape[0]
        self.nu_cols = self.data.shape[1]


    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400, end_date=1699) -> pl.LazyFrame | pl.DataFrame:

        all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

        train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date))
        if self.single_symbol is not None:
            train_data = train_data.filter(pl.col("symbol_id") == self.single_symbol)

        if self.collect_data_at_loading:
            train_data = train_data.collect()

        if self.sort_symbols:
            train_data = train_data.sort("symbol_id", maintain_order=True)

        return train_data

    def _extract_train_data(self, train_data: pl.LazyFrame | pl.DataFrame) -> torch.Tensor:
        # Generate features
        feature_features = [f"feature_{i:02d}" for i in range(79)]
        responders_features = [f"responder_{i}" for i in range(9)]
        symbol_feature = ['symbol_id']
        temporal_features = ['date_id', 'time_id']
        weight_feature = ['weight']

        # Combine features
        required_features = feature_features + responders_features + symbol_feature + temporal_features + weight_feature

        # Create a dictionary mapping features to their indices
        feature_index_mapping = {feature: index for index, feature in enumerate(required_features)}

        required_data = train_data.select(required_features)
        required_data = required_data.fill_null(0)

        if self.collect_data_at_loading:
            data = torch.tensor(required_data.to_numpy(), dtype=torch.float32)
        else:
            data = torch.tensor(required_data.collect().to_numpy(), dtype=torch.float32)

        print(f'Numpy {self.data_type} data created with shape: {data.shape}')
        if self.data_type == 'train':
            print(f'Numpy dataframe columns and indexes:{feature_index_mapping}')

        return data

    def _normalize_features(self):
        data = self.get_features()
        means = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        std = torch.where(std == 0, 0, std)

        self.data[:, :79] = (self.data[:, :79] - means) / std

    def _skew_weights(self):
        adjust_tensor = torch.arange(self.nu_rows, dtype=torch.float32, device=self.data.device)
        adjust_tensor = adjust_tensor / torch.max(adjust_tensor) + 0.5

        self.data[:, 91] = self.data[:, 91] * adjust_tensor

    def get_all(self, idx=None):
        if idx is None:
            return self.data[:, :]
        return self.data[idx, :]

    def get_features(self, idx=None):
        if idx is None:
            return self.data[:, :79]
        return self.data[idx, :79]

    def get_responders(self, idx=None):
        if idx is None:
            return self.data[:, 79:88]
        return self.data[idx, 79:88]

    def get_features_and_responders(self, idx=None):
        if idx is None:
            return self.data[:, :88]
        return self.data[idx, :88]

    def get_symbols(self, idx=None):
        if idx is None:
            return self.data[:, 88].to(torch.int32)
        return self.data[idx, 88].to(torch.int32)

    def get_dates(self, idx=None):
        if idx is None:
            return self.data[:, 89].to(torch.int32)
        return self.data[idx, 89].to(torch.int32)

    def get_times(self, idx=None):
        if idx is None:
            return self.data[:, 90].to(torch.int32)
        return self.data[idx, 90].to(torch.int32)

    def get_weights(self, idx=None):
        if idx is None:
            return self.data[:, 91]
        return self.data[idx, 91]

    def get_temporal(self, idx=None):
        if idx is None:
            return self.data[:, 89:91]
        return self.data[idx, 89:91]


class PartialDataset(GeneralDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 dual_loading):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols,
                         collect_data_at_loading, normalize_features, device, None, dual_loading)


class SingleRowPD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, sort_symbols: bool,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device, dual_loading):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols, collect_data_at_loading,
                         normalize_features, device, dual_loading)

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        X = self.get_features(idx)
        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        return X, Y, temporal, weights, symbol


class GPULoaderGeneral:
    def __init__(self, dataset, shuffle, batch_size, device, min_row_offset=0, max_row_offset=0):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.min_row_offset = min_row_offset
        self.max_row_offset = max_row_offset

        self.dataset.data = self.dataset.data.to(device)

        self.batch_number = 0
        self.nu_batches = self._len()

        self.shuffled_indices = torch.arange(min_row_offset, self.dataset.nu_rows - max_row_offset)
        if self.shuffle:
            self._shuffle_data()

    def __len__(self):
        return self.nu_batches

    def _len(self):
        return math.ceil((self.dataset.nu_rows - self.min_row_offset - self.max_row_offset) / self.batch_size)

    def _shuffle_data(self):
        self.shuffled_indices = torch.randperm(self.dataset.nu_rows - self.min_row_offset - self.max_row_offset) + self.min_row_offset - self.max_row_offset

    def _get_batch_indexes(self):
        start_idx = self.batch_number * self.batch_size
        self.batch_number += 1
        end_idx = min(self.batch_number * self.batch_size, self.dataset.nu_rows)

        return self.shuffled_indices[start_idx:end_idx]

    def get_batch(self):
        batch_indexes = self._get_batch_indexes()

        X = self.dataset.get_features(batch_indexes)
        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol


class GPULoaderLastTwenty(GPULoaderGeneral):
    def __init__(self, dataset, shuffle, batch_size, device, min_row_offset):
        super().__init__(dataset, shuffle, batch_size, device, min_row_offset)

    def get_batch(self):
        batch_indexes = self._get_batch_indexes()

        sequence_batch_indexes = torch.arange(self.min_row_offset, dtype=torch.int32).repeat(batch_indexes.shape[0], 1)

        sequence_batch_indexes = batch_indexes.unsqueeze(-1) + sequence_batch_indexes - self.min_row_offset

        X = self.dataset.data[sequence_batch_indexes, :79]
        Y = self.dataset.data[batch_indexes, 79:88]
        temporal = self.dataset.data[batch_indexes, 89:91]
        weights = self.dataset.data[batch_indexes, 88].unsqueeze(-1)
        symbol = self.dataset.data[batch_indexes, 91]

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol


class GeneralTrainEvalClass:
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type):
        super().__init__()

        self.model = model

        self.train_eval_type = train_eval_type

        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.total_iterations = len(self.loader)

        self.batch_size = batch_size
        self.mini_epoch_size = mini_epoch_size
        self.out_size = out_size

        self.epoch_loss = 0
        self.mini_epoch_loss = 0

        self.all_pred = []
        self.all_targets = []
        self.all_weights = []

        self.iteration = 0
        self.len_eval = self.loader.dataset.nu_rows

        self.stats_day_mapping = np.empty((0,))
        # if train_eval_type == 'eval':
        #     self._create_day_mapping()

    def _reset_cache(self):
        self.all_pred = torch.zeros((self.len_eval, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.len_eval, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.len_eval, 1), dtype=torch.float32)
        self.all_temporal = torch.zeros((self.len_eval, 2), dtype=torch.float32)
        self.epoch_loss = 0
        self.mini_epoch_loss = 0
        self.iteration = 0
        self.last_stats_row = 0

    def _update_cache(self, y_batch, weights_batch, loss, outputs, temporal_batch):
        self.epoch_loss += loss.item()
        self.mini_epoch_loss += loss.item()

        start_idx = self.last_stats_row
        end_idx = self.last_stats_row + y_batch.shape[0]

        self.all_pred[start_idx:end_idx] = outputs
        self.all_targets[start_idx:end_idx] = y_batch
        self.all_weights[start_idx:end_idx] = weights_batch
        self.all_temporal[start_idx:end_idx] = temporal_batch

        self.last_stats_row = end_idx


    def _log(self):
        if self.iteration % self.mini_epoch_size == 0 or self.iteration == self.total_iterations:
            nu_iterations_since_last_log = min(self.mini_epoch_size, ((self.iteration - 1) % self.mini_epoch_size) + 1)
            print(
                f"Iteration {self.iteration}/{self.total_iterations}, Average_Loss: {self.mini_epoch_loss / nu_iterations_since_last_log:.4f}")
            self.mini_epoch_loss = 0

    def _calculate_statistics(self):
        avg_epoch_loss = self.loss_function(self.all_pred, self.all_targets, self.all_weights)
        avg_epoch_r2 = r2_score(self.all_pred, self.all_targets, self.all_weights)
        avg_epoch_mse = weighted_mse(self.all_pred, self.all_targets, self.all_weights)

        avg_epoch_r2_responders = [
            r2_score(self.all_pred[:, i], self.all_targets[:, i], self.all_weights.squeeze())
            for i in range(9)
        ]

        # if self.train_eval_type == "eval":
        #     self._calculate_day_statistics()

        return avg_epoch_loss, avg_epoch_r2, avg_epoch_mse, avg_epoch_r2_responders

    def _create_day_mapping(self):
        @numba.njit()
        def _create_day_mapping_numba(dates, day_map):
            c_day = dates[0]
            last_index = 0
            c_day_idx = 0
            idx = 0
            for idx in range(dates.shape[0]):
                if c_day != dates[idx]:
                    c_day = dates[idx]
                    day_map[c_day_idx] = np.array([last_index, idx])
                    c_day_idx += 1
                    last_index = idx
            day_map[c_day_idx] = np.array([last_index, idx+1])
            return day_map

        all_temporal = self.loader.dataset.get_temporal().to(torch.int32).cpu().numpy()
        nu_days = self.loader.dataset.nu_days
        self.stats_day_mapping = np.zeros((nu_days, 2), dtype=np.int32)

        self.stats_day_mapping = _create_day_mapping_numba(all_temporal[:, 0], self.stats_day_mapping)

    def _calculate_day_statistics(self):
        r2_score_responder_6_per_day = []

        pred = self.all_pred[:, 6]
        target = self.all_targets[:, 6]
        weights = self.all_weights.squeeze()

        for i in range(self.stats_day_mapping.shape[0]):
            start_idx = self.stats_day_mapping[i, 0]
            end_idx = self.stats_day_mapping[i, 1]
            day_loss = r2_score(pred[start_idx:end_idx], target[start_idx:end_idx], weights[start_idx:end_idx]).item()
            r2_score_responder_6_per_day.append(day_loss)

        # Convert to numpy array for regression analysis
        days = np.arange(len(r2_score_responder_6_per_day))
        r2_scores = np.array(r2_score_responder_6_per_day)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(days, r2_scores)

        # Print regression statistics
        print("Linear Regression Statistics:")
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value ** 2}")
        print(f"P-value: {p_value}")
        print(f"Standard Error: {std_err}")

        # Plot the data and regression line
        plt.figure(figsize=(10, 6))
        plt.plot(days, r2_scores, label='R2 Scores', marker='o')
        plt.plot(days, slope * days + intercept, label='Regression Line', color='red', linestyle='--')
        plt.xlabel('Day Index')
        plt.ylabel('R2 Score')
        plt.title('R2 Score Responder 6 Per Day with Regression Line')
        plt.legend()
        plt.show()

    def _calculate_time_id_statistics(self):
        r2_score_per_day = []
        for i in range(978):
            mask = self.all_temporal[:, 1] == i
            pred = self.all_pred[:, 6][mask]
            target = self.all_targets[:, 6][mask]
            weights = self.all_weights[mask]

            r2_score_per_day.append(r2_score(pred, target, weights).item())
        plt.plot(r2_score_per_day)
        plt.show()

    def step_epoch(self):
        if self.train_eval_type == "eval":
            self.model.eval()
        elif self.train_eval_type == "train":
            self.model.train()

        self._reset_cache()

        for X_batch, Y_batch, temporal_batch, weights_batch, symbol_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            loss, outputs = self._run_model(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()

    def _run_model(self, x_batch, y_batch, weights_batch, symbol_batch=None):
        if self.train_eval_type == "eval":
            with torch.no_grad():
                # Forward pass
                outputs = self.model(x_batch)

                if outputs.shape != y_batch.shape:
                    raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

                loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)

                return loss, outputs
        elif self.train_eval_type == "train":
            outputs = self.model(x_batch)

            if outputs.shape != y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

            loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)

            # Backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Reset gradient to 0
            self.optimizer.zero_grad()

            return loss, outputs


class GPUTrainEvalClass(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type)

    def step_epoch(self):
        if self.train_eval_type == "eval":
            self.model.eval()
        elif self.train_eval_type == "train":
            self.model.train()

        self._reset_cache()

        for i in range(self.loader.nu_batches):
            self.iteration += 1
            x_batch, y_batch, temporal_batch, weights_batch, symbol_batch = self.loader.get_batch()

            loss, outputs = self._run_model(x_batch, y_batch, weights_batch)
            self._update_cache(y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()