import torch
from torch import optim, nn

import torch
import polars as pl
import numpy as np
from einops import repeat, rearrange
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import numba
from rotary_embedding_torch import RotaryEmbedding
from scipy.stats import linregress


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class RotationalPositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim=channels // 2)

    def forward(self, q, k):
        return self.rotary_emb.rotate_queries_or_keys(q), self.rotary_emb.rotate_queries_or_keys(k)


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, dim_out, rotary_emb, dropout=0.0):
        super().__init__()

        self.scale = dim_qk ** -0.5

        self.norm = nn.LayerNorm(dim_in)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim_in, dim_qk, bias=False)
        self.to_k = nn.Linear(dim_in, dim_qk, bias=False)
        self.to_v = nn.Linear(dim_in, dim_v, bias=False)

        self.rotary_emb = RotationalPositionalEncoding1D(dim_qk) if rotary_emb else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(dim_v, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # N, Seq_L, dim_in -> N, Seq_L, dim_in
        x = self.norm(x)

        # N, Seq_L, dim_in -> N, Seq_L, dim_qk
        q = self.to_q(x)
        k = self.to_k(x)
        # N, Seq_L, dim_in -> N, Seq_L, dim_v
        v = self.to_v(x)

        q, k = self.rotary_emb(q, k)

        # N, Seq_L, dim_qk @ N, Seq_L, dim_qk -> N, Seq_L, Seq_L
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # N, Seq_L, Seq_L @ N, Seq_L, dim_v -> N, Seq_L, dim_v
        out = torch.matmul(attn, v)

        # N, Seq_L, dim_v -> N, Seq_L, dim_out
        return self.to_out(out)


class FeedForwardGeneral(nn.Module):
    def __init__(self, layer_widths: list, activation_fct, dropout=0.):
        super().__init__()
        layers = []
        depth = len(layer_widths) - 1

        for i in range(depth - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < depth - 2:
                if activation_fct == 'relu':
                    layers.append(nn.ReLU())
                elif activation_fct == 'gelu':
                    layers.append(nn.GELU())
                elif activation_fct == 'tanh':
                    layers.append(nn.Tanh())
                elif activation_fct == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise NotImplementedError(activation_fct)
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def r2_score(y_pred, y_true, weights):
    numerator = torch.sum(weights * (y_true - y_pred) ** 2)
    denominator = torch.sum(weights * y_true ** 2)

    r2_score = 1 - numerator / denominator
    return r2_score


def weighted_mse(y_pred, y_true, weights):
    loss_fct = nn.MSELoss(reduction='none')

    unweighted_loss = loss_fct(y_pred, y_true)

    weighted_loss = weights * unweighted_loss

    return weighted_loss.mean()


class GeneralTrainEvalClass:
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size,
                 train_eval_type):
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
            day_map[c_day_idx] = np.array([last_index, idx + 1])
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
                    raise ValueError(
                        "Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

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

        return self._calculate_statistics()\


class GPURowSampler:
    def __init__(self, dataset, shuffle, batch_size, device):
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.dataset.data = self.dataset.data.to(device)

        self.batch_size = batch_size

        self.batch_number = 0
        self.shuffle = shuffle
        self.nu_batches = self._len()

        self.shuffled_indices = torch.arange(self.dataset.nu_days * 968 * 39)
        if self.shuffle:
            self._shuffle_data()

    def __len__(self):
        return self.nu_batches

    def _len(self):
        return math.ceil(self.dataset.nu_rows / self.batch_size)

    def _shuffle_data(self):
        self.shuffled_indices = torch.randperm(self.dataset.nu_rows)

    def get_batch(self):
        start_idx = self.batch_number * self.batch_size
        self.batch_number += 1
        end_idx = min(self.batch_number * self.batch_size, self.dataset.nu_rows)

        batch_indexes = self.shuffled_indices[start_idx:end_idx]

        X = torch.zeros((self.batch_size, 2, 90), dtype=torch.float32, device=self.device)

        X[:, 1, :79] = self.dataset.data[batch_indexes, :79]

        rand_int_last_day = torch.randint(0, 968, (self.batch_size,), dtype=torch.int32)

        times = self.dataset.data[batch_indexes, 90].to(torch.int32)

        rand_times_indexes = batch_indexes - times - 968 + rand_int_last_day
        rand_times_indexes = torch.where(rand_times_indexes < 0, 0, rand_times_indexes)

        X[:, 0, :88] = self.dataset.data[rand_times_indexes, :88]
        X[:, 0, 89] = times + 968 - rand_int_last_day


        Y = self.dataset.data[batch_indexes, 79:88]
        temporal = self.dataset.data[batch_indexes, 89:91].to(torch.int32)
        weights = self.dataset.data[batch_indexes, 91].unsqueeze(-1)
        symbol = self.dataset.data[batch_indexes, 88].to(torch.int32)

        X[:, :, 88] = torch.ones_like(X[:, :, 88], dtype=torch.float32, device=self.device) * symbol.unsqueeze(-1)

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol


def r2_loss(y_pred, y_true, weights):
    numerator = torch.sum(weights * (y_true - y_pred) ** 2)
    denominator = torch.sum(weights * y_true ** 2)

    r2_loss = numerator / denominator
    return r2_loss


class GeneralDataset(Dataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 single_symbol: int = None, dual_loading: bool = False):
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

        # if self.data_type == 'train':
        #     self._skew_weights()

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400,
                           end_date=1699) -> pl.LazyFrame | pl.DataFrame:

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


class RowSamplerSequence(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading,
                         normalize_features, device, False)
        self.rows_to_sample = rows_to_sample

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        X = torch.zeros((self.rows_to_sample, self.in_size), dtype=torch.float32)
        X[-1, :79] = self.get_features(idx)

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        X[:, 88] = torch.ones((self.rows_to_sample,), dtype=torch.float32) * symbol

        c_time = self.get_times(idx)
        c_date = self.get_dates(idx)
        c_symbol = self.get_symbols(idx)

        X[-1, 89] = c_time

        prev_day_end_idx = idx - c_time - 1
        if prev_day_end_idx < 0 or self.get_symbols(prev_day_end_idx) != c_symbol:
            return X, Y, temporal, weights, symbol

        if self.data_type == "eval":
            rand_time_id = torch.arange(self.rows_to_sample - 1)

            prev_idx = prev_day_end_idx - rand_time_id
            X[:-1, :88] = self.get_features_and_responders(prev_idx)
            X[:-1, 89] = self.get_times(prev_idx)
        elif self.data_type == "train":
            max_time_id_last_day = self.get_times(prev_day_end_idx)
            rand_time_id = torch.randint(0, max_time_id_last_day + 1, (self.rows_to_sample - 1,))

            prev_idx = prev_day_end_idx - rand_time_id
            X[:-1, :88] = self.get_features_and_responders(prev_idx)
            X[:-1, 89] = self.get_times(prev_idx)
        else:
            raise Exception

        return X, Y, temporal, weights, symbol


class SymbolAndTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim_symbol, embedding_dim_time):
        super(SymbolAndTimeEmbedding, self).__init__()

        self.embedding_dim_symbol = embedding_dim_symbol
        self.embedding_dim_time = embedding_dim_time

        self.embedding_s = nn.Embedding(100, embedding_dim_symbol)
        self.embedding_t = nn.Embedding(2000, embedding_dim_time)
    def forward(self, x):
        # x: N, in_dim
        # symbol_id: N, 1
        # time_id: N, 1
        symbol_id = x[:, :, -2].to(torch.int32)
        time_id = x[:, :, -1].to(torch.int32)

        # N -> N, embedding_dim
        emb_symbol = self.embedding_s(symbol_id)
        emb_time = self.embedding_t(time_id)

        x = torch.cat((x[:, :, :-2], emb_symbol, emb_time), dim=2)

        return x


class TransformerLayerGeneral(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, attention_depth, rotary_emb, mlp_layer_widths, activation_fct, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.layers = nn.ModuleList([])
        for _ in range(attention_depth):
            self.layers.append(nn.ModuleList([
                SingleHeadSelfAttention(dim_in=dim_in, dim_qk=dim_qk, dim_v=dim_v, dim_out=dim_in,
                                        rotary_emb=rotary_emb, dropout=dropout),
                FeedForwardGeneral(layer_widths=mlp_layer_widths, activation_fct=activation_fct, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TransformerGeneral(nn.Module):
    def __init__(self, *, dim_in, dim_attn, dim_qk, dim_v, attention_depth, dim_out, rotary_emb=True, mlp_layer_widths, activation_fct, dropout=0., noise=0.0):
        super().__init__()

        self.embedding = SymbolAndTimeEmbedding(1, 1)

        self.batch_norm = nn.BatchNorm1d(dim_in)

        self.noise = GaussianNoise(std=noise)

        self.to_input = nn.Linear(dim_in, dim_attn)

        # 1, 1, dim_attn
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_attn))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerLayerGeneral(dim_in=dim_attn, dim_qk=dim_qk, dim_v=dim_v, attention_depth=attention_depth, rotary_emb=rotary_emb, mlp_layer_widths=mlp_layer_widths, activation_fct=activation_fct, dropout=dropout)
        self.to_responders = nn.Linear(dim_attn, dim_out)

    def forward(self, x):

        x = self.embedding(x)

        x = self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.noise(x)
        x = self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)

        # N, Seq_L, dim_in -> N, Seq_L, dim_attn
        x = self.to_input(x)
        b, s, _ = x.shape

        # 1, 1, dim_attn -> N, 1, dim_attn
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        # N, Seq_L, dim_attn + N, 1, dim_attn -> N, Seq_L + 1, dim_attn
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        # N, Seq_L + 1, dim_attn -> N, Seq_L + 1, dim_attn
        x = self.transformer(x)

        # N, Seq_L + 1, dim_attn -> N, 1, dim_attn
        x = x[:, 0]

        # N, 1, dim_attn -> N, 1, dim_out -> N, dim_out
        x = self.to_responders(x).squeeze(1)

        return x

    def save(self, path=".\\savedModels\\attention_nn.pt"):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
in_size = 90
out_size = 9
epochs = 20
mini_epoch_size = 100
batch_size = 4096
lr = 1e-5
noise = 1

model = TransformerGeneral(dim_in=in_size, dim_attn=256, dim_qk=256, dim_v=256, attention_depth=1, dim_out=out_size,
                           rotary_emb=True,
                           mlp_layer_widths=[256, 256, 256, 256], activation_fct="relu", dropout=0.3,
                           noise=noise).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

trainDataset = RowSamplerSequence(data_type='train', path="../JaneStreetRealTimeMarketDataForecasting",
                                  start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device,
                                  collect_data_at_loading=False, normalize_features=False, rows_to_sample=17)
evalDataset = RowSamplerSequence(data_type='eval', path="../JaneStreetRealTimeMarketDataForecasting",
                                 start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device,
                                 collect_data_at_loading=False, normalize_features=False, rows_to_sample=17)

train_loader = GPURowSampler(trainDataset, shuffle=True, batch_size=batch_size, device=device)
eval_loader = GPURowSampler(evalDataset, shuffle=False, batch_size=batch_size, device=device)

trainClass = GPUTrainEvalClass(model, train_loader, optimizer, r2_loss, device, out_size, batch_size,
                               mini_epoch_size, "train")
evalClass = GPUTrainEvalClass(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size,
                              "eval")

for epoch in range(epochs):
    train_loss, train_r2_score, train_mse, train_r2_scores_responders = trainClass.step_epoch()

    # Print the general statistics
    print(
        f'Train statistics: Epoch: {epoch} Loss: {train_loss:.4f}, R2 score: {train_r2_score:.4f}, MSE: {train_mse:.4f}')

    # Print the R² scores for each responder
    for i, r2_score_responder in enumerate(train_r2_scores_responders):
        print(f'R2 score responder {i}: {r2_score_responder:.4f}')

    eval_loss, eval_r2_score, eval_mse, eval_r2_score_responders = evalClass.step_epoch()

    # Print the general statistics
    print(
        f'Eval statistics: Epoch: {epoch} Loss: {eval_loss:.4f}, R2 score: {eval_r2_score:.4f}, MSE: {eval_mse:.4f}')

    # Print the R² scores for each responder
    for i, eval_r2_score_responder in enumerate(eval_r2_score_responders):
        print(f'R2 score responder {i}: {eval_r2_score_responder:.4f}')
