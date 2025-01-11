import math

from torch.utils.data import Dataset
import polars as pl
import torch


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


class GeneralDatasetResponder(GeneralDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, sort_symbols: bool, device: torch.device):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols, False, False, device, None, False)

    def _extract_train_data(self, train_data: pl.LazyFrame | pl.DataFrame) -> torch.Tensor:
        # Generate features
        responders_features = [f"responder_{i}" for i in range(9)]
        symbol_feature = ['symbol_id']
        temporal_features = ['date_id', 'time_id']
        weight_feature = ['weight']

        # Combine features
        required_features = responders_features + symbol_feature + temporal_features + weight_feature

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

    def get_all(self, idx=None):
        if idx is None:
            return self.data[:, :]
        return self.data[idx, :]

    def get_responders(self, idx=None):
        if idx is None:
            return self.data[:, :9]
        return self.data[idx, :9]

    def get_symbols(self, idx=None):
        if idx is None:
            return self.data[:, 9].to(torch.int32)
        return self.data[idx, 9].to(torch.int32)

    def get_dates(self, idx=None):
        if idx is None:
            return self.data[:, 10].to(torch.int32)
        return self.data[idx, 10].to(torch.int32)

    def get_times(self, idx=None):
        if idx is None:
            return self.data[:, 11].to(torch.int32)
        return self.data[idx, 11].to(torch.int32)

    def get_weights(self, idx=None):
        if idx is None:
            return self.data[:, 12]
        return self.data[idx, 12]

    def get_temporal(self, idx=None):
        if idx is None:
            return self.data[:, 10:12]
        return self.data[idx, 10:12]


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
        self.nu_rows = self.dataset.data.shape[0]

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


