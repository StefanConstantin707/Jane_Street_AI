import torch
import polars as pl
import math
import numba

class GpuDataLoader:
    def __init__(self, path: str, start_date: int, end_date: int, shuffle: bool, normalize_features: bool, batch_size:int, device: torch.device):

        self.device = device
        self.shuffle = shuffle
        self.batch_size = batch_size

        train_data = self._load_partial_data(path, start_date, end_date)
        self.data = self._extract_train_data(train_data)

        if normalize_features:
            self._normalize_features()

        self.nu_rows = self.data.shape[0]
        self.nu_cols = self.data.shape[1]
        self.len_loader = self.len()

        self.nu_days = end_date - start_date

        self.index = 0

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400,
                           end_date=1699) -> pl.LazyFrame | pl.DataFrame:

        all_train_data = pl.scan_parquet(
            f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

        train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date))

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

        data = torch.tensor(required_data.collect().to_numpy(), device=self.device)

        return data

    def _normalize_features(self):
        data = self.data[:, :79]
        means = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        std = torch.where(std == 0, 1, std)

        self.data[:, :79] = (self.data[:, :79] - means) / std

    def len(self):
        return math.ceil(self.nu_rows/self.batch_size)

    def get_batch(self):
        start_idx = self.index * self.batch_size
        self.index += 1
        end_idx = min(self.index * self.batch_size, self.nu_rows)

        X = self.data[start_idx:end_idx, :79]
        Y = self.data[start_idx:end_idx, 79:88]
        temporal = self.data[start_idx:end_idx, 89:91]
        weights = self.data[start_idx:end_idx, 91]
        symbol = self.data[start_idx:end_idx, 88]

        return X, Y, temporal, weights, symbol

class DayEvalLoader:
    def __init__(self, path: str, start_date: int, end_date: int, shuffle: bool, normalize_features: bool, batch_size:int, device: torch.device):

        self.device = device
        self.shuffle = shuffle
        self.batch_size = batch_size

        train_data = self._load_partial_data(path, start_date, end_date)
        self.data = self._extract_train_data(train_data)

        if normalize_features:
            self._normalize_features()

        self.nu_rows = self.data.shape[0]
        self.nu_cols = self.data.shape[1]
        self.len_loader = self.len()

        self.nu_days = end_date - start_date

        self.index = 0

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400,
                           end_date=1699) -> pl.LazyFrame | pl.DataFrame:

        all_train_data = pl.scan_parquet(
            f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

        train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date))

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

        data = torch.tensor(required_data.collect().to_numpy(), device=self.device)

        return data

    def _normalize_features(self):
        data = self.data[:, :79]
        means = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        std = torch.where(std == 0, 1, std)

        self.data[:, :79] = (self.data[:, :79] - means) / std

    def _create_day_start_mapping(self):
        @numba.njit()
        def day_start_mapping(days, mapping):
            c_day = 0
            iteration = 0
            for i in range(days.shape[0]):
                if c_day != days[i]:
                    c_day = days[i]
                    mapping[iteration] = i
                    iteration += 1
            return mapping

    def len(self):
        return math.ceil(self.nu_rows/self.batch_size)

    def get_batch(self):
        start_idx = self.index * self.batch_size
        self.index += 1
        end_idx = min(self.index * self.batch_size, self.nu_rows)

        X = self.data[start_idx:end_idx, :79]
        Y = self.data[start_idx:end_idx, 79:88]
        temporal = self.data[start_idx:end_idx, 89:91]
        weights = self.data[start_idx:end_idx, 91]
        symbol = self.data[start_idx:end_idx, 88]

        return X, Y, temporal, weights, symbol

    def step_epoch(self):
        for i in range(self.len_loader):
            X, Y, temporal, weights, symbol = self.get_batch()

            with torch.no_grad():
                pred = self.model(X)

            loss = self.loss_fct(pred, Y, weights)






