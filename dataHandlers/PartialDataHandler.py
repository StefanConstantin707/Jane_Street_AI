import torch
import polars as pl
import numpy as np
import numba
from sympy.stats.sampling.sample_numpy import numpy

from torch.utils.data import Dataset

from dataHandlers.DataTransformation import fill_null_values

class PartialDataset(Dataset):
    def __init__(self, data_type, path, start_date, end_date, data_percentage, in_size, out_size, device):
        self.data_type = data_type
        self.data_percentage = data_percentage
        if self.data_type != 'train' and self.data_type != 'eval':
            raise ValueError('Type must be either train or eval')

        train_data = self._load_partial_data(path, start_date, end_date)
        self.data = self._extract_train_data(train_data)

        self.nu_rows = self.data.shape[0]
        self.nu_cols = self.data.shape[1]

        self.out_size = out_size
        self.nu_features = in_size
        self.device = device

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400, end_date=1699):

        all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}jane-street-real-time-market-data-forecasting/train.parquet")

        train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date))

        return train_data

    def _extract_train_data(self, train_data: pl.LazyFrame) -> [np.array]:
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
        required_data = fill_null_values(required_data)

        data = torch.tensor(required_data.collect().to_numpy())

        print(f'Numpy data created with shape: {data.shape}')
        print(f'Numpy dataframe columns and indexes:{feature_index_mapping}')

        return data

    def get_features(self, idx=None):
        if idx is None:
            return self.data[:, :79]
        return self.data[idx, :79]

    def get_responders(self, idx=None):
        if idx is None:
            return self.data[:, 79:88]
        return self.data[idx, 79:88]

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

    def get_y(self, idx=None):
        return self.get_responders(idx=idx)

    def get_x(self, idx=None):
        if idx is None:
            return self.data[:, :80]
        return self.data[idx, :80]

    def get_temporal(self, idx=None):
        if idx is None:
            return self.data[:, 89:91]
        return self.data[idx, 89:91]
    
class SingleRowDataset(PartialDataset):
    def __init__(self, data_type, path, start_date, end_date, data_percentage, in_size, out_size, device):
        super().__init__(data_type, path, start_date, end_date, data_percentage, in_size, out_size, device)

        self.x_indexes = torch.cat((torch.arange(0, 80), torch.tensor([88])), dim=0)

    # def get_x(self, idx=None):
    #     if idx is None:
    #         return self.data[:, self.x_indexes]
    #     return self.data[idx, self.x_indexes]

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        # nu_features,
        X = self.get_x(idx)
        # out_size,
        Y = self.get_y(idx)
        # 2,
        temporal = self.get_temporal(idx)
        # 1,
        weights = self.get_weights(idx).unsqueeze(0)
        # 1,
        symbol = self.get_symbols(idx)

        return X, Y, temporal, weights, symbol

class SequentialDualDataset(PartialDataset):
    def __init__(self, data_type, path, start_date, end_date, data_percentage, in_size, out_size, device, seq_len):
        super().__init__(data_type, path, start_date, end_date, data_percentage, in_size, out_size, device)

        self.prev_time_id_mapping = self._create_prev_time_id_mapping()

        self.seq_len = seq_len

    def _create_prev_time_id_mapping(self):
        @numba.njit
        def _create_mapping(array, prev__mapping, prev_symbol):
            for row in range(array.shape[0]):
                symbol = array[row, 0]
                date = array[row, 1]
                time = array[row, 2]

                prev__mapping[symbol, date, time] = prev_symbol[symbol]
                prev_symbol[symbol] = row
            return prev__mapping

        nu_dates = self.get_dates().max() + 1
        nu_time_ids = self.get_times().max() + 1
        nu_symbols = self.get_symbols().max() + 1

        print(f'nu_dates: {nu_dates}, nu_time_ids: {nu_time_ids}, nu_symbols: {nu_symbols}')

        prev_time_id_mapping = np.zeros((nu_symbols, nu_dates, nu_time_ids), dtype=np.int32)
        prev_symbol_idx = np.ones((nu_symbols,), dtype=np.int32) * (-1)

        prev_time_id_mapping = _create_mapping(self.data[:, 88:91].to(torch.int32).numpy(), prev_time_id_mapping,
                                               prev_symbol_idx)
        print("Mapping Created")

        return prev_time_id_mapping

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        # seq_len, nu_features
        X = torch.zeros((self.seq_len, self.nu_features), dtype=torch.float32)
        # out_size,
        Y = self.get_y(idx)
        # 2,
        temporal = self.get_temporal(idx)
        # 1,
        weights = self.get_weights(idx)
        # 1,
        symbol = self.get_symbols(idx)

        X[-1] = self.get_x(idx).clone()
        X[-1, 79:88] = torch.zeros((self.out_size,))

        date = temporal[0]
        time = temporal[1]

        if time != 0:
            prev_idx = self.prev_time_id_mapping[symbol, date, time]
            X[-2] = self.get_x(prev_idx).clone()

        return X, Y, weights, time