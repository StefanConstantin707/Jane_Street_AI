import torch
import polars as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

from dataHandlers.DataTransformation import fill_null_values


def load_partial_data(start_date=1400):
    jane_street_real_time_market_data_forecasting_path = './jane-street-real-time-market-data-forecasting'

    all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

    train_data = all_train_data.filter(pl.col("date_id") >= start_date).collect()

    return train_data

def create_prev_time_id_mapping(train_data):
    time_and_symbol_names = ["date_id", "time_id", "symbol_id"]
    nu_dates = train_data.select(pl.col("date_id").unique()).collect().to_numpy().max() + 1
    nu_time_ids = train_data.select(pl.col("time_id").unique()).collect().to_numpy().max() + 1
    nu_symbols = train_data.select(pl.col("symbol_id").unique()).collect().to_numpy().max() + 1

    print(f'nu_dates: {nu_dates}, nu_time_ids: {nu_time_ids}, nu_symbols: {nu_symbols}')

    time_and_symbol_data = train_data.select(time_and_symbol_names).collect().to_numpy()

    nu_rows, _ = time_and_symbol_data.shape

    prev_time_id_mapping = np.zeros((nu_symbols, nu_dates, nu_time_ids), dtype=np.int32)

    prev_symbol_idx = np.ones((nu_symbols,), dtype=np.int32) * (-1)

    for row in range(nu_rows):
        date = time_and_symbol_data[row, 0]
        time = time_and_symbol_data[row, 1]
        symbol = time_and_symbol_data[row, 2]

        prev_time_id_mapping[symbol, date, time] = prev_symbol_idx[symbol]
        prev_symbol_idx[symbol] = row

    return prev_time_id_mapping

def extract_train_data(train_data: pl.LazyFrame) -> [np.array]:
    feature_features = [f"feature_{i:02d}" for i in range(79)]
    responders_features = [f"responder_{i}" for i in range(9)]
    symbol_feature = ['symbol_id']
    train_features = feature_features + responders_features + symbol_feature

    temporal_features = ['date_id', 'time_id']

    weight_feature = ['weight']

    data_XY = train_data.select(train_features)
    data_time =  train_data.select(temporal_features)
    data_weights = train_data.select(weight_feature)

    data_XY = fill_null_values(data_XY)

    data_XY = data_XY.collect().to_numpy()
    data_time = data_time.collect().to_numpy()
    data_weights = data_weights.collect().to_numpy()

    print(f'data_XY shape: {data_XY.shape}; data_time shape: {data_time.shape}; data_weights shape: {data_weights.shape}')

    return data_XY, data_time, data_weights

def split_data_train_eval(XY, time, weights, train_percentage, eval_percentage):
    nu_rows = XY.shape[0]
    nu_train_rows = nu_rows * train_percentage
    nu_eval_rows = nu_rows * eval_percentage

    train_XY, eval_XY = XY[:nu_train_rows], XY[nu_train_rows:]
    train_time, eval_time = time[:nu_train_rows], time[nu_train_rows:]
    train_weights, eval_weights = weights[:nu_train_rows], weights[nu_train_rows:]

    return train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights

def create_seq_dataloaders(train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights, batch_size, shuffle, seq_len):
    train_dataset = SequenceDataset(train_XY, train_time, train_weights, seq_len)
    eval_dataset = SequenceDataset(eval_XY, eval_time, eval_weights, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return train_loader, eval_loader

class SequenceDataset(Dataset):
    def __init__(self, data_XY, data_time, data_weights, seq_len, out_size, device):
        self.data_XY = data_XY
        self.data_time = data_time
        self.data_weights = data_weights

        self.seq_len = seq_len
        self.device = device

        self.nu_features = data_XY.shape[1]
        self.out_size = out_size

    def __len__(self):
        return self.data_weights.shape[0]

    def __getitem__(self, idx):
        # seq_len, nu_features
        X = torch.zeros((self.seq_len, self.nu_features), dtype=torch.float32, device=self.device)
        # out_size,
        Y = self.data_XY[idx, -self.out_size-1:-1]
        # 2,
        temporal = self.data_time[idx].to(self.device)
        # 1,
        weights = self.data_weights[idx].to(self.device)
        # 1,
        symbol = self.data_XY[idx, -1].to(self.device)

        date = temporal[1]
        time = temporal[0]

        X[1] = self.data_XY[idx]
        prev_time_id =

        return X, Y, weights, time
