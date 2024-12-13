import torch
import polars as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

from means_and_stds import means, stds


def load_data():
    jane_street_real_time_market_data_forecasting_path = './jane-street-real-time-market-data-forecasting'

    valid_from = 1400

    all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

    train = all_train_data.filter(pl.col("date_id") >= valid_from).collect()

    feature_names = [f"feature_{i:02d}" for i in range(79)] + ["symbol_id"]

    train_features = train.select(feature_names)

    train_features = train_features.fill_null(strategy='forward').fill_null(0)

    train_features = normalize_dataframe(train_features, means, stds)

    X = train_features.to_numpy()

    Y = train.select('responder_6').to_numpy().reshape(-1)

    weights = train.select('weight').to_numpy().reshape(-1)

    return X, Y, weights

def load_data_symbol(symbol_id):
    jane_street_real_time_market_data_forecasting_path = './jane-street-real-time-market-data-forecasting'

    all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

    train = all_train_data.filter(pl.col("symbol_id") == symbol_id).collect()

    feature_names = [f"feature_{i:02d}" for i in range(79)]
    # responders_features = ['responder_0','responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5', 'responder_7', 'responder_8']
    # feature_names = feature_names + responders_features

    train_features = train.select(feature_names)

    train_features = train_features.fill_null(strategy='forward').fill_null(0)

    train_features = normalize_dataframe(train_features, means, stds)

    X = train_features.to_numpy()

    responders_features = [f"responder_{i}" for i in range(9)]
    Y = train.select(responders_features).to_numpy()

    weights = train.select('weight').to_numpy().reshape(-1)

    return X, Y, weights

def load_data_xy_symbol(symbol_id):
    jane_street_real_time_market_data_forecasting_path = './jane-street-real-time-market-data-forecasting'

    all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

    unfilled_train = all_train_data.filter(pl.col("symbol_id") == symbol_id).collect()

    # train = fill_missing_rows(unfilled_train)
    train = unfilled_train

    feature_features = [f"feature_{i:02d}" for i in range(79)]
    responders_features = [f"responder_{i}" for i in range(9)]
    temporal_features = ['date_id', 'time_id']

    feature_names = feature_features + responders_features

    train_features = train.select(feature_names)

    train_features = train_features.fill_null(strategy='zero')

    train_features = normalize_dataframe(train_features, means, stds)

    XY = train_features.to_numpy()

    time = train.select(temporal_features).to_numpy()

    weights = train.select('weight').to_numpy().reshape(-1)

    return XY, time, weights

def split_data(X, Y, weights, train_percentage, val_percentage, test_percentage):
    nu_train_rows = int(len(X) * train_percentage)
    nu_val_rows = int(len(X) * val_percentage)
    nu_test_rows = int(len(X) * test_percentage)

    train_X, val_X, test_X = X[:nu_train_rows], X[nu_train_rows:(nu_train_rows + nu_val_rows)], X[-nu_test_rows:]
    train_Y, val_Y, test_Y = Y[:nu_train_rows], Y[nu_train_rows:(nu_train_rows + nu_val_rows)], Y[-nu_test_rows:]
    train_weights, val_weights, test_weights = weights[:nu_train_rows], weights[nu_train_rows:(
                nu_train_rows + nu_val_rows)], weights[-nu_test_rows:]

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    train_weights = torch.tensor(train_weights, dtype=torch.float32)

    val_X = torch.tensor(val_X, dtype=torch.float32)
    val_Y = torch.tensor(val_Y, dtype=torch.float32)
    val_weights = torch.tensor(val_weights, dtype=torch.float32)

    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.float32)
    test_weights = torch.tensor(test_weights, dtype=torch.float32)

    return train_X, train_Y, train_weights, val_X, val_Y, val_weights, test_X, test_Y, test_weights

def split_data_train_eval(XY, time, weights, train_percentage, eval_percentage):
    # XY: 1645600, 79+9
    # time: 1645600, 2
    # weights: 1645600

    nu_rows = len(XY)
    nu_train_rows = int(len(XY) * train_percentage)
    nu_eval_rows = int(len(XY) * eval_percentage)

    train_XY, eval_XY = XY[:nu_train_rows], XY[nu_train_rows:]
    train_time, eval_time = time[:nu_train_rows], time[nu_train_rows:]
    train_weights, eval_weights = weights[:nu_train_rows], weights[nu_train_rows:]

    train_XY = torch.tensor(train_XY, dtype=torch.float32)
    train_time = torch.tensor(train_time, dtype=torch.int32)
    train_weights = torch.tensor(train_weights, dtype=torch.float32)

    eval_XY = torch.tensor(eval_XY, dtype=torch.float32)
    eval_time = torch.tensor(eval_time, dtype=torch.int32)
    eval_weights = torch.tensor(eval_weights, dtype=torch.float32)

    # XY: nu_days, 968, 79+9
    # time: nu_rows, 2
    # weights: nu_rows
    return train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights

def create_dataloaders(train_X, train_Y, train_weights, val_X, val_Y, val_weights, test_X, test_Y, test_weights, batch_size, shuffle):
    train_dataset = TensorDataset(train_X, train_Y, train_weights)
    val_dataset = TensorDataset(val_X, val_Y, val_weights)
    test_dataset = TensorDataset(test_X, test_Y, test_weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader

def create_seq_dataloaders(train_X, train_Y, train_weights, val_X, val_Y, val_weights, test_X, test_Y, test_weights, batch_size, shuffle, seq_len):
    train_dataset = SequenceDataset(train_X, train_Y, train_weights, seq_len)
    val_dataset = SequenceDataset(val_X, val_Y, val_weights, seq_len)
    test_dataset = SequenceDataset(test_X, test_Y, test_weights, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader

def create_feature_dataloaders(train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights, batch_size, shuffle, seq_len):
    train_dataset = TrainFeaturesDataset(train_XY, train_time, train_weights, seq_len)
    eval_dataset = TrainFeaturesDataset(eval_XY, eval_time, eval_weights, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, eval_loader

def create_day_data_loader(train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights, batch_size, shuffle, nu_days):
    train_dataset = SequenceDataset2(train_XY, train_time, train_weights, nu_days)
    eval_dataset = SequenceDataset2(eval_XY, eval_time, eval_weights, nu_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    return train_loader, eval_loader

class SequenceDataset(Dataset):
    def __init__(self, data_X, data_Y, data_weights, seq_len):
        self.data_X = data_X
        self.data_Y = data_Y
        self.data_weights = data_weights

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_Y)

    def __getitem__(self, idx):
        start = max(idx - self.seq_len + 1, 0)

        X = self.data_X[start:idx+1]
        Y = self.data_Y[idx]
        weights = self.data_weights[idx]

        if idx < self.seq_len:
            zeros_X = torch.zeros((self.seq_len - idx - 1, X.shape[1]), dtype=torch.float32)
            X = torch.cat((zeros_X, X), dim=0)

        return X, Y, weights

class SequenceDataset2(Dataset):
    def __init__(self, data_XY, data_time, data_weights, seq_len):
        self.data_XY = data_XY
        self.data_time = data_time
        self.data_weights = data_weights

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_weights)

    def __getitem__(self, idx):
        start = max(idx - self.seq_len + 1, 0)

        while self.data_weights[idx] == 0.0:
            idx = torch.randint(0, self.__len__(), (1,))
            start = max(idx - self.seq_len + 1, 0)

        X = self.data_XY[start:idx+1].clone()

        Y = self.data_XY[idx, -9:]
        weights = self.data_weights[idx]

        time = self.data_time[idx]

        if idx < self.seq_len:
            zeros_X = torch.zeros((self.seq_len - idx - 1, X.shape[1]), dtype=torch.float32)
            X = torch.cat((zeros_X, X), dim=0)

        X[-1, -9:] = torch.zeros((9,), dtype=torch.float32)

        return X, Y, weights.unsqueeze(0), time

class DaySequenceDataset(Dataset):
    def __init__(self, train_XY, train_time, train_weights, nu_days):
        self.nu_rows = train_XY.shape[0]

        self.data_XY = train_XY.reshape(self.nu_rows//968, 968, -1)
        self.data_time = train_time - train_time[0]
        self.data_weights = train_weights

        self.nu_days = nu_days

    def __len__(self):
        return len(self.data_weights)

    def __getitem__(self, idx):
        X = torch.zeros((968*2, self.data_XY.shape[-1]), dtype=torch.float32)

        date = self.data_time[idx, 0]
        time = self.data_time[idx, 1]

        if date >= 0:
            X[:968] = self.data_XY[date-1]
        X[968:968+time+1] = self.data_XY[date, :time+1]
        # X[968:, -9:] = torch.zeros((968, 9), dtype=torch.float32)
        X[time+968, -9:] = torch.zeros((9,), dtype=torch.float32)

        Y = self.data_XY[date, time, -9:]
        weights = self.data_weights[idx]

        return X, Y, weights

class TrainFeaturesDataset(Dataset):
    def __init__(self, data_XY, data_time, data_weights, seq_len):
        self.data_XY = data_XY
        self.data_time = data_time
        self.data_weights = data_weights

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_weights)

    def __getitem__(self, idx):
        if self.seq_len == 1:
            X = self.data_XY[idx, 1:]
            Y = self.data_XY[idx, 0]
            weights = self.data_weights[idx]
        else:
            raise Exception

        return X, Y, weights

