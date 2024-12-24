import torch
import polars as pl
import numpy as np
import numba

from dataHandlers.DataHandlerGeneral import GeneralDataset

class PartialDataset(GeneralDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols, collect_data_at_loading, normalize_features, device, None)

class SingleRowPD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading, normalize_features, device)

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        X = self.get_features(idx)
        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        return X, Y, temporal, weights, symbol


class RowSamplerDatasetPD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading, normalize_features, device)
        self.rows_to_sample = rows_to_sample

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        c_time = self.get_times(idx)
        c_date = self.get_dates(idx)
        c_symbol = self.get_symbols(idx)

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        if self.data_type == "eval":
            X = torch.zeros((self.rows_to_sample, self.in_size), dtype=torch.float32)

            prev_idx = idx - self.rows_to_sample + 1
            prev_symbol = self.get_symbols(prev_idx)
            if prev_symbol != c_symbol:
                prev_idx = idx - c_time

            nu_rows = idx - prev_idx + 1

            prev_time = self.get_times(prev_idx)
            prev_date = self.get_dates(prev_idx)

            X[-nu_rows:, :79] = self.data[prev_idx:idx+1, :79]
            X[-nu_rows:, 88] = c_time - self.data[prev_idx:idx+1, 90]

            if prev_date < c_date:
                X[-nu_rows:-c_time, 79:88] = self.data[prev_idx:idx-c_time, 79:88]

        elif self.data_type == "train":
            X = torch.zeros((self.in_size,), dtype=torch.float32)
            rows_step_back = torch.randint(0, self.rows_to_sample, (1,)).item()
            prev_idx = idx - rows_step_back

            prev_symbol = self.get_symbols(prev_idx)
            if prev_symbol != c_symbol:
                prev_idx = idx - c_time

            prev_time = self.get_times(prev_idx)
            prev_date = self.get_dates(prev_idx)

            X[:79] = self.data[prev_idx, :79]
            X[88] = c_time - self.data[prev_idx, 90]

            if prev_date < c_date:
                X[79:88] = self.data[prev_idx, 79:88]
        else:
            raise Error

        return X, Y, temporal, weights, symbol


class RowSamplerSequence(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading, normalize_features, device)
        self.rows_to_sample = rows_to_sample

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        X = torch.zeros((2, self.in_size), dtype=torch.float32)
        X[-1, :79] = self.get_features(idx)

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        X[-1, 88] = symbol
        X[-2, 88] = symbol

        c_time = self.get_times(idx)
        c_date = self.get_dates(idx)
        c_symbol = self.get_symbols(idx)

        X[-1, 89] = c_time

        prev_day_end_idx = idx - c_time - 1
        if prev_day_end_idx < 0 or self.get_symbols(prev_day_end_idx) != c_symbol:
            return X, Y, temporal, weights, symbol

        if self.data_type == "eval":
            X[0, :88] = self.get_features_and_responders(prev_day_end_idx)
            X[-2, 89] = self.get_times(prev_day_end_idx)
        elif self.data_type == "train":
            max_time_id_last_day = self.get_times(prev_day_end_idx)
            rand_time_id = torch.randint(0, max_time_id_last_day + 1, (1,)).item()

            prev_idx = idx - rand_time_id
            X[0, :88] = self.get_features_and_responders(prev_idx)
            X[-2, 89] = self.get_times(prev_idx)
        else:
            raise Error

        return X, Y, temporal, weights, symbol
