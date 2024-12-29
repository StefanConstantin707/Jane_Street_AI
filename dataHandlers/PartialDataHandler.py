import math

import torch
import polars as pl
import numpy as np
import numba

from dataHandlers.DataHandlerGeneral import GeneralDataset


class PartialDataset(GeneralDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, dual_loading):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols,
                         collect_data_at_loading, normalize_features, device, None, dual_loading)


class SingleRowPD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device, dual_loading):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, False, collect_data_at_loading,
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


class SingleRowNoisePD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading,
                         normalize_features, device)

    def __len__(self):
        return self.nu_rows * 2

    def __getitem__(self, idx):
        if idx % 2 == 1 and self.data_type == "train":
            X = torch.randn((self.in_size,), dtype=torch.float32)
            Y = torch.ones((self.out_size,), dtype=torch.float32) * 5
            temporal = torch.zeros((2,), dtype=torch.float32)
            weights = torch.ones((1,), dtype=torch.float32)
            symbol = torch.zeros((1,), dtype=torch.int32).squeeze()
            return X, Y, temporal, weights, symbol

        if idx % 2 == 1:
            idx -= 1

        idx = idx // 2
        X = self.get_features(idx)
        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)
        symbol = self.get_symbols(idx)

        return X, Y, temporal, weights, symbol


class RowSamplerDatasetPD(PartialDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading,
                         normalize_features, device)
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

            X[-nu_rows:, :79] = self.data[prev_idx:idx + 1, :79]
            X[-nu_rows:, 88] = c_time - self.data[prev_idx:idx + 1, 90]

            if prev_date < c_date:
                X[-nu_rows:-c_time, 79:88] = self.data[prev_idx:idx - c_time, 79:88]

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
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int,
                 collect_data_at_loading: bool, normalize_features: bool, device: torch.device,
                 rows_to_sample: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, True, collect_data_at_loading,
                         normalize_features, device)
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
            raise Error

        return X, Y, temporal, weights, symbol


class GPULoader:
    def __init__(self, dataset, shuffle, batch_size, device):
        super().__init__()
        self.dataset = dataset
        self.dataset.data = self.dataset.data.to(device)

        self.batch_size = batch_size

        self.batch_number = 0
        self.shuffle = shuffle
        self.nu_batches = self._len()

        self.shuffled_indices = torch.arange(self.dataset.nu_rows)
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

class GPUOnlineLoader:
    def __init__(self, dataset, device):
        super().__init__()
        self.dataset = dataset
        self.dataset.data = self.dataset.data.to(device)

        self.batch_number = 0
        self.nu_batches = self._len()
        self.start_time_id_mapping = np.empty((0,), dtype=np.int32)
        self._create_start_time_id_mapping()

        self.shuffled_indices = torch.arange(self.dataset.nu_rows)

    def __len__(self):
        return self.nu_batches

    def _len(self):
        len_data = self.dataset.nu_days * 968
        return int(len_data)

    def _create_start_time_id_mapping(self):
        @numba.njit()
        def create_start_time_id_mapping(time_ids_ordered, time_map):
            c_time = time_ids_ordered[0]
            prev_idx = 0
            map_index = 0
            for i in range(time_ids_ordered.shape[0]):
                if time_ids_ordered[i] != c_time:
                    time_map[map_index] = np.array([prev_idx, i], dtype=np.int32)
                    map_index += 1
                    prev_idx = i
                    c_time = time_ids_ordered[i]
                time_map[map_index] = np.array([prev_idx, i + 1], dtype=np.int32)

            return time_map

        time_ids = self.dataset.get_times().cpu().numpy()
        self.start_time_id_mapping = np.zeros((self.nu_batches, 2), dtype=np.int32)
        self.start_time_id_mapping = create_start_time_id_mapping(time_ids, self.start_time_id_mapping)

    def get_batch(self):
        start_idx = self.start_time_id_mapping[self.batch_number, 0]
        end_idx = self.start_time_id_mapping[self.batch_number, 1]
        self.batch_number += 1

        batch_indexes = self.shuffled_indices[start_idx:end_idx]

        data = self.dataset.get_all(batch_indexes)

        X = self.dataset.get_features(batch_indexes)
        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        if self.batch_number == self.nu_batches:
            self.batch_number = 0

        return data, X, Y, temporal, weights, symbol

class GPUOnlineCacheLoader:
    def __init__(self, dataset, shuffle, batch_size, device):
        super().__init__()
        self.dataset = dataset
        self.dataset.data = self.dataset.data[-batch_size*968:].to(device)
        self.nu_rows = batch_size * 968

        self.batch_size = batch_size

        self.batch_number = 0
        self.shuffle = shuffle
        self.nu_batches = self._len()

        self.shuffled_indices = torch.arange(self.nu_rows)
        if self.shuffle:
            self._shuffle_data()

        self.added_indices = 0

        self.day_cache = torch.empty([0], device=device)

    def __len__(self):
        return self.nu_batches

    def _len(self):
        return math.ceil(self.nu_rows / self.batch_size)

    def _shuffle_data(self):
        self.shuffled_indices = torch.randperm(self.nu_rows)

    def add_data(self, data):
        if self.added_indices == 0:
            self.day_cache = torch.zeros((968*data.shape[0], data.shape[1]))
        self.day_cache[self.added_indices:self.added_indices+data.shape[0]] = data
        self.added_indices += data.shape[0]

    def _update_data(self):
        self.dataset.data = torch.cat((self.dataset.data[self.day_cache.shape[0]:], self.day_cache), dim=0)
        self.added_indices = 0

    def get_batch(self):
        start_idx = self.batch_number * self.batch_size
        self.batch_number += 1
        end_idx = min(self.batch_number * self.batch_size, self.nu_rows)

        batch_indexes = self.shuffled_indices[start_idx:end_idx]

        data = self.dataset.get_all(batch_indexes)

        X = self.dataset.get_features(batch_indexes)
        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0
            self._update_data()

        return data, X, Y, temporal, weights, symbol

