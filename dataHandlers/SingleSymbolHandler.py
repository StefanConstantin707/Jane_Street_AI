import torch
import numpy as np
import numba

from dataHandlers.DataHandlerGeneral import GeneralDataset


class SingleSymbolDataset(GeneralDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, False, collect_data_at_loading, normalize_features, device, single_symbol)

class SingleRowDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)

        # self.x_indexes = torch.cat((torch.arange(0, 79), torch.tensor([88])), dim=0)
        self.x_indexes = torch.arange(0, 79)

    def get_x(self, idx=None):
        if idx is None:
            return self.data[:, self.x_indexes]
        return self.data[idx, self.x_indexes]

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        # nu_features,
        X = self.get_x(idx)
        # out_size,
        Y = self.get_responders(idx)
        # 2,
        temporal = self.get_temporal(idx)
        # 1,
        weights = self.get_weights(idx).unsqueeze(0)

        symbol = self.single_symbol

        return X, Y, temporal, weights, symbol

class SequenceDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0, seq_len: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)
        self.seq_len = seq_len
        self.x_indexes = torch.arange(0, 88)

    def get_x(self, idx=None):
        if idx is None:
            return self.data[:, self.x_indexes]
        return self.data[idx, self.x_indexes]

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        time = self.get_times(idx)
        start_idx = max(idx - time, idx - self.seq_len + 1)
        nu_filled_rows = idx - start_idx + 1
        X = torch.zeros((self.seq_len, self.in_size), dtype=torch.float32)

        if start_idx == idx - time:
            assert self.get_times(start_idx) == 0
        else:
            assert idx - start_idx + 1 == self.seq_len

        X[-nu_filled_rows:, :] = self.data[start_idx:idx+1, :88].clone()
        X[-1, 79:88] = torch.zeros((9,), dtype=torch.float32)

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)

        return X, Y, temporal, weights

class SequenceFeatureDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0, seq_len: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)
        self.seq_len = seq_len

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        start_idx = max(0, idx - self.seq_len + 1)
        nu_filled_rows = idx - start_idx + 1
        X = torch.zeros((self.seq_len, self.in_size), dtype=torch.float32)

        X[-nu_filled_rows:, :] = self.data[start_idx:idx+1, :79]

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)

        return X, Y, temporal, weights

class SeparateSequenceDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0, seq_len: int = 1):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)
        self.seq_len = seq_len
        self.last_day_index_mapping = self._get_last_day_mapping()
        self.x_indexes = torch.arange(0, 88)

    def _get_last_day_mapping(self):
        @numba.njit()
        def get_last_day_mapping(days, idx_map):
            prev_idx = 0
            middle_idx = -1
            day = days[0]
            for i in range(len(days)):
                if day != days[i]:
                    day = days[i]
                    if middle_idx == -1:
                        middle_idx = i
                    else:
                        prev_idx = middle_idx
                        middle_idx = i
                idx_map[i] = prev_idx
            return idx_map

        dates = self.get_dates().numpy()
        mapping = np.zeros((self.nu_rows,), dtype=np.int32)

        mapping = get_last_day_mapping(dates, mapping)

        return torch.tensor(mapping)


    def get_x(self, idx=None):
        if idx is None:
            return self.data[:, self.x_indexes]
        return self.data[idx, self.x_indexes]

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        X = torch.zeros((2, self.in_size), dtype=torch.float32)
        X[0, :] = self.get_x(self.last_day_index_mapping[idx])
        X[1, :79] = self.get_features(idx)

        Y = self.get_responders(idx)
        temporal = self.get_temporal(idx)
        weights = self.get_weights(idx).unsqueeze(0)

        return X, Y, temporal, weights

class TwoDayDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)

        self.prev_day_mapping = self._get_last_day_mapping()

        self.x_indexes = torch.arange(0, 79)

    def _get_last_day_mapping(self):
        @numba.njit()
        def get_last_day_mapping(days, idx_map):
            prev_idx = 0
            middle_idx = -1
            day = days[0]
            for i in range(len(days)):
                if day != days[i]:
                    day = days[i]
                    if middle_idx == -1:
                        middle_idx = i
                    else:
                        prev_idx = middle_idx
                        middle_idx = i
                idx_map[i] = prev_idx
            return idx_map

        dates = self.get_dates().numpy()
        mapping = np.zeros((self.nu_rows,), dtype=np.int32)

        mapping = get_last_day_mapping(dates, mapping)

        return torch.tensor(mapping)

    def get_x(self, idx=None):
        if idx is None:
            return self.data[:, self.x_indexes]
        return self.data[idx, self.x_indexes]

    def __len__(self):
        return self.nu_rows

    def __getitem__(self, idx):
        start_idx = self.prev_day_mapping[idx]
        X = torch.zeros((968*2, self.in_size), dtype=torch.float32)
        X[:(idx-start_idx+1)] = self.data[start_idx:idx+1, :88].clone()
        # out_size,
        Y = self.get_responders(idx)
        # 2,
        temporal = self.get_temporal(idx)
        # 1,
        weights = self.get_weights(idx).unsqueeze(0)

        X[968:, 79:88] = torch.zeros_like(X[968:, 79:88], dtype=torch.float32)

        return X, Y, temporal, weights

class SingleRowSamplingDatasetSS(SingleSymbolDataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = 0):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, collect_data_at_loading, normalize_features, device, single_symbol)

    def __len__(self):
        return self.nu_rows - 256

    def __getitem__(self, idx):
        idx += 256
        if self.data_type == "eval":
            X = torch.zeros((256, 92), dtype=torch.float32)
            X[:79] = self.data[idx-255:idx+1, :79]
            time = self.get_times(idx)
            X[-1, 88:] = torch.tensor([0, 0, self.single_symbol, time], dtype=torch.float32)

            Y = self.get_responders(idx)
            # 2,
            temporal = self.get_temporal(idx)
            # 1,
            weights = self.get_weights(idx).unsqueeze(0)

            return X, Y, temporal, weights, self.single_symbol


        X = torch.zeros((92,), dtype=torch.float32)

        time = self.get_times(idx)
        date = self.get_dates(idx)

        rand_prev_time_separation = torch.randint(0, 256, (1,))
        prev_idx = idx - rand_prev_time_separation
        prev_time = self.get_times(prev_idx)
        prev_date = self.get_dates(prev_idx)

        if prev_date < date:
            X[:88] = self.get_features_and_responders(prev_idx)
        else:
            X[:79] = self.get_features(prev_idx)

        date_difference = date - prev_date
        time_difference = time - prev_time

        X[88:] = torch.tensor([date_difference, time_difference, self.single_symbol, time], dtype=torch.float32)

        # out_size,
        Y = self.get_responders(idx)
        # 2,
        temporal = self.get_temporal(idx)
        # 1,
        weights = self.get_weights(idx).unsqueeze(0)

        return X, Y, temporal, weights, self.single_symbol