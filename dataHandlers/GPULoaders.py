import math
from turtledemo.penrose import start

import numba
import numpy as np
import torch

from dataHandlers.DataHandlerGeneral import GPULoaderGeneral


class GPULoaderResponderLag(GPULoaderGeneral):
    def __init__(self, dataset, shuffle, batch_size, device):
        super().__init__(dataset, shuffle, batch_size, device)

    def get_batch(self):

        batch_indexes = self._get_batch_indexes()

        responder_indexes = torch.arange(21) + batch_indexes.unsqueeze(-1) - 40

        X = self.dataset.get_features(batch_indexes)
        X = torch.cat((X, self.dataset.data[responder_indexes, 85]), dim=1)
        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol

class GPUOnlineLoader(GPULoaderGeneral):
    def __init__(self, dataset, device):
        super().__init__(dataset, False, 0, device)
        self.start_time_id_mapping = np.empty((0,), dtype=np.int32)
        self._create_start_time_id_mapping()

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
        self.dataset = dataset
        self.dataset.data = self.dataset.data[-batch_size * 968:].to(device)
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
            self.day_cache = torch.zeros((968 * data.shape[0], data.shape[1]))
        self.day_cache[self.added_indices:self.added_indices + data.shape[0]] = data
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

class GPUOnlineCacheLoaderSequential(GPULoaderGeneral):
    def __init__(self, dataset, shuffle, batch_size, device):
        super().__init__(dataset, shuffle, batch_size, device)

        self.added_indices = 0
        self.nu_symbols = 0
        self.last_nu_symbols = 0

        self.day_cache = torch.empty([0], device=device)
        self.start_index = self.nu_rows

        self.first_pass = True

    def add_data(self, data):
        if data[0, 90] == 0:
            self.new_day_reset(data)

        self.day_cache[self.added_indices:self.added_indices + data.shape[0]] = data
        self.added_indices += data.shape[0]

    def new_day_reset(self, data):
        self.added_indices = 0
        self.last_nu_symbols = self.nu_symbols
        self.nu_symbols = data.shape[0]
        self.start_index = self.nu_rows

        if self.first_pass:
            self.last_nu_symbols = self.nu_symbols
            self.day_cache = self.dataset.data[-data.shape[0] * 968:]
            self.first_pass = False
            self.dataset.data = torch.cat((self.dataset.data, self.day_cache.clone()), dim=0)
        else:
            self.dataset.data = torch.cat((self.dataset.data[:self.nu_rows], self.day_cache.clone()), dim=0)

        self.day_cache = torch.zeros((968 * data.shape[0], data.shape[1]), device=self.device)

    def get_batch(self):
        batch_indexes = torch.randint(low=0, high=self.nu_rows, size=((self.batch_size - self.last_nu_symbols),))
        new_data_indexes = torch.arange(self.start_index, self.start_index + self.last_nu_symbols)
        self.start_index += self.last_nu_symbols

        if self.dataset.data.shape[0] < self.start_index:
            a=2
        batch_indexes = torch.cat((batch_indexes, new_data_indexes), dim=0)

        data = self.dataset.get_all(batch_indexes)

        X = self.dataset.get_features(batch_indexes)
        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        return data, X, Y, temporal, weights, symbol