import math

import torch
import polars as pl
import numpy as np
import numba

from dataHandlers.DataHandlerGeneral import GPULoaderGeneral


class GPULoaderLastTwenty(GPULoaderGeneral):
    def __init__(self, dataset, shuffle, batch_size, device, min_row_offset):
        super().__init__(dataset, shuffle, batch_size, device, min_row_offset)

    def get_batch(self):
        batch_indexes = self._get_batch_indexes()

        sequence_batch_indexes = torch.arange(self.min_row_offset, dtype=torch.int32).repeat(batch_indexes.shape[0], 1)

        sequence_batch_indexes = batch_indexes.unsqueeze(-1) + sequence_batch_indexes

        X = self.dataset.data[sequence_batch_indexes, :79]
        Y = self.dataset.data[batch_indexes, 79:88]
        temporal = self.dataset.data[batch_indexes, 89:91]
        weights = self.dataset.data[batch_indexes, 88].unsqueeze(-1)
        symbol = self.dataset.data[batch_indexes, 91]

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol
