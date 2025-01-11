import torch


from dataHandlers.DataHandlerGeneral import GeneralDatasetResponder, GPULoaderGeneral


class SingleRowPDResponder(GeneralDatasetResponder):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, sort_symbols: bool, device: torch.device):
        super().__init__(data_type, path, start_date, end_date, in_size, out_size, sort_symbols, device)

    def __len__(self):
        return self.nu_rows


class GPULoaderResponder(GPULoaderGeneral):
    def __init__(self, dataset, shuffle, batch_size, device, min_row_offset, max_row_offset):
        super().__init__(dataset, shuffle, batch_size, device, min_row_offset, max_row_offset)

    def get_batch(self):
        batch_indexes = self._get_batch_indexes()

        X = torch.empty((batch_indexes.shape[0], self.dataset.in_size), device=self.device)

        X[:, :79] = self.dataset.get_features(batch_indexes)
        X[:, 79:-1] = self.dataset.get_features(batch_indexes - 20)
        X[:, -1] = self.dataset.get_responders(batch_indexes - 20)[:, 6]

        Y = self.dataset.get_responders(batch_indexes)
        temporal = self.dataset.get_temporal(batch_indexes)
        weights = self.dataset.get_weights(batch_indexes).unsqueeze(-1)
        symbol = self.dataset.get_symbols(batch_indexes)

        if self.batch_number == self.nu_batches:
            if self.shuffle:
                self._shuffle_data()
            self.batch_number = 0

        return X, Y, temporal, weights, symbol