from torch.utils.data import Dataset
import polars as pl
import torch


class GeneralDataset(Dataset):
    def __init__(self, data_type: str, path: str, start_date: int, end_date: int, in_size: int, out_size: int, sort_symbols: bool, collect_data_at_loading: bool, normalize_features: bool, device: torch.device, single_symbol: int = None):
        self.data_type = data_type
        if self.data_type != 'train' and self.data_type != 'eval':
            raise ValueError('Type must be either train or eval')

        self.in_size = in_size
        self.out_size = out_size
        self.single_symbol = single_symbol
        self.sort_symbols = sort_symbols
        self.collect_data_at_loading = collect_data_at_loading
        self.device = device

        train_data = self._load_partial_data(path, start_date, end_date)
        self.data = self._extract_train_data(train_data)
        if normalize_features:
            self._normalize_features()

        self.nu_rows = self.data.shape[0]
        self.nu_cols = self.data.shape[1]



        self.nu_days = end_date - start_date

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1400, end_date=1699) -> pl.LazyFrame | pl.DataFrame:

        all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}JaneStreetRealTimeMarketDataForecasting/train.parquet")

        train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date))
        if self.single_symbol is not None:
            train_data = train_data.filter(pl.col("symbol_id") == self.single_symbol)

        if self.collect_data_at_loading:
            train_data = train_data.collect()

        if self.sort_symbols:
            train_data = train_data.sort("symbol_id", maintain_order=True)

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

        if self.collect_data_at_loading:
            data = torch.tensor(required_data.to_numpy())
        else:
            data = torch.tensor(required_data.collect().to_numpy())

        print(f'Numpy {self.data_type} data created with shape: {data.shape}')
        if self.data_type == 'train':
            print(f'Numpy dataframe columns and indexes:{feature_index_mapping}')

        return data

    def _normalize_features(self):
        data = self.get_features()
        means = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        std = torch.where(std == 0, 0, std)

        self.data[:, :79] = (self.data[:, :79] - means) / std

    def get_features(self, idx=None):
        if idx is None:
            return self.data[:, :79]
        return self.data[idx, :79]

    def get_responders(self, idx=None):
        if idx is None:
            return self.data[:, 79:88]
        return self.data[idx, 79:88]

    def get_features_and_responders(self, idx=None):
        if idx is None:
            return self.data[:, :88]
        return self.data[idx, :88]

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

    def get_temporal(self, idx=None):
        if idx is None:
            return self.data[:, 89:91]
        return self.data[idx, 89:91]
