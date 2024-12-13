import torch
import polars as pl
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

def load_all_data(path):
    jane_street_real_time_market_data_forecasting_path = f'{path}/jane-street-real-time-market-data-forecasting'

    all_train_data = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")

    return all_train_data

