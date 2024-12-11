import torch
import polars as pl
from torch.utils.data import TensorDataset, DataLoader

from means_and_stds import means, stds

def normalize_dataframe(df: pl.DataFrame, means: dict, stds: dict) -> pl.DataFrame:
    # We normalize the polars dataframe using the provided means and standard deviations
    normalize_exprs = []

    for col in df.columns:
        if col in means and col in stds: #only normalize columns present in the means and std
            if stds[col] != 0: #avoid division by 0
                #Normalize the column and alias it with the same name
                normalize_exprs.append(
                    ((pl.col(col) - means[col]) / stds[col]).alias(col)
                )
            else:
                normalize_exprs.append(pl.col(col) - means[col]).alias(col)
        else:
            normalize_exprs.append(pl.col(col) / pl.col(col).max().alias(col))


    normalized_df = df.select(normalize_exprs) #Dataframe with normalized expressions
    return normalized_df

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

def create_dataloaders(train_X, train_Y, train_weights, val_X, val_Y, val_weights, test_X, test_Y, test_weights, batch_size, shuffle):
    train_dataset = TensorDataset(train_X, train_Y, train_weights)
    val_dataset = TensorDataset(val_X, val_Y, val_weights)
    test_dataset = TensorDataset(test_X, test_Y, test_weights)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
