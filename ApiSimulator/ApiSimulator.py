import time

import torch
import polars as pl
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch import optim

from torch.utils.data import Dataset

from Models.MiniAttention import FullDualTransformer
from Models.SimpleNN import SimpleNN
from TrainClass import GeneralEval
from Utillity.LossFunctions import r2_score, r2_loss, r2_score_batch, r2_score_numpy
from Utillity.means_and_stds import means, stds
from dataHandlers.PartialDataHandler import SingleRowRespDataset

lags_: pl.DataFrame | None = None
cache: pl.DataFrame | None = None
cache_add_columns = []
tensor_columns = []
old_feature_columns = []
old_features = None

numpy_cache = None
symbols = np.array([0], dtype=np.int8)

def predict_two_day(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global cache, nu_symbols
    if lags is not None and lags.height == 0:
        lags = None
        nu_symbols = test.n_unique("symbol_id")

    if lags is not None:
        # Get current day when the lags exists and the time_id is therefore time_id = 0
        current_day = cache.select(pl.last("date_id")).item()

        # Drop the cache from two days ago
        cache.filter(pl.col("date_id") == current_day - 1)

        # Add the new lags to the cache, replacing the values that before were 0
        cache.update(other=lags, on=identifier_columns, how="left")

        # Get the nu of symbols for the day so we can know the shape of our input tensor (ASSUMES A DAY HAS A CONSTANT NUMBER OF SYMBOL_IDs)
        nu_symbols = test.n_unique("symbol_id")

    # (Test assumption for each time_id)
    assert nu_symbols == test.n_unique("symbol_id")

    test_row_ids = test.select("row_id")
    test = test.drop("row_id")
    test = normalize_dataframe(test)
    test = test.with_columns((pl.lit(0.0, dtype=pl.Float32).alias(responders_feature) for responders_feature in responders_features))

    # (Only happens at the first timestamp when date_id=0, time_id=0 to initialize the cache with zeros (ASSUMES THE FIRST TIMESTEP DOESN'T COME WITH A LAGS DATAFRAME))
    if cache is None:
        responders_features = [f"responder_{i}" for i in range(9)]
        cache_columns = ["date_id", "time_id", "symbol_id"] + [f"feature_{i:02d}" for i in range(79)] + [
            f"responder_{i}" for i in range(9)]
        identifier_columns = ["date_id", "time_id", "symbol_id"]
        feature_columns = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{i}" for i in range(9)]
        cache_add_columns = ["date_id", "time_id", "symbol_id"] + [f"feature_{i:02d}" for i in range(79)]
        full_feature_columns = feature_columns + ["symbol_id"]

        cache = test.select(identifier_columns).with_columns(
            ( pl.lit(0.0, dtype=pl.Float32).alias(feature_column) for feature_column in feature_columns )
        )
        cache = cache.with_columns(pl.col("date_id") - 1)

    # Add the new timestep to the cache, we do not need to fill_null for the new DataFrame, as joining cache and test will result in null values for the responder features which we do not yet have
    cache = pl.concat([cache, test.select(cache_columns)], rechunk=True)
    # After updating the cache, we can now fill_null only once
    cache = cache.fill_null(0)

    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = torch.tensor(cache.select(feature_columns).to_numpy(), dtype=torch.float32, device=device).view(-1, nu_symbols, in_size).transpose(0, 1)

    # Adds padding to the X tensor to match the 2-day interval
    nu_rows_X = X.shape[1]
    padded_X = torch.zeros((X.shape[0], 968*2, in_size), dtype=X.dtype, device=device)
    padded_X[:, :nu_rows_X] = X

    with torch.no_grad():
        responder_predictions = model(padded_X)
    responder_six_predictions = responder_predictions[:, 6]

    predictions = test_row_ids.select(
        'row_id',
        pl.Series(responder_six_predictions.cpu().numpy()).alias('responder_6'),
    )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

def predict_last_time_id(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global cache, nu_symbols, cache_add_columns, tensor_columns, old_feature_columns, old_features

    symbols = pl.Series(test.select('symbol_id')).to_list()

    time_id = test.select("time_id").head(1).item()
    if time_id == 0 and cache is not None:
        old_features = cache.select(old_feature_columns)
        old_features = old_features.rename(lambda column_name: column_name[:10] + "old_" + column_name[10:])
        cache = None

    if cache is None:
        tensor_columns = [f"feature_old_{i:02d}" for i in range(79)] + [f"responder_old_{i}" for i in range(9)] + [f"feature_{i:02d}" for i in range(79)] + ["symbol_id"] + ["time_id"]
        cache_add_columns = [f"feature_old_{i:02d}" for i in range(79)] + [f"responder_old_{i}" for i in range(9)] + [f"feature_{i:02d}" for i in range(79)] + ["time_id"]
        old_feature_columns = [f"feature_{i:02d}" for i in range(79)] + ["symbol_id"]
        cache = pl.from_numpy(np.arange(0, stop=50, dtype=np.int8), schema=["symbol_id"])
        cache = cache.with_columns(
            ( pl.lit(0.0, dtype=pl.Float32).alias(past_feature_and_responses_column) for past_feature_and_responses_column in cache_add_columns )
        )

        cache = cache.select(tensor_columns)

        if old_features is not None:
            cache = cache.update(other=old_features, how="left", on="symbol_id")
            old_features = None

    if lags is not None and lags.height == 0:
        lags = None

    if lags is not None:
        # Filter the lags to only collect the rows at the last timestep of yesterday (time_id == 967)
        last_time_id_of_yesterday = lags.select(pl.last("time_id")).item()
        lags = lags.filter(pl.col("time_id") == last_time_id_of_yesterday)
        lags = lags.rename(lambda column_name: column_name[:10] + "old_" + column_name[10:] )

        # Add the new lags to the cache, replacing the values that before were 0
        cache = cache.update(other=lags, on="symbol_id", how="left")

    test_row_ids = test.select("row_id")
    test = test.drop("row_id")
    test = normalize_dataframe(test)

    cache = cache.update(test, "symbol_id", "left")

    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = torch.tensor(cache.filter(pl.col("symbol_id").is_in(symbols)).to_numpy(), dtype=torch.float32, device=device)

    with torch.no_grad():
        responder_predictions = model(X)
    responder_six_predictions = responder_predictions[:, 6]

    predictions = test_row_ids.select(
        'row_id',
        pl.Series(responder_six_predictions.cpu().numpy()).alias('responder_6'),
    )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

def predict_last_time_id_numpy(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global numpy_cache, symbols

    if numpy_cache is None:
        numpy_cache = np.zeros((50, 169))
        numpy_cache[:, 167] = np.arange(50)

    time_id = test.select("time_id").head(1).item()
    if time_id == 0:
        symbols = pl.Series(test.select('symbol_id')).to_numpy()

    if lags is not None and lags.height != 0:
        last_time_id = lags.select(pl.last("time_id")).item()
        lags = lags.filter(pl.col("time_id") == last_time_id)
        lags_symbols = lags.select('symbol_id').to_numpy().reshape(-1)
        numpy_cache[lags_symbols, 79:88] = lags.select([f"responder_{i}" for i in range(9)]).to_numpy()

    test_row_ids = test.select("row_id")
    test = test.drop("row_id")
    test = normalize_dataframe(test)
    test = test.fill_null(0)

    numpy_cache[symbols, 88:167] = test.select([f"feature_{i:02d}" for i in range(79)]).to_numpy()
    numpy_cache[:, 168] = np.ones((50,)) * time_id

    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = torch.tensor(numpy_cache[symbols], dtype=torch.float32, device=device)

    with torch.no_grad():
        responder_predictions = model(X)
    responder_six_predictions = responder_predictions[:, 6]

    predictions = test_row_ids.select(
        'row_id',
        pl.Series(responder_six_predictions.cpu().numpy()).alias('responder_6'),
    )

    if time_id == 967:
        numpy_cache[symbols, :79] = test.select([f"feature_{i:02d}" for i in range(79)]).to_numpy().copy()

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

def predict_last_time_id_v2(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global cache, symbols, numpy_cache

    test_row_ids = test.select("row_id")
    test = test.drop("row_id").fill_null(0)
    # test = normalize_dataframe(test)

    if cache is None:
        cache_columns = [f"feature_{i:02d}" for i in range(79)] + [f"responder_{i}" for i in range(9)]
        cache = pl.DataFrame({
            "symbol_id": np.arange(50, dtype=np.int8)
        })
        cache = cache.with_columns(
            (pl.lit(0.0, dtype=pl.Float32).alias(cache_column) for
             cache_column in cache_columns)
        )
    cache = cache.update(other=test, on="symbol_id", how="left")

    time_id = test.select("time_id").head(1).item()
    if time_id == 0:
        symbols = pl.Series(test.select('symbol_id')).to_numpy()
        nu_symbols = symbols.shape[0]

        numpy_cache = np.zeros((nu_symbols, 167), dtype=np.float32)
        if lags is not None and lags.height != 0:
            last_time_id = lags.select(pl.last("time_id")).item()
            lags = lags.filter(pl.col("time_id") == last_time_id)
            cache = cache.update(other=lags, on="symbol_id", how="left")

            numpy_cache[symbols, :88] = cache.filter(pl.col("symbol_id").is_in(symbols)).drop("symbol_id").to_numpy()

    numpy_cache[symbols, 88:] = test.drop("date_id", "time_id", "symbol_id").to_numpy()



    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = torch.tensor(numpy_cache[symbols], dtype=torch.float32, device=device)

    with torch.no_grad():
        responder_predictions = model(X)
    responder_six_predictions = responder_predictions[:, 6]

    predictions = test_row_ids.select(
        'row_id',
        pl.Series(responder_six_predictions.cpu().numpy()).alias('responder_6'),
    )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

def predict_iteratively(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global symbols, numpy_cache

    test_row_ids = test.select("row_id")
    test = test.drop("row_id").fill_null(0)

    if numpy_cache is None:
        numpy_cache = np.zeros((50, 88))
        symbols = test.select('symbol_id').to_numpy().reshape(-1)

    if lags is not None and lags.height != 0:
        symbols = test.select('symbol_id').to_numpy().reshape(-1)
        last_time_id = lags.select(pl.last("time_id")).item()
        lags = lags.filter(pl.col("time_id") == last_time_id)
        lags_symbols = lags.select('symbol_id').to_numpy().reshape(-1)

        numpy_cache[lags_symbols, 79:] = lags.filter(pl.col("symbol_id").is_in(lags_symbols)).drop("date_id", "time_id", "symbol_id").to_numpy()

    numpy_cache[symbols, :79] = test.drop("date_id", "time_id", "symbol_id", "weight", "is_scored").to_numpy()

    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = torch.tensor(numpy_cache[symbols], dtype=torch.float32, device=device)

    with torch.no_grad():
        responder_predictions = model(X)
    responder_six_predictions = responder_predictions[:, 6]

    numpy_cache[symbols, 79:] = responder_predictions.cpu().numpy()

    predictions = test_row_ids.select(
        'row_id',
        pl.Series(responder_six_predictions.cpu().numpy()).alias('responder_6'),
    )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame)
    # with columns 'row_id', 'responder_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

class TestDataset(Dataset):
    def __init__(self, path, start_date):

        self.responders_features = [f"responder_{i}" for i in range(9)]
        self.feature_features = [f"feature_{i:02d}" for i in range(79)]

        test_data, self.nu_rows = self._load_partial_data(path, start_date)
        self.valid_test_data_input = self.build_test_data_input(test_data)
        self.lags_data = self.build_lags(test_data)
        self.date_time_tensor = self.build_date_time_vector(test_data)
        self.targets_and_weights = self.build_targets_and_weights(test_data)

    def _load_partial_data(self, jane_street_real_time_market_data_forecasting_path, start_date=1577) -> [pl.LazyFrame, int]:
        all_train_data = pl.scan_parquet(
            f"{jane_street_real_time_market_data_forecasting_path}JaneStreetRealTimeMarketDataForecasting/train.parquet")

        test_data = all_train_data.filter((pl.col("date_id") >= start_date))
        nu_rows = test_data.select(pl.len()).collect().item()

        return test_data, nu_rows

    def build_test_data_input(self, test_data: pl.LazyFrame) -> pl.DataFrame:
        test_data_with_extra_columns = test_data.with_columns(pl.Series(range(self.nu_rows)).alias("row_id"), pl.lit(True).alias("is_scored"))

        preV_features = ["row_id", "date_id", "time_id", "symbol_id", "weight", "is_scored"]
        required_features = preV_features + self.feature_features

        valid_test_data_input = test_data_with_extra_columns.select(required_features).collect()

        return valid_test_data_input

    def build_lags(self, test_data: pl.LazyFrame) -> pl.DataFrame:
        lags_features = ["date_id", "time_id", "symbol_id"] + self.responders_features
        lags_data = test_data.select(lags_features).collect()

        return lags_data

    def build_date_time_vector(self, test_data: pl.LazyFrame) -> torch.Tensor:
        date_time_features = ["date_id", "time_id"]
        date_time_frame = test_data.select(date_time_features).unique(keep="first", maintain_order=True)

        data_time_tensor = torch.tensor(date_time_frame.collect().to_numpy())

        return data_time_tensor

    def build_targets_and_weights(self, test_data: pl.LazyFrame) -> pl.DataFrame:
        targets_and_weights = test_data.select("date_id", "responder_6", "weight").collect()
        return targets_and_weights

    def send_data(self):
        all_submission_dataframe = []
        all_inference_times = []
        timeout = 60

        for timestep in range(self.date_time_tensor.shape[0]):
            date, c_time = self.date_time_tensor[timestep]

            test = self.valid_test_data_input.filter((pl.col("date_id") == date) & (pl.col("time_id") == c_time))
            if c_time == 0:
                lag = self.lags_data.filter((pl.col("date_id") == (date - 1)))
            else:
                lag = None

            start_time = time.time()

            submission_dataframe = predict_iteratively(test, lag)

            all_submission_dataframe.append(submission_dataframe)

            end_time = time.time()

            diff = end_time - start_time

            all_inference_times.append(diff)

            if diff > timeout:
                print(f"{date=},{c_time=}{diff=}")

            if (timestep + 1) % 968 == 0:
                plt.plot(all_inference_times)
                plt.show()

                targets_and_weights = self.targets_and_weights.filter(pl.col("date_id") == date)
                targets = targets_and_weights.select("responder_6").to_numpy().reshape(-1)
                weights = targets_and_weights.select("weight").to_numpy().reshape(-1)

                all_submission_dataframe = [pl.concat(all_submission_dataframe)]
                pred = all_submission_dataframe[0].select("responder_6").tail(weights.shape[0]).to_numpy().reshape(-1)

                score = r2_score_numpy(pred, targets, weights)
                print(score)

                self.plot_data_by_time_id(pred, targets, weights)


        all_submission_dataframe = pl.concat(all_submission_dataframe)

        return all_submission_dataframe

    def plot_data_by_time_id(self, pred: np.array, targets: np.array, weights: np.array):
        pred = pred.reshape((968, -1))
        targets = targets.reshape((968, -1))
        weights = weights.reshape((968, -1))

        r2_scores_per_time_id = r2_score_batch(pred, targets, weights)

        plt.plot(r2_scores_per_time_id)
        plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
in_size = 88
out_size = 9
batch_size = 4096
mini_epoch_size = 100
noise = 0.01

model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
model.load_state_dict(torch.load("..\\savedModels\\simple_nn.pt", weights_only=True))
model.eval()

# evalDataset = SingleRowRespDataset(data_type='eval', path="../", start_date=1580, end_date=1699, out_size=out_size,
#                                    in_size=in_size, device=device)
# eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=0,
#                                           pin_memory=True)
# evalClass = GeneralEval(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)
# eval_loss, eval_r2_score, eval_mse, eval_r2_score_responders = evalClass.step_epoch()
#
# # Print the general statistics
# print(
#     f'Eval statistics: Epoch: 1 Loss: {eval_loss:.4f}, R2 score: {eval_r2_score:.4f}, MSE: {eval_mse:.4f}')
#
# # Print the R² scores for each responder
# for i, eval_r2_score_responder in enumerate(eval_r2_score_responders):
#     print(f'R2 score responder {i}: {eval_r2_score_responder:.4f}')


test_dataframe = TestDataset("../", start_date=1577)

test_dataframe.send_data()


