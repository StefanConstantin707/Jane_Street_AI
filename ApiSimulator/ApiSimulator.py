import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import polars as pl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from Models.SimpleNN import SimpleNN
from torch.utils.data import Dataset
from Utillity.LossFunctions import r2_score_batch, r2_score_numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
in_size = 79
hidden_size = 256
out_size = 9
num_layers = 2
batch_size = 4096

lr = 3e-5
weight_decay = 1e-4

dropout = 0.3
noise = 1.0

loss_fct = nn.MSELoss(reduction='none')

model = SimpleNN(input_size=in_size, hidden_dim=hidden_size, output_size=out_size, num_layers=num_layers, dropout_prob=dropout, noise=noise).to(device)
model.load_state_dict(torch.load("../savedModels/nn_model_weights_0.008_0.0108max.pth", weights_only=True, map_location=torch.device('cpu')))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

batch_number = 0
cache_index = 0
shuffled_indices = torch.randperm(968 * 4096)

predict_features = [f"feature_{i:02d}" for i in range(79)]
lags_features = [f"responder_{i}_lag_1" for i in range(9)]
lags_features = [f"responder_{i}" for i in range(9)]
cache_features = ["date_id", "time_id", "symbol_id", "weight"] + [f"feature_{i:02d}" for i in range(79)]

cache_size = 968 * batch_size
day_cache = None

path = "../JaneStreetRealTimeMarketDataForecasting"
all_train_data = pl.scan_parquet(f"{path}/train.parquet")
start_date = 1400
end_date = 1577

train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date)).drop("partition_id").collect()
cache = train_data.sample(n=(cache_size), with_replacement=False).cast(pl.Float32).fill_null(0)

torch_cache = torch.tensor(cache.to_numpy(), dtype=torch.float32, device=device)

def weighted_mse(y_pred, y_true, weights):
    unweighted_loss = loss_fct(y_pred, y_true)
    weighted_loss = weights * unweighted_loss
    return weighted_loss.mean()

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global day_cache, torch_cache, batch_number, shuffled_indices, cache_index

    if lags is not None:
        new_rows = lags.height
        added_rows = new_rows // 2

        if day_cache is not None:
            day_cache[:, 83:92] = torch.tensor(lags.select(lags_features).to_numpy(), dtype=torch.float32,
                                               device=device)

            selected_indices = torch.randint(0, new_rows, (added_rows,), dtype=torch.int32, device=device)
            day_cache = day_cache[selected_indices]

            torch_cache = torch.cat((torch_cache[added_rows:], day_cache), dim=0)

        day_cache = torch.zeros((test.height * 968, 92), dtype=torch.float32, device=device)

        batch_number = 0
        cache_index = 0
        shuffled_indices = torch.randperm(cache_size)

    start_idx = batch_number * batch_size
    batch_number += 1
    end_idx = batch_number * batch_size
    batch_indexes = shuffled_indices[start_idx:end_idx]

    x_batch = torch_cache[batch_indexes, 4:83]
    y_batch = torch_cache[batch_indexes, 83:92]
    weights_batch = torch_cache[batch_indexes, 3].unsqueeze(-1)

    outputs = model(x_batch)
    loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    test_row_ids = test.select("row_id")
    test = test.fill_null(0)

    test_tensor = torch.tensor(test.select(cache_features).to_numpy(), device=device, dtype=torch.float32)

    day_cache[cache_index:cache_index + test.height, :83] = test_tensor
    cache_index += test.height

    # Creates the input tensor for the model (It doesn't have values for the remaining of the day yet)
    X = test_tensor[:, 4:]

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

lags_: pl.DataFrame | None = None
cache: pl.DataFrame | None = None

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

            submission_dataframe = predict(test, lag)

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

test_dataframe = TestDataset("../", start_date=1577)
test_dataframe.send_data()


