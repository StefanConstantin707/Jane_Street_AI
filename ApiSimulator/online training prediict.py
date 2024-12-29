import polars as pl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from ApiSimulator.ApiSimulator import TestDataset
from Models.SimpleNN import SimpleNN

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
model.load_state_dict(torch.load("/savedModels/nn_model_weights_0.008_0.0108max.pth", weights_only=True))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


batch_number = 0
cache_index = 0
shuffled_indices = torch.randperm(968 * 4096)

predict_features = [f"feature_{i:02d}" for i in range(79)]
lags_features = [f"responder_{i}_lag_1" for i in range(9)]
cache_features = ["date_id", "time_id", "symbol_id", "weight"] + [f"feature_{i:02d}" for i in range(79)]

cache_size = 968 * batch_size
day_cache = None

path = "./JaneStreetRealTimeMarketDataForecasting"
all_train_data = pl.scan_parquet(f"{path}/train.parquet")
start_date = 1400
end_date = 1699

train_data = all_train_data.filter((pl.col("date_id") >= start_date) & (pl.col("date_id") < end_date)).drop("partition_id").collect()
cache = train_data.sample(n=(cache_size), with_replacement=False).cast(pl.Float32)

torch_cache = torch.tensor(cache.to_numpy(), dtype=torch.float32, device=device)

test_dataframe = TestDataset("../", start_date=1577)

test_dataframe.send_data()


def weighted_mse(y_pred, y_true, weights):
    unweighted_loss = loss_fct(y_pred, y_true)
    weighted_loss = weights * unweighted_loss
    return weighted_loss.mean()

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    """Make a prediction."""
    global day_cache, torch_cache, batch_number, shuffled_indices, cache_index

    if lags is not None and lags.height != 0:
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

    day_cache[cache_index:cache_index + test.height, :83] = test_tensor[:]
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