import numpy as np
import polars as pl

from dataHandlers.FullDataHandler import load_all_data


def main():
    train_data = load_all_data('../')
    time_and_symbol_names = ["date_id", "time_id", "symbol_id"]
    nu_dates = train_data.select(pl.col("date_id").unique()).collect().to_numpy().max() + 1
    nu_time_ids = train_data.select(pl.col("time_id").unique()).collect().to_numpy().max() + 1
    nu_symbols = train_data.select(pl.col("symbol_id").unique()).collect().to_numpy().max() + 1

    print(f'nu_dates: {nu_dates}, nu_time_ids: {nu_time_ids}, nu_symbols: {nu_symbols}')

    time_and_symbol_data = train_data.select(time_and_symbol_names).collect().to_numpy()

    nu_rows, _ = time_and_symbol_data.shape

    prev_time_id_mapping = np.zeros((nu_symbols, nu_dates, nu_time_ids), dtype=np.int32)

    prev_symbol_idx = np.ones((nu_symbols,), dtype=np.int32) * (-1)

    for row in range(nu_rows):
        date = time_and_symbol_data[row, 0]
        time = time_and_symbol_data[row, 1]
        symbol = time_and_symbol_data[row, 2]

        prev_time_id_mapping[symbol, date, time] = prev_symbol_idx[symbol]
        prev_symbol_idx[symbol] = row

if __name__ == '__main__':
    main()