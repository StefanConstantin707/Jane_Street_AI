import torch
import numpy as np
import polars as pl


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

def fill_missing_rows(df, max_date_id=1700, max_time_id=968):
    # Explicitly cast date_id and time_id to Int16
    df = df.with_columns([
        pl.col("date_id").cast(pl.Int16),
        pl.col("time_id").cast(pl.Int16)
    ])

    # Create a complete grid of date_id and time_id
    complete_grid = pl.DataFrame({
        "date_id": np.arange(max_date_id).repeat(max_time_id),
        "time_id": np.tile(np.arange(max_time_id), max_date_id)
    }).with_columns([
        pl.col("date_id").cast(pl.Int16),
        pl.col("time_id").cast(pl.Int16),
    ])

    # Left join the original dataframe with the complete grid
    filled_df = complete_grid.join(
        df,
        on=["date_id", "time_id"],
        how="left"
    )

    # Fill missing values with 0 for non-key columns
    non_key_columns = [
        col for col in df.columns
        if col not in ["date_id", "time_id"]
    ]

    filled_df = filled_df.with_columns(
        [pl.col(col).fill_null(0) for col in non_key_columns]
    )

    return filled_df

def fill_null_values(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.fill_null(0)
