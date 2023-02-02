import pandas as pd


def get_dataset(
    path="data/european_wholesale_electricity_price_data_hourly.csv",
    year="2020",
    country="Germany",
    usecols=["Datetime (Local)", "Price (EUR/MWhe)", "Country"],
    starttime=None,
    endtime=None,
):
    df = pd.read_csv(path, usecols=usecols)
    df = df[df.Country == country]
    if year:
        df = df[df["Datetime (Local)"] > f"{year}-01-01 00:00:00"]
    df.drop(["Country"], axis=1, inplace=True)
    df.rename(
        {"Datetime (Local)": "timestamp", "Price (EUR/MWhe)": "price"},
        axis=1,
        inplace=True,
        errors="raise",
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%Y-%m-%dT%H:%M:%S")

    if starttime:
        df = df[df["timestamp"] >= starttime]
    if endtime:
        df = df[df["timestamp"] <= endtime]
    df.reset_index(drop=True, inplace=True)
    return df


def add_derivatives(df, colname="price", nders=2, rolling_windows=None, shift=0):
    rolling_windows = rolling_windows if rolling_windows is not None else [
        5, 5]
    df[f"{colname}_der1"] = df[colname].shift(
        shift) - df[colname].shift(rolling_windows[0]+shift)
    for i in range(2, nders + 1):
        df[f"{colname}_der{i}"] = df[f"{colname}_der{i-1}"].shift(
            shift) - df[f"{colname}_der{i-1}"].shift(rolling_windows[i-1]+shift)
    return df
