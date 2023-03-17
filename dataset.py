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
    df["vgc"] = 0
    return df


def add_derivatives(df, colname, output_col_name, rolling_windows=5, shift=0):
    df[output_col_name] = df[colname].shift(
        shift) - df[colname].shift(rolling_windows+shift)

    return df


def get_charge_cycles(df_optim):
    return (df_optim.SOC - df_optim.SOC.shift(1)).abs().sum()/200


def get_action_switches(df_optim):
    return ((df_optim.SOC - df_optim.SOC.shift(1)) != (df_optim.SOC.shift(1) - df_optim.SOC.shift(2))).sum()
