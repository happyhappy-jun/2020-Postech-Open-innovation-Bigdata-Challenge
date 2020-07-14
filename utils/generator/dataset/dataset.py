import pandas as pd
import glob
import os

problem = pd.read_csv("src/SolarPV_Elec_Problem.csv", index_col=0, parse_dates=True)
weather = pd.read_csv("src/AVG_weather_15min.csv", index_col=1, parse_dates=True)



def get_average_table(weather_frame: pd.DataFrame, period_min: int):
    return weather_frame.resample(f'{str(period_min)}T').mean()


def df_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(f'out/{filename}', sep=',', na_rep='NaN')


def make_avg_file():
    weather_file_list = glob.glob("src/weather/*.csv")
    for f in weather_file_list:
        print(f)
        target = pd.read_csv(f, index_col=1, parse_dates=True)
        k = get_average_table(target, 15)
        print(k)
        df_to_csv(k, f"weather_avg_15min/AVG_{os.path.basename(f)}")


def concat_csv_vertical(l, index_col: int, filename: str):
    return pd.concat((pd.read_csv(f, index_col=index_col) for f in l))


def concat_dataset():
    return pd.concat([problem, weather], axis=1)


df_to_csv(concat_dataset(), "dataset.csv")
