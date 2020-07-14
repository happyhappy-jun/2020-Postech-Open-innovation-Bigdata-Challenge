import pandas as pd
import glob
import os

problem = pd.read_csv("src/target/SolarPV_Elec_Problem.csv")

weather_file_list = glob.glob("src/weather/*.csv")


def get_average_table(weather_frame: pd.DataFrame, period_min: int):
    return weather_frame.resample(f'{str(period_min)}T').mean()


def df_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(f'out/{filename}', sep=',', na_rep='NaN')


def make_avg_file():
    for f in weather_file_list:
        print(f)
        target = pd.read_csv(f, index_col=1, parse_dates=True)
        k = get_average_table(target, 15)
        print(k)
        df_to_csv(k, f"AVG_{os.path.basename(f)}")

make_avg_file()