# -*- coding:utf-8 -*-
import pandas as pd
import sys
import glob
import os

problem = pd.read_csv("src/SolarPV_Elec_Problem.csv", index_col=0, parse_dates=True)
weather = pd.read_csv("src/AVG_weather_15min.csv", index_col=1, parse_dates=True)


def get_average_table(weather_frame: pd.DataFrame, period_min: int):
    # 풍향(deg),풍속(m/s),현지기압(hPa),해면기압(hPa),습도(%),일사(MJ/m^2),일조(Sec)
    # location, datetime, temperature, percipitation, wind_degree, wind_speed, air_pressure, sea_level_pressure, humidity, solar_radiation, solr_intensity
    weather_frame.columns = ["datetime", "temperature", "percipitation", "wind_degree", "wind_speed", "air_pressure",
                             "sea_level_pressure", "humidity", "solar_radiation", "solar_intensity"]
    temp = pd.DataFrame()
    # temp['datetime'] = weather_frame['datetime'].resample(f'{str(period_min)}T')
    temp['temperature'] = weather_frame['temperature'].resample(f'{str(period_min)}T').mean()
    temp['percipitation'] = weather_frame['percipitation'].resample(f'{str(period_min)}T').mean().diff()
    temp['wind_degree'] = weather_frame['wind_degree'].resample(f'{str(period_min)}T').mean()
    temp['wind_speed'] = weather_frame['wind_speed'].resample(f'{str(period_min)}T').mean()
    temp['air_pressure'] = weather_frame['air_pressure'].resample(f'{str(period_min)}T').mean()
    temp['sea_level_pressure'] = weather_frame['sea_level_pressure'].resample(f'{str(period_min)}T').mean()
    temp['humidity'] = weather_frame['humidity'].resample(f'{str(period_min)}T').mean()
    temp['solar_radiation'] = weather_frame['solar_radiation'].resample(f'{str(period_min)}T').mean().diff()
    temp['solar_intensity'] = weather_frame['solar_intensity'].resample(f'{str(period_min)}T').mean().diff()
    print(temp.head(5))
    return temp


def df_to_csv(df: pd.DataFrame, filename: str):
    df.to_csv(f'out/{filename}', sep=',', na_rep='NaN')


def make_avg_file():
    weather_file_list = glob.glob("src/weather/*.csv")
    for f in weather_file_list:
        print(f)
        target = pd.read_csv(f, index_col=1, parse_dates=True)
        k = get_average_table(target, 15)
        df_to_csv(k, f"weather_avg_15min/diff_{os.path.basename(f)}")


def concat_csv_vertical(l, index_col: int):
    return pd.concat((pd.read_csv(f, index_col=index_col) for f in l))


def concat_dataset(a,b):
    return pd.concat([a, b], axis=1)


# make_avg_file()
processed_file_list = glob.glob("out/weather_avg_15min/diff*.csv")
w = concat_csv_vertical(processed_file_list, 0)
size = len(w.index)
print(size)
# df_to_csv(concat_csv_vertical(processed_file_list, 0), "dataset1.csv")
# print(concat_csv_vertical(processed_file_list, 0))
# print(pd.concat([w, problem], axis=1).head(5))
pd.concat([w, problem], axis=1).to_csv("dataset_dff.csv")