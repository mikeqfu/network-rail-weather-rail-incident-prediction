import os
import pandas as pd
from utils import cdd, load_pickle, save_pickle
from converters import osgb36_to_wgs84


# Change directory to "Weather"
def cdd_weather(*directories):
    path = cdd("Weather")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Gridded observations of daily maximum temperature
def read_daily_timeseries_maximum_temperature(update=False):
    filename = "ukcp09_gridded-land-obs-daily_timeseries_maximum-temperature_000000E_500000N_19600101-20161231.csv"
    path_to_file = cdd_weather(os.path.splitext(filename)[0] + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        temperature_data = load_pickle(path_to_file)
    else:
        try:
            cartesian_coords_temp = pd.read_csv(cdd_weather(filename), header=None, index_col=0, nrows=2)
            cartesian_coordinates = [tuple(x) for x in cartesian_coords_temp.T.values]
            long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_coordinates]

            max_temperatures = pd.read_csv(cdd_weather(filename), header=None, skiprows=[0, 1],
                                           parse_dates=[0], dayfirst=True)
            max_temperatures[0] = max_temperatures[0].map(lambda x: x.date())
            max_temperatures.set_index(0, inplace=True)

            idx = pd.MultiIndex.from_product([long_lat, max_temperatures.index.tolist()], names=['LongLat', 'Date'])
            temperature_data = pd.DataFrame(data=max_temperatures.T.values.flatten(), index=idx,
                                            columns=['Maximum_Temperature'])

            save_pickle(temperature_data, path_to_file)

        except Exception as e:
            temperature_data = None

            print("Failed to get temperate data, {}, due to {}".format(filename, e))
            print("Data directory is located at {}.".format(os.path.dirname(path_to_file)))

    return temperature_data


#
def append_temperature_variables():
    dmax_temp = read_daily_timeseries_maximum_temperature()



    return data
