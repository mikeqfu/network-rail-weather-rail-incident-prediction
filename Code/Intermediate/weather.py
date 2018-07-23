""" Gridded weather observations """

import itertools
import os
import zipfile

import natsort
import pandas as pd

from converters import osgb36_to_wgs84
from utils import cdd, load_pickle, save_pickle


# Change directory to "Weather"
def cdd_weather(*directories):
    path = cdd("Weather")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Find coordinates for each corner of the weather observation grid
def find_square_corners(centre_point, side_length=5000, rotation=None):
    """

    :param centre_point: (easting, northing)
    :param side_length:
    :param rotation: angle
    :return:

    Easting and northing coordinates are commonly measured in metres from the axes of some horizontal datum.
    However, other units (e.g. survey feet) are also used.

    """
    assert (isinstance(centre_point, tuple) or isinstance(centre_point, list)) and len(centre_point) == 2

    x, y = centre_point

    if rotation:
        sin_theta, cos_theta = pd.np.sin(rotation), pd.np.cos(rotation)
        lower_left = (x - 1 / 2 * side_length * sin_theta, y - 1 / 2 * side_length * cos_theta)
        upper_left = (x - 1 / 2 * side_length * cos_theta, y + 1 / 2 * side_length * sin_theta)
        upper_right = (x + 1 / 2 * side_length * sin_theta, y + 1 / 2 * side_length * cos_theta)
        lower_right = (x + 1 / 2 * side_length * cos_theta, y - 1 / 2 * side_length * sin_theta)
    else:
        lower_left = (x - 1/2*side_length, y - 1/2*side_length)
        upper_left = (x - 1/2*side_length, y + 1/2*side_length)
        upper_right = (x + 1/2*side_length, y + 1/2*side_length)
        lower_right = (x + 1/2*side_length, y - 1/2*side_length)
    return lower_left, upper_left, upper_right, lower_right


# Read gridded weather observations from the raw zipped file
def read_daily_gridded_weather_obs(filename, col_name='variable_name', start_date='2006-01-01'):
    # Centres
    cartesian_centres_temp = pd.read_csv(filename, header=None, index_col=0, nrows=2)
    cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

    # Temperature observations
    timeseries_data = pd.read_csv(filename, header=None, skiprows=[0, 1], parse_dates=[0], dayfirst=True)
    timeseries_data[0] = timeseries_data[0].map(lambda x: x.date())
    if start_date is not None:
        mask = (timeseries_data[0] >= pd.to_datetime(start_date).date())
        timeseries_data = timeseries_data.loc[mask]
    timeseries_data.set_index(0, inplace=True)

    # Reshape the dataframe
    idx = pd.MultiIndex.from_product([cartesian_centres, timeseries_data.index.tolist()], names=['Centre', 'Date'])
    data = pd.DataFrame(timeseries_data.T.values.flatten(), index=idx, columns=[col_name])
    data.reset_index(inplace=True)

    # Add levels of Grid corners (and LongLat centres)
    grid = [find_square_corners(centre, 5000, rotation=None) for centre in cartesian_centres]
    data['Grid'] = list(
        itertools.chain.from_iterable(itertools.repeat(x, len(timeseries_data)) for x in grid))

    long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]
    data['LongLat'] = list(
        itertools.chain.from_iterable(itertools.repeat(x, len(timeseries_data)) for x in long_lat))

    data.set_index(['Grid', 'Centre', 'LongLat', 'Date'], inplace=True)

    return data


# Get gridded weather observations
def get_daily_gridded_weather_obs(filename, col_name='variable_name', start_date='2006-01-01', update=False):
    """

    :param filename:
    :param col_name:
    :param start_date:
    :param update:
    :return:
    """
    path_to_file = cdd_weather("UKCP gridded data", filename + "_{}".format(start_date.replace("-", "")) + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        gridded_obs = load_pickle(path_to_file)
    else:
        path_to_zip_file = cdd_weather("UKCP gridded data", filename.replace('_', ' ').capitalize() + ".zip")

        with zipfile.ZipFile(path_to_zip_file, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            temp_dat = [read_daily_gridded_weather_obs(zf.open(f), col_name, start_date) for f in filename_list]

        gridded_obs = pd.concat(temp_dat, axis=0, sort=False)

        save_pickle(gridded_obs, path_to_file)

    return gridded_obs


# Combine weather observations of different variables
def integrate_daily_gridded_weather_obs(update=False, start_date='2006-01-01'):
    filename = "daily_gridded_weather_obs_{}".format(start_date.replace('-', ''))
    path_to_file = cdd_weather("UKCP gridded data", filename + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        daily_gridded_weather_obs = load_pickle(path_to_file)
    else:
        try:
            daily_max_temp = \
                get_daily_gridded_weather_obs("daily_maximum_temperature", 'Maximum_Temperature', start_date)
            daily_min_temp = \
                get_daily_gridded_weather_obs("daily_minimum_temperature", 'Minimum_Temperature', start_date)
            daily_rainfall = \
                get_daily_gridded_weather_obs("daily_rainfall", 'Rainfall', start_date)

            daily_gridded_weather_obs = pd.concat([daily_max_temp, daily_min_temp, daily_rainfall], axis=1)

            save_pickle(daily_gridded_weather_obs, path_to_file)

        except Exception as e:
            print("Failed to get integrated daily gridded weather observations due to {}.".format(e))
            daily_gridded_weather_obs = None

    return daily_gridded_weather_obs


# Met station locations
def get_met_station_locations(update=False):
    spreadsheet_filename = "Met station locations.xlsx"
    path_to_spreadsheet = cdd_weather(spreadsheet_filename)
    path_to_pickle = path_to_spreadsheet.replace(".xlsx", ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        met_stn_loc = load_pickle(path_to_pickle)
    else:
        try:
            met_stn_loc = pd.read_excel(path_to_spreadsheet)
            met_stn_loc.Name = met_stn_loc.Name.str.strip()
            save_pickle(met_stn_loc, path_to_pickle)
        except Exception as e:
            print("Failed to get {} due to {}".format(spreadsheet_filename, e))
            met_stn_loc = None
    return met_stn_loc
