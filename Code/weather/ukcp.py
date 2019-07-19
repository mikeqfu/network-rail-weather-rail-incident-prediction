""" UKCP gridded weather observations: maximum temperature, minimum temperature and rainfall """

import os
import zipfile

import natsort
import pandas as pd
import shapely.geometry
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

from weather.utils import cdd_weather, create_grid

pd_preferences()


# Observation grids --------------------------------------------------------------------------------------------------
def parse_observation_grids(filename: str):
    """
    :param filename: [str] e.g. filename="ukcp09_gridded-land-obs-daily_timeseries_maximum
                                            -temperature_000000E_450000N_19600101-20161231.csv"
    :return:
    """
    cartesian_centres_temp = pd.read_csv(filename, header=None, index_col=0, nrows=2)
    cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

    grid = [create_grid(centre, 5000, rotation=None) for centre in cartesian_centres]

    long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]

    obs_grids = pd.DataFrame({'Centroid': cartesian_centres,
                              'Centroid_XY': [shapely.geometry.Point(x) for x in cartesian_centres],
                              'Centroid_LongLat': [shapely.geometry.Point(x) for x in long_lat],
                              'Grid': [shapely.geometry.Polygon(x) for x in grid]})
    return obs_grids


def prep_observation_grids(zip_filename: str):
    """
    :param zip_filename: e.g. zip_filename="daily-maximum-temperature.zip"
    """
    path_to_zip = cdd_weather("UKCP", zip_filename)
    try:
        with zipfile.ZipFile(path_to_zip, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            obs_grids = [parse_observation_grids(zf.open(f)) for f in filename_list]
        zf.close()

        observation_grids = pd.concat(obs_grids, ignore_index=True)

        # Add a pseudo id for each observation grid
        observation_grids.sort_values('Centroid', inplace=True)
        observation_grids.index = pd.Index(range(len(observation_grids)), name='Pseudo_Grid_ID')

        path_to_pickle = cdd_weather("UKCP", cdd_weather("UKCP", "observation-grids.pickle"))
        save_pickle(observation_grids, path_to_pickle)

    except Exception as e:
        print("Failed to get \"Observation Grids\". {}".format(e))


def fetch_observation_grids(zip_filename="daily-rainfall.zip", update=False):
    """
    :param zip_filename: [str]
    :param update: [bool]
    :return: [pd.DataFrame]
    """
    path_to_pickle = cdd_weather("UKCP", "observation-grids.pickle")

    if not os.path.isfile(path_to_pickle) or update:
        prep_observation_grids(zip_filename)
    try:
        observation_grids = load_pickle(path_to_pickle)
        return observation_grids
    except Exception as e:
        print(e)


# Gridded weather observations from the raw zipped file --------------------------------------------------------------
def parse_daily_gridded_weather_obs(filename, var_name, start_date='2006-01-01'):
    """
    :param filename:
    :param var_name: [str] Variable name, e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Rainfall'
    :param start_date: [str] The start date from which the observation data was collected, formatted as 'yyyy-mm-dd'
    :return:
    """
    # Centres
    cartesian_centres_temp = pd.read_csv(filename, header=None, index_col=0, nrows=2)
    cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

    # Temperature observations
    timeseries_data = pd.read_csv(filename, header=None, skiprows=[0, 1], parse_dates=[0], dayfirst=True)
    timeseries_data[0] = timeseries_data[0].map(lambda x: x.date())
    if start_date is not None and isinstance(pd.to_datetime(start_date), pd.Timestamp):
        mask = (timeseries_data[0] >= pd.to_datetime(start_date).date())
        timeseries_data = timeseries_data.loc[mask]
    timeseries_data.set_index(0, inplace=True)

    # Reshape the dataframe
    idx = pd.MultiIndex.from_product([cartesian_centres, timeseries_data.index.tolist()], names=['Centroid', 'Date'])
    data = pd.DataFrame(timeseries_data.T.values.flatten(), index=idx, columns=[var_name])
    # data.reset_index(inplace=True)

    # data.Centre = data.Centre.map(shapely.geometry.asPoint)

    # # Add levels of Grid corners (and centres' LongLat)
    # num = len(timeseries_data)

    # import itertools

    # grid = [find_square_corners(centre, 5000, rotation=None) for centre in cartesian_centres]
    # data['Grid'] = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in grid))

    # long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]
    # data['Centroid_LongLat'] = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in long_lat))

    return data


def make_ukcp_pickle_path(filename: str, start_date: str):
    """
    :param filename: e.g. filename="daily-maximum-temperature"
    :param start_date: e.g. start_date='2006-01-01'
    :return: [str]
    """
    filename_suffix = "" if start_date is None else "-{}".format(start_date.replace("-", ""))
    pickle_filename = filename + filename_suffix + ".pickle"
    path_to_pickle = cdd_weather("UKCP", pickle_filename)
    return path_to_pickle


def prep_daily_gridded_weather_obs(filename: str, var_name: str, start_date='2006-01-01', use_pseudo_grid_id=False,
                                   update=False):
    """
    :param filename: [str] e.g. filename="daily-maximum-temperature"
    :param var_name: [str] e.g. var_name='Maximum_Temperature'
    :param start_date: [str] start_date='2006-01-01' (default)
    :param use_pseudo_grid_id: [bool] False (default)
    :param update: [bool]
    """
    assert isinstance(pd.to_datetime(start_date), pd.Timestamp) or start_date is None
    try:
        path_to_zip = cdd_weather("UKCP", filename + ".zip")
        with zipfile.ZipFile(path_to_zip, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            obs_data = [parse_daily_gridded_weather_obs(zf.open(f), var_name, start_date) for f in filename_list]
        zf.close()

        gridded_obs = pd.concat(obs_data, axis=0)

        # Add a pseudo id for each observation grid
        if use_pseudo_grid_id:
            observation_grids = fetch_observation_grids(update=update)
            observation_grids = observation_grids.reset_index().set_index('Centroid')
            gridded_obs = gridded_obs.reset_index(level='Date').join(observation_grids[['Pseudo_Grid_ID']])
            gridded_obs = gridded_obs.reset_index().set_index(['Pseudo_Grid_ID', 'Centroid', 'Date'])

        path_to_pickle = make_ukcp_pickle_path(filename, start_date)
        save_pickle(gridded_obs, path_to_pickle)

    except Exception as e:
        print("Failed to get \"{}\". {}.".format(os.path.splitext(make_ukcp_pickle_path(filename, start_date))[0], e))


def fetch_daily_gridded_weather_obs(filename: str, var_name: str, start_date='2006-01-01',
                                    use_pseudo_grid_id=False, update=False) -> pd.DataFrame:
    """
    :param filename: [str] e.g. filename="daily-rainfall"
    :param var_name: [str] variable name, e.g. var_name='Rainfall' (or 'Maximum_Temperature', 'Minimum_Temperature')
    :param start_date: start_date: [str] The start date from which the observation data was collected; 'yyyy-mm-dd'
    :param use_pseudo_grid_id: [bool]
    :param update: [bool]
    :return: [pd.DataFrame]
    """
    path_to_pickle = make_ukcp_pickle_path(filename, start_date)

    if not os.path.isfile(path_to_pickle) or update:
        prep_daily_gridded_weather_obs(filename, var_name, start_date, use_pseudo_grid_id)
    try:
        gridded_obs = load_pickle(path_to_pickle)
        return gridded_obs
    except Exception as e:
        print(e)


# Combine Weather observations of different variables ----------------------------------------------------------------
def integrate_daily_gridded_weather_obs(start_date='2006-01-01', use_pseudo_grid_id=True, update=False):
    try:
        d_max_temp = fetch_daily_gridded_weather_obs(
            "daily-maximum-temperature", 'Maximum_Temperature', start_date, use_pseudo_grid_id=False, update=update)
        d_min_temp = fetch_daily_gridded_weather_obs(
            "daily-minimum-temperature", 'Minimum_Temperature', start_date, use_pseudo_grid_id=False, update=update)
        d_rainfall = fetch_daily_gridded_weather_obs(
            "daily-rainfall", 'Rainfall', start_date, use_pseudo_grid_id=False, update=update)

        gridded_obs = pd.concat([d_max_temp, d_min_temp, d_rainfall], axis=1)
        gridded_obs['Temperature_Change'] = abs(gridded_obs.Maximum_Temperature - gridded_obs.Minimum_Temperature)

        if use_pseudo_grid_id:
            observation_grids = fetch_observation_grids(update=update)
            observation_grids = observation_grids.reset_index().set_index('Centroid')
            gridded_obs = gridded_obs.reset_index('Date').join(observation_grids[['Pseudo_Grid_ID']])
            gridded_obs = gridded_obs.reset_index().set_index(['Pseudo_Grid_ID', 'Centroid', 'Date'])

        path_to_pickle = make_ukcp_pickle_path("ukcp-daily-gridded-weather", start_date)
        save_pickle(gridded_obs, path_to_pickle)

    except Exception as e:
        print("Failed to integrate the UKCP gridded weather observations. {}".format(e))


def fetch_integrated_daily_gridded_weather_obs(start_date='2006-01-01', use_pseudo_grid_id=True,
                                               update=False) -> pd.DataFrame:
    """
    :param start_date: [str]
    :param use_pseudo_grid_id: [bool]
    :param update: [bool]
    :return:
    """
    path_to_pickle = make_ukcp_pickle_path("ukcp-daily-gridded-weather", start_date)
    if not os.path.isfile(path_to_pickle) or update:
        integrate_daily_gridded_weather_obs(start_date, use_pseudo_grid_id, update)
    try:
        ukcp_gridded_weather_obs = load_pickle(path_to_pickle)
        return ukcp_gridded_weather_obs
    except Exception as e:
        print(e)
