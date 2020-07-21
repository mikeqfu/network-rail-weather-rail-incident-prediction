""" UKCP gridded weather observations: maximum temperature, minimum temperature and rainfall """

import os
import zipfile

import natsort
import pandas as pd
import shapely.geometry
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

from utils import cdd_weather
from weather.tools import create_grid

pd_preferences()


def parse_observation_grids(filename):
    """
    Parse observation grids.

    :param filename: filename of the observation grid data
    :type filename: str
    :return: parsed data of the observation grids
    :rtype: pandas.DataFrame

    **Example**::

        filename = "ukcp09_gridded-land-obs-daily_timeseries_maximum-temperature_000000E_450000N_19600101-20161231.csv"

        obs_grids = parse_observation_grids(filename)
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


def prep_observation_grids(zip_filename, verbose=False):
    """
    Get ready the data of observation grids from the zipped file.

    :param zip_filename: filename of a zipped file for the data of observation grids
    :type zip_filename: str
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int

    **Example**::

        zip_filename = "daily-maximum-temperature.zip"

        prep_observation_grids(zip_filename)
    """

    try:
        path_to_zip = cdd_weather("ukcp", zip_filename)

        with zipfile.ZipFile(path_to_zip, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            obs_grids = [parse_observation_grids(zf.open(f)) for f in filename_list]
        zf.close()

        observation_grids = pd.concat(obs_grids, ignore_index=True)

        # Add a pseudo id for each observation grid
        observation_grids.sort_values('Centroid', inplace=True)
        observation_grids.index = pd.Index(range(len(observation_grids)), name='Pseudo_Grid_ID')

        path_to_pickle = cdd_weather("ukcp", "observation-grids.pickle")
        save_pickle(observation_grids, path_to_pickle, verbose=verbose)

    except Exception as e:
        print("Failed to get \"Observation Grids\". {}".format(e))


def fetch_observation_grids(zip_filename="daily-rainfall.zip", update=False, verbose=False):
    """
    Fetch data of observation grids from local pickle.

    :param zip_filename: defaults to ``"daily-rainfall.zip"``
    :type zip_filename: str
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: MIDAS RADTOB (Radiation data)
    :rtype: pandas.DataFrame

    **Example**::

        from weather.ukcp import fetch_observation_grids

        zip_filename = "daily-rainfall.zip"
        update = True
        verbose = True

        observation_grids = fetch_observation_grids(zip_filename, update, verbose)
    """

    path_to_pickle = cdd_weather("ukcp", "observation-grids.pickle")

    if not os.path.isfile(path_to_pickle) or update:
        prep_observation_grids(zip_filename, verbose=verbose)
    try:
        observation_grids = load_pickle(path_to_pickle)
        return observation_grids
    except Exception as e:
        print(e)


def parse_daily_gridded_weather_obs(filename, var_name, start_date='2006-01-01'):
    """
    Parse gridded weather observations from the raw zipped file.

    :param filename: filename of raw data
    :type filename: str
    :param var_name: variable name, e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Rainfall'
    :type var_name: str
    :param start_date: start date on which the observation data was collected, formatted as 'yyyy-mm-dd'
    :type start_date: str
    :return: parsed data of the daily gridded weather observations
    :rtype: pandas.DataFrame
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


def make_ukcp_pickle_path(filename, start_date):
    """
    Make a full path to the pickle file of the UKCP data.

    :param filename: e.g. filename="daily-maximum-temperature"
    :type filename: str
    :param start_date: e.g. start_date='2006-01-01'
    :type start_date: str
    :return: a full path to the pickle file of the UKCP data
    :rtype: str
    """

    filename_suffix = "" if start_date is None else "-{}".format(start_date.replace("-", ""))
    pickle_filename = filename + filename_suffix + ".pickle"
    path_to_pickle = cdd_weather("ukcp", pickle_filename)
    return path_to_pickle


def prep_daily_gridded_weather_obs(filename, var_name, start_date='2006-01-01', use_pseudo_grid_id=False,
                                   update=False, verbose=False):
    """
    Get ready the data of daily gridded weather observations from original file.

    :param filename: e.g. filename="daily-maximum-temperature"
    :type filename: str
    :param var_name: e.g. var_name='Maximum_Temperature'
    :type var_name: str
    :param start_date: defaults to ``'2006-01-01'``
    :type start_date: str
    :param use_pseudo_grid_id: defaults to ``False``
    :type use_pseudo_grid_id: bool
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int

    **Example**::

        filename = "daily-maximum-temperature"
        var_name = 'Maximum_Temperature'
        start_date = '2006-01-01'
        use_pseudo_grid_id = False
        update = True
        verbose = True
    """

    assert isinstance(pd.to_datetime(start_date), pd.Timestamp) or start_date is None

    try:
        path_to_zip = cdd_weather("ukcp", filename + ".zip")
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
        save_pickle(gridded_obs, path_to_pickle, verbose=verbose)

    except Exception as e:
        print("Failed to get \"{}\". {}.".format(os.path.splitext(make_ukcp_pickle_path(filename, start_date))[0], e))


def fetch_daily_gridded_weather_obs(filename, var_name, start_date='2006-01-01', use_pseudo_grid_id=False,
                                    update=False, verbose=False):
    """
    :param filename: e.g. filename="daily-rainfall"
    :type filename: str
    :param var_name: variable name, e.g. var_name='Rainfall' (or 'Maximum_Temperature', 'Minimum_Temperature')
    :type var_name: str
    :param start_date: start date from which the observation data was collected, defaults to ``'2006-01-01'``
    :type start_date: str
    :param use_pseudo_grid_id: defaults to ``False``
    :type use_pseudo_grid_id: bool
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of daily gridded weather observations
    :rtype: pandas.DataFrame

    **Example**::

        from weather.ukcp import fetch_daily_gridded_weather_obs

        filename = "daily-maximum-temperature"
        var_name = 'Maximum_Temperature'
        start_date = '2006-01-01'
        use_pseudo_grid_id = False
        update = True
        verbose = True

        gridded_obs = fetch_daily_gridded_weather_obs(filename, var_name, start_date, use_pseudo_grid_id,
                                                      update, verbose)
    """

    path_to_pickle = make_ukcp_pickle_path(filename, start_date)

    if not os.path.isfile(path_to_pickle) or update:
        prep_daily_gridded_weather_obs(filename, var_name, start_date, use_pseudo_grid_id, verbose=verbose)
    try:
        gridded_obs = load_pickle(path_to_pickle)
        return gridded_obs
    except Exception as e:
        print(e)


def integrate_daily_gridded_weather_obs(start_date='2006-01-01', use_pseudo_grid_id=True, update=False, verbose=False):
    """
    Integrate weather observations of different variables.

    :param start_date: start date from which the observation data was collected, defaults to ``'2006-01-01'``
    :type start_date: str
    :param use_pseudo_grid_id: defaults to ``False``
    :type use_pseudo_grid_id: bool
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of integrated daily gridded weather observations
    :rtype: pandas.DataFrame
    """

    try:
        d_max_temp = fetch_daily_gridded_weather_obs("daily-maximum-temperature", 'Maximum_Temperature', start_date,
                                                     use_pseudo_grid_id=False, update=update, verbose=verbose)
        d_min_temp = fetch_daily_gridded_weather_obs("daily-minimum-temperature", 'Minimum_Temperature', start_date,
                                                     use_pseudo_grid_id=False, update=update, verbose=verbose)
        d_rainfall = fetch_daily_gridded_weather_obs("daily-rainfall", 'Rainfall', start_date,
                                                     use_pseudo_grid_id=False, update=update, verbose=verbose)

        gridded_obs = pd.concat([d_max_temp, d_min_temp, d_rainfall], axis=1)
        gridded_obs['Temperature_Change'] = abs(gridded_obs.Maximum_Temperature - gridded_obs.Minimum_Temperature)

        if use_pseudo_grid_id:
            observation_grids = fetch_observation_grids(update=update)
            observation_grids = observation_grids.reset_index().set_index('Centroid')
            gridded_obs = gridded_obs.reset_index('Date').join(observation_grids[['Pseudo_Grid_ID']])
            gridded_obs = gridded_obs.reset_index().set_index(['Pseudo_Grid_ID', 'Centroid', 'Date'])

        path_to_pickle = make_ukcp_pickle_path("ukcp-daily-gridded-weather", start_date)
        save_pickle(gridded_obs, path_to_pickle, verbose=verbose)

    except Exception as e:
        print("Failed to integrate the UKCP gridded weather observations. {}".format(e))


def fetch_integrated_daily_gridded_weather_obs(start_date='2006-01-01', use_pseudo_grid_id=True, update=False,
                                               verbose=False):
    """
    Fetch integrated weather observations of different variables from local pickle.

    :param start_date: start date from which the observation data was collected, defaults to ``'2006-01-01'``
    :type start_date: str
    :param use_pseudo_grid_id: defaults to ``False``
    :type use_pseudo_grid_id: bool
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of integrated daily gridded weather observations
    :rtype: pandas.DataFrame
    :return: data of integrated daily gridded weather observations
    :rtype: pandas.DataFrame

    **Example**::

        from weather.ukcp import fetch_integrated_daily_gridded_weather_obs

        start_date = '2006-01-01'
        use_pseudo_grid_id = False
        update = True
        verbose = True

        gridded_obs = fetch_integrated_daily_gridded_weather_obs(start_date, use_pseudo_grid_id, update,
                                                                 verbose)
    """

    path_to_pickle = make_ukcp_pickle_path("ukcp-daily-gridded-weather", start_date)

    if not os.path.isfile(path_to_pickle) or update:
        integrate_daily_gridded_weather_obs(start_date=start_date, use_pseudo_grid_id=use_pseudo_grid_id,
                                            update=update, verbose=verbose)

    try:
        ukcp_gridded_weather_obs = load_pickle(path_to_pickle)
        return ukcp_gridded_weather_obs
    except Exception as e:
        print(e)
