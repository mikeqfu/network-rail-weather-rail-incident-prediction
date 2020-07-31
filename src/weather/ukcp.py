""" UKCP gridded weather observations: maximum temperature, minimum temperature and precipitation """

import gc
import os
import tempfile
import zipfile

import datetime_truncate
import natsort
import pandas as pd
import shapely.geometry
import sqlalchemy.types
from pyhelpers.geom import osgb36_to_wgs84
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

from mssqlserver.tools import create_mssql_connectable_engine
from utils import cdd_weather
from weather.tools import create_grid

pd_preferences()


# == Observation grids ================================================================================

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


def get_observation_grids(zip_filename="daily-precipitation.zip", update=False, verbose=False):
    """
    Fetch data of observation grids from local pickle.

    :param zip_filename: filename of a zipped file for the data of observation grids
    :type zip_filename: str
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: MIDAS RADTOB observation grids
    :rtype: pandas.DataFrame

    **Example**::

        from weather.ukcp import get_observation_grids

        update = True
        verbose = True

        zip_filename = "daily-precipitation.zip"
        observation_grids = get_observation_grids(zip_filename, update, verbose)
    """

    path_to_pickle = cdd_weather("ukcp", "observation-grids.pickle")

    if os.path.isfile(path_to_pickle) and not update:
        observation_grids = load_pickle(path_to_pickle)

    else:
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
            observation_grids = None

    return observation_grids


# == UKCP09 data ======================================================================================

def parse_daily_gridded_weather_obs(filename, var_name, start_date='2006-01-01'):
    """
    Parse gridded weather observations from the raw zipped file.

    :param filename: filename of raw data
    :type filename: str
    :param var_name: variable name, e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Precipitation'
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


def get_ukcp09_var_obs(zip_filename, var_name, start_date='2006-01-01', use_pseudo_grid_id=False, update=False,
                       verbose=False):
    """
    :param zip_filename: "daily-maximum-temperature", "daily-minimum-temperature", or "daily-precipitation"
    :type zip_filename: str
    :param var_name: variable name; 'Precipitation' or 'Maximum_Temperature', 'Minimum_Temperature'
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

        from weather.ukcp import get_ukcp09_var_obs

        zip_filename = "daily-maximum-temperature"
        var_name = 'Maximum_Temperature'
        start_date = '2006-01-01'
        use_pseudo_grid_id = False
        update = False
        verbose = True

        gridded_obs = get_ukcp09_var_obs(zip_filename, var_name, start_date, use_pseudo_grid_id, update, verbose)
    """

    assert isinstance(pd.to_datetime(start_date), pd.Timestamp) or start_date is None

    filename = os.path.splitext(zip_filename)[0]
    path_to_pickle = make_ukcp_pickle_path(filename, start_date)

    if os.path.isfile(path_to_pickle) and not update:
        gridded_obs = load_pickle(path_to_pickle)

    else:
        try:
            path_to_zip = cdd_weather("ukcp", zip_filename + ".zip")

            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                obs_data = [parse_daily_gridded_weather_obs(zf.open(f), var_name, start_date) for f in filename_list]
            zf.close()

            gridded_obs = pd.concat(obs_data, axis=0)

            # Add a pseudo id for each observation grid
            if use_pseudo_grid_id:
                observation_grids = get_observation_grids(update=update)
                observation_grids = observation_grids.reset_index().set_index('Centroid')
                gridded_obs = gridded_obs.reset_index(level='Date').join(observation_grids[['Pseudo_Grid_ID']])
                gridded_obs = gridded_obs.reset_index().set_index(['Pseudo_Grid_ID', 'Centroid', 'Date'])

            save_pickle(gridded_obs, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(filename.replace("-", " "), e))
            gridded_obs = None

    return gridded_obs


def get_ukcp09_data(start_date='2006-01-01', use_pseudo_grid_id=True, update=False, verbose=False):
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

        from weather.ukcp import get_ukcp09_data

        start_date = '2006-01-01'
        use_pseudo_grid_id = False
        update = False
        verbose = True

        ukcp09_data = get_ukcp09_data(start_date, use_pseudo_grid_id, update, verbose)
    """

    filename = "ukcp-daily-gridded-weather"
    path_to_pickle = make_ukcp_pickle_path(filename, start_date)

    if os.path.isfile(path_to_pickle) and not update:
        ukcp09_data = load_pickle(path_to_pickle)

    else:
        try:
            d_max_temp = get_ukcp09_var_obs("daily-maximum-temperature", 'Maximum_Temperature', start_date,
                                            use_pseudo_grid_id=False, update=update, verbose=verbose)
            d_min_temp = get_ukcp09_var_obs("daily-minimum-temperature", 'Minimum_Temperature', start_date,
                                            use_pseudo_grid_id=False, update=update, verbose=verbose)
            d_precipitation = get_ukcp09_var_obs("daily-precipitation", 'Precipitation', start_date,
                                                 use_pseudo_grid_id=False, update=update, verbose=verbose)

            ukcp09_data = pd.concat([d_max_temp, d_min_temp, d_precipitation], axis=1)

            del d_max_temp, d_min_temp, d_precipitation
            gc.collect()

            ukcp09_data['Temperature_Change'] = abs(ukcp09_data.Maximum_Temperature - ukcp09_data.Minimum_Temperature)

            if use_pseudo_grid_id:
                observation_grids = get_observation_grids(update=update)
                observation_grids = observation_grids.reset_index().set_index('Centroid')
                ukcp09_data = ukcp09_data.reset_index('Date').join(observation_grids[['Pseudo_Grid_ID']])
                ukcp09_data = ukcp09_data.reset_index().set_index(['Pseudo_Grid_ID', 'Centroid', 'Date'])

            path_to_pickle = make_ukcp_pickle_path(filename, start_date)
            save_pickle(ukcp09_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to integrate the UKCP09 gridded weather observations. {}".format(e))
            ukcp09_data = None

    return ukcp09_data


def dump_ukcp09_data_to_mssql(table_name='UKCP091', if_exists='append', chunk_size=100000, update=False,
                              verbose=False):
    """
    See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

    :param table_name:
    :param if_exists:
    :param chunk_size:
    :param update:
    :param verbose:
    :return:
    """

    ukcp09_data = get_ukcp09_data(update=update, verbose=verbose)
    ukcp09_data.reset_index(inplace=True)

    ukcp09_engine = create_mssql_connectable_engine(database_name='Weather')

    ukcp09_data_ = pd.DataFrame(ukcp09_data.Centroid.to_list(), columns=['Centroid_X', 'Centroid_Y'])
    ukcp09_data = pd.concat([ukcp09_data.drop('Centroid', axis=1), ukcp09_data_], axis=1)

    print("Importing UKCP09 data to MSSQL Server", end=" ... ")

    with tempfile.NamedTemporaryFile() as temp_file:
        ukcp09_data.to_csv(temp_file.name + ".csv", index=False, chunksize=chunk_size)

        tsql_chunksize = 2100 // len(ukcp09_data.columns)
        temp_file_ = pd.read_csv(temp_file.name + ".csv", chunksize=tsql_chunksize)
        for chunk in temp_file_:
            # e.g. chunk = temp_file_.get_chunk(chunk_size)
            chunk.to_sql(table_name, ukcp09_engine, schema='dbo', if_exists=if_exists, index=False,
                         dtype={'Date': sqlalchemy.types.DATE}, method='multi')
            gc.collect()

        temp_file.close()

    os.remove(temp_file.name)

    print("Done. ")


def query_ukcp09_by_grid_datetime(grids, period, update=False, dat_dir=None, pickle_it=False, verbose=False):
    """
    Get UKCP09 data by observation grids (Query from the database) for the given ``period``.

    :param grids: a list of weather observation IDs
    :type grids: list
    :param period: prior-incident / non-incident period
    :type period:
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param dat_dir: directory where the queried data is saved, defaults to ``None``
    :type dat_dir: str, None
    :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
    :type pickle_it: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: UKCP09 data by ``grids`` and ``period``
    :rtype: pandas.DataFrame

    **Examples**::

        from weather.ukcp import query_ukcp09_by_grid_datetime

        dat_dir = None
        update = False
        verbose = True

        pickle_it = False
        grids = incidents.Weather_Grid.iloc[0]
        period = incidents.Critical_Period.iloc[0]
        ukcp09_dat = query_ukcp09_by_grid_datetime(grids, period, verbose=verbose)

        pickle_it = True
        grids = incidents.Weather_Grid.iloc[1]
        period = incidents.Critical_Period.iloc[1]
        ukcp09_dat = query_ukcp09_by_grid_datetime(grids, period, pickle_it=pickle_it, verbose=verbose)
    """

    # Make a pickle filename
    pickle_filename = "{}-{}.pickle".format(
        "".join(str(x)[0] + str(x)[-1] for x in grids),
        "-".join([period.min().strftime('%Y%m%d%H'), period.max().strftime('%Y%m%d%H')]))

    # Specify a directory/path to store the pickle file (if appropriate)
    dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else cdd_weather("ukcp", "dat")
    path_to_pickle = cdd_weather(dat_dir, pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        ukcp09_dat = load_pickle(path_to_pickle)

    else:
        # Create an engine to the MSSQL server
        conn_metex = create_mssql_connectable_engine(database_name='Weather')
        # Specify database sql query
        grids_ = tuple(grids) if len(grids) > 1 else grids[0]
        period_ = tuple(x.strftime('%Y-%m-%d %H:%M:%S') for x in period)
        sql_query = "SELECT * FROM dbo.[UKCP09] WHERE [Pseudo_Grid_ID] {} {} AND [Date] IN {};".format(
            'IN' if len(grids) > 1 else '=', grids_, period_)
        # Query the weather data
        ukcp09_dat = pd.read_sql(sql_query, conn_metex)

        if pickle_it:
            save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

    return ukcp09_dat


def query_ukcp09_by_grid_datetime_(grids, period, update=False, dat_dir=None, pickle_it=False, verbose=False):
    """
    Get UKCP09 data by observation grids and date (Query from the database)
    from the beginning of the year to the start of the ``period``.

    :param grids: a list of weather observation IDs
    :type grids: list
    :param period: prior-incident / non-incident period
    :type period:
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param dat_dir: directory where the queried data is saved, defaults to ``None``
    :type dat_dir: str, None
    :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
    :type pickle_it: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: UKCP09 data by ``grids`` and ``period``
    :rtype: pandas.DataFrame

    **Examples**::

        from weather.ukcp import query_ukcp09_by_grid_datetime

        update = False
        verbose = True

        pickle_it = False
        grids = incidents.Weather_Grid.iloc[0]
        period = incidents.Critical_Period.iloc[0]
        ukcp09_dat = query_ukcp09_by_grid_datetime_(grids, period, verbose=verbose)

        pickle_it = True
        grids = incidents.Weather_Grid.iloc[1]
        period = incidents.Critical_Period.iloc[1]
        ukcp09_dat = query_ukcp09_by_grid_datetime_(grids, period, pickle_it=pickle_it, verbose=verbose)
    """

    y_start = datetime_truncate.truncate_year(period.min()).strftime('%Y-%m-%d')
    p_start = period.min().strftime('%Y-%m-%d')

    # Make a pickle filename
    pickle_filename = "{}-{}.pickle".format(
        "".join(str(x)[0] + str(x)[-1] for x in grids),
        "-".join([y_start.replace("-", ""), p_start.replace("-", "")]))

    # Specify a directory/path to store the pickle file (if appropriate)
    dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else cdd_weather("ukcp", "dat")
    path_to_pickle = cdd_weather(dat_dir, pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        ukcp09_dat = load_pickle(path_to_pickle)

    else:
        # Create an engine to the MSSQL server
        conn_metex = create_mssql_connectable_engine(database_name='Weather')
        # Specify database sql query
        grids_ = tuple(grids) if len(grids) > 1 else grids[0]
        sql_query = "SELECT * FROM dbo.[UKCP09] " \
                    "WHERE [Pseudo_Grid_ID] {} {} " \
                    "AND [Date] >= '{}' AND [Date] <= '{}';".format('IN' if len(grids) > 1 else '=', grids_,
                                                                    y_start, p_start)
        # Query the weather data
        ukcp09_dat = pd.read_sql(sql_query, conn_metex)

        if pickle_it:
            save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

    return ukcp09_dat
