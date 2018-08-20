""" Gridded weather observations """

import os
import zipfile

import natsort
import pandas as pd
import shapely.geometry

from converters import osgb36_to_wgs84
from utils import cdd, load_pickle, save_pickle


# Change directory to "Weather"
def cdd_weather(*directories):
    path = cdd("Weather")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Find coordinates for each corner of the weather observation grid
def create_grid(centre_point, side_length=5000, rotation=None):
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

    # corners = shapely.geometry.Polygon([lower_left, upper_left, upper_right, lower_right])

    return lower_left, upper_left, upper_right, lower_right


# ====================================================================================================================
""" UKCP gridded data: maximum/minimum temperatures and rainfall """


# Read observation grids
def read_observation_grids(obs_filename):
    cartesian_centres_temp = pd.read_csv(obs_filename, header=None, index_col=0, nrows=2)
    cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

    grid = [create_grid(centre, 5000, rotation=None) for centre in cartesian_centres]

    long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]

    obs_grids = pd.DataFrame({'Centroid': cartesian_centres,
                              'Centroid_XY': [shapely.geometry.Point(x) for x in cartesian_centres],
                              'Centroid_LongLat': [shapely.geometry.Point(x) for x in long_lat],
                              'Grid': [shapely.geometry.Polygon(x) for x in grid]})

    return obs_grids


# Get all observation grids
def get_observation_grids(obs_zip_filename="daily-rainfall.zip", update=False):
    pickle_filename = "observation-grids.pickle"
    path_to_pickle = cdd_weather("UKCP gridded obs", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        observation_grids = load_pickle(path_to_pickle)
    else:
        try:
            path_to_zip = cdd_weather("UKCP gridded obs", obs_zip_filename)

            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                obs_grids = [read_observation_grids(zf.open(f)) for f in filename_list]
            zf.close()

            observation_grids = pd.concat(obs_grids, ignore_index=True)

            # Add a pseudo id for each observation grid
            observation_grids = observation_grids.sort_values('Centroid').set_index('Centroid')
            observation_grids['Pseudo_Grid_ID'] = range(len(observation_grids))

            save_pickle(observation_grids, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Observation Grids\". {}".format(e))
            observation_grids = pd.DataFrame()

    return observation_grids


# Read gridded weather observations from the raw zipped file
def read_daily_gridded_weather_obs(filename, col_name, start_date='2006-01-01'):
    """
    :param filename:
    :param col_name: [str] Variable name, e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Rainfall'
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
    data = pd.DataFrame(timeseries_data.T.values.flatten(), index=idx, columns=[col_name])
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


# Get gridded weather observations
def get_daily_gridded_weather_obs(filename, col_name, start_date='2006-01-01', pseudo_grid_id=False, update=False):
    """
    :param filename:
    :param col_name: [str] Variable name, e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Rainfall'
    :param start_date: start_date: [str] The start date from which the observation data was collected; 'yyyy-mm-dd'
    :param pseudo_grid_id: [bool]
    :param update: [bool]
    :return:
    """
    filename_suffix = "" if start_date is None else "-{}".format(start_date.replace("-", ""))
    pickle_filename = filename + filename_suffix + ".pickle"
    path_to_pickle = cdd_weather("UKCP gridded obs", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        gridded_obs = load_pickle(path_to_pickle)
    else:
        try:
            path_to_zip = cdd_weather("UKCP gridded obs", filename + ".zip")

            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                obs_data = [read_daily_gridded_weather_obs(zf.open(f), col_name, start_date) for f in filename_list]
            zf.close()

            gridded_obs = pd.concat(obs_data, axis=0)

            # Add a pseudo id for each observation grid
            if pseudo_grid_id:
                observation_grids = get_observation_grids(update=update)
                gridded_obs = gridded_obs.reset_index(level='Date').join(observation_grids[['Pseudo_Grid_ID']])
                gridded_obs.set_index(['Pseudo_Grid_ID', 'Date'], inplace=True)

            save_pickle(gridded_obs, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(filename, e))
            gridded_obs = pd.DataFrame()

    return gridded_obs


# Combine weather observations of different variables
def get_integrated_daily_gridded_weather_obs(start_date='2006-01-01', pseudo_grid_id=True, update=False):

    assert isinstance(pd.to_datetime(start_date), pd.Timestamp) or start_date is None

    filename_suffix = "" if start_date is None else "-{}".format(start_date.replace("-", ""))
    pickle_filename = "daily-gridded-weather-obs{}.pickle".format(filename_suffix)
    path_to_file = cdd_weather("UKCP gridded obs", pickle_filename)

    if os.path.isfile(path_to_file) and not update:
        daily_gridded_obs = load_pickle(path_to_file)
    else:
        try:
            d_max_temp = get_daily_gridded_weather_obs(
                "daily-maximum-temperature", 'Maximum_Temperature', start_date, pseudo_grid_id=False, update=update)
            d_min_temp = get_daily_gridded_weather_obs(
                "daily-minimum-temperature", 'Minimum_Temperature', start_date, pseudo_grid_id=False, update=update)
            d_rainfall = get_daily_gridded_weather_obs(
                "daily-rainfall", 'Rainfall', start_date, pseudo_grid_id=False, update=update)

            daily_gridded_obs = pd.concat([d_max_temp, d_min_temp, d_rainfall], axis=1)

            if pseudo_grid_id:
                obs_grids = get_observation_grids(update=update)
                daily_gridded_obs = daily_gridded_obs.reset_index(level='Date').join(obs_grids[['Pseudo_Grid_ID']])
                daily_gridded_obs.set_index(['Pseudo_Grid_ID', 'Date'], inplace=True)

            save_pickle(daily_gridded_obs, path_to_file)

        except Exception as e:
            print("Failed to get integrated daily gridded weather observations. {}.".format(e))
            daily_gridded_obs = pd.DataFrame()

    return daily_gridded_obs


# ====================================================================================================================
""" Met Office RADTOB (Radiation values currently being reported) """


# Met station locations
def get_meteorological_stations(update=False):

    pickle_filename = "meteorological-stations.pickle"
    path_to_pickle = cdd_weather(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        met_stations = load_pickle(path_to_pickle)
    else:
        try:
            path_to_spreadsheet = path_to_pickle.replace(".pickle", ".xlsx")
            met_stations = pd.read_excel(path_to_spreadsheet, parse_dates=['Station start date'])

            met_stations.columns = [x.replace(' ', '_').upper() for x in met_stations.columns]
            met_stations = met_stations.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # Convert coordinates to shapely.geometry.Point
            met_stations['LONG_LAT'] = met_stations.apply(
                lambda x: shapely.geometry.Point((x['LONGITUDE'], x['LATITUDE'])), axis=1)
            met_stations['E_N'] = met_stations.apply(
                lambda x: shapely.geometry.Point((x['EASTING'], x['NORTHING'])), axis=1)

            met_stations.sort_values(['SRC_ID', 'NAME'], inplace=True)
            met_stations.set_index('SRC_ID', inplace=True)

            save_pickle(met_stations, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Meteorological stations\" data. {}".format(e))
            met_stations = pd.DataFrame()

    return met_stations


# Read each txt file of MIDAS RADTOB
def read_radiation_data(filename, headers, full_data=False, met_stn=True):
    """
    :param filename:
    :param headers:
    :param full_data:
    :param met_stn:
    :return:

    SRC_ID:         Unique source identifier or station site number
    OB_END_TIME:    Date and time at end of observation
    OB_HOUR_COUNT:  Observation hour count
    VERSION_NUM:    Observation version number - Use the row with '1', which has been quality checked by the Met Office
    GLBL_IRAD_AMT:  Global solar irradiation amount Kjoules/sq metre over the observation period

    """
    raw_txt = pd.read_csv(filename, header=None, names=headers, parse_dates=[2, 12], dtype={8: str, 10: str, 13: str})
    if full_data:
        ro_data = raw_txt
    else:
        use_dat = raw_txt[raw_txt.OB_HOUR_COUNT == 24]
        use_dat.index = range(len(use_dat))
        ro_data = use_dat[['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT', 'VERSION_NUM', 'GLBL_IRAD_AMT']]

        # Keep VERSION_NUM == 2 only if available
        checked = ro_data.groupby(['SRC_ID', 'OB_END_TIME']).agg({'VERSION_NUM': 'count'})
        checked_idx = checked[checked.VERSION_NUM == 2]
        idx = [ro_data[(ro_data.SRC_ID == i) & (ro_data.OB_END_TIME == d)].sort_values('VERSION_NUM').index[0]
               for i, d in checked_idx.index]
        ro_data.drop(idx, axis='index', inplace=True)

        ro_data.rename(columns={'OB_END_TIME': 'OB_END_DATE_TIME'}, inplace=True)
        ob_end_date = ro_data.OB_END_DATE_TIME.map(lambda x: x.date())
        ro_data.insert(ro_data.columns.get_loc('OB_END_DATE_TIME'), column='OB_END_DATE', value=ob_end_date)

        ro_data.sort_values(['SRC_ID', 'OB_END_DATE_TIME'], inplace=True)
        ro_data.index = range(len(ro_data))

        if met_stn:
            met_stn = get_meteorological_stations()
            met_stn.rename(columns={'NAME': 'MET_STATION'}, inplace=True)
            ro_data = ro_data.join(met_stn, on='SRC_ID')

    return ro_data


# Headers of the midas_radtob data set
def get_ro_headers():
    headers_raw = pd.read_excel(cdd_weather("Radiation obs", "RO-column-headers.xlsx"), header=None)
    headers = [x.strip() for x in headers_raw.iloc[0, :].values]
    return headers


# MIDAS RADTOB
def get_midas_radtob(full_data=False, update=False):
    """
    :param full_data:
    :param update:
    :return:

    MIDAS   -   Met Office Integrated Data Archive System
    RADTOB 	-   RADT-OB table. Radiation values currently being reported

    """

    pickle_filename = "midas-radtob-20060101-20141231{}.pickle".format("-full" if full_data else "")
    path_to_pickle = cdd_weather("Radiation obs", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        radiation_data = load_pickle(path_to_pickle)
    else:
        headers = get_ro_headers()
        try:
            path_to_zip = path_to_pickle.replace("-full.pickle" if full_data else ".pickle", ".zip")

            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                temp_dat = [read_radiation_data(zf.open(f), headers, full_data, met_stn=False) for f in filename_list]
            zf.close()

            radiation_data = pd.concat(temp_dat, axis=0, sort=False, ignore_index=True)

            met_stn = get_meteorological_stations()
            met_stn.rename(columns={'NAME': 'MET_STATION'}, inplace=True)
            radiation_data = radiation_data.join(met_stn, on='SRC_ID')

            save_pickle(radiation_data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Radiation obs\". {}".format(e))
            radiation_data = pd.DataFrame(columns=headers)

    return radiation_data
