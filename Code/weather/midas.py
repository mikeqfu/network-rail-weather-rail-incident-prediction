""" Met Office RADTOB (Radiation values currently being reported) """

import os
import zipfile

import natsort
import pandas as pd
import shapely.geometry
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

from weather.utils import cdd_weather

pd_preferences()


# Locations of the meteorological stations ---------------------------------------------------------------------------
def prep_meteorological_stations_locations():
    path_to_spreadsheet = cdd_weather("meteorological-stations.xlsx")
    try:
        met_stations_info = pd.read_excel(path_to_spreadsheet, parse_dates=['Station start date'])

        met_stations_info.columns = [x.replace(' ', '_').upper() for x in met_stations_info.columns]
        met_stations_info = met_stations_info.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert coordinates to shapely.geometry.Point
        met_stations_info['LONG_LAT'] = met_stations_info.apply(lambda x: (x['LONGITUDE'], x['LATITUDE']), axis=1)
        met_stations_info['LONG_LAT_GEOM'] = met_stations_info.apply(
            lambda x: shapely.geometry.Point((x['LONGITUDE'], x['LATITUDE'])), axis=1)
        met_stations_info['E_N'] = met_stations_info.apply(lambda x: (x['EASTING'], x['NORTHING']), axis=1)
        met_stations_info['E_N_GEOM'] = met_stations_info.apply(
            lambda x: shapely.geometry.Point((x['EASTING'], x['NORTHING'])), axis=1)

        met_stations_info.rename(columns={'NAME': 'MET_STATION'}, inplace=True)

        met_stations_info.sort_values(['SRC_ID', 'MET_STATION'], inplace=True)
        met_stations_info.set_index('SRC_ID', inplace=True)

        save_pickle(met_stations_info, path_to_spreadsheet.replace(".xlsx", ".pickle"))

    except Exception as e:
        print("Failed to get the locations of the meteorological stations. {}".format(e))


def fetch_meteorological_stations_locations(update=False) -> pd.DataFrame:
    """
    :param update: [bool]
    """
    path_to_pickle = cdd_weather("meteorological-stations.pickle")
    if not os.path.isfile(path_to_pickle) or update:
        prep_meteorological_stations_locations()
    try:
        met_stations_info = load_pickle(path_to_pickle)
        return met_stations_info
    except Exception as e:
        print(e)


# MIDAS RADTOB (Radiation data) --------------------------------------------------------------------------------------
def parse_midas_radtob(filename: str, headers: list, daily=False, met_stn=False):
    """
    MIDAS  - Met Office Integrated Data Archive System
    RADTOB - RADT-OB table. Radiation values currently being reported

    :param filename: [str] e.g. filename = "midas_radtob_200601-200612.txt"
    :param headers: [list]
    :param daily: [bool] if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :param met_stn: [bool] if True, add the location of meteorological station; False (default)

    SRC_ID:        Unique source identifier or station site number
    OB_END_TIME:   Date and time at end of observation
    OB_HOUR_COUNT: Observation hour count
    VERSION_NUM:   Observation version number - Use the row with '1', which has been quality checked by the Met Office
    GLBL_IRAD_AMT: Global solar irradiation amount Kjoules/sq metre over the observation period

    """
    raw_txt = pd.read_csv(filename, header=None, names=headers, parse_dates=[2, 12], infer_datetime_format=True,
                          skipinitialspace=True)

    ro_data = raw_txt[['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT', 'VERSION_NUM', 'GLBL_IRAD_AMT']]
    ro_data.drop_duplicates(inplace=True)

    if daily:
        ro_data = ro_data[ro_data.OB_HOUR_COUNT == 24]
        ro_data.index = range(len(ro_data))

    # Rename 'OB_END_TIME'
    ro_data.rename(columns={'OB_END_TIME': 'OB_END_DATE_TIME'}, inplace=True)

    # Cleanse the data
    key_cols = ['SRC_ID', 'OB_END_DATE_TIME', 'OB_HOUR_COUNT']

    # Remove records where VERSION_NUM == 0 if higher versions (i.e. VERSION_NUM == 1) are available
    checked_tmp = ro_data.groupby(key_cols).agg({'VERSION_NUM': 'count'})
    checked_dat = checked_tmp[checked_tmp.VERSION_NUM >= 2]
    temp = ro_data.join(checked_dat, on=key_cols, rsuffix='_checked')
    to_drop_tmp = temp[temp.VERSION_NUM_checked.notnull()]

    to_drop_idx = to_drop_tmp[to_drop_tmp.VERSION_NUM == 0].index
    ro_data.drop(to_drop_idx, axis='index', inplace=True)

    # Remove records with duplicated "VERSION_NUM == 1"
    checked_tmp = ro_data.groupby(key_cols).agg({'VERSION_NUM': 'count'})
    checked_dat = checked_tmp[checked_tmp.VERSION_NUM >= 2]
    temp = ro_data.join(checked_dat, on=key_cols, rsuffix='_checked')
    to_drop_tmp = temp[temp.VERSION_NUM_checked.notnull()]

    to_drop_idx = to_drop_tmp.drop_duplicates(key_cols, keep='first').index
    ro_data.drop(to_drop_idx, axis='index', inplace=True)

    # Sort rows
    ro_data.sort_values(key_cols, inplace=True)
    ro_data.index = range(len(ro_data))

    # Insert dates of 'OB_END_DATE_TIME'
    ob_end_date = ro_data.OB_END_DATE_TIME.map(lambda x: x.date())
    ro_data.insert(ro_data.columns.get_loc('OB_END_DATE_TIME'), column='OB_END_DATE', value=ob_end_date)

    if met_stn:
        met_stn = fetch_meteorological_stations_locations()
        met_stn.rename(columns={'NAME': 'MET_STATION'}, inplace=True)
        ro_data = ro_data.join(met_stn, on='SRC_ID')

    return ro_data


def make_radtob_pickle_path(data_filename: str, daily: bool, met_stn: bool):
    """
    :param data_filename: e.g. data_filename="midas-radtob-20060101-20141231"
    :param daily: e.g. daily=False
    :param met_stn: e.g. met_stn=False
    :return: [str]
    """
    filename_suffix = "-agg" if daily else "", "-met_stn" if met_stn else ""
    path_to_radtob_pickle = cdd_weather("MIDAS", "{}{}.pickle".format(data_filename, *filename_suffix))
    return path_to_radtob_pickle


def prep_midas_radtob(data_filename, daily=False, met_stn=False):
    """
    :param data_filename: [str] e.g. data_filename="midas-radtob-20060101-20141231"
    :param daily: [bool] if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :param met_stn: [bool] if True, add the location of meteorological station; False (default)
    """
    try:
        # Headers of the midas_radtob data set
        header_filename = "RO-column-headers.xlsx"
        path_to_header_file = cdd_weather("MIDAS", header_filename)
        headers_raw = pd.read_excel(path_to_header_file, header=None)
        headers = [x.strip() for x in headers_raw.iloc[0, :].values]

        path_to_zip = cdd_weather("MIDAS", data_filename + ".zip")
        with zipfile.ZipFile(path_to_zip, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            temp_dat = [parse_midas_radtob(zf.open(f), headers, daily, met_stn=False) for f in filename_list]
        zf.close()

        radtob = pd.concat(temp_dat, axis=0, ignore_index=True, sort=False)

        # Note: The following line is questionable
        radtob.loc[(radtob.GLBL_IRAD_AMT < 0) | radtob.GLBL_IRAD_AMT.isna(), 'GLBL_IRAD_AMT'] = 0  # or pd.np.nan

        if met_stn:
            met_stn = fetch_meteorological_stations_locations()
            radtob = radtob.join(met_stn, on='SRC_ID')

        radtob.set_index(['SRC_ID', 'OB_END_DATE'], inplace=True)

        path_to_pickle = make_radtob_pickle_path(data_filename, daily, met_stn)
        save_pickle(radtob, path_to_pickle)

    except Exception as e:
        print("Failed to get the radiation observations. {}".format(e))


def fetch_midas_radtob(data_filename="midas-radtob-20060101-20141231", daily=False, met_stn=False,
                       update=False) -> pd.DataFrame:
    """
    :param data_filename: [str] data_filename="midas-radtob-20060101-20141231"
    :param daily: [bool] if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :param met_stn: [bool] if True, add the location of meteorological station; False (default)
    :param update:
    """
    path_to_pickle = make_radtob_pickle_path(data_filename, daily, met_stn)
    if not os.path.isfile(path_to_pickle) or update:
        prep_midas_radtob(daily, met_stn)
    try:
        radtob = load_pickle(path_to_pickle)
        return radtob
    except Exception as e:
        print(e)
