""" Met Office RADTOB (Radiation values currently being reported) """

import gc
import os
import zipfile

import natsort
import numpy as np
import pandas as pd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.store import load_pickle, save_pickle

import settings
from weather.tools import cdd_weather

settings.pd_preferences()


# Locations of the meteorological stations ---------------------------------------------------------------------------
def get_radiation_stations_information(update=False) -> pd.DataFrame:
    filename = "radiation-stations-information"
    path_to_pickle = cdd_weather("MIDAS", filename + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)
    else:
        try:
            path_to_spreadsheet = cdd_weather("MIDAS", filename + ".xlsx")
            rad_stations_info = pd.read_excel(path_to_spreadsheet, parse_dates=['Station start date'])
            rad_stations_info.columns = [x.title().replace(' ', '') if x != 'src_id' else x.upper()
                                         for x in rad_stations_info.columns]
            rad_stations_info.StationName = rad_stations_info.StationName.str.replace('\xa0\xa0\xa0\xa0Locate', '')

            osgb_en = np.array(wgs84_to_osgb36(rad_stations_info.Longitude.values, rad_stations_info.Latitude.values))
            rad_stations_info[['Easting', 'Northing']] = pd.DataFrame(osgb_en.T)

            rad_stations_info.sort_values(['SRC_ID'], inplace=True)
            rad_stations_info.set_index('SRC_ID', inplace=True)

            save_pickle(rad_stations_info, path_to_pickle.replace(".xlsx", ".pickle"))

            return rad_stations_info

        except Exception as e:
            print("Failed to get the locations of the meteorological stations. {}".format(e))


# MIDAS RADTOB (Radiation data) --------------------------------------------------------------------------------------
def parse_midas_radtob(filename: str, headers: list, daily=False, rad_stn=False):
    """
    MIDAS  - Met Office Integrated Data Archive System
    RADTOB - RADT-OB table. Radiation values currently being reported

    :param filename: [str] e.g. filename = "midas_radtob_200601-200612.txt"
    :param headers: [list]
    :param daily: [bool] if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :param rad_stn: [bool] if True, add the location of meteorological station; False (default)

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

    # Cleanse the data
    key_cols = ['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT']

    checked_tmp = ro_data.groupby(key_cols).agg({'VERSION_NUM': max})
    checked_tmp.reset_index(inplace=True)
    key_cols_ = key_cols + ['VERSION_NUM']
    ro_data_ = checked_tmp.join(ro_data.set_index(key_cols_), on=key_cols_)

    # Note: The following line is questionable
    ro_data_.loc[(ro_data_.GLBL_IRAD_AMT < 0.0) | ro_data_.GLBL_IRAD_AMT.isna(), 'GLBL_IRAD_AMT'] = 0.0  # or np.nan

    ro_data_.sort_values(['SRC_ID', 'OB_END_TIME'], inplace=True)  # Sort rows
    ro_data_.index = range(len(ro_data_))

    # Insert 'OB_END_DATE'
    ro_data_.insert(ro_data_.columns.get_loc('OB_END_TIME'), column='OB_END_DATE', value=ro_data_.OB_END_TIME.dt.date)

    ro_data_.rename(columns={'OB_END_TIME': 'OB_END_DATE_TIME'}, inplace=True)  # Rename 'OB_END_TIME'

    if rad_stn:
        rad_stn_info = get_radiation_stations_information()
        ro_data_ = ro_data_.join(rad_stn_info, on='SRC_ID')

    return ro_data_


def make_radtob_pickle_path(data_filename: str, daily: bool, rad_stn: bool):
    """
    :param data_filename: e.g. data_filename="midas-radtob-20060101-20141231"
    :param daily: e.g. daily=False
    :param rad_stn: e.g. met_stn=False
    :return: [str]
    """
    filename_suffix = "-agg" if daily else "", "-met_stn" if rad_stn else ""
    path_to_radtob_pickle = cdd_weather("MIDAS", "{}{}.pickle".format(data_filename, *filename_suffix))
    return path_to_radtob_pickle


def get_radtob(data_filename="midas-radtob-2006-2019", daily=False, update=False) -> pd.DataFrame:
    """
    :param data_filename: [str]
    :param daily: [bool] if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :param update: [bool]

    Testing parameters:
    e.g.
        data_filename="midas-radtob-2006-2019"
        daily=False
        update=False
    """
    path_to_pickle = make_radtob_pickle_path(data_filename, daily, rad_stn=False)
    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)
    else:
        try:
            # Headers of the midas_radtob data set
            headers_raw = pd.read_excel(cdd_weather("MIDAS", "radiation-observation-data-headers.xlsx"), header=None)
            headers = [x.strip() for x in headers_raw.iloc[0, :].values]

            path_to_zip = cdd_weather("MIDAS", data_filename + ".zip")
            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                temp_dat = [parse_midas_radtob(zf.open(f), headers, daily, rad_stn=False) for f in filename_list]
            zf.close()

            radtob = pd.concat(temp_dat, axis=0, ignore_index=True, sort=False)
            radtob.set_index(['SRC_ID', 'OB_END_DATE_TIME'], inplace=True)

            # Save data as a pickle
            save_pickle(radtob, path_to_pickle)

            gc.collect()

            return radtob

        except Exception as e:
            print("Failed to get the radiation observations. {}".format(e))
