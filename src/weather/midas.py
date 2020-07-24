""" Met Office RADTOB (Radiation values currently being reported) """

import gc
import os
import zipfile

import natsort
import numpy as np
import pandas as pd
import shapely.geometry
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.store import load_pickle, save_pickle

from settings import pd_preferences
from utils import cdd_weather

pd_preferences()


def get_radiation_stations_information(update=False, verbose=False):
    """
    Get locations and relevant information of meteorological stations.

    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of meteorological stations
    :rtype: pandas.DataFrame

    **Example**::

        from weather.midas import get_radiation_stations_information

        update = True
        verbose = True

        rad_stations_info = get_radiation_stations_information(update, verbose)
    """

    filename = "radiation-stations-information"
    path_to_pickle = cdd_weather("midas", filename + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        rad_stations_info = load_pickle(path_to_pickle)
        return rad_stations_info

    else:
        try:
            path_to_spreadsheet = path_to_pickle.replace(".pickle", ".xlsx")
            rad_stations_info = pd.read_excel(path_to_spreadsheet, parse_dates=['Station start date'])
            rad_stations_info.columns = [x.title().replace(' ', '') if x != 'src_id' else x.upper()
                                         for x in rad_stations_info.columns]
            rad_stations_info.StationName = rad_stations_info.StationName.str.replace('\xa0\xa0\xa0\xa0Locate', '')

            osgb_en = np.array(wgs84_to_osgb36(rad_stations_info.Longitude.values, rad_stations_info.Latitude.values))
            rad_stations_info[['Easting', 'Northing']] = pd.DataFrame(osgb_en.T)
            rad_stations_info['E_N_GEOM'] = [shapely.geometry.Point(xy)
                                             for xy in zip(rad_stations_info.Easting, rad_stations_info.Northing)]

            rad_stations_info.sort_values(['SRC_ID'], inplace=True)
            rad_stations_info.set_index('SRC_ID', inplace=True)

            save_pickle(rad_stations_info, path_to_pickle, verbose=verbose)

            return rad_stations_info

        except Exception as e:
            print("Failed to get the locations of the meteorological stations. {}".format(e))


def parse_midas_radtob(filename, headers, daily=False, rad_stn=False):
    """
    Parse original MIDAS RADTOB (Radiation data).

    MIDAS  - Met Office Integrated Data Archive System
    RADTOB - RADT-OB table. Radiation values currently being reported

    :param filename: e.g. filename = "midas_radtob_200601-200612.txt"
    :type filename: str
    :param headers: column names of the data frame
    :type headers: list
    :param daily: if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :type daily: bool
    :param rad_stn: if True, add the location of meteorological station; False (default)
    :type rad_stn: bool

    .. note::

        SRC_ID:        Unique source identifier or station site number
        OB_END_TIME:   Date and time at end of observation
        OB_HOUR_COUNT: Observation hour count
        VERSION_NUM:   Observation version number - Use the row with '1',
                       which has been quality checked by the Met Office
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


def make_radtob_pickle_path(data_filename, daily, rad_stn):
    """
    Make a full path to the pickle file of radiation data.

    :param data_filename: e.g. data_filename="midas-radtob-20060101-20141231"
    :type data_filename: str
    :param daily: e.g. daily=False
    :type daily: bool
    :param rad_stn: e.g. met_stn=False
    :type rad_stn: bool
    :return: a full path to the pickle file of radiation data
    :rtype: str
    """

    filename_suffix = "-agg" if daily else "", "-met_stn" if rad_stn else ""
    path_to_radtob_pickle = cdd_weather("MIDAS", "{}{}.pickle".format(data_filename, *filename_suffix))
    return path_to_radtob_pickle


def get_radtob(data_filename="midas-radtob-2006-2019", daily=False, update=False, verbose=False):
    """
    Get MIDAS RADTOB (Radiation data).

    :param data_filename: defaults to ``"midas-radtob-2006-2019"``
    :type data_filename: str
    :param daily: if True, 'OB_HOUR_COUNT' == 24, i.e. aggregate value in one day 24 hours; False (default)
    :type daily: bool
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: MIDAS RADTOB (Radiation data)
    :rtype: pandas.DataFrame

    **Example**::

        from weather.midas import get_radtob

        data_filename = "midas-radtob-2006-2019"
        daily = False
        update = True
        verbose = True

        radtob = get_radtob(data_filename, daily, update, verbose)
    """

    path_to_pickle = make_radtob_pickle_path(data_filename, daily, rad_stn=False)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            # Headers of the midas_radtob data set
            headers_raw = pd.read_excel(cdd_weather("midas", "radiation-observation-data-headers.xlsx"), header=None)
            headers = [x.strip() for x in headers_raw.iloc[0, :].values]

            path_to_zip = cdd_weather("midas", data_filename + ".zip")
            with zipfile.ZipFile(path_to_zip, 'r') as zf:
                filename_list = natsort.natsorted(zf.namelist())
                temp_dat = [parse_midas_radtob(zf.open(f), headers, daily, rad_stn=False) for f in filename_list]
            zf.close()

            radtob = pd.concat(temp_dat, axis=0, ignore_index=True, sort=False)
            radtob.set_index(['SRC_ID', 'OB_END_DATE_TIME'], inplace=True)

            # Save data as a pickle
            save_pickle(radtob, path_to_pickle, verbose=verbose)

            gc.collect()

            return radtob

        except Exception as e:
            print("Failed to get the radiation observations. {}".format(e))
