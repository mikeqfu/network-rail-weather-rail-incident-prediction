""" Met Office RADTOB (Radiation values currently being reported) """

import gc
import glob
import os
import tempfile
import zipfile

import natsort
import pandas as pd
import shapely.geometry
import sqlalchemy.types
from pyhelpers import wgs84_to_osgb36, load_pickle, save_pickle

from mssqlserver.tools import create_mssql_connectable_engine
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
            rad_stations_info.StationName = rad_stations_info.StationName.str.replace(r'(\xa0)+Locate', '', regex=True)

            rad_stations_info['Easting'], rad_stations_info['Northing'] = wgs84_to_osgb36(
                rad_stations_info.Longitude.values, rad_stations_info.Latitude.values)
            rad_stations_info['EN_GEOM'] = [
                shapely.geometry.Point(xy) for xy in zip(rad_stations_info.Easting, rad_stations_info.Northing)]

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


def make_midas_radtob_pickle_path(data_filename, daily, rad_stn):
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


def get_midas_radtob(data_filename="midas-radtob-2006-2019", daily=False, update=False, verbose=False):
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

        from weather.midas import get_midas_radtob

        data_filename = "midas-radtob-2006-2019"
        daily = False
        update = True
        verbose = True

        radtob = get_midas_radtob(data_filename, daily, update, verbose)
    """

    path_to_pickle = make_midas_radtob_pickle_path(data_filename, daily, rad_stn=False)

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

            return radtob

        except Exception as e:
            print("Failed to get the radiation observations. {}".format(e))


def dump_midas_radtob_to_mssql(table_name='MIDAS_RADTOB', if_exists='append', chunk_size=100000, update=False,
                               verbose=False):
    """
    See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

    :param table_name:
    :param if_exists:
    :param chunk_size:
    :param update:
    :param verbose:
    :return:

    **Example**::

        table_name = 'MIDAS_RADTOB'
        if_exists = 'append'
        chunk_size = 100000
        update = False
        verbose = False
    """

    midas_radtob = get_midas_radtob(update=update, verbose=verbose)
    midas_radtob.reset_index(inplace=True)

    midas_radtob_engine = create_mssql_connectable_engine(database_name='Weather')

    print("Importing MIDAS RADTOB data to MSSQL Server", end=" ... ")

    temp_file = tempfile.NamedTemporaryFile()
    csv_filename = temp_file.name + ".csv"
    midas_radtob.to_csv(csv_filename, index=False, chunksize=chunk_size)

    tsql_chunksize = 2097 // len(midas_radtob.columns)
    temp_file_ = pd.read_csv(csv_filename, chunksize=tsql_chunksize)
    for chunk in temp_file_:
        # e.g. chunk = temp_file_.get_chunk(tsql_chunksize)
        chunk.to_sql(table_name, midas_radtob_engine, schema='dbo', if_exists=if_exists, index=False,
                     dtype={'OB_END_DATE_TIME': sqlalchemy.types.DATETIME, 'OB_END_DATE': sqlalchemy.types.DATE},
                     method='multi')
        gc.collect()

    temp_file.close()

    os.remove(csv_filename)

    print("Done. ")


def process_midas_supplement(update=False, verbose=False):
    path_to_pickle = cdd_weather("midas\\suppl", "suppl-dat.pickle")

    if os.path.isfile(path_to_pickle) and not update:
        supplement_data = load_pickle(path_to_pickle)

    else:
        rad_stations_info = get_radiation_stations_information()

        try:
            suppl_dat = []
            for f in glob.glob(cdd_weather("midas\\suppl", "*.csv")):
                dat = pd.read_csv(f, names=['OB_END_DATE', 'GLBL_IRAD_AMT'], skiprows=1)

                filename = os.path.basename(f)

                src_id = (rad_stations_info.StationName == filename.split('_')[0].upper()).idxmax()
                dat.insert(0, 'SRC_ID', src_id)
                dat.insert(2, 'OB_HOUR_COUNT', 24)

                if 'Wattisham' in filename:
                    dat['Route'] = 'Anglia'
                if 'Hurn' in filename:
                    dat['Route'] = 'Wessex'
                if 'Valley' in filename:
                    dat['Route'] = 'Wales'
                if 'Durham' in filename:
                    dat['Route'] = 'North and East'

                suppl_dat.append(dat)

            supplement_data = pd.concat(suppl_dat)

            save_pickle(supplement_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print(e)
            supplement_data = None

    return supplement_data


def dump_midas_supplement_to_mssql(table_name='MIDAS_RADTOB_suppl', if_exists='replace', update=False, verbose=False):
    """

    :param table_name:
    :param if_exists:
    :param update:
    :param verbose:

    **Example**::

        table_name = 'MIDAS_RADTOB_suppl'
        if_exists = 'replace'
        update = True
        verbose = True
    """

    supplement_data = process_midas_supplement(update=update, verbose=verbose)

    if supplement_data is not None and not supplement_data.empty:
        midas_radtob_engine = create_mssql_connectable_engine(database_name='Weather')
        supplement_data.to_sql(table_name, midas_radtob_engine, schema='dbo', if_exists=if_exists, index=False,
                               dtype={'OB_END_DATE': sqlalchemy.types.DATE})


def query_midas_radtob_by_grid_datetime(met_stn_id, period, route_name, use_suppl_dat=False, update=False, dat_dir=None,
                                        pickle_it=False, verbose=False):
    """
    Get MIDAS RADTOB (Radiation data) by met station ID (Query from the database) for the given ``period``.

    :param met_stn_id: met station ID
    :type met_stn_id: list
    :param period: prior-incident / non-incident period
    :type period:
    :param route_name: name of Route
    :type route_name: str
    :param use_suppl_dat: defaults to ``False``
    :type use_suppl_dat: bool
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

        from weather.midas import query_midas_radtob_by_grid_datetime

        update = False
        verbose = True
        dat_dir = None

        pickle_it = False
        met_stn_id = incidents.Met_SRC_ID.iloc[0]
        period = incidents.Critical_Period.iloc[0]
        route_name = incidents.Route.iloc[0]
        midas_radtob = query_midas_radtob_by_grid_datetime(met_stn_id, period, verbose=verbose)

        pickle_it = True
        met_stn_id = incidents.Met_SRC_ID.iloc[3]
        period = incidents.Critical_Period.iloc[3]
        route_name = incidents.Route.iloc[3]
        midas_radtob = query_midas_radtob_by_grid_datetime(met_stn_id, period, pickle_it=pickle_it, verbose=verbose)
    """

    p_start, p_end = period.left.min().strftime('%Y%m%d%H'), period.right.max().strftime('%Y%m%d%H')

    # Make a pickle filename
    pickle_filename = "{}-{}.pickle".format(str(met_stn_id[0]), "-".join([p_start, p_end]))

    # Specify a directory/path to store the pickle file (if appropriate)
    dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else cdd_weather("midas", "dat")
    path_to_pickle = cdd_weather(dat_dir, pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        midas_radtob = load_pickle(path_to_pickle)

    else:
        # Create an engine to the MSSQL server
        conn_metex = create_mssql_connectable_engine(database_name='Weather')
        # Specify database sql query
        ms_id = met_stn_id[0]
        dates = tuple(x.strftime('%Y-%m-%d %H:%M:%S') for x in [period.left.min(), period.right.max()])
        sql_query = "SELECT * FROM dbo.[MIDAS_RADTOB] " \
                    "WHERE [SRC_ID] = {} " \
                    "AND [OB_END_DATE_TIME] BETWEEN '{}' AND '{}';".format(ms_id, dates[0], dates[1])
        # Query the weather data
        midas_radtob = pd.read_sql(sql_query, conn_metex)

        if midas_radtob.empty and use_suppl_dat:
            dates = tuple(x.strftime('%Y-%m-%d') for x in [period.left.min(), period.right.max()])
            sql_query = "SELECT * FROM dbo.[MIDAS_RADTOB_suppl] " \
                        "WHERE [Route] = '{}' " \
                        "AND [OB_END_DATE] BETWEEN '{}' AND '{}';".format(route_name, dates[0], dates[1])
            midas_radtob = pd.read_sql(sql_query, conn_metex)

        if pickle_it:
            save_pickle(midas_radtob, path_to_pickle, verbose=verbose)

    return midas_radtob
