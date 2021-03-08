"""
Weather
"""

import gc
import glob
import os
import tempfile
import zipfile

import datetime_truncate
import natsort
import pandas as pd
import shapely.geometry
import shapely.ops
import shapely.wkt
import sqlalchemy.types
from pyhelpers.dir import cd, validate_input_data_dir
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.store import load_pickle, save_pickle

from utils import cdd_weather, establish_mssql_connection


class MIDAS:
    """
    Met Office RADTOB.

    :param database_name: name of the database, defaults to ``'Weather'``
    :type database_name: str

    :ivar str Name: name of the data resource
    :ivar str Acronym: acronym of the data resource name
    :ivar str RadStnInfoFilename: filename of the radiation stations information
    :ivar str RadtobFilename: filename of the radiation observation data
    :ivar str HeadersFilename: filename of the headers for the radiation observation data
    :ivar sqlalchemy.engine.Connection DatabaseConn: connection to the database
    :ivar str SchemaName: name of the schema for storing the radiation observation data
    :ivar str RadtobTblName: name of the table for storing the radiation observation data
    :ivar str RadtobSupplTblName: name of the table for storing supplementary data

    **Test**::

        >>> from preprocessor import MIDAS

        >>> midas = MIDAS()

        >>> midas.Name
        'Met Office RADTOB (Radiation values currently being reported).'
    """

    def __init__(self, database_name='Weather'):
        self.Name = 'Met Office RADTOB (Radiation values currently being reported).'
        self.Acronym = 'MIDAS'

        self.RadStnInfoFilename = "radiation-stations-information"
        self.RadtobFilename = "midas-radtob-2006-2019"
        self.HeadersFilename = "radiation-observation-data-headers"

        # Create an engine to the MSSQL server
        self.DatabaseConn = establish_mssql_connection(database_name=database_name)

        self.SchemaName = self.Acronym
        self.RadtobTblName = 'RADTOB'
        self.RadtobSupplTblName = self.RadtobTblName + '_suppl'

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\weather\\midas" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data\\weather\\midas"`` and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> os.path.relpath(midas.cdd())
            'data\\weather\\midas'
        """

        path = cdd_weather(self.Acronym.lower(), *sub_dir, mkdir=mkdir)

        return path

    def get_radiation_stations(self, update=False, verbose=False):
        """
        Get locations and relevant information of meteorological stations.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of meteorological stations
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> # dat = midas.get_radiation_stations(update=True, verbose=True)
            >>> dat = midas.get_radiation_stations()

            >>> dat.tail()
                                    StationName  ...                                      EN_GEOM
            SRC_ID                               ...
            61986             TIBENHAM AIRFIELD  ...  POINT (615070.0334468307 288957.3690573045)
            62034   ROUDSEA WOOD AND MOSSES NNR  ...  POINT (333410.2674705166 482772.0753328541)
            62041           EXETER AIRPORT NO 2  ...  POINT (300979.5951061826 93939.67916976719)
            62122                   ALMONDSBURY  ...  POINT (361406.5017112559 183704.4311121872)
            62139                   WISLEY NO 2  ...   POINT (506466.6675712555 157848.166606131)
            [5 rows x 11 columns]
        """

        path_to_pickle = self.cdd(self.RadStnInfoFilename + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            rad_stn = load_pickle(path_to_pickle)
            return rad_stn

        else:
            try:
                path_to_spreadsheet = path_to_pickle.replace(".pickle", ".xlsx")
                rad_stn = pd.read_excel(path_to_spreadsheet, parse_dates=['Station start date'])
                rad_stn.columns = [
                    x.title().replace(' ', '') if x != 'src_id' else x.upper() for x in rad_stn.columns]
                rad_stn.StationName = rad_stn.StationName.str.replace(r'(\xa0)+Locate', '', regex=True)

                rad_stn['Easting'], rad_stn['Northing'] = wgs84_to_osgb36(
                    rad_stn.Longitude.values, rad_stn.Latitude.values)
                rad_stn['EN_GEOM'] = [
                    shapely.geometry.Point(xy) for xy in zip(rad_stn.Easting, rad_stn.Northing)]

                rad_stn.sort_values(['SRC_ID'], inplace=True)
                rad_stn.set_index('SRC_ID', inplace=True)

                save_pickle(rad_stn, path_to_pickle, verbose=verbose)

                return rad_stn

            except Exception as e:
                print("Failed to get the locations of the meteorological stations. {}".format(e))

    def parse_radtob(self, file, headers, daily=False, rad_stn=False):
        """
        Parse original MIDAS RADTOB (Radiation data).

        MIDAS  - Met Office Integrated Data Archive System
        RADTOB - RADT-OB table. Radiation values currently being reported

        :param file: e.g. ``"midas_radtob_200601-200612.txt"``
        :type file: str or typing.IO[bytes]
        :param headers: column names of the data frame
        :type headers: list
        :param daily: if ``True``, ``'OB_HOUR_COUNT'`` equals ``24``,
            i.e. aggregate value in one day 24 hours; defaults to ``False``
        :type daily: bool
        :param rad_stn: if ``True``, add the location of meteorological station;
            defaults to ``False``
        :type rad_stn: bool
        :return: data of MIDAS RADTOB
        :rtype: pandas.DataFrame

        .. note::

            - SRC_ID:        Unique source identifier or station site number
            - OB_END_TIME:   Date and time at end of observation
            - OB_HOUR_COUNT: Observation hour count
            - VERSION_NUM:   Observation version number - Use the row with '1',
                             which has been quality checked by the Met Office
            - GLBL_IRAD_AMT: Global solar irradiation amount Kjoules/sq metre
                             over the observation period

        **Test**::

            >>> import zipfile
            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> zf = zipfile.ZipFile(midas.cdd(midas.RadtobFilename + ".zip"))
            >>> example_file = zf.open(zf.filelist[0])
            >>> zf.close()

            >>> col_names = midas.get_radtob_headers()

            >>> dat = midas.parse_radtob(file=example_file, headers=col_names)

            >>> print(dat)
                    SRC_ID OB_END_DATE  ... VERSION_NUM  GLBL_IRAD_AMT
            0            9  2006-01-01  ...           1            0.0
            1            9  2006-01-01  ...           1            0.0
            2            9  2006-01-01  ...           1            0.0
            3            9  2006-01-01  ...           1            0.0
            4            9  2006-01-01  ...           1            0.0
                    ...         ...  ...         ...            ...
            744241   55511  2006-12-31  ...           1            0.0
            744242   55511  2006-12-31  ...           1            0.0
            744243   55511  2006-12-31  ...           1            0.0
            744244   55511  2006-12-31  ...           1            0.0
            744245   55511  2006-12-31  ...           1         1870.0
            [744246 rows x 6 columns]
        """

        raw_txt = pd.read_csv(file, header=None, names=headers, parse_dates=[2, 12],
                              infer_datetime_format=True, skipinitialspace=True)

        selected_feat = ['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT', 'VERSION_NUM', 'GLBL_IRAD_AMT']
        ro_data = raw_txt[selected_feat].drop_duplicates()

        if daily:
            ro_data = ro_data[ro_data.OB_HOUR_COUNT == 24]
            ro_data.index = range(len(ro_data))

        # Cleanse the data
        key_cols = ['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT']

        checked_tmp = ro_data.groupby(key_cols).agg({'VERSION_NUM': max}).reset_index()
        key_cols_ = key_cols + ['VERSION_NUM']
        ro_data_ = checked_tmp.join(ro_data.set_index(key_cols_), on=key_cols_)

        # Note: The following line is questionable
        ro_data_.loc[
            (ro_data_.GLBL_IRAD_AMT < 0.0) | ro_data_.GLBL_IRAD_AMT.isna(), 'GLBL_IRAD_AMT'] = 0.0
        # or np.nan

        ro_data_.sort_values(['SRC_ID', 'OB_END_TIME'], inplace=True)  # Sort rows
        ro_data_.index = range(len(ro_data_))

        # Insert 'OB_END_DATE'
        ro_data_.insert(ro_data_.columns.get_loc('OB_END_TIME'), column='OB_END_DATE',
                        value=ro_data_.OB_END_TIME.dt.date)

        # Rename 'OB_END_TIME'
        ro_data_.rename(columns={'OB_END_TIME': 'OB_END_DATE_TIME'}, inplace=True)

        if rad_stn:
            rad_stn_info = self.get_radiation_stations()
            ro_data_ = ro_data_.join(rad_stn_info, on='SRC_ID')

        return ro_data_

    def make_radtob_pickle_path(self, filename, daily, rad_stn):
        """
        Make a full path to the pickle file of radiation data.

        :param filename: e.g. ``"midas-radtob-20060101-20141231"``
        :type filename: str
        :param daily: e.g. ``False``
        :type daily: bool
        :param rad_stn: e.g. met_stn=False
        :type rad_stn: bool
        :return: a full path to the pickle file of radiation data
        :rtype: str

        **Test**::

            >>> import os
            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> fn = "midas-radtob-20060101-20141231"
            >>> pf = midas.make_radtob_pickle_path(filename=fn, daily=False, rad_stn=False)

            >>> os.path.relpath(pf)
            data\\weather\\midas\\midas-radtob-20060101-20141231.pickle
        """

        filename_suffix = "-agg" if daily else "", "-met_stn" if rad_stn else ""

        pickle_filename = "{}{}.pickle".format(filename, *filename_suffix)

        path_to_radtob_pickle = self.cdd(pickle_filename)

        return path_to_radtob_pickle

    def get_radtob_headers(self):
        """
        Get a list of column names for RADTOB data.

        :return: list of column names
        :rtype: list

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> col_names = midas.get_radtob_headers()

            >>> col_names[:5]
            ['ID', 'ID_TYPE', 'OB_END_TIME', 'OB_HOUR_COUNT', 'VERSION_NUM']
        """
        # Headers of the midas_radtob data set
        headers_raw = pd.read_excel(self.cdd(self.HeadersFilename + ".xlsx"), header=None)

        headers = [x.strip() for x in headers_raw.iloc[0, :].values]

        return headers

    def get_radtob(self, daily=False, update=False, verbose=False):
        """
        Get MIDAS RADTOB (Radiation data).

        :param daily: if ``True``, ``'OB_HOUR_COUNT'`` equals ``24``,
            i.e. aggregate value in one day 24 hours; defaults to ``False``
        :type daily: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: MIDAS RADTOB (Radiation data)
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> # dat = midas.get_radtob(update=True, verbose=True)
            >>> # Updating "midas-radtob-2006-2019.pickle" at "data\\weather\\midas" ... Done.

            >>> dat = midas.get_radtob()

            >>> dat.tail()
                                       OB_END_DATE  ...  GLBL_IRAD_AMT
            SRC_ID OB_END_DATE_TIME                 ...
            62139  2019-06-30 20:00:00  2019-06-30  ...          379.0
                   2019-06-30 21:00:00  2019-06-30  ...           26.0
                   2019-06-30 22:00:00  2019-06-30  ...            0.0
                   2019-06-30 23:00:00  2019-06-30  ...            0.0
                   2019-06-30 23:59:00  2019-06-30  ...        20900.0
            [5 rows x 4 columns]
        """

        path_to_pickle = self.make_radtob_pickle_path(self.RadtobFilename, daily, rad_stn=False)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                headers = self.get_radtob_headers()

                path_to_zip = self.cdd(self.RadtobFilename + ".zip")
                with zipfile.ZipFile(path_to_zip, 'r') as zf:
                    filename_list = natsort.natsorted(zf.namelist())
                    temp_dat = [self.parse_radtob(zf.open(f), headers, daily, rad_stn=False)
                                for f in filename_list]
                zf.close()

                radtob = pd.concat(temp_dat, axis=0, ignore_index=True, sort=False)
                radtob.set_index(['SRC_ID', 'OB_END_DATE_TIME'], inplace=True)

                # Save data as a pickle
                save_pickle(radtob, path_to_pickle, verbose=verbose)

                return radtob

            except Exception as e:
                print("Failed to get the radiation observations. {}".format(e))

    def import_radtob(self, if_exists='fail', chunk_size=100000, update=False, verbose=False):
        """
        Import the radiation data.

        See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

        :param if_exists: whether to replace, append or raise an error if the database already exists
        :type if_exists: str
        :param chunk_size: size of a chunk to import
        :type chunk_size: int or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> # midas.import_radtob(if_exists='replace', chunk_size=100000, verbose=True)
        """

        midas_radtob = self.get_radtob(update=update, verbose=verbose)
        midas_radtob.reset_index(inplace=True)

        print("Importing MIDAS RADTOB data to MSSQL Server", end=" ... ")

        temp_file = tempfile.NamedTemporaryFile()
        csv_filename = temp_file.name + ".csv"
        midas_radtob.to_csv(csv_filename, index=False, chunksize=chunk_size)

        tsql_chunksize = 2097 // len(midas_radtob.columns)
        temp_file_ = pd.read_csv(csv_filename, chunksize=tsql_chunksize)
        for chunk in temp_file_:
            # e.g. chunk = temp_file_.get_chunk(tsql_chunksize)
            dtype_ = {'OB_END_DATE_TIME': sqlalchemy.types.DATETIME,
                      'OB_END_DATE': sqlalchemy.types.DATE}

            chunk.to_sql(con=self.DatabaseConn, schema='dbo', name=self.RadtobTblName,
                         if_exists=if_exists, index=False, dtype=dtype_, method='multi')

            del chunk
            gc.collect()

        temp_file.close()

        os.remove(csv_filename)

        print("Done. ")

    def process_suppl_dat(self, update=False, verbose=False):
        """
        Parse supplementary data.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: parsed supplementary data
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> suppl_data = midas.process_suppl_dat()

            >>> print(suppl_data)
                  SRC_ID  OB_END_DATE  OB_HOUR_COUNT  GLBL_IRAD_AMT           Route
            0        326   2001-01-01             24            864  North and East
            1        326   2001-01-01             24            864  North and East
            2        326   2001-01-02             24            253  North and East
            3        326   2001-01-03             24           2690  North and East
            4        326   2001-01-04             24            670  North and East
                  ...          ...            ...            ...             ...
            6867     440   2020-06-26             24          28925          Anglia
            6868     440   2020-06-27             24          14716          Anglia
            6869     440   2020-06-28             24          24003          Anglia
            6870     440   2020-06-29             24          16254          Anglia
            6871     440   2020-06-30             24          13861          Anglia
            [20783 rows x 5 columns]
        """

        path_to_pickle = self.cdd("suppl", "suppl-dat.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            supplement_data = load_pickle(path_to_pickle)

        else:
            rad_stations_info = self.get_radiation_stations()

            try:
                suppl_dat = []
                for f in glob.glob(self.cdd("suppl", "*.csv")):
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

    def import_suppl_dat(self, if_exists='fail', update=False, verbose=False):
        """
        Import the supplementary data.

        :param if_exists: whether to replace, append or raise an error if the database already exists
        :type if_exists: str
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> midas.import_suppl_dat(verbose=True)
        """

        supplement_data = self.process_suppl_dat(update=update, verbose=verbose)

        if supplement_data is not None and not supplement_data.empty:
            supplement_data.to_sql(con=self.DatabaseConn,
                                   name=self.RadtobSupplTblName, schema='dbo',
                                   if_exists=if_exists,
                                   index=False,
                                   dtype={'OB_END_DATE': sqlalchemy.types.DATE})

    def query_radtob_by_grid_datetime(self, met_stn_id, period, route_name, use_suppl_dat=False,
                                      update=False, dat_dir=None, pickle_it=False, verbose=False):
        """
        Query (from database) MIDAS RADTOB (Radiation data) by met station ID for the given ``period``.

        :param met_stn_id: met station ID
        :type met_stn_id: list
        :param period: prior-incident / non-incident period
        :type period:
        :param route_name: name of Route
        :type route_name: str
        :param use_suppl_dat: defaults to ``False``
        :type use_suppl_dat: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str, None
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor.weather import MIDAS

            >>> midas = MIDAS()

            >>> incidents = pd.DataFrame()

            >>> m_stn_id = incidents.Met_SRC_ID.iloc[0]
            >>> p = incidents.Critical_Period.iloc[0]
            >>> rte = incidents.Route.iloc[0]
            >>> dat = midas.query_radtob_by_grid_datetime(m_stn_id, p, rte, verbose=True)

            >>> m_stn_id = incidents.Met_SRC_ID.iloc[3]
            >>> p = incidents.Critical_Period.iloc[3]
            >>> rte = incidents.Route.iloc[3]
            >>> dat = midas.query_radtob_by_grid_datetime(m_stn_id, p, rte, pickle_it=True, verbose=True)
        """

        p_start = period.left.min().strftime('%Y%m%d%H')
        p_end = period.right.max().strftime('%Y%m%d%H')

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(str(met_stn_id[0]), "-".join([p_start, p_end]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if dat_dir is None:
            dat_dir_ = self.cdd("dat")
        else:
            dat_dir_ = validate_input_data_dir(dat_dir)
        path_to_pickle = cd(dat_dir_, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            midas_radtob = load_pickle(path_to_pickle)

        else:
            # Specify database sql query
            ms_id = met_stn_id[0]
            dates = tuple(
                x.strftime('%Y-%m-%d %H:%M:%S') for x in [period.left.min(), period.right.max()])

            sql_query = \
                f"SELECT * FROM dbo.[MIDAS_RADTOB] " \
                f"WHERE [SRC_ID] = {ms_id} " \
                f"AND [OB_END_DATE_TIME] BETWEEN '{dates[0]}' AND '{dates[1]}';"

            midas_radtob = pd.read_sql(sql=sql_query, con=self.DatabaseConn)  # Query the weather data

            if midas_radtob.empty and use_suppl_dat:
                dates = tuple(
                    x.strftime('%Y-%m-%d') for x in [period.left.min(), period.right.max()])

                sql_query = \
                    f"SELECT * FROM dbo.[MIDAS_RADTOB_suppl] " \
                    f"WHERE [Route] = '{route_name}' " \
                    f"AND [OB_END_DATE] BETWEEN '{dates[0]}' AND '{dates[1]}';"

                midas_radtob = pd.read_sql(sql=sql_query, con=self.DatabaseConn)

            if pickle_it:
                save_pickle(midas_radtob, path_to_pickle, verbose=verbose)

        return midas_radtob


class UKCP09:
    """
    UKCP09 gridded weather observations.

    :param start_date: start date on which the observation data was collected, formatted as 'yyyy-mm-dd'
    :type start_date: str
    :param database_name: name of the database, defaults to ``'Weather'``
    :type database_name: str

    :ivar str Name:
    :ivar str Acronym:
    :ivar str Description:
    :ivar str StartDate: (specified with the creation of the instance)
    :ivar sqlalchemy.engine.Connection DatabaseConn: connection to the database

    **Test**::

        >>> from preprocessor.weather import UKCP09

        >>> ukcp = UKCP09()

        >>> print(ukcp.Name)
        UK Climate Projections
    """

    def __init__(self, start_date='2006-01-01', database_name='Weather'):
        self.Name = 'UK Climate Projections'
        self.Acronym = 'UKCP09'
        self.Description = 'UKCP09 gridded weather observations: ' \
                           'maximum temperature, minimum temperature and precipitation.'

        self.StartDate = start_date

        # Create an engine to the MSSQL server
        self.DatabaseConn = establish_mssql_connection(database_name=database_name)

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "data\\weather\\ukcp" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data\\weather\\ukcp"`` and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            >>> os.path.relpath(ukcp.cdd())
            'data\\weather\\ukcp09'
        """

        path = cdd_weather("ukcp", *sub_dir, mkdir=mkdir)

        return path

    @staticmethod
    def create_grid(centre_point, side_length=5000, rotation=None):
        """
        Find coordinates for each corner of the Weather observation grid.

        :param centre_point: (easting, northing)
        :type centre_point: tuple
        :param side_length: side length
        :type side_length: int, float
        :param rotation: e.g. rotation=90; defaults to ``None``
        :type rotation; int, float, None
        :return: coordinates of four corners of the created grid
        :rtype: tuple

        .. note::

            Easting and northing coordinates are commonly measured in metres from the axes of some
            horizontal datum. However, other units (e.g. survey feet) are also used.
        """

        assert isinstance(centre_point, (tuple, list)) and len(centre_point) == 2
        
        x, y = centre_point
        
        if rotation:
            sin_theta, cos_theta = pd.np.sin(rotation), pd.np.cos(rotation)
            lower_left = (x - 1 / 2 * side_length * sin_theta, y - 1 / 2 * side_length * cos_theta)
            upper_left = (x - 1 / 2 * side_length * cos_theta, y + 1 / 2 * side_length * sin_theta)
            upper_right = (x + 1 / 2 * side_length * sin_theta, y + 1 / 2 * side_length * cos_theta)
            lower_right = (x + 1 / 2 * side_length * cos_theta, y - 1 / 2 * side_length * sin_theta)
        
        else:
            lower_left = (x - 1 / 2 * side_length, y - 1 / 2 * side_length)
            upper_left = (x - 1 / 2 * side_length, y + 1 / 2 * side_length)
            upper_right = (x + 1 / 2 * side_length, y + 1 / 2 * side_length)
            lower_right = (x + 1 / 2 * side_length, y - 1 / 2 * side_length)
        
        # corners = shapely.geometry.Polygon([lower_left, upper_left, upper_right, lower_right])
        
        return lower_left, upper_left, upper_right, lower_right

    def parse_observation_grids(self, filename):
        """
        Parse observation grids.

        :param filename: file of the observation grid data
        :type filename: str or typing.IO[bytes]
        :return: parsed data of the observation grids
        :rtype: pandas.DataFrame
        """

        cartesian_centres_temp = pd.read_csv(filename, header=None, index_col=0, nrows=2)
        cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

        grid = [self.create_grid(centre, 5000, rotation=None) for centre in cartesian_centres]

        long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]

        obs_grids = pd.DataFrame({'Centroid': cartesian_centres,
                                  'Centroid_XY': [shapely.geometry.Point(x) for x in cartesian_centres],
                                  'Centroid_LongLat': [shapely.geometry.Point(x) for x in long_lat],
                                  'Grid': [shapely.geometry.Polygon(x) for x in grid]})

        return obs_grids

    def get_observation_grids(self, update=False, verbose=False):
        """
        Fetch data of observation grids from local pickle.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: MIDAS RADTOB observation grids
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            >>> zip_filename = "daily-precipitation.zip"
            >>> obs_grid_dat = ukcp.get_observation_grids(zip_filename, update=True, verbose=True)
        """

        path_to_pickle = self.cdd("observation-grids.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            observation_grids = load_pickle(path_to_pickle)

        else:
            try:
                path_to_zip = self.cdd("daily-precipitation.zip")

                with zipfile.ZipFile(path_to_zip, 'r') as zf:
                    filename_list = natsort.natsorted(zf.namelist())
                    obs_grids = [self.parse_observation_grids(zf.open(f)) for f in filename_list]
                    zf.close()

                observation_grids = pd.concat(obs_grids, ignore_index=True)

                # Add a pseudo id for each observation grid
                observation_grids.sort_values('Centroid', inplace=True)
                observation_grids.index = pd.Index(range(len(observation_grids)), name='Pseudo_Grid_ID')

                path_to_pickle = self.cdd("observation-grids.pickle")
                save_pickle(observation_grids, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Observation Grids\". {}".format(e))
                observation_grids = None

        return observation_grids

    def parse_daily_gridded_weather_obs(self, filename, var_name):
        """
        Parse gridded weather observations from the raw zipped file.

        :param filename: file of raw data
        :type filename: str or typing.IO[bytes]
        :param var_name: variable name,
            e.g. ``'Maximum_Temperature'``, ``'Minimum_Temperature'`` and ``'Precipitation'``
        :type var_name: str
        :return: parsed data of the daily gridded weather observations
        :rtype: pandas.DataFrame
        """

        # Centres
        cartesian_centres_temp = pd.read_csv(filename, header=None, index_col=0, nrows=2)
        cartesian_centres = [tuple(x) for x in cartesian_centres_temp.T.values]

        # Temperature observations
        timeseries_data = pd.read_csv(filename, header=None, skiprows=[0, 1], parse_dates=[0],
                                      dayfirst=True)
        timeseries_data[0] = timeseries_data[0].map(lambda x: x.date())
        if self.StartDate is not None and isinstance(pd.to_datetime(self.StartDate), pd.Timestamp):
            mask = (timeseries_data[0] >= pd.to_datetime(self.StartDate).date())
            timeseries_data = timeseries_data.loc[mask]
        timeseries_data.set_index(0, inplace=True)

        # Reshape the dataframe
        idx = pd.MultiIndex.from_product([cartesian_centres, timeseries_data.index.tolist()],
                                         names=['Centroid', 'Date'])
        data = pd.DataFrame(timeseries_data.T.values.flatten(), index=idx, columns=[var_name])
        # data.reset_index(inplace=True)

        # data.Centre = data.Centre.map(shapely.geometry.asPoint)

        # # Add levels of Grid corners (and centres' LongLat)
        # num = len(timeseries_data)

        # import itertools

        # grid = [find_square_corners(centre, 5000, rotation=None) for centre in cartesian_centres]
        # data['Grid'] = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in grid))

        # long_lat = [osgb36_to_wgs84(x[0], x[1]) for x in cartesian_centres]
        # data['Centroid_LongLat'] = list(
        #     itertools.chain.from_iterable(itertools.repeat(x, num) for x in long_lat))

        return data

    def make_pickle_path(self, filename):
        """
        Make a full path to the pickle file of the UKCP09 data.

        :param filename: e.g. file="daily-maximum-temperature"
        :type filename: str
        :return: a full path to the pickle file of the UKCP09 data
        :rtype: str
        """

        filename_suffix = "" if self.StartDate is None else "-{}".format(self.StartDate.replace("-", ""))
        
        pickle_filename = filename + filename_suffix + ".pickle"
        
        path_to_pickle = cdd_weather("ukcp", pickle_filename)
        
        return path_to_pickle

    def get_obs_data_by_category(self, zip_filename, var_name, use_pseudo_grid_id=False, update=False,
                                 pickle_it=False, verbose=False):
        """
        Get observation data for a given category (i.e. weather variable).

        :param zip_filename: "daily-maximum-temperature", "daily-minimum-temperature",
            or "daily-precipitation"
        :type zip_filename: str
        :param var_name: variable name;
            'Precipitation' or 'Maximum_Temperature', 'Minimum_Temperature'
        :type var_name: str
        :param use_pseudo_grid_id: defaults to ``False``
        :type use_pseudo_grid_id: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of daily gridded weather observations
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            >>> zip_fn = "daily-maximum-temperature"
            >>> vn = 'Maximum_Temperature'

            >>> gridded_obs_dat = ukcp.get_obs_data_by_category(zip_fn, vn)

            >>> gridded_obs_dat.tail()
                                         Maximum_Temperature
            Centroid         Date
            (652500, 302500) 2016-12-27                 7.44
                             2016-12-28                 5.89
                             2016-12-29                 6.08
                             2016-12-30                 4.46
                             2016-12-31                 7.81
        """

        assert isinstance(pd.to_datetime(self.StartDate), pd.Timestamp) or self.StartDate is None

        filename = os.path.splitext(zip_filename)[0]
        path_to_pickle = self.make_pickle_path(filename)

        if os.path.isfile(path_to_pickle) and not update:
            gridded_obs = load_pickle(path_to_pickle)

        else:
            try:
                path_to_zip = self.cdd(zip_filename + ".zip")

                with zipfile.ZipFile(path_to_zip, 'r') as zf:
                    filename_list = natsort.natsorted(zf.namelist())
                    obs_data = [
                        self.parse_daily_gridded_weather_obs(zf.open(f), var_name)
                        for f in filename_list]
                zf.close()

                gridded_obs = pd.concat(obs_data, axis=0)

                # Add a pseudo id for each observation grid
                if use_pseudo_grid_id:
                    observation_grids = self.get_observation_grids(update=update)
                    observation_grids = observation_grids.reset_index().set_index('Centroid')
                    gridded_obs = gridded_obs.reset_index(level='Date').join(
                        observation_grids[['Pseudo_Grid_ID']])
                    gridded_obs = gridded_obs.reset_index().set_index(
                        ['Pseudo_Grid_ID', 'Centroid', 'Date'])

                if pickle_it:
                    save_pickle(gridded_obs, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(filename.replace("-", " "), e))
                gridded_obs = None

        return gridded_obs

    def get_obs_data(self, use_pseudo_grid_id=True, update=False, pickle_it=False, verbose=False):
        """
        Fetch integrated weather observations of different variables (from local pickle, if available).

        :param use_pseudo_grid_id: defaults to ``True``
        :type use_pseudo_grid_id: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of integrated daily gridded weather observations
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            >>> ukcp09_dat = ukcp.get_obs_data()

            >>> ukcp09_dat.tail()
                                                        Maximum_Temperature  ...  Temperature_Change
            Pseudo_Grid_ID Centroid         Date                             ...
            10358          (652500, 312500) 2016-12-27                 7.50  ...                4.34
                                            2016-12-28                 5.86  ...                6.22
                                            2016-12-29                 6.10  ...                5.15
                                            2016-12-30                 4.61  ...                3.40
                                            2016-12-31                 7.98  ...                7.55
            [5 rows x 4 columns]
        """

        filename = "ukcp-daily-gridded-weather"
        path_to_pickle = self.make_pickle_path(filename)

        if os.path.isfile(path_to_pickle) and not update:
            ukcp09_data = load_pickle(path_to_pickle)

        else:
            try:
                d_max_temp = self.get_obs_data_by_category(
                    zip_filename="daily-maximum-temperature", var_name='Maximum_Temperature',
                    use_pseudo_grid_id=False, update=update, verbose=verbose)
                d_min_temp = self.get_obs_data_by_category(
                    zip_filename="daily-minimum-temperature", var_name='Minimum_Temperature',
                    use_pseudo_grid_id=False, update=update, verbose=verbose)
                d_precipitation = self.get_obs_data_by_category(
                    zip_filename="daily-precipitation", var_name='Precipitation',
                    use_pseudo_grid_id=False, update=update, verbose=verbose)

                ukcp09_data = pd.concat([d_max_temp, d_min_temp, d_precipitation], axis=1)

                del d_max_temp, d_min_temp, d_precipitation
                gc.collect()

                ukcp09_data['Temperature_Change'] = abs(
                    ukcp09_data.Maximum_Temperature - ukcp09_data.Minimum_Temperature)

                if use_pseudo_grid_id:
                    observation_grids = self.get_observation_grids(update=update)
                    observation_grids = observation_grids.reset_index().set_index('Centroid')
                    ukcp09_data = ukcp09_data.reset_index('Date').join(
                        observation_grids[['Pseudo_Grid_ID']])
                    ukcp09_data = ukcp09_data.reset_index().set_index(
                        ['Pseudo_Grid_ID', 'Centroid', 'Date'])

                if pickle_it:
                    path_to_pickle = self.make_pickle_path(filename)
                    save_pickle(ukcp09_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to integrate the UKCP09 gridded weather observations. {}".format(e))
                ukcp09_data = None

        return ukcp09_data

    def import_data(self, table_name='UKCP09', if_exists='fail', chunk_size=100000, update=False,
                    verbose=False):
        """
        See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

        :param table_name:
        :param if_exists:
        :param chunk_size:
        :param update:
        :param verbose:

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()
        """

        ukcp09_data = self.get_obs_data(update=update, verbose=verbose)
        ukcp09_data.reset_index(inplace=True)

        ukcp09_data_ = pd.DataFrame(ukcp09_data.Centroid.to_list(), columns=['Centroid_X', 'Centroid_Y'])
        ukcp09_data = pd.concat([ukcp09_data.drop('Centroid', axis=1), ukcp09_data_], axis=1)

        print("Importing UKCP09 data to MSSQL Server", end=" ... ")

        with tempfile.NamedTemporaryFile() as temp_file:
            ukcp09_data.to_csv(temp_file.name + ".csv", index=False, chunksize=chunk_size)

            tsql_chunksize = 2100 // len(ukcp09_data.columns)
            temp_file_ = pd.read_csv(temp_file.name + ".csv", chunksize=tsql_chunksize)
            for chunk in temp_file_:
                # e.g. chunk = temp_file_.get_chunk(chunk_size)
                chunk.to_sql(name=table_name, con=self.DatabaseConn, schema='dbo', if_exists=if_exists,
                             index=False, dtype={'Date': sqlalchemy.types.DATE}, method='multi')

                del chunk
                gc.collect()

            temp_file.close()

        os.remove(temp_file.name)

        print("Done. ")

    def query_by_grid_datetime(self, grids, period, update=False, dat_dir=None, pickle_it=False,
                               verbose=False):
        """
        Get UKCP09 data by observation grids (Query from the database) for the given ``period``.

        :param grids: a list of weather observation IDs
        :type grids: list
        :param period: prior-incident / non-incident period
        :type period:
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str, None
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            dat_dir = None
            update = False
            verbose = True

            pickle_it = False
            grids = incidents.Weather_Grid.iloc[0]
            period = incidents.Critical_Period.iloc[0]
            ukcp09_dat = query_by_grid_datetime(grids, period, verbose=verbose)

            pickle_it = True
            grids = incidents.Weather_Grid.iloc[1]
            period = incidents.Critical_Period.iloc[1]
            ukcp09_dat = query_by_grid_datetime(grids, period, pickle_it=pickle_it,
                                                       verbose=verbose)
        """

        period = pd.date_range(period.left.date[0], period.right.date[0], normalize=True)

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(
            "-".join([str(grids[0]), str(len(grids) * sum(grids[1:-1])), str(grids[-1])]),
            "-".join([period.min().strftime('%Y%m%d'), period.max().strftime('%Y%m%d')]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if isinstance(dat_dir, str) and os.path.isabs(dat_dir):
            dat_dir_ = dat_dir
        else:
            dat_dir_ = self.cdd("dat")
        path_to_pickle = cdd_weather(dat_dir_, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            ukcp09_dat = load_pickle(path_to_pickle)

        else:
            # Specify database sql query
            grids_ = tuple(grids) if len(grids) > 1 else grids[0]
            period_ = tuple(x.strftime('%Y-%m-%d') for x in period)
            in_ = 'IN' if len(grids) > 1 else '='

            sql_query = f"SELECT * FROM dbo.[UKCP09] " \
                        f"WHERE [Pseudo_Grid_ID] {in_} {grids_} " \
                        f"AND [Date] IN {period_};"

            # Query the weather data
            ukcp09_dat = pd.read_sql(sql=sql_query, con=self.DatabaseConn)

            if pickle_it:
                save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

        return ukcp09_dat

    def query_by_grid_datetime_(self, grids, period, update=False, dat_dir=None, pickle_it=False,
                                verbose=False):
        """
        Get UKCP09 data by observation grids and date (Query from the database)
        from the beginning of the year to the start of the ``period``.

        :param grids: a list of weather observation IDs
        :type grids: list
        :param period: prior-incident / non-incident period
        :type period:
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str, None
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import UKCP09

            >>> ukcp = UKCP09()

            update = False
            verbose = True

            pickle_it = False
            grids = incidents.Weather_Grid.iloc[0]
            period = incidents.Critical_Period.iloc[0]
            ukcp09_dat = query_by_grid_datetime_(grids, period, verbose=verbose)

            pickle_it = True
            grids = incidents.Weather_Grid.iloc[1]
            period = incidents.Critical_Period.iloc[1]
            ukcp09_dat = query_by_grid_datetime_(grids, period, pickle_it=pickle_it,
                                                        verbose=verbose)
        """

        period = pd.date_range(period.left.date[0], period.right.date[0], normalize=True)
        y_start = datetime_truncate.truncate_year(period.min()).strftime('%Y-%m-%d')
        p_start = period.min().strftime('%Y-%m-%d')

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(
            "-".join([str(grids[0]), str(len(grids) * sum(grids[1:-1])), str(grids[-1])]),
            "-".join([y_start.replace("-", ""), p_start.replace("-", "")]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if isinstance(dat_dir, str) and os.path.isabs(dat_dir):
            dat_dir_ = dat_dir
        else:
            dat_dir_ = cdd_weather("ukcp", "dat")
        path_to_pickle = cdd_weather(dat_dir_, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            ukcp09_dat = load_pickle(path_to_pickle)

        else:
            # Specify database sql query
            grids_ = tuple(grids) if len(grids) > 1 else grids[0]
            in_ = 'IN' if len(grids) > 1 else '='

            sql_query = f"SELECT * FROM dbo.[UKCP09] " \
                        f"WHERE [Pseudo_Grid_ID] {in_} {grids_} " \
                        f"AND [Date] >= '{y_start}' AND [Date] <= '{p_start}';"

            # Query the weather data
            ukcp09_dat = pd.read_sql(sql=sql_query, con=self.DatabaseConn)

            if pickle_it:
                save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

        return ukcp09_dat
