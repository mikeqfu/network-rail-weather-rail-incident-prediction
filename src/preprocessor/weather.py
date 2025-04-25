"""
Weather
"""

import gc
import glob
import multiprocessing
import os
import re
import tempfile
import zipfile

import datetime_truncate
import natsort
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import shapely.wkt
import sqlalchemy.types
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import get_square_vertices, osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.store import load_data, save_data, xlsx_to_csv

from src.utils import Handler, WxRailIncidentsPred


# noinspection PyShadowingNames
class MIDAS(Handler):
    """
    Met Office RADTOB.
    """

    #: Name of the data.
    DATA_NAME: str = 'Met Office Integrated Data Archive System'
    #: Acronym of the data resource.
    ACRONYM: str = 'MIDAS'
    #: Brief description of the data.
    DESCRIPTION: str = 'Met Office RADTOB (Radiation values currently being reported).'
    #: Schema name.
    SCHEMA_NAME: str = 'MIDAS_RADTOB'

    def __init__(self, db_instance=None):
        """
        :param db_instance:
        :type db_instance:

        :ivar str | os.PathLike[str] DATA_DIR: Pathname of the data directory.
        :ivar str | pathlib.Path supplements_data_dir: Pathname of the supplimentary data direcotry.
        :ivar pandas.DataFrame | None Radiation_monitoring_stations: radiation monitoring stations.
        :ivar list | None radtob_table_header: Column names of the radiation observation data.
        :ivar pandas.DataFrame | None radtob_table: Radiation observation data.

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> midas.DATA_NAME
            'Met Office Integrated Data Archive System'
        """

        super().__init__(db_instance=db_instance)

        self.data_dir = cdd("weather/midas")
        self.supplements_data_dir = cd(self.data_dir, "supplements")

        self.radiation_monitoring_stations = None
        self.radtob_table_header = None
        self.radtob_table = None

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data/Weather/midas" and subdirectories or a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data/Weather/midas"`` and subdirectories or a file
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> import os
            >>> midas = MIDAS()
            >>> os.path.relpath(midas.cdd())
            'data\\weather\\midas'
        """

        path = cd(self.data_dir, *sub_dir, mkdir=mkdir)

        return path

    def _radiation_monitoring_stations(self, verbose=False, **kwargs):
        """
        Get locations and relevant information of meteorological stations.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of meteorological stations
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> data = midas._radiation_monitoring_stations(verbose=True)
            >>> data.shape
            (240, 11)
        """

        filename = "radiation_monitoring_stations"

        xlsx_pathname = self.cdd(filename + ".xlsx")
        temp_csv_pathname = xlsx_to_csv(xlsx_pathname, sheet_name='1')
        dat = pd.read_csv(
            temp_csv_pathname, parse_dates=['Station start date'], encoding='ISO-8859-1')
        os.remove(temp_csv_pathname)

        dat.columns = [
            x.title().replace(' ', '') if x != 'src_id' else x.upper() for x in dat.columns]
        dat['StationName'] = dat['StationName'].str.replace(r'(\xa0)+Locate', '', regex=True)

        dat['Easting'], dat['Northing'] = wgs84_to_osgb36(dat.Longitude.values, dat.Latitude.values)
        dat['XY'] = [shapely.geometry.Point(xy) for xy in zip(dat.Easting, dat.Northing)]

        index_name = 'SRC_ID'
        data = dat.set_index(index_name).sort_index()

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        self.dump_preprocessed_data(
            data=data, table_name=filename, verbose=verbose, pkey=[index_name], **kwargs)
        self.db_instance.null_text_to_empty_string(
            table_name=filename, schema_name=self.SCHEMA_NAME)

        return data

    def read_radiation_monitoring_stations(self, **kwargs):
        """
        Get locations and relevant information of meteorological stations.

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> # midas.read_radiation_monitoring_stations(update=True, verbose=True)
            >>> midas.read_radiation_monitoring_stations()
            >>> midas.radiation_monitoring_stations.shape
            (240, 11)
        """

        self.read_data(table_name='radiation_monitoring_stations', index_col='SRC_ID', **kwargs)

        if self.radiation_monitoring_stations.XY.convert_dtypes().dtype.name == 'string':
            self.radiation_monitoring_stations.XY = self.radiation_monitoring_stations.XY.map(
                shapely.wkt.loads)

    def _radtob_table_header(self, as_list=True, verbose=False, **kwargs):
        """
        Get a list of column names for RADTOB data.

        :return: list of column names
        :rtype: list

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> data = midas._radtob_table_header()
            >>> len(data)
            22
        """

        filename = "radtob_table_header"

        xlsx_pathname = self.cdd(filename + ".xlsx")
        temp_csv_pathname = xlsx_to_csv(xlsx_pathname, sheet_name='1')
        raw = pd.read_csv(temp_csv_pathname, header=None)
        os.remove(temp_csv_pathname)

        data = pd.DataFrame([x.strip() for x in raw.iloc[0, :].values], columns=['Header'])

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        self.dump_preprocessed_data(
            data=data, table_name=filename, verbose=verbose, pkey=[], index=False, **kwargs)

        if as_list:
            data = data.Header.to_list()

        return data

    def read_radtob_table_header(self, as_list=True, **kwargs):
        """
        Get a list of column names for RADTOB data.

        :param as_list:
        :param kwargs:
        :return:

        Get a list of column names for RADTOB data.

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> midas.read_radtob_table_header()
            >>> len(midas.radtob_table_header)
            22
        """

        self.read_data(table_name='radtob_table_header', **kwargs)

        if as_list:
            if isinstance(self.radtob_table_header, pd.DataFrame):
                self.radtob_table_header = self.radtob_table_header.Header.to_list()

    def prep_radtob_table(self, raw_tbl, daily=False, rad_stn=False):
        """
        Parse original MIDAS RADTOB (Radiation data).

        MIDAS  - Met Office Integrated Data Archive System
        RADTOB - RADT-OB table. Radiation values currently being reported

        :param raw_tbl:
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

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> import zipfile
            >>> import natsort
            >>> midas = MIDAS()
            >>> zf = zipfile.ZipFile(midas.cdd("radtob_2006_2019.zip"))
            >>> filename_list = natsort.natsorted(zf.namelist())
            >>> f = filename_list[0]
            >>> midas.read_radtob_table_header()
            >>> idx_names = ['ID', 'ID_TYPE', 'SRC_ID', 'OB_END_TIME']
            >>> dat = pd.read_csv(
            ...     zf.open(f), header=None, names=midas.radtob_table_header, index_col=idx_names,
            ...     parse_dates=[2, 12], skipinitialspace=True)
            >>> dat.shape
            (1366349, 18)
        """

        # selected_feat = ['SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT', 'VERSION_NUM', 'GLBL_IRAD_AMT']
        # raw_dat = raw_dat[selected_feat].drop_duplicates()

        raw_dat = raw_tbl.reset_index()

        if daily:
            raw_dat = raw_dat[raw_dat.OB_HOUR_COUNT == 24]

        # Cleanse the data
        key_cols = ['ID', 'SRC_ID', 'OB_END_TIME', 'OB_HOUR_COUNT']

        temp = raw_dat.groupby(key_cols).agg({'VERSION_NUM': 'max'}).reset_index()
        key_cols_ = key_cols + ['VERSION_NUM']
        prep_dat = temp.join(raw_dat.set_index(key_cols_), on=key_cols_)

        # Note: The following line is questionable
        idx = (prep_dat.GLBL_IRAD_AMT < 0.0) | prep_dat.GLBL_IRAD_AMT.isna()
        prep_dat.loc[idx, 'GLBL_IRAD_AMT'] = 0.0  # np.nan

        # Insert 'OB_END_DATE'
        loc = prep_dat.columns.get_loc('OB_END_TIME')
        prep_dat.insert(loc=loc, column='OB_END_DATE', value=prep_dat.OB_END_TIME.dt.date)

        # Rename 'OB_END_TIME'
        prep_dat.rename(columns={'OB_END_TIME': 'OB_END_DATETIME'}, inplace=True)

        # Sort rows
        index_names = ['ID', 'SRC_ID', 'OB_END_DATETIME', 'OB_END_DATE']
        prep_data = prep_dat.set_index(index_names).sort_index()

        if rad_stn:
            if self.radiation_monitoring_stations is None:
                self.read_radiation_monitoring_stations()
            prep_data = pd.merge(
                prep_data, self.radiation_monitoring_stations, left_index=True, right_index=True)

        return prep_data

    def _radtob_table(self, prep=True, daily=False, rad_stn=False, ret_data=False, verbose=True,
                      **kwargs):
        """

        :param prep:
        :param daily:
        :param rad_stn:
        :param ret_data:
        :param verbose:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> midas._radtob_table(prep=False, verbose=True)

            >>> midas._radtob_table(prep=True, verbose=True)

        """

        filename = "radtob_2006_2019"

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        if self.radtob_table_header is None:
            self.read_radtob_table_header()

        # zf = zipfile.ZipFile(midas.cdd(midas.radtob_filename + ".zip"))
        with zipfile.ZipFile(self.cdd(filename + ".zip"), mode='r') as zf:
            index_names = ['ID', 'ID_TYPE', 'SRC_ID', 'OB_END_TIME']

            data_list = []
            filename_list = natsort.natsorted(zf.namelist())
            for f in filename_list:
                raw_dat = pd.read_csv(
                    zf.open(f), header=None, names=self.radtob_table_header, parse_dates=[2, 12],
                    skipinitialspace=True, index_col=index_names)

                table_name = '_'.join(re.findall(r'\d+', f))

                if prep:
                    raw_dat = self.prep_radtob_table(raw_tbl=raw_dat, daily=daily, rad_stn=rad_stn)
                    table_name += '_prep'
                    if daily:
                        table_name += '_daily'

                self.dump_preprocessed_data(
                    data=raw_dat, table_name=table_name, verbose=verbose, pkey=[], **kwargs)
                self.db_instance.null_text_to_empty_string(
                    table_name=table_name, schema_name=self.SCHEMA_NAME)

                data_list.append(raw_dat)

        if ret_data:
            data = pd.concat(data_list, axis=0)
            return data

    def _supplement_data(self, verbose=False, **kwargs):
        """
        Parse supplementary data.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: parsed supplementary data
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> midas = MIDAS()
            >>> supplement_data = midas._supplement_data(verbose=True)
            >>> supplement_data.shape
            (20783, 3)
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        if self.radiation_monitoring_stations is None:
            self.read_radiation_monitoring_stations()

        suppl_dat = []
        for f in glob.glob(self.cdd("Supplements", "*.csv")):
            dat = pd.read_csv(f, parse_dates=['Date'])
            dat.rename(columns={'Date': 'OB_END_DATE', 'Rad': 'GLBL_IRAD_AMT'}, inplace=True)

            filename = os.path.basename(f)
            station_name = filename.split('_')[0].upper()

            src_id = self.radiation_monitoring_stations.StationName.eq(station_name).idxmax()
            dat.insert(loc=0, column='SRC_ID', value=src_id)
            dat.insert(loc=2, column='OB_HOUR_COUNT', value=24)

            if 'Wattisham' in filename:
                dat['Route'] = 'Anglia'
            if 'Hurn' in filename:
                dat['Route'] = 'Wessex'
            if 'Valley' in filename:
                dat['Route'] = 'Wales'
            if 'Durham' in filename:
                dat['Route'] = 'North and East'

            suppl_dat.append(dat)

        suppl_data = pd.concat(suppl_dat, axis=0, ignore_index=True)

        index_names = ['SRC_ID', 'OB_END_DATE']
        supplement_data = suppl_data.set_index(index_names)

        self.dump_preprocessed_data(
            data=supplement_data, table_name='supplement_prep', verbose=verbose, pkey=[],
            **kwargs)

        return supplement_data

    def __query_radtob_by_grid_datetime(self, src_id, period, route_name):
        """
        Query (from Database) MIDAS RADTOB (Radiation data) by met station ID
        for the given ``period``.

        :param src_id: met station ID
        :type src_id: list
        :param period: prior-incident / non-incident period
        :type period:
        :param route_name: name of Route
        :type route_name: str
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame
        """

        # Specify Database sql query
        ms_id = tuple(src_id)
        dates = tuple(
            x.strftime('%Y-%m-%d %H:%M:%S') for x in [period.left.min(), period.right.max()])

        sql_query = \
            f"SELECT * FROM dbo.[MIDAS_RADTOB] " \
            f"WHERE [SRC_ID] IN {ms_id} " \
            f"AND [OB_END_DATE_TIME] BETWEEN '{dates[0]}' AND '{dates[1]}';"

        midas_radtob = pd.read_sql(sql=sql_query, con=self.db_instance.engine)

        if midas_radtob.empty:
            dates = tuple(x.strftime('%Y-%m-%d') for x in [period.left.min(), period.right.max()])

            sql_query = \
                f"SELECT * FROM dbo.[MIDAS_RADTOB_Supplement] " \
                f"WHERE [Route] = '{route_name}' " \
                f"AND [OB_END_DATE] BETWEEN '{dates[0]}' AND '{dates[1]}';"

            midas_radtob = pd.read_sql(sql=sql_query, con=self.db_instance.engine)

        return midas_radtob

    def query_radtob_by_grid_datetime(self, src_id, period, route_name, **kwargs):
        # noinspection PyShadowingNames
        """
        Query (from Database) MIDAS RADTOB (Radiation data) by met station ID
        for the given ``period``.

        :param src_id: met station ID
        :type src_id: list
        :param period: prior-incident / non-incident period
        :type period:
        :param route_name: name of Route
        :type route_name: str
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.weather import MIDAS
            >>> from src.utils import WxRailIncidentsPred
            >>> import datetime
            >>> import pandas as pd

            >>> db_instance = WxRailIncidentsPred()
            >>> midas = MIDAS(db_instance)
            >>> src_id = 9
            >>> route_name = 'Anglia'

            >>> start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> end_dt = datetime.datetime(2018, 6, 1, 13)  # '2018-06-01 13:00:00'
            >>> period = pd.date_range(start=start_dt, end=end_dt, freq='h')
            >>> midas.query_radtob_by_grid_datetime(src_id, period, route_name)

            >>> start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> end_dt = datetime.datetime(2018, 12, 1, 12)  # '2018-12-01 12:00:00'
            >>> period = pd.date_range(start=start_dt, end=end_dt, freq='h')
            >>> midas.query_radtob_by_grid_datetime(src_id, period, route_name)

            >>> start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> end_dt = datetime.datetime(2019, 6, 1, 12)  # '2018-12-01 12:00:00'
            >>> period = pd.date_range(start=start_dt, end=end_dt, freq='h')
            >>> midas.query_radtob_by_grid_datetime(src_id, period, route_name)
        """

        # Specify Database sql query
        _src_id = f'= {src_id}' if isinstance(src_id, int) else f'IN {tuple(src_id)}'
        dt_start, dt_end = [x.strftime('%Y-%m-%d %H:%M:%S') for x in [period.min(), period.max()]]

        table_names = [f'{year}01_{year}12_prep' for year in set(p.year for p in period.date)]

        midas_radtob_ = []
        for table_name in table_names:
            sql_query = \
                f'SELECT * FROM "{self.SCHEMA_NAME}"."{table_name}" ' \
                f'WHERE "SRC_ID" {_src_id} ' \
                f'AND "OB_END_DATETIME" BETWEEN \'{dt_start}\' AND \'{dt_end}\''

            midas_radtob_.append(self.db_instance.read_sql_query(sql_query, **kwargs))

        midas_radtob = pd.concat(midas_radtob_, axis=0, ignore_index=True)

        if midas_radtob.empty:
            dt_start, dt_end = [x.strftime('%Y-%m-%d') for x in [period.min(), period.max()]]

            sql_query = \
                f'SELECT * FROM "{self.SCHEMA_NAME}"."supplement_prep" ' \
                f'WHERE "Route" = \'{route_name}\' ' \
                f'AND "OB_END_DATE" BETWEEN \'{dt_start}\' AND \'{dt_end}\';'

            midas_radtob = self.db_instance.read_sql_query(sql_query, **kwargs)

        return midas_radtob


class UKCP09(Handler):
    """
    UKCP09 gridded Weather observations.
    """

    #: Name of the data
    DATA_NAME = 'UK Climate Projections 2009'
    #: Acronym of the data name
    ACRONYM = 'UKCP09'
    #: Brief description of the data
    DESCRIPTION = 'UKCP09 gridded Weather observations: ' \
                  'maximum temperature, minimum temperature and precipitation.'
    #: str: Schema name.
    SCHEMA_NAME = 'UKCP09'

    def __init__(self, start_date='2006-01-01', db_instance=None):
        """
        :param start_date: start date on which the observation data was collected,
            formatted as 'yyyy-mm-dd', defaults to ``'2006-01-01'``
        :type start_date: str
        :param db_instance:
        :type db_instance:

        :ivar str start_date: (specified with the creation of the instance)

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> ukcp09 = UKCP09()
            >>> ukcp09.DATA_NAME
            'UK Climate Projections 2009'
        """

        super().__init__(db_instance=db_instance)

        self.data_dir = cdd("weather/ukcp09")

        self.start_date = pd.to_datetime(start_date)

        self.observation_grids = None
        self.observation_data = None

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\Weather\\UKCP09" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data\\Weather\\UKCP09"`` and subdirectories / a file
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> import os
            >>> ukcp09 = UKCP09()
            >>> os.path.relpath(ukcp09.cdd())
            'data\\weather\\ukcp09'
            >>> os.path.isdir(ukcp09.cdd())
            True
        """

        path = cd(self.data_dir, *sub_dir, mkdir=mkdir)

        return path

    # == Read data of Weather observation grids ====================================================

    @staticmethod
    def _poly_osgb36_to_wgs84(poly):
        lonlat = [osgb36_to_wgs84(xy[0], xy[1]) for xy in poly.exterior.coords]
        poly_ = shapely.geometry.Polygon(lonlat)
        return poly_

    def parse_observation_grids(self, filename):
        """
        Parse observation grids.

        :param filename: file of the observation grid data
        :type filename: str or typing.IO[bytes]
        :return: parsed data of the observation grids
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> import zipfile
            >>> ukcp09 = UKCP09()
            >>> path_to_zip = ukcp09.cdd("daily_precipitation.zip")
            >>> zf = zipfile.ZipFile(path_to_zip, 'r')
            >>> f = zf.open(zf.namelist()[0])
            >>> obs_grid_data = ukcp09.parse_observation_grids(f)  # filename = f
            >>> zf.close()
            >>> obs_grid_data.shape
            (9, 7)
        """

        centroids_xy_raw = [
            tuple(ctr) for ctr in pd.read_csv(filename, header=None, index_col=0, nrows=2).T.values]

        centroids_xy = [shapely.geometry.Point(ctr) for ctr in centroids_xy_raw]

        grids_xy = [
            shapely.geometry.Polygon(get_square_vertices(xy[0], xy[1], side_length=5000))
            for xy in centroids_xy_raw]

        centroids_lonlat = [
            shapely.geometry.Point(ctr)
            for ctr in [osgb36_to_wgs84(xy[0], xy[1]) for xy in centroids_xy_raw]]

        # grids_lonlat = [
        #     shapely.geometry.Polygon(
        #         [osgb36_to_wgs84(xy[0], xy[1]) for xy in grid.exterior.coords])
        #     for grid in grids_xy]
        with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
            grids_lonlat = p.map(self._poly_osgb36_to_wgs84, grids_xy)

        dat = {
            'Centroid': centroids_xy_raw,
            'Centroid_X': np.array(centroids_xy_raw)[:, 0],
            'Centroid_Y': np.array(centroids_xy_raw)[:, 1],
            'Centroid_XY': centroids_xy,
            'Grid_XY': grids_xy,
            'Centroid_LonLat': centroids_lonlat,
            'Grid_LonLat': grids_lonlat,
        }

        data = pd.DataFrame(dat, index=range(len(centroids_xy_raw)))

        return data

    def _observation_grids(self):
        """

        :return:

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09

            >>> ukcp09 = UKCP09()

            >>> obs_grids_data = UKCP09._observation_grids()
        """

        path_to_zip = self.cdd("daily_precipitation.zip")

        with zipfile.ZipFile(path_to_zip, 'r') as zf:
            filename_list = natsort.natsorted(zf.namelist())
            obs_grids = [self.parse_observation_grids(zf.open(f)) for f in filename_list]

        data = pd.concat(obs_grids, ignore_index=True)

        # Add a pseudo id for each observation grid
        # obs_grids.sort_values('Grid_XY', inplace=True)
        data.index = pd.Index(range(len(data)), name='Pseudo_Grid_ID')

        return data

    def read_observation_grids(self, ret_data=False, update=False, verbose=False):
        """
        Fetch data of observation grids from local pickle.

        :param ret_data:
        :type ret_data:
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: MIDAS RADTOB observation grids
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> ukcp09 = UKCP09()
            >>> # ukcp09.read_observation_grids(update=True, verbose=True)
            >>> ukcp09.read_observation_grids()
            >>> ukcp09.observation_grids.shape
            (10359, 7)
        """

        path_to_pickle = self.cdd("observation_grids.pkl")

        if os.path.isfile(path_to_pickle) and not update:
            self.observation_grids = load_data(path_to_pickle)

        else:
            if verbose:
                print(f"Reading {self.ACRONYM} Weather observation grids", end=" ... ")

            try:
                data = self._observation_grids()

                if verbose:
                    print("Done.")

                self.observation_grids = data

                save_data(self.observation_grids, path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")

        if ret_data:
            return self.observation_grids

    # == Read data of Weather observations =========================================================

    def _make_pickle_pathname(self, filename):
        """
        Make a full path to the pickle file of the UKCP09 data.

        :param filename: e.g. file="daily_maximum_temperature"
        :type filename: str
        :return: a full path to the pickle file of the UKCP09 data
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> import os

            >>> ukcp09 = UKCP09()

            >>> os.path.relpath(UKCP09._make_pickle_pathname("daily_maximum_temperature"))
            'data\\Weather\\UKCP09\\daily_maximum_temperature_20060101.pkl'
        """

        if self.start_date is None:
            filename_suffix = ""
        else:
            filename_suffix = "_{}".format(str(self.start_date.date()).replace("-", ""))

        pickle_filename = filename + filename_suffix + ".pkl"

        path_to_pickle = self.cdd(pickle_filename)

        return path_to_pickle

    def parse_daily_gridded_observations(self, filename, variable_name='Weather_Variable'):
        """
        Parse gridded Weather observations from the raw zipped file.

        :param filename: filename of raw data
        :type filename: str or typing.IO[bytes]
        :param variable_name: variable name,
            e.g. ``'Maximum_Temperature'``, ``'Minimum_Temperature'`` and ``'Precipitation'``
        :type variable_name: str
        :return: parsed data of the daily gridded Weather observations
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> import natsort
            >>> import zipfile
            >>> ukcp09 = UKCP09()
            >>> path_to_zip = ukcp09.cdd("daily_precipitation.zip")
            >>> zf = zipfile.ZipFile(path_to_zip, 'r')
            >>> filename_list = natsort.natsorted(zf.namelist())
            >>> f = zf.open(filename_list[0])  # filename = f
            >>> dat = ukcp09.parse_daily_gridded_observations(f, variable_name='Precipitation')
            >>> dat.shape
            (56252, 1)
        """

        # Centroids of Weather observation grids
        centroids_xy_raw = pd.read_csv(filename, header=None, index_col=0, nrows=2)
        centroid_xy_tuples = [tuple(x) for x in centroids_xy_raw.T.values]

        # Weather observations (Timeseries data)
        ts_data = pd.read_csv(
            filename, header=None, skiprows=[0, 1], parse_dates=[0], date_format='%Y-%m-%d')
        ts_data[0] = pd.to_datetime(ts_data[0]).dt.date

        if isinstance(self.start_date, pd.Timestamp):
            ts_data = ts_data[ts_data[0] >= self.start_date.date()]
        ts_data.set_index(0, inplace=True)

        # Reshape the dataframe
        new_idx = pd.MultiIndex.from_product(
            [centroid_xy_tuples, ts_data.index], names=['Centroid', 'Date'])
        data = pd.DataFrame(
            ts_data.T.values.flatten(), index=new_idx, columns=[variable_name]).reset_index()

        centroids_xy = pd.DataFrame(
            data['Centroid'].to_list(), index=data.index, columns=['Centroid_X', 'Centroid_Y'])
        data = pd.concat([centroids_xy, data.drop('Centroid', axis=1)], axis=1)

        data.set_index(['Centroid_X', 'Centroid_Y', 'Date'], inplace=True)

        return data

    def make_pseudo_grid_id(self, data):
        if self.observation_grids is None:
            self.read_observation_grids()
        obs_grids = self.observation_grids.reset_index().set_index(['Centroid_X', 'Centroid_Y'])

        data_ = data.reset_index(level='Date').join(obs_grids[['Pseudo_Grid_ID']])
        data_ = data_.reset_index().set_index(
            ['Pseudo_Grid_ID', 'Centroid_X', 'Centroid_Y', 'Date'])

        return data_

    def read_observations_by_category(self, zip_filename, variable_name='Weather_Variable',
                                      use_pseudo_grid_id=False, update=False, pickle_it=False,
                                      verbose=False):
        """
        Get observation data for a given category (i.e. Weather variable).

        :param zip_filename: "daily_maximum_temperature", "daily_minimum_temperature",
            or "daily_precipitation"
        :type zip_filename: str
        :param variable_name: variable name;
            'Precipitation' or 'Maximum_Temperature', 'Minimum_Temperature'
        :type variable_name: str
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
        :return: data of daily gridded Weather observations
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> ukcp09 = UKCP09()
            >>> example_zip_filename = "daily_precipitation.zip"
            >>> example_var_name = 'Precipitation'
            >>> dat = ukcp09.read_observations_by_category(example_zip_filename, example_var_name)
            >>> dat.shape
            (41622462, 1)
        """

        assert isinstance(self.start_date, pd.Timestamp) or self.start_date is None

        filename = os.path.splitext(zip_filename)[0]
        path_to_pickle = self._make_pickle_pathname(filename)

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            if verbose:
                print(f'Reading "{filename}"', end=" ... ")

            try:
                path_to_zip = self.cdd(zip_filename)

                with zipfile.ZipFile(path_to_zip, 'r') as zf:
                    filename_list = natsort.natsorted(zf.namelist())
                    obs_data = [
                        self.parse_daily_gridded_observations(
                            zf.open(f), variable_name=variable_name)
                        for f in filename_list]

                data = pd.concat(obs_data, axis=0)

                if use_pseudo_grid_id:  # Add a pseudo id for each observation grid
                    data = self.make_pseudo_grid_id(data=data)

                if verbose:
                    print("Done.")

                if pickle_it:
                    save_data(data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                data = None

        return data

    def read_observations(self, use_pseudo_grid_id=True, update=False, pickle_it=False,
                          verbose=False):
        """
        Fetch integrated Weather observations of different variables
        (from local pickle, if available).

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
        :return: data of integrated daily gridded Weather observations
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> ukcp09 = UKCP09()
            >>> ukcp09.read_observations(use_pseudo_grid_id=False, verbose=True)
            >>> ukcp09.observation_data.shape
            (41622462, 4)
            >>> ukcp09.observation_data.index.nlevels
            3
            >>> ukcp09.read_observations(verbose=True)
            >>> ukcp09.observation_data.shape
            (41622462, 4)
            >>> ukcp09.observation_data.index.nlevels
            4
        """

        filename = "daily_weather"
        path_to_pickle = self._make_pickle_pathname(filename)

        if os.path.isfile(path_to_pickle) and not update:
            self.observation_data = load_data(path_to_pickle)

        else:
            names_dict = {
                "daily_max_temperature.zip": 'Maximum_Temperature',
                "daily_min_temperature.zip": 'Minimum_Temperature',
                "daily_precipitation.zip": 'Precipitation',
            }

            try:
                dat_list = []
                for zip_filename, variable_name in names_dict.items():
                    # if verbose:
                    #     print(f"Reading \"{zip_filename}\"", end=" ... ")

                    dat = self.read_observations_by_category(
                        zip_filename=zip_filename, variable_name=variable_name,
                        use_pseudo_grid_id=False, update=update, verbose=verbose)

                    dat_list.append(dat)

                    # if verbose:
                    #     print("Done.")

                if verbose:
                    print("Integrating the above data", end=" ... ")

                data = pd.concat(dat_list, axis=1)

                del dat_list
                gc.collect()

                data['Temperature_Change'] = \
                    (data['Maximum_Temperature'] - data['Minimum_Temperature']).abs()

                if use_pseudo_grid_id:
                    data = self.make_pseudo_grid_id(data=data)

                if verbose:
                    print("Done.")

                self.observation_data = data

                if pickle_it:
                    save_data(data, path_to_pickle, verbose=verbose)

                if verbose:
                    print("Completed.")

            except Exception as e:
                print(f"Failed to integrate the UKCP09 gridded Weather observations. {e}.")

    def __import_data(self, table_name='UKCP09', if_exists='fail', chunk_size=100000, update=False,
                      verbose=False):
        """
        See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

        :param table_name:
        :param if_exists:
        :param chunk_size:
        :param update:
        :param verbose:

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09

            >>> ukcp09 = UKCP09()
        """

        data = self.read_observations(update=update, verbose=verbose)
        data.reset_index(inplace=True)

        data_ = pd.DataFrame(data['Centroid'].to_list(), columns=['Centroid_X', 'Centroid_Y'])
        data = pd.concat([data.drop('Centroid', axis=1), data_], axis=1)

        if verbose:
            print("Importing UKCP09 data to MSSQL Server", end=" ... ")

        with tempfile.NamedTemporaryFile() as temp_file:
            # noinspection PyTypeChecker
            data.to_csv(temp_file.name + ".csv", index=False, chunksize=chunk_size)

            tsql_chunksize = 2100 // len(data.columns)
            temp_file_ = pd.read_csv(temp_file.name + ".csv", chunksize=tsql_chunksize)

            with self.db_instance.engine.connect() as connection:
                for chunk in temp_file_:
                    # e.g. chunk = temp_file_.get_chunk(chunk_size)
                    chunk.to_sql(
                        name=table_name, con=connection, schema='dbo', if_exists=if_exists,
                        index=False, dtype={'Date': sqlalchemy.types.DATE}, method='multi')

                    del chunk
                    gc.collect()

        os.remove(temp_file.name)

        if verbose:
            print("Done.")

    def import_data(self, chunk_size=10**6, if_exists='fail', update=False, verbose=False,
                    **kwargs):
        """

        :param chunk_size:
        :param if_exists:
        :param update:
        :param verbose:

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> ukcp09 = UKCP09()
            >>> # ukcp09.import_data(if_exists='replace', verbose=True)
            >>> ukcp09.import_data(verbose=True)
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        if self.observation_data is None:
            self.read_observations(update=update, verbose=verbose)
        data = self.observation_data.copy()
        data.reset_index(inplace=True)

        gc.collect()

        if verbose:
            print("Importing UKCP09 data (Max / Min temperature, and Max precipitation) ... ")

        import_args = {
            'schema_name': self.SCHEMA_NAME,
            'chunk_size': chunk_size,
            'if_exists': if_exists,
            'index': True,
            'confirmation_required': False,
            'verbose': verbose,
            'pkey': ['Date', 'Centroid_X', 'Centroid_Y', 'Pseudo_Grid_ID'],
        }
        kwargs.update(import_args)

        for year, dat in data.groupby(pd.to_datetime(data['Date']).dt.year):
            dat_ = dat.set_index(import_args['pkey'])
            self.dump_preprocessed_data(data=dat_, table_name=str(year), **kwargs)

        if verbose:
            print("Completed.")

    def query_by_datetime_grid(self, period, grids, update=False, dat_dir=None, pickle_it=False,
                               verbose=False):
        """
        Get UKCP09 data by observation grids (Query from the Database) for the given ``period``.

        :param grids: a list of Weather observation IDs
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
        :return: UKCP09 data by ``period`` and ``grids``
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> from pyhelpers.settings import pd_preferences
            >>> pd_preferences()
            >>> ukcp09 = UKCP09()

            period = Incidents['Critical_Period'].iloc[0]
            grids = Incidents['Weather_Grid'].iloc[0]
            ukcp09_data = ukcp09.query_by_datetime_grid(period, grids)

            period = Incidents['Critical_Period'].iloc[1]
            grids = Incidents.Weather_Grid.iloc[1]
            ukcp09_data = ukcp09.query_by_datetime_grid(period, grids)
        """

        date_range = pd.date_range(period.left.date[0], period.right.date[0], normalize=True)

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(
            "-".join([str(grids[0]), str(len(grids) * sum(grids[1:-1])), str(grids[-1])]),
            "-".join([date_range.min().strftime('%Y%m%d'), date_range.max().strftime('%Y%m%d')]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if isinstance(dat_dir, str) and os.path.isabs(dat_dir):
            dat_dir_ = dat_dir
        else:
            dat_dir_ = self.cdd("dat")
        path_to_pickle = self.cdd(dat_dir_, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            if self.db_instance is None:
                self.db_instance = WxRailIncidentsPred()

            grids_ = tuple(grids) if len(grids) > 1 else grids[0]
            period_ = tuple(x.strftime('%Y-%m-%d') for x in date_range)
            in_ = 'IN' if len(grids) > 1 else '='

            # # MSSQL
            # sql_query = f"SELECT * FROM dbo.[UKCP09] " \
            #             f"WHERE [Pseudo_Grid_ID] {in_} {grids_} " \
            #             f"AND [Date] IN {period_};"
            # with self.db_instance.engine.connect() as connection:
            #     data = pd.read_sql(sql=sql_query, con=connection)

            # PostgreSQL
            table_name = period_[0][:4]
            sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{table_name}" ' \
                        f'WHERE "Pseudo_Grid_ID" {in_} {grids_} ' \
                        f'AND "Date" IN {period_}'

            data = self.db_instance.read_sql_query(sql_query)

            if pickle_it:
                save_data(data, path_to_pickle, verbose=verbose)

        return data

    def query_by_grid_datetime_heretofore(self, period, grids, update=False, dat_dir=None,
                                          pickle_it=False, verbose=False):
        """
        Get UKCP09 data by observation grids and date (Query from the Database)
        from the beginning of the year to the start of the ``period``.

        :param grids: a list of Weather observation IDs
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

        **Examples**::

            >>> from src.preprocessor.weather import UKCP09
            >>> from pyhelpers.settings import pd_preferences
            >>> pd_preferences()
            >>> ukcp09 = UKCP09()

            period = Incidents['Critical_Period'].iloc[0]
            grids = Incidents['Weather_Grid'].iloc[0]
            ukcp09_data = ukcp09.query_by_grid_datetime_heretofore(period, grids)

            period = Incidents['Critical_Period'].iloc[1]
            grids = Incidents.Weather_Grid.iloc[1]
            ukcp09_data = ukcp09.query_by_grid_datetime_heretofore(period, grids)
        """

        date_range = pd.date_range(period.left.date[0], period.right.date[0], normalize=True)
        y_start = datetime_truncate.truncate_year(date_range.min()).strftime('%Y-%m-%d')
        p_start = date_range.min().strftime('%Y-%m-%d')

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(
            "-".join([str(grids[0]), str(len(grids) * sum(grids[1:-1])), str(grids[-1])]),
            "-".join([y_start.replace("-", ""), p_start.replace("-", "")]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if isinstance(dat_dir, str) and os.path.isabs(dat_dir):
            dat_dir_ = dat_dir
        else:
            dat_dir_ = self.cdd("dat")
        path_to_pickle = self.cdd(dat_dir_, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            data = load_data(path_to_pickle)

        else:
            if self.db_instance is None:
                self.db_instance = WxRailIncidentsPred()

            grids_ = tuple(grids) if len(grids) > 1 else grids[0]
            in_ = 'IN' if len(grids) > 1 else '='

            # # MSSQL
            # sql_query = f"SELECT * FROM dbo.[UKCP09] " \
            #             f"WHERE [Pseudo_Grid_ID] {in_} {grids_} " \
            #             f"AND [Date] >= '{y_start}' AND [Date] <= '{p_start}';"
            # with self.db_instance.engine.connect() as connection:
            #     ukcp09_dat = pd.read_sql(sql=sql_query, con=connection)

            # PostgreSQL
            table_name = y_start[:4]
            sql_query = f'SELECT * FROM "{self.SCHEMA_NAME}"."{table_name}" ' \
                        f'WHERE "Pseudo_Grid_ID" {in_} {grids_} ' \
                        f'AND "Date" >= \'{y_start}\' AND "Date" <= \'{p_start}\''

            data = self.db_instance.read_sql_query(sql_query)

            if pickle_it:
                save_data(data, path_to_pickle, verbose=verbose)

        return data
