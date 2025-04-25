import copy
import datetime
import os
import re
import string
import zipfile

import numpy as np
import pandas as pd
import shapely.geometry
import shapely.wkt
from pyhelpers.dbms import MSSQL
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.ops import update_dict_keys
from pyhelpers.store import load_data, save_data
from pyhelpers.text import find_similar_str
from pyrcs.converter import mileage_num_to_str, mileage_str_to_num, yard_to_mileage
from pyrcs.line_data import LocationIdentifiers
from pyrcs.utils import fetch_location_names_errata

from src.preprocessor.glossary import DelayAttributionGlossary
from src.utils import WxRailIncidentsPred, get_subset, update_route_names


class BaseMETEX:
    #: Name of the data.
    DATA_NAME: str = 'METEX'
    #: Pathname of the local data directory.
    DATA_DIR: str = os.path.relpath(cdd(f"{DATA_NAME.lower()}/database"))

    #: Name of the database in Microsoft SQL Server.
    MSSQL_DATABASE_NAME: str = 'NR_METEx_20190203'

    #: Schema name for the original data in the PostgreSQL database.
    POSTGRES_SCHEMA_NAME: str = copy.copy(MSSQL_DATABASE_NAME)

    #: Schema name for the preprocessed data.
    SCHEMA_NAME: str = POSTGRES_SCHEMA_NAME + '_prep'

    def __init__(self):
        self.db_instance = None

        self.imdm = None
        self.imdm_alias = None
        self.weather_cell_map = None
        self.incident_reason_info = None
        self.weather_codes = None
        self.incident_record = None
        self.location = None
        self.pfpi = None
        self.route = None
        self.stanox_location = None
        self.stanox_section = None
        self.trust_incident = None
        self.weather_cell = None
        self.weather_cell_map_boundary = None
        self.weather = None
        self.track = None
        self.track_summary = None

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change to the data directory.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: Whether to create a directory; defaults to ``False``.
        :type mkdir: bool
        :return: absolute path to "data\\METEX\\Database" and subdirectories / a file
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> import os
            >>> mt = METEX()
            >>> os.path.relpath(mt.cdd())
            'data\\METEX\\Database'
        """

        path = cd(self.DATA_DIR, *sub_dir, mkdir=mkdir)

        return path

    # == Read table data from the Microsoft SQL database ===========================================

    def read_mssql_table(self, mssql_table_name, route_name=None, weather_category=None, **kwargs):
        """
        Read data of a table in the METEX (Lite) database.

        :param mssql_table_name: name of a table
        :type mssql_table_name: str
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :param weather_category: Weather category; if ``None`` (default), all Weather categories
        :type weather_category: str | None
        :param kwargs: optional parameters of `pandas.read_sql`_
        :return: Data of the queried table stored in NR_METEX_* database
        :rtype: pandas.DataFrame

        .. _`pandas.read_sql`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html

        **Examples**::

            >>> from src.preprocessor._base import BaseMETEX
            >>> base_mtx = BaseMETEX()
            >>> imdm_tbl = base_mtx.read_mssql_table(mssql_table_name='IMDM')
            >>> imdm_tbl.head()
        """

        if self.db_instance is None:
            self.db_instance = MSSQL(database_name=self.MSSQL_DATABASE_NAME)

        sql_query = f'SELECT * FROM %s' % f'dbo.[{mssql_table_name}]'

        if route_name is not None and weather_category is None:
            sql_query += f" WHERE [Route]='{route_name}'"  # given Route
        elif route_name is None and weather_category is not None:
            sql_query += f" WHERE [weather_category]='{weather_category}'"
        elif route_name and weather_category:
            sql_query += f" WHERE [Route]='{route_name}' AND [weather_category]='{weather_category}'"

        index_col = self.db_instance.get_primary_keys(mssql_table_name)

        table_data = pd.read_sql(
            sql_query, con=self.db_instance.engine, index_col=index_col, **kwargs)

        return table_data

    # == Write/read table data to/from the PostgreSQL database =====================================

    @staticmethod
    def _table_name(table_name):
        if table_name.upper() == table_name:
            table_name_ = table_name.lower()
        elif table_name.lower() == table_name:
            table_name_ = table_name
        else:
            table_name_ = '_'.join(re.findall(r'[A-Z][^A-Z]*', table_name)).lower()

        return table_name_

    def _sql_query(self, table_name, raw=False):
        schema_name = self.POSTGRES_SCHEMA_NAME if raw else self.SCHEMA_NAME
        sql_query = 'SELECT * FROM {}'.format(f'"{schema_name}"."{table_name}"')

        return sql_query

    def _dump_prep_data(self, data, table_name, verbose, pkey=None, **kwargs):
        if verbose:
            tbl = f'"{self.SCHEMA_NAME}"."{table_name}"'
            if self.db_instance.table_exists(table_name, self.SCHEMA_NAME):
                msg = "Updating "
            else:
                msg = "Importing "
            print(msg + f"{tbl}", end=" ... ")

        try:
            self.db_instance.import_data(
                data=data, table_name=table_name, schema_name=self.SCHEMA_NAME, if_exists='replace',
                index=True, method=self.db_instance.psql_insert_copy, confirmation_required=False,
                **kwargs)

            if pkey is None:
                primary_keys = self.db_instance.get_primary_keys(
                    table_name=table_name, schema_name=self.POSTGRES_SCHEMA_NAME)
            else:
                primary_keys = pkey.copy()
            if primary_keys:
                self.db_instance.add_primary_keys(
                    primary_keys=primary_keys, table_name=table_name, schema_name=self.SCHEMA_NAME)

            if verbose:
                print("Done.")

        except Exception as e:
            if verbose:
                print(f"Failed. {e}")

    def _prep_data(self, table_name, verbose=False, **kwargs):
        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        data = self.db_instance.read_sql_query(sql_query, index_col=primary_keys, **kwargs)

        self._dump_prep_data(data=data, table_name=table_name, verbose=verbose)

        return data

    def _read_data(self, table_name, ivar_name=None, pkey=None, update=False, verbose=False,
                   ret_data=False, **kwargs):
        """
        Get data of a table.

        :param update:
        :type update: bool
        :param ret_data:
        :type ret_data: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of a table
        :rtype: pandas.DataFrame | None

        **Tests**::

            >>> from src.preprocessor._base import BaseMETEX
            >>> base_mtx = BaseMETEX()
            >>> base_mtx._read_data(table_name='IMDM')
            >>> base_mtx.imdm.shape
            (42, 3)
        """

        if ivar_name is None:
            name = self._table_name(table_name)
        else:
            name = copy.copy(ivar_name)
        path_to_pkl = self.cdd("tables", f"{name}.pkl.xz")

        if os.path.isfile(path_to_pkl) and not update:
            data = load_data(path_to_pkl, verbose=verbose)

        else:
            if self.db_instance is None:
                self.db_instance = WxRailIncidentsPred(verbose=False)

            if self.db_instance.table_exists(table_name, self.SCHEMA_NAME) and not update:
                sql_query = self._sql_query(table_name)

                if pkey is None:
                    primary_keys = self.db_instance.get_primary_keys(table_name, self.SCHEMA_NAME)
                else:
                    primary_keys = pkey.copy()

                if not primary_keys:
                    primary_keys = None
                data = self.db_instance.read_sql_query(sql_query, index_col=primary_keys, **kwargs)

            else:
                data = getattr(self, f'_{name}')(verbose=verbose)

            save_data(data, path_to_pkl, verbose=verbose)

        self.__setattr__(name, data)

        if ret_data:
            return data

    def _imdm(self, verbose=False, **kwargs):
        """
        Read data of 'IMDM'.

        :return: Data of the table 'IMDM'
        :rtype: pandas.DataFrame | None

        **Tests**::

            >>> from src.preprocessor.metex import METEX
            >>> from src.utils import WxRailIncidentsPred
            >>> db_instance = WxRailIncidentsPred()
            >>> mtx = METEX(db_instance=db_instance)
            >>> query = mtx._sql_query(table_name='IMDM', raw=True)
            >>> data = mtx.db_instance.read_sql_query(query)
            >>> data.shape
            (42, 2)
        """

        table_name = 'IMDM'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)

        imdm = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        imdm.index.rename(name='IMDM', inplace=True)  # Rename index
        imdm = update_route_names(imdm)  # Update names of 'Routes'

        # Add information about 'Regions'
        regions_and_routes = load_data(cdd("Network/Regions", "Routes.json"))
        regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]

        regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
        regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
        imdm = imdm.join(regions, on='Route')

        imdm = imdm.where((pd.notnull(imdm)), None)

        self._dump_prep_data(
            data=imdm, table_name=table_name, verbose=verbose, pkey=['IMDM'], **kwargs)

        return imdm

    def _imdm_alias(self, as_dict=False, verbose=False, **kwargs):
        """
        Read data of 'ImdmAlias'.

        :param as_dict: Whether to return the data as a dictionary; defaults to ``False``.
        :type as_dict: bool
        :return: Data of the table 'ImdmAlias'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'ImdmAlias'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        imdm_alias = self.db_instance.read_sql_query(
            sql_query, dtype={'Alias': str}, keep_default_na=False)

        imdm_alias.rename(columns={'Alias': 'IMDMAlias', 'Imdm': 'IMDM'}, inplace=True)

        # imdm_alias.IMDMAlias.fillna(imdm_alias.IMDM.str.upper(), inplace=True)
        imdm_alias.set_index('IMDMAlias', inplace=True)

        self._dump_prep_data(
            data=imdm_alias, table_name=table_name, verbose=verbose, pkey=['IMDMAlias'], **kwargs)

        if as_dict:
            imdm_alias = imdm_alias.to_dict()  # imdm_alias = imdm_alias['IMDM']

        return imdm_alias

    def _weather_cell_map(self, grouped=False, verbose=False, **kwargs):
        """
        Get data of the table 'IMDMWeatherCellMap'.

        :param grouped: Whether to group the data by either ``'Route'``; defaults to ``False``.
        :type grouped: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'IMDMWeatherCellMap'
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.metex import METEX

            >>> mt = METEX()

            >>> mt._weather_cell_map(verbose=True)
        """

        table_name = 'IMDMWeatherCellMap_pc'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        weather_cell_map = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        weather_cell_map.set_index('Id', inplace=True)

        weather_cell_map = update_route_names(weather_cell_map)

        index_name = 'IMDMWeatherCellMap' + weather_cell_map.index.name
        weather_cell_map.index.rename(index_name, inplace=True)
        weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'}, inplace=True)

        self._dump_prep_data(
            data=weather_cell_map, table_name='IMDMWeatherCellMap', verbose=verbose, pkey=[index_name],
            **kwargs)

        if grouped:
            weather_cell_map = weather_cell_map.groupby('Route').aggregate(
                lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))

        return weather_cell_map

    def _incident_reason_info(self, verbose=False, **kwargs):
        """
        Get data of the table 'IncidentReasonInfo'.

        :param plus: defaults to ``True``
        :type plus: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'IncidentReasonInfo'
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.metex import METEX

            >>> mt = METEX()
        """

        table_name = 'IncidentReasonInfo'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        incident_reason_info = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        index_name = 'IncidentReasonCode'
        incident_reason_info.index.rename(index_name, inplace=True)
        incident_reason_info.rename(
            columns={'Description': 'IncidentReasonDescription',
                     'Category': 'IncidentCategory',
                     'CategoryDescription': 'IncidentCategoryDescription'},
            inplace=True)

        # To include data of more detailed description about incident reasons
        dag = DelayAttributionGlossary()
        ir = dag._read_dag_data_from_db('Incident Reason')
        ir.columns = [x.replace('_', '') for x in ir.columns]
        ir.set_index('IncidentReason', inplace=True)
        ir.index.rename(index_name, inplace=True)
        incident_reason_info = ir.join(incident_reason_info, rsuffix='_drop').dropna(axis=1)

        self._dump_prep_data(
            data=incident_reason_info, table_name=table_name, verbose=verbose, pkey=[index_name],
            **kwargs)

        return incident_reason_info

    def _weather_codes(self, as_dict=False, verbose=False, **kwargs):
        """
        Get data of the table 'WeatherCategoryLookup'.

        :param as_dict: Whether to return the data as a dictionary; defaults to ``False``.
        :type as_dict: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'WeatherCategoryLookup'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'WeatherCodes'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        weather_codes = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        index_name = 'WeatherCategoryCode'
        weather_codes.rename(
            columns={'Code': index_name, 'Weather Category': 'WeatherCategory'}, inplace=True)

        weather_codes.set_index(index_name, inplace=True)

        self._dump_prep_data(
            data=weather_codes, table_name=table_name, verbose=verbose, pkey=[index_name],
            **kwargs)

        if as_dict:
            weather_codes = weather_codes.to_dict()

        return weather_codes

    def _incident_record(self, verbose=False, **kwargs):
        """
        Get data of the table 'IncidentRecord'.

        :param use_amendment_csv: Whether to use a supplementary .csv file
            to amend the original table data in the database; defaults to ``True``.
        :type use_amendment_csv: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'IncidentRecord'
        :rtype: pandas.DataFrame | None

        .. note::

            None values are filled with ``NaN``.
        """

        table_name = 'IncidentRecord'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        incident_record = self.db_instance.read_sql_query(
            sql_query, index_col=primary_keys, parse_dates=['CreateDate'])

        amendment_csv = pd.read_csv(
            self.cdd("Updates", table_name + ".zip"), index_col='Id', parse_dates=['CreateDate'],
            dayfirst=True)
        amendment_csv.columns = incident_record.columns
        idx = amendment_csv[amendment_csv.WeatherCategory.isna()].index
        amendment_csv.loc[idx, 'WeatherCategory'] = None

        incident_record.drop(
            incident_record[incident_record.CreateDate >= pd.to_datetime('2018-01-01')].index,
            inplace=True)
        incident_record = pd.concat([incident_record, amendment_csv], axis=0)

        index_name = table_name + 'Id'
        incident_record.index.rename(index_name, inplace=True)
        incident_record.rename(
            columns={'CreateDate': table_name + 'CreateDate', 'Reason': 'IncidentReasonCode'},
            inplace=True)

        # Replace each Weather category code with its full name
        weather_codes_dict = self._weather_codes(as_dict=True)
        incident_record.replace(weather_codes_dict, inplace=True)
        incident_record.fillna({'WeatherCategory': ''}, inplace=True)

        self._dump_prep_data(
            data=incident_record, table_name=table_name, verbose=verbose, pkey=[index_name],
            **kwargs)

        self.db_instance.null_text_to_empty_string(table_name, schema_name=self.SCHEMA_NAME)

        return incident_record

    def _location(self, verbose=False, **kwargs):
        """
        Get data of the table 'Location'.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'Location'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'Location'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        location = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        index_name = table_name + location.index.name
        location.index.rename('LocationId', inplace=True)
        location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
        location[['WeatherCell', 'SMDCell']] = location[['WeatherCell', 'SMDCell']].map(
            lambda x: 0 if np.isnan(x) else int(x))
        # location.loc[610096, 'StartLongitude':'EndLatitude'] = [-0.0751, 51.5461, -0.0751, 51.5461]

        self._dump_prep_data(
            data=location, table_name=table_name, verbose=verbose, pkey=[index_name], **kwargs)

        return location

    def _pfpi(self, verbose=False, **kwargs):
        """
        Get data of the table 'PfPI' (Process for Performance Improvement).

        :param plus: defaults to ``True``
        :type plus: bool
        :param update: Whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data; defaults to ``None``.
        :type save_original_as: str | None
        :param use_amendment_csv: Whether to use a supplementary .csv file
            to amend the original table data in the database; defaults to ``True``.
        :type use_amendment_csv: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'PfPI'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'PfPI'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        pfpi = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        # Amend
        incident_record = self._incident_record()
        incident_record['IncidentRecordCreateDate'] = pd.to_datetime(
            incident_record['IncidentRecordCreateDate'])
        min_id = incident_record[
            incident_record['IncidentRecordCreateDate'] >= datetime.datetime(2018, 1, 1)].index.min()
        pfpi.drop(pfpi[pfpi['IncidentRecordId'] >= min_id].index, inplace=True)
        amendments = pd.read_csv(self.cdd("updates", table_name + ".zip"), index_col='Id')
        pfpi = pd.concat([pfpi, amendments], axis=0)

        index_name = table_name + pfpi.index.name
        pfpi.index.rename(index_name, inplace=True)

        # To include more information for 'PerformanceEventCode'
        dag = DelayAttributionGlossary()
        perf_event_code = dag._read_dag_data_from_db('Performance Event Code')
        perf_event_code.columns = [x.replace('_', '') for x in perf_event_code.columns]
        perf_event_code.set_index('PerformanceEventCode', inplace=True)
        # Merge pfpi and performance_event_code
        pfpi = pfpi.join(perf_event_code, on='PerformanceEventCode')

        self._dump_prep_data(
            data=pfpi, table_name=table_name, verbose=verbose, pkey=[index_name], **kwargs)

        return pfpi

    def _route(self, as_dict=False, verbose=False, **kwargs):
        """
        Get data of the table 'Route'.

        :param as_dict: Whether to return the data as a dictionary; defaults to ``False``.
        :type as_dict: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'Route'
        :rtype: pandas.DataFrame | None

        .. note::

            There is only one column in the original table.
        """

        table_name = "Route"

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        # primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        route = self.db_instance.read_sql_query(sql_query, index_col=None)

        route.rename(columns={'Name': 'Route'}, inplace=True)
        route = update_route_names(route)

        # Add Regions
        regions_and_routes = load_data(cdd("network/regions", "routes.json"))
        regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]

        regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
        regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
        route = route.join(regions, on='Route')

        route = route.where((pd.notnull(route)), None)

        index_name = 'RouteAlias'
        route.set_index(index_name, inplace=True)

        self._dump_prep_data(
            data=route, table_name=table_name, verbose=verbose, pkey=[index_name], **kwargs)

        if as_dict:
            route = dict(route.Route)

        return route

    @staticmethod
    def _cleanse_raw_stanox_location(raw_stanox_location):
        dat = raw_stanox_location.copy()

        dat['Name'] = dat['Name'].str.upper()

        # Use external data - Railway Codes
        errata = load_data(cdd("network/railway_codes", "metex_location_name_errata.json"))
        errata = update_dict_keys(
            errata, replacements=dict(zip(errata.keys(), ['Stanox', 'Description', 'Name'])))
        # Note that {'CLAPS47': 'CLPHS47'} in err_tiploc is dubious.
        dat.replace(errata, inplace=True)

        duplicated_stanox = dat[dat['Stanox'].duplicated(keep=False)].sort_values('Stanox')
        # nan_idx = pd.isna(duplicated_stanox[['ELR', 'Yards', 'LocationId']]).all(axis=1)
        dat.drop(duplicated_stanox[duplicated_stanox['LocationId'].isna()].index, inplace=True)
        # dat.drop_duplicates(subset=['Stanox'], keep='last', inplace=True)

        lid = LocationIdentifiers()
        rc_codes = lid.fetch_codes()[lid.KEY]
        rc_codes.drop_duplicates(subset=['Location', 'TIPLOC', 'STANME', 'STANOX'], inplace=True)

        # Fill in NA 'Description' (i.e. Location names)
        na_desc_dat = dat[dat['Description'].isnull()]
        if not na_desc_dat.empty:
            temp = na_desc_dat.join(rc_codes.set_index('STANOX'), on='Stanox')
            dat.loc[na_desc_dat.index, ['Description', 'Name']] = temp[['Location', 'STANME']].values

        # Fill in NA 'Name's (i.e. STANME)
        na_name_dat = dat[dat['Name'].isna()]
        if not na_name_dat.empty:
            # Some records of the 'Description' are recorded by 'TIPLOC' instead
            temp = na_name_dat.join(
                rc_codes.set_index(['STANOX', 'TIPLOC']), on=['Stanox', 'Description'])
            temp = temp[temp['Location'].notnull() & temp['STANME'].notnull()]
            dat.loc[temp.index, ['Description', 'Name']] = temp[['Location', 'STANME']].values

        # Use manually-created dictionary of regular expressions
        dat.replace(fetch_location_names_errata(k='Description'), inplace=True)
        dat.replace(fetch_location_names_errata(k='Description', regex=True), inplace=True)

        # Check if 'Description' has STANOX instead of location name using STANOX-dictionary
        rc_stanox_dict = lid.make_xref_dict('STANOX')
        temp = dat.join(rc_stanox_dict, on='Description')
        valid_loc = temp[temp['Location'].notnull()][['Description', 'Name', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                lambda x: find_similar_str(x['Name'], x['Location'], engine='fuzz')
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Check if 'Description' has TIPLOC instead of location name using STANOX-TIPLOC-dictionary
        rc_stanox_tiploc_dict = lid.make_xref_dict(['STANOX', 'TIPLOC'])
        temp = dat.join(rc_stanox_tiploc_dict, on=['Stanox', 'Description'])
        valid_loc = temp[temp['Location'].notnull()][['Description', 'Name', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                lambda x: find_similar_str(x['Name'], x['Location'], engine='fuzz')
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Check if 'Description' has STANME instead of location name using STANOX-STANME-dictionary
        rc_stanox_stanme_dict = lid.make_xref_dict(['STANOX', 'STANME'])
        temp = dat.join(rc_stanox_stanme_dict, on=['Stanox', 'Description'])
        valid_loc = temp[temp['Location'].notnull()][['Description', 'Name', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                lambda x: find_similar_str(x['Name'], x['Location'], engine='fuzz')
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Finalize cleansing 'Description' (i.e. location names) using STANOX dictionary
        temp = dat.join(rc_stanox_dict, on='Stanox')[['Description', 'Location']]
        na_loc_idx = temp['Location'].isnull()
        temp.loc[na_loc_idx, 'Location'] = temp.loc[na_loc_idx, 'Description']
        dat.loc[temp.index, 'Description'] = temp.apply(
            lambda x: find_similar_str(x['Description'], x['Location'])
            if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        stanox_loc_ref = rc_codes.set_index(['STANOX', 'Location'])[['STANME']]
        temp = dat.join(stanox_loc_ref, on=['Stanox', 'Description'])
        temp.drop_duplicates(subset=['Stanox', 'Description', 'LocationId'], inplace=True)
        if temp.index.equals(dat.index):
            dat['Name'] = temp['STANME']

        for k in ['NLC', 'TIPLOC', 'STANME', 'STANOX']:
            temp = dat.join(rc_codes.set_index(['STANOX', k]), on=['Stanox', 'Description'])
            temp = temp[temp['Location'].notna()]
            if not temp.empty:
                dat.loc[temp.index, ['Description', 'Name']] = temp[['Location', 'STANME']].values

        dat.rename(columns={'Description': 'Location', 'Name': 'Stanme'}, inplace=True)

        return dat

    def _stanox_location(self, verbose=False, fix_known_errors=True, **kwargs):
        """
        Get data of the table 'StanoxLocation'.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'StanoxLocation'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'StanoxLocation'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        raw_dat = self.db_instance.read_sql_query(sql_query, dtype={'Stanox': str}, index_col=None)

        if fix_known_errors:
            raw_dat.loc[raw_dat['Stanox'] == '52053', :] = (
                '52053', 'Highbury & Islington (North London Lines)', 'HIGH&ISNL', 'BOK1',
                6072, 592571)
            raw_dat.loc[raw_dat['Stanox'] == '52074', :] = (
                '52074', 'Dalston Junction (East London Line)', '', 'ELL1',
                242, 610096)
            raw_dat.loc[raw_dat['Stanox'] == '32009', ['Description', 'Name']] = (
                'Ardwick Junction', 'ARDWICKJN')

        # Cleanse raw stanox_location
        stanox_location = self._cleanse_raw_stanox_location(raw_stanox_location=raw_dat)

        # For 'ELR', replace NaN with ''
        stanox_location['ELR'] = stanox_location['ELR'].fillna('')

        # For 'LocationId', replace NaN with -1
        stanox_location['LocationId'] = stanox_location['LocationId'].fillna(-1).astype(int)

        # Add 'Mileage'
        stanox_location['Mileage'] = stanox_location['Yards'].map(yard_to_mileage)

        # Set index
        # primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        stanox_location.set_index(['LocationId', 'Stanox'], inplace=True)

        # Note: 'Stanox' has duplicated values in it, so that `pkey=[]`
        self._dump_prep_data(
            data=stanox_location, table_name=table_name, verbose=verbose, pkey=[], **kwargs)

        self.db_instance.null_text_to_empty_string(table_name, schema_name=self.SCHEMA_NAME)

        return stanox_location

    def _cleanse_raw_stanox_section(self, raw_stanox_section):
        dat = raw_stanox_section.copy()

        # For 'LocationId', replace NaN with -1
        dat['LocationId'] = dat['LocationId'].fillna(-1).astype(int)

        # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
        errata = load_data(cdd("network/railway_codes/metex_location_name_errata.json"))
        errata_stanox = errata['STANOX']

        dat.replace({'StartStanox': errata_stanox, 'EndStanox': errata_stanox}, inplace=True)

        #
        temp = pd.DataFrame(dat['Description'].str.split(' : ').to_list())
        dat[['Start_Raw', 'End_Raw']] = dat[['Start', 'End']] = temp.values
        # Solve duplicated STANOX
        unknown_loc = load_data(cdd("network/railway_codes/problematic_data.json"))
        dat.replace({'Start': unknown_loc, 'End': unknown_loc}, inplace=True)

        # Use cleansed 'stanox_location'
        stanox_location = self._stanox_location()

        temp = dat.join(stanox_location, on=['LocationId', 'StartStanox'])
        temp = temp[temp['Location'].notna()]
        dat.loc[temp.index, 'Start'] = temp['Location'].values
        temp = dat.join(stanox_location, on=['LocationId', 'EndStanox'])
        temp = temp[temp['Location'].notna()]
        dat.loc[temp.index, 'End'] = temp['Location'].values

        # Use Railway Codes
        lid = LocationIdentifiers()

        rc_codes = lid.fetch_codes()[lid.KEY]
        rc_codes.drop_duplicates(subset=['Location', 'TIPLOC', 'STANME', 'STANOX'], inplace=True)
        rc_codes.set_index('STANOX', inplace=True)

        # Check if 'Start' and 'End' have STANOX instead of location names using STANOX-dictionary
        rc_stanox_dict = lid.make_xref_dict('STANOX')
        temp = dat.join(rc_stanox_dict, on='StartStanox')
        valid_loc = temp[temp['Location'].notnull()]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                lambda x: find_similar_str(x['Start'], x['Location'], engine='fuzz')
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)
        temp = dat.join(rc_stanox_dict, on='EndStanox')
        valid_loc = temp[temp['Location'].notnull()]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'End'] = valid_loc.apply(
                lambda x: find_similar_str(x['End'], x['Location'], engine='fuzz')
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Check if 'Start' and 'End' have STANOX/TIPLOC codes using STANOX-TIPLOC-dictionary
        rc_stanox_tiploc_dict = lid.make_xref_dict(['STANOX', 'TIPLOC'])
        temp = dat.join(rc_stanox_tiploc_dict, on=['StartStanox', 'Start'])
        valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                lambda x: find_similar_str(x['End'], x['Location'])
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)
        temp = dat.join(rc_stanox_tiploc_dict, on=['EndStanox', 'End'])
        valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'End'] = valid_loc.apply(
                lambda x: find_similar_str(x['End'], x['Location'])
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Check if 'Start' and 'End' have STANOX/STANME codes using STANOX-STANME-dictionary
        rc_stanox_stanme_dict = lid.make_xref_dict(['STANOX', 'STANME'])
        temp = dat.join(rc_stanox_stanme_dict, on=['StartStanox', 'Start'])
        valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                lambda x: find_similar_str(x['Start'], x['Location'])
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)
        temp = dat.join(rc_stanox_stanme_dict, on=['EndStanox', 'End'])
        valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
        if not valid_loc.empty:
            dat.loc[valid_loc.index, 'End'] = valid_loc.apply(
                lambda x: find_similar_str(x['End'], x['Location'])
                if isinstance(x['Location'], (list, tuple)) else x['Location'], axis=1)

        # Apply manually-created dictionaries
        loc_name_replacement_dict = fetch_location_names_errata('Start')
        dat.replace(loc_name_replacement_dict, inplace=True)
        loc_name_regexp_replacement_dict = fetch_location_names_errata('Start', regex=True)
        dat.replace(loc_name_regexp_replacement_dict, inplace=True)
        loc_name_replacement_dict = fetch_location_names_errata('End')
        dat.replace(loc_name_replacement_dict, inplace=True)
        loc_name_regexp_replacement_dict = fetch_location_names_errata('End', regex=True)
        dat.replace(loc_name_regexp_replacement_dict, inplace=True)

        # Finalize cleansing
        na_start_dat = dat[dat['Start'].isnull()]
        temp = na_start_dat.join(rc_stanox_dict, on='StartStanox')
        temp = temp[temp['Location'].notnull()][['Start', 'Location']]
        dat.loc[temp.index, 'Start'] = temp.apply(
            lambda x: x['Location'][0] if isinstance(x['Location'], (list, tuple)) else x['Location'],
            axis=1).values

        na_end_idx = dat[dat['StartStanox'].eq(dat['EndStanox'])].index
        dat.loc[na_end_idx, 'End'] = dat.loc[na_end_idx, 'Start'].values

        na_end_dat = dat[dat['End'].isnull()]
        if not na_end_dat.empty:
            temp = na_end_dat.join(rc_stanox_dict, on='EndStanox')
            temp = temp[temp['Location'].notnull()][['End', 'Location']]
            dat.loc[temp.index, 'End'] = temp.apply(
                lambda x: x['Location'][0] if isinstance(x['Location'], (list, tuple))
                else x['Location'], axis=1)

        section = dat['Start'].str.strip() + ' : ' + dat['End'].str.strip()
        non_section_dat_idx = dat['Start'].eq(dat['End'])
        section.loc[non_section_dat_idx] = dat[non_section_dat_idx]['Start'].values

        dat['Description'] = section

        dat.insert(dat.columns.get_loc('StartStanox') + 1, 'StartLocation', dat['Start'].str.strip())
        dat.insert(dat.columns.get_loc('EndStanox') + 1, 'EndLocation', dat['End'].str.strip())

        # Finalizing the cleaning process
        dat.drop(columns=['Start_Raw', 'End_Raw', 'Start', 'End'], inplace=True)
        dat.rename(columns={'Description': 'StanoxSection'}, inplace=True)

        return dat

    def _stanox_section(self, verbose=False, **kwargs):
        """
        Get data of the table 'StanoxSection'.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'StanoxSection'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'StanoxSection'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        raw_dat = self.db_instance.read_sql_query(
            sql_query, dtype={'StartStanox': str, 'EndStanox': str}, index_col=primary_keys,
            true_values=['t', 'true'], false_values=['f', 'false'])

        stanox_section = self._cleanse_raw_stanox_section(raw_stanox_section=raw_dat)

        index_name = table_name + 'Id'
        stanox_section.index.name = index_name

        self._dump_prep_data(
            data=stanox_section, table_name=table_name, verbose=verbose, pkey=[index_name],
            **kwargs)

        return stanox_section

    def _trust_incident(self, start_year=2006, end_year=None, verbose=False, **kwargs):
        """
        Get data of the table 'TrustIncident'.

        :param start_year: defaults to ``2006``
        :type start_year: int | None
        :param end_year: defaults to ``None``
        :type end_year: int | None
        :param use_amendment_csv: Whether to use a supplementary .csv file
            to amend the original table data in the database; defaults to ``True``.
        :type use_amendment_csv: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'TrustIncident'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'TrustIncident'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        trust_incident_ = self.db_instance.read_sql_query(
            sql_query, index_col=primary_keys, parse_dates=['StartDate', 'EndDate'],
            true_values=['t', 'true'], false_values=['f', 'false'], keep_default_na=False,
            low_memory=False)

        # Amend
        with zipfile.ZipFile(self.cdd("Updates", table_name + ".zip")) as zip_file:
            dat_list = []
            for f in zip_file.infolist():
                dat = pd.read_csv(
                    zip_file.open(f), index_col='Id', parse_dates=['StartDate', 'EndDate'],
                    keep_default_na=False)
                dat_list.append(dat)
            amendments = pd.concat(dat_list, axis=0)

        # Remove raw data >= '2018-01-01', pd.to_datetime('2018-01-01')
        trust_incident_.drop(
            trust_incident_[trust_incident_['StartDate'] >= pd.to_datetime('2018-01-01')].index,
            inplace=True)
        trust_incident = pd.concat([trust_incident_, amendments], axis=0)

        index_name = table_name + trust_incident.index.name
        trust_incident.index.rename(index_name, inplace=True)
        trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
        # Extract a subset of data, where StartDateTime is between 'start_year' and 'end_year'
        trust_incident = trust_incident[
            (trust_incident['FinancialYear'] >=
             (start_year if start_year else 0)) &
            (trust_incident['FinancialYear'] <=
             (end_year if end_year else datetime.datetime.now().year))]

        self._dump_prep_data(
            data=trust_incident, table_name=table_name, verbose=verbose, pkey=[index_name], **kwargs)

        self.db_instance.null_text_to_empty_string(table_name, schema_name=self.SCHEMA_NAME)

        return trust_incident

    def _weather_cell(self, route_name=None, verbose=False, **kwargs):
        """
        Get data of the table 'WeatherCell'.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :param show_map: Whether to show a Map of the Weather cells; defaults to ``False``.
        :type show_map: bool
        :param save_map_as: Whether to save the created Map or what format the created Map is saved as,
            defaults to ``None``
        :type save_map_as: str | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'WeatherCell'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'WeatherCell'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        weather_cell_ = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        index_name = table_name + weather_cell_.index.name
        weather_cell_.index.rename(index_name, inplace=True)

        # Get IMDM Weather cell Map
        weather_cell_map = self._weather_cell_map()

        # Merge the acquired data set
        weather_cell = weather_cell_map.join(
            weather_cell_, on=index_name).sort_values(index_name).reset_index()
        weather_cell.set_index(index_name, inplace=True)

        # Lower left corner:
        weather_cell['ll_Longitude'] = weather_cell.Longitude  # - weather_cell.width / 2
        weather_cell['ll_Latitude'] = weather_cell.Latitude  # - weather_cell.height / 2
        # Upper left corner:
        weather_cell['ul_Longitude'] = weather_cell.ll_Longitude  # - weather_cell.width / 2
        weather_cell['ul_Latitude'] = weather_cell.ll_Latitude + weather_cell.height  # / 2
        # Upper right corner:
        weather_cell['ur_Longitude'] = weather_cell.ul_Longitude + weather_cell.width  # / 2
        weather_cell['ur_Latitude'] = weather_cell.ul_Latitude  # + weather_cell.height / 2
        # Lower right corner:
        weather_cell['lr_Longitude'] = weather_cell.ur_Longitude  # + weather_cell.width  # / 2
        weather_cell['lr_Latitude'] = weather_cell.ur_Latitude - weather_cell.height  # / 2

        # Create polygons WGS84 (Longitude, Latitude)
        weather_cell['polygon_WGS84'] = weather_cell.apply(
            lambda x: shapely.geometry.Polygon(
                zip([x['ll_Longitude'], x['ul_Longitude'], x['ur_Longitude'], x['lr_Longitude']],
                    [x['ll_Latitude'], x['ul_Latitude'], x['ur_Latitude'], x['lr_Latitude']])),
            axis=1)

        # Create polygons OSGB36 (Easting, Northing)
        weather_cell['ll_Easting'], weather_cell['ll_Northing'] = \
            wgs84_to_osgb36(weather_cell.ll_Longitude.values, weather_cell.ll_Latitude.values)
        weather_cell['ul_Easting'], weather_cell['ul_Northing'] = \
            wgs84_to_osgb36(weather_cell.ul_Longitude.values, weather_cell.ul_Latitude.values)
        weather_cell['ur_Easting'], weather_cell['ur_Northing'] = \
            wgs84_to_osgb36(weather_cell.ur_Longitude.values, weather_cell.ur_Latitude.values)
        weather_cell['lr_Easting'], weather_cell['lr_Northing'] = \
            wgs84_to_osgb36(weather_cell.lr_Longitude.values, weather_cell.lr_Latitude.values)

        weather_cell['polygon_OSGB36'] = weather_cell.apply(
            lambda x: shapely.geometry.Polygon(
                zip([x['ll_Easting'], x['ul_Easting'], x['ur_Easting'], x['lr_Easting']],
                    [x['ll_Northing'], x['ul_Northing'], x['ur_Northing'], x['lr_Northing']])),
            axis=1)

        regions_and_routes = load_data(cdd("network/regions/routes.json"))
        regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
        # noinspection PyTypeChecker
        regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
        weather_cell['Region'] = weather_cell['Route'].replace(regions_and_routes_dict)

        kwargs.update({'verbose': verbose})

        self._dump_prep_data(data=weather_cell, table_name=table_name, pkey=[], **kwargs)

        weather_cell = get_subset(data=weather_cell, route_name=route_name)

        return weather_cell

    def _track(self, verbose=False, **kwargs):
        """
        Get data of the table 'Track'.

        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'Track'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'Track'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        track_ = self.db_instance.read_sql_query(
            sql_query, dtype={'StartMileage': str, 'EndMileage': str}, index_col=primary_keys)

        # track_.geom = track_.geom.Map(shapely.wkt.loads)

        track_.rename(
            columns={'S_MILEAGE': 'StartMileage', 'F_MILEAGE': 'EndMileage',
                     'S_YARDAGE': 'StartYard', 'F_YARDAGE': 'EndYard',
                     'MAINTAINER': 'Maintainer', 'ROUTE': 'Route',
                     'DELIVERY_U': 'IMDM',
                     'StartEasti': 'StartEasting', 'StartNorth': 'StartNorthing',
                     'EndNorthin': 'EndNorthing'},
            inplace=True)

        # Mileage and Yardage
        mileage_cols, yardage_cols = ['StartMileage', 'EndMileage'], ['StartYard', 'EndYard']
        track_[mileage_cols] = track_[mileage_cols].map(mileage_num_to_str)
        track_[yardage_cols] = track_[yardage_cols].map(int)

        # Delivery Unit and IMDM
        track_.IMDM = track_.IMDM.map(lambda x: '' if pd.isnull(x) else 'IMDM ' + x)

        # Start and end longitude and latitude coordinates
        track_['StartLongitude'], track_['StartLatitude'] = osgb36_to_wgs84(
            track_.StartEasting.values, track_.StartNorthing.values)
        track_['EndLongitude'], track_['EndLatitude'] = osgb36_to_wgs84(
            track_.EndEasting.values, track_.EndNorthing.values)

        track_[['StartMileage_num', 'EndMileage_num']] = track_[
            ['StartMileage', 'EndMileage']].map(mileage_str_to_num)

        # Route
        track = update_route_names(track_, route_col_name='Route').drop_duplicates()

        index_names = ['ELR', 'TrackID', 'StartMileage', 'EndMileage']
        track.set_index(index_names, inplace=True)
        track.sort_index(inplace=True)

        self._dump_prep_data(
            data=track, table_name=table_name, verbose=verbose, pkey=index_names, **kwargs)

        return track

    def _track_summary(self, verbose=False, **kwargs):
        """
        Get data of the table 'Track Summary'.

        :param update: Whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data; defaults to ``None``.
        :type save_original_as: str | None
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of the table 'Track Summary'
        :rtype: pandas.DataFrame | None
        """

        table_name = 'Track Summary'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        track_summary_ = self.db_instance.read_sql_query(
            sql_query, index_col=primary_keys, low_memory=False)

        track_summary_.iloc[:, 45] = track_summary_.iloc[:, 45].fillna('')

        # Change column names
        rename_cols = {
            'GEOGIS Switch ID': 'GeoGISSwitchID',
            'TID': 'TrackID',
            'Start Yards': 'StartYards',
            'End Yards': 'EndYards',
            'Sub-route': 'SubRoute',
            'CP6 criticality': 'CP6Criticality',
            'CP5 Start Route': 'CP5StartRoute',
            'Adjacent S&C': 'AdjacentS&C',
            'Rail cumulative EMGT': 'RailCumulativeEMGT',
            'Sleeper cumulative EMGT': 'SleeperCumulativeEMGT',
            'Ballast cumulative EMGT': 'BallastCumulativeEMGT'
        }
        track_summary_.rename(columns=rename_cols, inplace=True)

        renamed_cols = list(rename_cols.values())
        upper_columns = [
            'ID', 'SRS', 'ELR', 'IMDM', 'TME', 'TSM', 'MGTPA', 'EMGTPA', 'LTSF', 'IRJs']
        track_summary_.columns = [
            string.capwords(x).replace(' ', '') if x not in upper_columns + renamed_cols else x
            for x in track_summary_.columns]

        # IMDM
        track_summary_.IMDM = track_summary_.IMDM.map(lambda x: 'IMDM ' + x)

        # Route
        route_names_changes = load_data(cdd("network/routes/name_changes.json"))

        temp1 = pd.DataFrame.from_dict(route_names_changes, orient='index', columns=['Route'])
        route_names_in_table = list(track_summary_.SubRoute.unique())
        route_alt = [
            find_similar_str(x, temp1.index.tolist(), engine='fuzz')
            for x in route_names_in_table]

        temp2 = pd.DataFrame.from_dict(
            dict(zip(route_names_in_table, route_alt)), orient='index', columns=['RouteAlias'])
        temp = temp2.join(temp1, on='RouteAlias').dropna()
        route_names_changes_alt = dict(zip(temp.index, temp.Route))
        track_summary_['Route'] = track_summary_.SubRoute.replace(route_names_changes_alt)

        # Mileages
        mileage_colnames, yard_colnames = ['StartMileage', 'EndMileage'], ['StartYards', 'EndYards']
        track_summary_[mileage_colnames] = track_summary_[yard_colnames].map(yard_to_mileage)

        index_names = ['ELR', 'TrackID', 'StartYards', 'EndYards']
        track_summary = track_summary_.set_index(index_names)
        track_summary.sort_index(inplace=True)

        new_table_name = table_name.replace(' ', '')

        self._dump_prep_data(
            data=track_summary, table_name=new_table_name, verbose=verbose, pkey=index_names,
            **kwargs)

        self.db_instance.null_text_to_empty_string(
            table_name=new_table_name, schema_name=self.SCHEMA_NAME)

        return track_summary


class BaseVegetation:
    #: Name of the data.
    DATA_NAME: str = 'Vegetation'
    #: Pathname of the local data directory.
    DATA_DIR: str = os.path.relpath(cdd(f"{DATA_NAME.lower()}/database"))

    #: Name of the database in Microsoft SQL Server.
    MSSQL_DATABASE_NAME: str = 'NR_Vegetation_20141031'

    #: Schema name for the original data in the PostgreSQL database.
    POSTGRES_SCHEMA_NAME: str = copy.copy(MSSQL_DATABASE_NAME)

    #: Schema name for the preprocessed data.
    SCHEMA_NAME: str = POSTGRES_SCHEMA_NAME + '_prep'

    def __init__(self):
        self.mssql = None

    # == Change directories ========================================================================

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\Vegetation\\Database\\" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\Vegetation\\Database\\" and subdirectories / a file
        :rtype: str

        **Examples**::

            >>> from src.preprocessor._base import BaseVegetation
            >>> import os
            >>> veg = BaseVegetation()
            >>> os.path.relpath(veg.cdd())
            'data\\Vegetation'
        """

        path = cd(self.DATA_DIR, *sub_dir, mkdir=mkdir)

        return path

    # == Read table data from the Microsoft SQL Database ===========================================

    def read_mssql_table(self, mssql_table_name, route_name=None, save_as=None, update=False,
                         **kwargs):
        """
        Read Tables stored in NR_Vegetation_* Database.

        :param mssql_table_name: name of a table
        :type mssql_table_name: str
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param save_as: file extension, defaults to ``None``
        :type save_as: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param kwargs: optional parameters of `pandas.read_sql`_
        :return: data of the queried table stored in NR_Vegetation_* Database
        :rtype: pandas.DataFrame

        .. _`pandas.read_sql`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html

        **Examples**::

            >>> from src.preprocessor._base import BaseVegetation
            >>> veg = BaseVegetation()
            >>> dat = veg.read_mssql_table(mssql_table_name='AdverseWind', route_name='Anglia')
            >>> dat.shape
            (1, 1)
        """

        if self.mssql is None:
            self.mssql = MSSQL(database_name=self.MSSQL_DATABASE_NAME)

        sql_query = f'SELECT * FROM %s' % f'dbo.[{mssql_table_name}]'

        if route_name is not None:
            sql_query += f" WHERE [Route] = '{route_name}'"  # given a specific Route

        index_col = self.mssql.get_primary_keys(mssql_table_name)

        data = pd.read_sql(sql=sql_query, con=self.mssql.engine, index_col=index_col, **kwargs)

        if save_as:  # Save data locally
            path_to_file = self.cdd("Tables", mssql_table_name + save_as)

            if not os.path.isfile(path_to_file) or update:
                save_data(data, path_to_file, index=False if index_col is None else True)

        return data

    # == Write/read table data to/from the PostgreSQL Database =====================================

    @staticmethod
    def _table_name(table_name):
        if table_name.upper() == table_name:
            table_name_ = table_name.lower()
        elif table_name.lower() == table_name:
            table_name_ = table_name
        else:
            table_name_ = '_'.join(re.findall(r'[A-Z][^A-Z]*', table_name)).lower()

        return table_name_

    def _sql_query(self, table_name, raw=False):
        schema_name = self.POSTGRES_SCHEMA_NAME if raw else self.SCHEMA_NAME
        sql_query = f'SELECT * FROM "{schema_name}"."{table_name}"'

        return sql_query

    def _dump_prep_data(self, data, table_name, verbose, pkey=None, **kwargs):
        if verbose:
            tbl = f'"{self.SCHEMA_NAME}"."{table_name}"'
            if self.db_instance.table_exists(table_name, self.SCHEMA_NAME):
                msg = "Updating "
            else:
                msg = "Importing "
            print(msg + f"{tbl}", end=" ... ")

        try:
            self.db_instance.import_data(
                data=data, table_name=table_name, schema_name=self.SCHEMA_NAME, if_exists='replace',
                index=True, method=self.db_instance.psql_insert_copy, confirmation_required=False,
                **kwargs)

            if pkey is None:
                primary_keys = self.db_instance.get_primary_keys(
                    table_name=table_name, schema_name=self.POSTGRES_SCHEMA_NAME)
            else:
                primary_keys = copy.copy(pkey)

            if primary_keys:
                self.db_instance.add_primary_keys(
                    primary_keys=primary_keys, table_name=table_name, schema_name=self.SCHEMA_NAME)

            if verbose:
                print("Done.")

        except Exception as e:
            if verbose:
                print("Failed. {}".format(e))

    def _prep_data(self, table_name, verbose=False, **kwargs):
        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(
            table_name=table_name, schema_name=self.POSTGRES_SCHEMA_NAME)

        data = self.db_instance.read_sql_query(sql_query, index_col=primary_keys, **kwargs)

        self._dump_prep_data(data=data, table_name=table_name, verbose=verbose)

        return data

    def _read_data(self, table_name, ivar_name=None, pkey=None, update=False, verbose=False,
                   ret_data=False, **kwargs):
        """
        Get data of a table.

        :param update:
        :type update: bool
        :param ret_data:
        :type ret_data: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: data of a table
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.preprocessor._base import BaseVegetation

            >>> veg = BaseVegetation()

            >>> veg._read_data(table_name='AdverseWind')
            >>> veg.__getattribute__('adverse_wind').shape
            (9, 2)
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        if ivar_name is None:
            name = self._table_name(table_name)
        else:
            name = copy.copy(ivar_name)

        if self.db_instance.table_exists(table_name, self.SCHEMA_NAME) and not update:
            if pkey is None:
                primary_keys = self.db_instance.get_primary_keys(table_name, self.SCHEMA_NAME)
            else:
                primary_keys = copy.copy(pkey)
            sql_query = self._sql_query(table_name)
            data = self.db_instance.read_sql_query(sql_query, index_col=primary_keys, **kwargs)
        else:
            data = getattr(self, '_' + name)(verbose=verbose)

        self.__setattr__(name, data)

        if ret_data:
            return data

    # == Get table data ============================================================================

    def _adverse_wind(self, verbose=False, **kwargs):
        table_name = 'AdverseWind'

        sql_query = self._sql_query(table_name=table_name, raw=True)
        adverse_wind_ = self.db_instance.read_sql_query(sql_query)

        adverse_wind = update_route_names(adverse_wind_, route_col_name='Route')  # Update route names

        adverse_wind = adverse_wind.groupby('Route').agg(list).map(
            lambda x: x if len(x) > 1 else x[0])

        kwargs.update(dict(data=adverse_wind, table_name=table_name, verbose=verbose))
        self._dump_prep_data(**kwargs)

        return adverse_wind

    def _cutting_angle_class(self, verbose=False, **kwargs):
        cutting_angle_class = self._prep_data(
            table_name='CuttingAngleClass', verbose=verbose, **kwargs)

        return cutting_angle_class

    def _cutting_depth_class(self, verbose=False, **kwargs):
        cutting_depth_class = self._prep_data(table_name='CuttingDepthClass', verbose=verbose, **kwargs)

        return cutting_depth_class

    def _du_list(self, verbose=False, **kwargs):
        # MDU: Maintenance Delivery Units
        du_list = self._prep_data(table_name='DUList', verbose=verbose, **kwargs)

        return du_list

    def _path_route(self, verbose=False, **kwargs):
        path_route = self._prep_data(
            table_name='PathRoute', verbose=verbose, keep_default_na=False,
            true_values=['t', 'true'], false_values=['f', 'false'],
            **kwargs)

        return path_route

    def _routes(self, verbose=False, **kwargs):
        table_name = 'Routes'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)

        # (Note that 'Routes' table contains information about Delivery Units)
        routes = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        # Replace values in (index) column 'DUName'
        routes.index = routes.index.to_series().replace(
            {'Lanc&Cumbria MDU - HR1': 'Lancashire & Cumbria MDU - HR1',
             'S/wel& Dud MDU - HS7': 'Sandwell & Dudley MDU - HS7'})
        # Replace values in column 'DUNameGIS'
        routes.DUNameGIS.replace({'IMDM  Lanc&Cumbria': 'IMDM Lancashire & Cumbria'}, inplace=True)
        # Update route names
        routes = update_route_names(routes, route_col_name='Route')

        self._dump_prep_data(data=routes, table_name=table_name, verbose=verbose, **kwargs)

        return routes

    def _s8data(self, verbose=False, **kwargs):
        """Get data of the table 'S8Data'."""
        table_name = 'S8Data'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)

        s8data = self.db_instance.read_sql_query(sql_query, index_col=primary_keys)

        s8data = update_route_names(s8data, route_col_name='Route')

        self._dump_prep_data(data=s8data, table_name=table_name, verbose=verbose, **kwargs)

        return s8data

    def _tree_age_class(self, verbose=False, **kwargs):
        """Get data of the table 'TreeAgeClass'."""
        tree_age_class = self._prep_data(table_name='TreeAgeClass', verbose=verbose, **kwargs)
        return tree_age_class

    def _tree_size_class(self, verbose=False, **kwargs):
        """Get data of the table 'TreeSizeClass'."""
        tree_size_class = self._prep_data(table_name='TreeSizeClass', verbose=verbose, **kwargs)

        return tree_size_class

    def _tree_type(self, verbose=False, **kwargs):
        """Get data of the table 'TreeType'."""
        tree_type = self._prep_data(table_name='TreeType', verbose=verbose, **kwargs)

        return tree_type

    def _felling_type(self, verbose=False, **kwargs):
        """Get data of the table 'FellingType'."""
        felling_type = self._prep_data(table_name='FellingType', verbose=verbose, **kwargs)
        return felling_type

    def _area_work_type(self, verbose=False, **kwargs):
        """Get data of the table 'AreaWorkType'."""
        area_work_type = self._prep_data(table_name='AreaWorkType', verbose=verbose, **kwargs)
        return area_work_type

    def _service_detail(self, verbose=False, **kwargs):
        """Get data of the table 'ServiceDetail'."""
        service_detail = self._prep_data(table_name='ServiceDetail', verbose=verbose, **kwargs)
        return service_detail

    def _service_path(self, verbose=False, **kwargs):
        """Get data of the table 'ServicePath'."""
        service_path = self._prep_data(table_name='ServicePath', verbose=verbose, **kwargs)
        return service_path

    def _supplier(self, verbose=False, **kwargs):
        """Get data of the table 'Supplier'."""
        supplier = self._prep_data(table_name='Supplier', verbose=verbose, **kwargs)
        return supplier

    def _supplier_costs(self, verbose=False, **kwargs):
        """Get data of the table 'SupplierCosts'."""

        table_name = 'SupplierCosts'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        supplier_costs = self.db_instance.read_sql_query(sql_query, index_col=None)

        supplier_costs = update_route_names(supplier_costs, route_col_name='Route')

        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        primary_keys = ['RouteAlias' if x == 'Route' else x for x in primary_keys]
        supplier_costs.set_index(primary_keys, inplace=True)

        self._dump_prep_data(
            data=supplier_costs, table_name=table_name, verbose=verbose, pkey=primary_keys, **kwargs)

        return supplier_costs

    def _supplier_costs_area(self, verbose=False, **kwargs):
        """
        Get data of the table 'SupplierCostsArea'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'SupplierCostsArea'
        :rtype: pandas.DataFrame or None
        """

        table_name = 'SupplierCostsArea'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        costs_area = self.db_instance.read_sql_query(sql_query, index_col=None)

        costs_area = update_route_names(costs_area, route_col_name='Route')

        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        primary_keys = ['RouteAlias' if x == 'Route' else x for x in primary_keys]
        costs_area.set_index(primary_keys, inplace=True)

        self._dump_prep_data(
            data=costs_area, table_name=table_name, verbose=verbose, pkey=primary_keys, **kwargs)

        return costs_area

    def _supplier_costs_simple(self, verbose=False, **kwargs):
        """
        Get data of the table 'SupplierCostsSimple'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'SupplierCostsSimple'
        :rtype: pandas.DataFrame or None
        """

        table_name = 'SupplierCostsSimple'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        supplier_cost_simple = self.db_instance.read_sql_query(sql_query, index_col=None)

        supplier_cost_simple = update_route_names(supplier_cost_simple, route_col_name='Route')

        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        primary_keys = ['RouteAlias' if x == 'Route' else x for x in primary_keys]
        supplier_cost_simple.set_index(primary_keys, inplace=True)

        self._dump_prep_data(
            data=supplier_cost_simple, table_name=table_name, verbose=verbose, pkey=primary_keys,
            **kwargs)

        return supplier_cost_simple

    def _tree_action_fractions(self, verbose=False, **kwargs):
        """
        Get data of the table 'TreeActionFractions'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TreeActionFractions'
        :rtype: pandas.DataFrame or None
        """

        table_name = 'TreeActionFractions'
        tree_action_fractions = self._prep_data(table_name=table_name, verbose=verbose, **kwargs)

        return tree_action_fractions

    def _veg_surv_type_class(self, verbose=False, **kwargs):
        """
        Get data of the table 'VegSurvTypeClass'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'VegSurvTypeClass'
        :rtype: pandas.DataFrame or None
        """

        veg_surv_type_class = self._prep_data(table_name='VegSurvTypeClass', verbose=verbose, **kwargs)

        return veg_surv_type_class

    def _wb_factors(self, verbose=False, **kwargs):
        """
        Get data of the table 'WBFactors'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WBFactors'
        :rtype: pandas.DataFrame or None
        """

        wb_factors = self._prep_data(table_name='WBFactors', verbose=verbose, **kwargs)

        return wb_factors

    def _weed_spray(self, verbose=False, **kwargs):
        """
        Get data of the table 'Weedspray'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Weedspray'
        :rtype: pandas.DataFrame or None
        """

        table_name = 'Weedspray'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        weed_spray = self.db_instance.read_sql_query(sql_query, index_col=None)

        weed_spray = update_route_names(weed_spray, route_col_name='Route')

        weed_spray.set_index(['RouteAlias'], inplace=True)

        self._dump_prep_data(
            data=weed_spray, table_name=table_name, verbose=verbose, pkey=['RouteAlias'], **kwargs)

        return weed_spray

    def _work_hours(self, verbose=False, **kwargs):
        """
        Get data of the table 'WorkHours'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WorkHours'
        :rtype: pandas.DataFrame or None
        """

        work_hours = self._prep_data(table_name='WorkHours', verbose=verbose, **kwargs)

        return work_hours

    def _furlong_data(self, pseudo_amendment=True, verbose=False, **kwargs):
        """
        Get data of the table 'FurlongData'.

        :param pseudo_amendment: whether to make an amendment with external data,
            defaults to ``True``
        :type pseudo_amendment: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'FurlongData'
        :rtype: pandas.DataFrame or None

        .. note::

            Equipment Class: VL ('VEGETATION - 1/8 MILE SECTION')
            1/8 mile = 220 yards = 1 furlong
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        table_name = 'FurlongData'

        sql_query = self._sql_query(table_name=table_name, raw=True)

        furlong_data = self.db_instance.read_sql_query(sql_query, index_col=None, **kwargs)

        # Re-format mileage data
        furlong_data[['StartMileage', 'EndMileage']] = furlong_data[
            ['StartMileage', 'EndMileage']].map(mileage_num_to_str)

        # Rename columns
        renamed_cols_dict = {
            'TEF307601': 'MainSpeciesScore',
            'TEF307602': 'TreeSizeScore',
            'TEF307603': 'SurroundingLandScore',
            'TEF307604': 'DistanceFromRailScore',
            'TEF307605': 'OtherVegScore',
            'TEF307606': 'TopographyScore',
            'TEF307607': 'AtmosphereScore',
            'TEF307608': 'TreeDensityScore'}
        furlong_data.rename(columns=renamed_cols_dict, inplace=True)
        # Edit the 'TEF' columns
        furlong_data.OtherVegScore.replace({-1: 0}, inplace=True)
        renamed_cols = list(renamed_cols_dict.values())
        furlong_data[renamed_cols] = furlong_data[renamed_cols].map(
            lambda x: 0 if np.isnan(x) else x + 1)
        # Re-format date of measure
        furlong_data.DateOfMeasure = furlong_data.DateOfMeasure.map(
            lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M'))
        # Edit route data
        furlong_data = update_route_names(furlong_data, route_col_name='Route')

        # Make amendment to "CoverPercent" data for which the total is not 0 or 100?
        if pseudo_amendment:
            # Find columns relating to "CoverPercent..."
            cp_cols = [x for x in furlong_data.columns if re.match('^CoverPercent[A-Z]', x)]

            temp = furlong_data[cp_cols].sum(1)
            if not temp.empty:

                # For all zero 'CoverPercent...'
                cpo_col = 'CoverPercentOther'
                furlong_data.loc[temp[temp == 0].index, cpo_col] = 100.0

                # For all non-100 'CoverPercent...'
                idx = temp[~temp.isin([0.0, 100.0])].index

                nonzero_cols = furlong_data.loc[idx, cp_cols].apply(lambda x: x != 0.0).apply(
                    lambda x: list(pd.Index(cp_cols)[x.values]), axis=1)

                errors = pd.Series(100.0 - temp[idx])

                for i in idx:
                    features = nonzero_cols[i].copy()
                    if len(features) == 1:
                        feature = features[0]
                        if feature == cpo_col:
                            furlong_data.loc[[i], cpo_col] = 100.0
                        else:
                            if errors.loc[i] > 0:
                                furlong_data.loc[[i], cpo_col] = np.sum([
                                    furlong_data.loc[i, cpo_col], errors.loc[i]])
                            else:  # errors.loc[i] < 0
                                furlong_data[feature].loc[[i]] = np.sum([
                                    furlong_data[feature].loc[i], errors.loc[i]])
                    else:  # len(nonzero_cols[i]) > 1
                        if cpo_col in features:
                            err = np.sum([furlong_data.loc[i, cpo_col], errors.loc[i]])
                            if err >= 0.0:
                                furlong_data.loc[[i], cpo_col] = err
                            else:
                                features.remove(cpo_col)
                                furlong_data.loc[[i], cpo_col] = 0.0
                                if len(features) == 1:
                                    feature = features[0]
                                    furlong_data.loc[[i], feature] = np.sum(
                                        [furlong_data.loc[i, feature], err])
                                else:
                                    err = np.divide(err, len(features))
                                    furlong_data.loc[i, features] += err
                        else:
                            err = np.divide(errors.loc[i], len(features))
                            furlong_data.loc[i, features] += err

        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        furlong_data.set_index(primary_keys, inplace=True)

        self._dump_prep_data(data=furlong_data, table_name=table_name, verbose=verbose)

        return furlong_data

    def _furlong_location(self, key_columns_only=True, verbose=False, **kwargs):
        """
        Get data of the table 'FurlongLocation'.

        :param key_columns_only: whether to return only the columns relevant to the project,
            defaults to ``True``
        :type key_columns_only: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'FurlongLocation'
        :rtype: pandas.DataFrame or None
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        table_name = 'FurlongLocation'

        # Read data from Database
        sql_query = self._sql_query(table_name=table_name, raw=True)
        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        furlong_location = self.db_instance.read_sql_query(
            sql_query, index_col=primary_keys, true_values=['t', 'true'], false_values=['f', 'false'],
            **kwargs)

        # Re-format mileage data
        furlong_location.loc[:, ['StartMileage', 'EndMileage']] = \
            furlong_location[['StartMileage', 'EndMileage']].map(mileage_num_to_str)

        # Replace boolean values with binary values
        furlong_location.loc[:, ['Electrified', 'HazardOnly']] = \
            furlong_location[['Electrified', 'HazardOnly']].map(int)
        # Replace Route names
        furlong_location = update_route_names(furlong_location, route_col_name='Route')

        self._dump_prep_data(data=furlong_location, table_name=table_name, verbose=verbose)

        # Select useful columns only?
        if key_columns_only:
            key_columns = [
                'Route',
                'RouteAlias',
                'DU',
                'ELR',
                'StartMileage',
                'EndMileage',
                'Electrified',
                'HazardOnly',
            ]
            furlong_location = furlong_location[key_columns]

        return furlong_location

    def _hazard_tree(self, verbose=False, **kwargs):
        """Get data of the table 'HazardTree'."""

        table_name = 'HazardTree'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred()

        sql_query = self._sql_query(table_name=table_name, raw=True)
        hazard_tree = self.db_instance.read_sql_query(
            sql_query, index_col=None, true_values=['t', 'true'], false_values=['f', 'false'],
            low_memory=False, **kwargs)

        # Re-format mileage data
        hazard_tree.Mileage = hazard_tree.Mileage.apply(mileage_num_to_str)

        # Edit the original data
        hazard_tree.drop(['Treesurvey', 'Treetunnel'], axis=1, inplace=True)
        hazard_tree.dropna(subset=['Northing', 'Easting'], inplace=True)
        hazard_tree.Treespecies.replace({'': 'No data'}, inplace=True)

        # Update route data
        hazard_tree = update_route_names(hazard_tree, route_col_name='Route')

        # Integrate information from several features in a DataFrame
        def sum_up_selected_features(data, selected_features, new_feature):
            """
            :param data: original data frame
            :type data: pandas.DataFrame
            :param selected_features: list of columns names
            :type selected_features: list
            :param new_feature: new column name
            :type new_feature: str
            :return: integrated data
            :rtype: pandas.DataFrame
            """
            data.replace({True: 1, False: 0}, inplace=True)
            data[new_feature] = data[selected_features].fillna(0).apply(sum, axis=1)
            data.drop(selected_features, axis=1, inplace=True)

        # Integrate TEF: Failure scores
        failure_scores = ['TEF30770' + str(i) for i in range(1, 6)]
        sum_up_selected_features(hazard_tree, failure_scores, new_feature='Failure_Score')
        # Integrate TEF: Target scores
        target_scores = ['TEF3077%02d' % i for i in range(6, 12)]
        sum_up_selected_features(hazard_tree, target_scores, new_feature='Target_Score')
        # Integrate TEF: Impact scores
        impact_scores = ['TEF3077' + str(i) for i in range(12, 16)]
        sum_up_selected_features(hazard_tree, impact_scores, new_feature='Impact_Score')
        # Rename the rest of TEF
        work_req = ['TEF3077' + str(i) for i in range(17, 27)]
        work_req_desc = [
            'WorkReq_ExpertInspection',
            'WorkReq_LocalisedPruning',
            'WorkReq_GeneralPruning',
            'WorkReq_CrownRemoval',
            'WorkReq_StumpRemoval',
            'WorkReq_TreeRemoval',
            'WorkReq_TargetManagement',
            'WorkReq_FurtherInvestigation',
            'WorkReq_LimbRemoval',
            'WorkReq_InstallSupport']
        hazard_tree.rename(columns=dict(zip(work_req, work_req_desc)), inplace=True)

        # Note the feasibility of the following operation is not guaranteed:
        hazard_tree[work_req_desc] = hazard_tree[work_req_desc].fillna(value=0)

        # Rearrange DataFrame index
        hazard_tree.index = range(len(hazard_tree))

        # Add two columns of Latitudes and Longitudes corresponding to the Easting and Northing
        hazard_tree['Longitude'], hazard_tree['Latitude'] = osgb36_to_wgs84(
            hazard_tree.Easting.values, hazard_tree.Northing.values)

        primary_keys = self.db_instance.get_primary_keys(table_name, self.POSTGRES_SCHEMA_NAME)
        hazard_tree.set_index(primary_keys, inplace=True)

        self._dump_prep_data(data=hazard_tree, table_name=table_name, verbose=verbose)

        return hazard_tree
