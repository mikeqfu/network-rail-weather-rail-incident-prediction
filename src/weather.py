"""
Weather
"""

import copy
import gc
import glob
import string
import tempfile
import zipfile

import datetime_truncate
import fuzzywuzzy.fuzz
import fuzzywuzzy.process
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import natsort
import networkx as nx
import shapely.geometry
import shapely.geometry
import shapely.ops
import shapely.wkt
import sqlalchemy.types
from pyhelpers.dir import cd
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.store import save_fig
from pyrcs.line_data import LocationIdentifiers
from pyrcs.other_assets import Stations
from pyrcs.utils import *

from misc.dag import DelayAttributionGlossary
from utils import *


class METEX:
    """
    Read and cleanse data of NR_METEX_* database.

    - Schedule 4 compensates train operators for the impact of planned service disruption, and
    - Schedule 8 compensates train operators for the impact of unplanned service disruption.

    .. todo::

        view_schedule8_incident_location_tracks()
    """

    def __init__(self):
        self.Name = 'METEX'
        self.Desc = 'METEX is a geographic information system (GIS) based decision support tool, ' \
                    'used to assess asset and system vulnerability to weather.'

        self.DatabaseName = 'NR_METEX_20190203'

    # == Change directories ===========================================================================

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to "data\\metex\\database" and sub-directories / a file
        :rtype: str
        """

        path = cdd_metex("database", *sub_dir, mkdir=mkdir)

        return path

    def cdd_tables(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database\\tables" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to "data\\metex\\database\\tables" and sub-directories / a file
        :rtype: str
        """

        path = self.cdd("tables", *sub_dir, mkdir=mkdir)

        return path

    def cdd_views(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database\\views" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to "data\\metex\\database\\views" and sub-directories / a file
        :rtype: str
        """

        path = self.cdd("views", *sub_dir, mkdir=mkdir)

        return path

    def cdd_figures(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database\\figures" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to "data\\metex\\database\\figures" and sub-directories / a file
        :rtype: str
        """

        path = self.cdd("figures", *sub_dir, mkdir=mkdir)

        return path

    # == Functions to read table data from the database ===============================================

    def read_table(self, table_name, schema_name='dbo', index_col=None,
                   route_name=None, weather_category=None, save_as=None, update=False, **kwargs):
        """
        Read tables stored in NR_METEX_* database.

        :param table_name: name of a table
        :type table_name: str
        :param schema_name: name of schema, defaults to ``'dbo'``
        :type schema_name: str
        :param index_col: column(s) set to be index of the returned data frame, defaults to ``None``
        :type index_col: str, None
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str, None
        :param weather_category: weather category; if ``None`` (default), all weather categories
        :type weather_category: str, None
        :param save_as: file extension, defaults to ``None``
        :type save_as: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param kwargs: optional parameters of `pandas.read_sql`_
        :return: data of the queried table stored in NR_METEX_* database
        :rtype: pandas.DataFrame

        .. _`pandas.read_sql`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html

        **Test**::

            >>> from weather import METEX

            table_name       = 'IMDM'
            index_col        = None
            route_name       = None
            weather_category = None
            schema_name      = 'dbo'
            save_as          = None
            update           = True

            data = metex.read_table(table_name, index_col, route_name, weather_category,
                                          schema_name, save_as, update)
        """

        table = '{}."{}"'.format(schema_name, table_name)
        # Connect to the queried database
        conn_metex = establish_mssql_connection(database_name=self.DatabaseName)
        # Specify possible scenarios:
        if not route_name and not weather_category:
            sql_query = "SELECT * FROM {}".format(table)  # Get all data of a given table
        elif route_name and not weather_category:
            sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route_name)  # given Route
        elif route_name is None and weather_category is not None:
            # given Weather
            sql_query = "SELECT * FROM {} WHERE WeatherCategory = '{}'".format(
                table, weather_category)
        else:
            # Get all data of a table,
            # given Route and Weather category e.g. data about wind-related events on Anglia Route
            sql_query = "SELECT * FROM {} WHERE Route = '{}' AND WeatherCategory = '{}'".format(
                table, route_name, weather_category)
        # Create a pd.DataFrame of the queried table
        table_data = pd.read_sql(sql_query, conn_metex, index_col=index_col, **kwargs)
        # Disconnect the database
        conn_metex.close()
        if save_as:
            path_to_file = self.cdd_tables(table_name + save_as)
            if not os.path.isfile(path_to_file) or update:
                save(table_data, path_to_file, index=True if index_col else False)
        return table_data

    def get_primary_key(self, table_name):
        """
        Get primary keys of a table stored in database 'NR_METEX_*'.

        :param table_name: name of a table stored in the database 'NR_METEX_*'
        :type table_name: str
        :return: a (list of) primary key(s)
        :rtype: list

        **Test**::

            from mssqlserver import metex

            table_name = 'IMDM'

            pri_key = metex.get_primary_key(table_name)
            print(pri_key)
        """

        pri_key = get_table_primary_keys(self.DatabaseName, table_name=table_name)

        return pri_key

    # == Functions to get table data ==================================================================

    def get_imdm(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'IMDM'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'IMDM'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            as_dict          = False
            update           = True
            save_original_as = None
            verbose          = True

            imdm = metex.get_imdm(as_dict, update, save_original_as, verbose)
            print(imdm)
        """

        METEX.IMDM = 'IMDM'
        path_to_file = self.cdd_tables("".join([METEX.IMDM, ".json" if as_dict else ".pickle"]))

        if os.path.isfile(path_to_file) and not update:
            imdm = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                imdm = self.read_table(METEX.IMDM, index_col=self.get_primary_key(METEX.IMDM),
                                       save_as=save_original_as, update=update)
                imdm.index.rename(name='IMDM', inplace=True)  # Rename index

                # Update route names
                update_nr_route_names(imdm)

                # Add regions
                regions_and_routes = load_json(cdd_network("Regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
                imdm = imdm.join(regions, on='Route')

                imdm = imdm.where((pd.notnull(imdm)), None)

                if as_dict:
                    imdm_dict = imdm.to_dict()
                    imdm = imdm_dict['Route']
                    imdm.pop('None')
                save(imdm, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METEX.IMDM, " as a dictionary" if as_dict else "", e))
                imdm = None

        return imdm

    def get_imdm_alias(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'ImdmAlias'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'ImdmAlias'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            as_dict = False
            imdm_alias = metex.get_imdm_alias(as_dict, update, save_original_as, verbose)
            print(imdm_alias)

            as_dict = True
            imdm_alias = metex.get_imdm_alias(as_dict, update, save_original_as, verbose)
            print(imdm_alias)
        """

        METEX.ImdmAlias = 'ImdmAlias'
        path_to_file = self.cdd_tables(METEX.ImdmAlias + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_file) and not update:
            imdm_alias = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                imdm_alias = self.read_table(METEX.ImdmAlias,
                                             index_col=self.get_primary_key(METEX.ImdmAlias),
                                             save_as=save_original_as, update=update)
                imdm_alias.index.rename(name='ImdmAlias', inplace=True)  # Rename index
                imdm_alias.rename(columns={'Imdm': 'IMDM'}, inplace=True)  # Rename a column
                if as_dict:
                    imdm_alias = imdm_alias.to_dict()
                    # imdm_alias = imdm_alias['IMDM']
                save(imdm_alias, path_to_file, verbose=verbose)
            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METEX.ImdmAlias, " as a dictionary" if as_dict else "", e))
                imdm_alias = None
        return imdm_alias

    def get_imdm_weather_cell_map(self, route_info=True, grouped=False, update=False,
                                  save_original_as=None, verbose=False):
        """
        Get data of the table 'IMDMWeatherCellMap'.

        :param route_info: get the data that contains route information, defaults to ``True``
        :type route_info: bool
        :param grouped: whether to group the data by either ``'Route'`` or ``'WeatherCellId'``,
            defaults to ``False``
        :type grouped: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'IMDMWeatherCellMap'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            route_info = True
            grouped = False
            weather_cell_map = metex.get_imdm_weather_cell_map(route_info, grouped, update,
                                                               save_original_as, verbose)
            print(weather_cell_map)

            route_info = True
            grouped = True
            weather_cell_map = metex.get_imdm_weather_cell_map(route_info, grouped, update,
                                                               save_original_as, verbose)
            print(weather_cell_map)

            route_info = False
            grouped = True
            weather_cell_map = metex.get_imdm_weather_cell_map(route_info, grouped, update,
                                                               save_original_as, verbose)
            print(weather_cell_map)

            route_info = False
            grouped = False
            weather_cell_map = metex.get_imdm_weather_cell_map(route_info, grouped, update,
                                                               save_original_as, verbose)
            print(weather_cell_map)
        """

        METEX.IMDMWeatherCellMap = 'IMDMWeatherCellMap_pc' if route_info else 'IMDMWeatherCellMap'
        path_to_pickle = self.cdd_tables(
            METEX.IMDMWeatherCellMap + ("-grouped.pickle" if grouped else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            weather_cell_map = load_pickle(path_to_pickle)

        else:
            try:
                # Read IMDMWeatherCellMap table
                weather_cell_map = self.read_table(
                    METEX.IMDMWeatherCellMap,
                    index_col=self.get_primary_key(METEX.IMDMWeatherCellMap),
                    coerce_float=False, save_as=save_original_as, update=update)

                if route_info:
                    update_nr_route_names(weather_cell_map)
                    weather_cell_map[['Id', 'WeatherCell']] = weather_cell_map[
                        ['Id', 'WeatherCell']].applymap(int)
                    weather_cell_map.set_index('Id', inplace=True)

                weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index
                weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'},
                                        inplace=True)  # Rename a column

                if grouped:  # To find out how many IMDMs each 'WeatherCellId' is associated with
                    weather_cell_map = weather_cell_map.groupby(
                        'Route' if route_info else 'WeatherCellId').aggregate(
                        lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))

                save_pickle(weather_cell_map, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METEX.IMDMWeatherCellMap, " (being grouped)" if grouped else "", e))
                weather_cell_map = None

        return weather_cell_map

    def get_incident_reason_info(self, plus=True, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'IncidentReasonInfo'.

        :param plus: defaults to ``True``
        :type plus: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'IncidentReasonInfo'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            plus = True
            incident_reason_info = metex.get_incident_reason_info(plus, update, save_original_as,
                                                                  verbose)
            print(incident_reason_info)

            plus = False
            incident_reason_info = metex.get_incident_reason_info(plus, update, save_original_as,
                                                                  verbose)
            print(incident_reason_info)
        """

        METEX.IncidentReasonInfo = 'IncidentReasonInfo'
        path_to_pickle = self.cdd_tables(
            METEX.IncidentReasonInfo + ("-plus.pickle" if plus else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            incident_reason_info = load_pickle(path_to_pickle)

        else:
            try:
                # Get data from the database
                incident_reason_info = self.read_table(
                    METEX.IncidentReasonInfo, index_col=self.get_primary_key(METEX.IncidentReasonInfo),
                    save_as=save_original_as, update=update)
                incident_reason_info.index.rename('IncidentReasonCode', inplace=True)
                incident_reason_info.rename(
                    columns={'Description': 'IncidentReasonDescription',
                             'Category': 'IncidentCategory',
                             'CategoryDescription': 'IncidentCategoryDescription'},
                    inplace=True)
                if plus:  # To include data of more detailed description about incident reasons
                    dag = DelayAttributionGlossary()
                    incident_reason_metadata = dag.read_incident_reason_metadata()
                    incident_reason_metadata.index.name = 'IncidentReasonCode'
                    incident_reason_metadata.columns = [
                        x.replace('_', '') for x in incident_reason_metadata.columns]
                    incident_reason_info = incident_reason_info.join(
                        incident_reason_metadata, rsuffix='_plus')
                    # incident_reason_info.dropna(axis=1, inplace=True)

                save_pickle(incident_reason_info, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METEX.IncidentReasonInfo, " with extra information" if plus else "", e))
                incident_reason_info = None

        return incident_reason_info

    def get_weather_codes(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'WeatherCategoryLookup'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'WeatherCategoryLookup'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            as_dict = False
            weather_codes = metex.get_weather_codes(as_dict, update, save_original_as, verbose)
            print(weather_codes)

            as_dict = True
            weather_codes = metex.get_weather_codes(as_dict, update, save_original_as, verbose)
            print(weather_codes)
        """

        METEX.WeatherCodes = 'WeatherCodes'  # WeatherCodes
        path_to_file = self.cdd_tables(METEX.WeatherCodes + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_file) and not update:
            weather_codes = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                weather_codes = self.read_table(
                    METEX.WeatherCodes, index_col=self.get_primary_key(METEX.WeatherCodes),
                    save_as=save_original_as, update=update)
                weather_codes.rename(
                    columns={'Code': 'WeatherCategoryCode', 'Weather Category': 'WeatherCategory'},
                    inplace=True)
                if as_dict:
                    weather_codes.set_index('WeatherCategoryCode', inplace=True)
                    weather_codes = weather_codes.to_dict()
                save(weather_codes, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.WeatherCodes, e))
                weather_codes = None

        return weather_codes

    def get_incident_record(self, update=False, save_original_as=None, use_amendment_csv=True,
                            verbose=False):
        """
        Get data of the table 'IncidentRecord'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'IncidentRecord'
        :rtype: pandas.DataFrame, None

        .. note::

            None values are filled with NaN.

        **Test**::

            from mssqlserver import metex

            update            = True
            save_original_as  = None
            verbose           = True

            use_amendment_csv = True
            incident_record = metex.get_incident_record(update, save_original_as, use_amendment_csv,
                                                        verbose)
            print(incident_record)

            use_amendment_csv = False
            incident_record = metex.get_incident_record(update, save_original_as, use_amendment_csv,
                                                        verbose)
            print(incident_record)
        """

        METEX.IncidentRecord = 'IncidentRecord'
        filename = METEX.IncidentRecord + "-amended" if use_amendment_csv else METEX.IncidentRecord
        path_to_pickle = self.cdd_tables(filename + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            incident_record = load_pickle(path_to_pickle)

        else:
            try:
                incident_record = self.read_table(
                    METEX.IncidentRecord, index_col=self.get_primary_key(METEX.IncidentRecord),
                    save_as=save_original_as, update=update)

                if use_amendment_csv:
                    amendment_csv = pd.read_csv(
                        cdd("updates", METEX.IncidentRecord + ".zip"), index_col='Id',
                        parse_dates=['CreateDate'], infer_datetime_format=True, dayfirst=True)
                    amendment_csv.columns = incident_record.columns
                    idx = amendment_csv[amendment_csv.WeatherCategory.isna()].index
                    amendment_csv.loc[idx, 'WeatherCategory'] = None
                    incident_record.drop(incident_record[incident_record.CreateDate >= pd.to_datetime(
                        '2018-01-01')].index, inplace=True)
                    incident_record = incident_record.append(amendment_csv)

                incident_record.index.rename(METEX.IncidentRecord + 'Id', inplace=True)
                incident_record.rename(
                    columns={'CreateDate': METEX.IncidentRecord + 'CreateDate',
                             'Reason': 'IncidentReasonCode'},
                    inplace=True)  # Rename column names

                # Get a weather category lookup dictionary
                weather_codes = self.get_weather_codes(as_dict=True)
                # Replace each weather category code with its full name
                incident_record.replace(weather_codes, inplace=True)
                incident_record.WeatherCategory.fillna(value='', inplace=True)

                save_pickle(incident_record, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.IncidentRecord, e))
                incident_record = None

        return incident_record

    def get_location(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Location'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'Location'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            location = metex.get_location(update, save_original_as, verbose)
            print(location)
        """

        METEX.Location = 'Location'
        path_to_pickle = self.cdd_tables(METEX.Location + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            location = load_pickle(path_to_pickle)

        else:
            try:
                location = self.read_table(
                    table_name=METEX.Location, index_col=self.get_primary_key(METEX.Location),
                    coerce_float=False, save_as=save_original_as, update=update)
                location.index.rename('LocationId', inplace=True)
                location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
                location[['WeatherCell', 'SMDCell']] = location[['WeatherCell', 'SMDCell']].applymap(
                    lambda x: 0 if np.isnan(x) else int(x))
                # location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]

                save_pickle(location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.Location, e))
                location = None

        return location

    def get_pfpi(self, plus=True, update=False, save_original_as=None, use_amendment_csv=True,
                 verbose=False):
        """
        Get data of the table 'PfPI' (Process for Performance Improvement).

        :param plus: defaults to ``True``
        :type plus: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'PfPI'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update            = True
            save_original_as  = None
            verbose           = True

            plus = True
            use_amendment_csv = True
            pfpi = metex.get_pfpi(plus, update, save_original_as, use_amendment_csv, verbose)
            print(pfpi)

            plus = False
            use_amendment_csv = True
            pfpi = metex.get_pfpi(plus, update, save_original_as, use_amendment_csv, verbose)
            print(pfpi)

            plus = True
            use_amendment_csv = False
            pfpi = metex.get_pfpi(plus, update, save_original_as, use_amendment_csv, verbose)
            print(pfpi)

            plus = False
            use_amendment_csv = False
            pfpi = metex.get_pfpi(plus, update, save_original_as, use_amendment_csv, verbose)
            print(pfpi)
        """

        METEX.PfPI = 'PfPI'
        table_name_ = (METEX.PfPI + "-plus" if plus else METEX.PfPI)
        path_to_pickle = self.cdd_tables(
            table_name_ + ("-amended.pickle" if use_amendment_csv else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            pfpi = load_pickle(path_to_pickle)

        else:
            try:
                pfpi = self.read_table(METEX.PfPI, index_col=self.get_primary_key(METEX.PfPI),
                                       save_as=save_original_as,
                                       update=update)

                if use_amendment_csv:
                    incident_record = self.read_table('IncidentRecord',
                                                      index_col=self.get_primary_key('IncidentRecord'))
                    min_id = incident_record[
                        incident_record.CreateDate >= pd.to_datetime('2018-01-01')].index.min()
                    pfpi.drop(pfpi[pfpi.IncidentRecordId >= min_id].index, inplace=True)
                    pfpi = pfpi.append(
                        pd.read_csv(cdd("updates", METEX.PfPI + ".zip", ), index_col='Id'))

                pfpi.index.rename(METEX.PfPI + pfpi.index.name, inplace=True)

                if plus:  # To include more information for 'PerformanceEventCode'
                    dag = DelayAttributionGlossary()
                    performance_event_code = dag.read_performance_event_code()
                    performance_event_code.index.rename('PerformanceEventCode', inplace=True)
                    performance_event_code.columns = [
                        x.replace('_', '') for x in performance_event_code.columns]
                    # Merge pfpi and pe_code
                    pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')

                save_pickle(pfpi, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METEX.PfPI, " with performance event name" if plus else "", e))
                pfpi = None

        return pfpi

    def get_route(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Route'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'Route'
        :rtype: pandas.DataFrame, None

        .. note::

            There is only one column in the original table.

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            as_dict = False
            route = metex.get_route(as_dict, update, save_original_as, verbose)
            print(route)

            as_dict = True
            route = metex.get_route(as_dict, update, save_original_as, verbose)
            print(route)
        """

        table_name = "Route"
        path_to_pickle = self.cdd_tables(table_name + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            route = load_pickle(path_to_pickle)

        else:
            try:
                route = self.read_table(table_name, save_as=save_original_as, update=update)
                route.rename(columns={'Name': 'Route'}, inplace=True)
                update_nr_route_names(route)

                # Add regions
                regions_and_routes = load_json(cdd_network("Regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
                route = route.join(regions, on='Route')

                route = route.where((pd.notnull(route)), None)

                if as_dict:
                    route.drop_duplicates('Route', inplace=True)
                    route = dict(zip(route.RouteAlias, route.Route))

                save(route, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                route = None

        return route

    def get_stanox_location(self, use_nr_mileage_format=True, update=False, save_original_as=None,
                            verbose=False):
        """
        Get data of the table 'StanoxLocation'.

        :param use_nr_mileage_format: defaults to ``True``
        :type use_nr_mileage_format: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'StanoxLocation'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            use_nr_mileage_format = True
            stanox_location = metex.get_stanox_location(use_nr_mileage_format, update, save_original_as,
                                                        verbose)
            print(stanox_location)

            use_nr_mileage_format = False
            stanox_location = metex.get_stanox_location(use_nr_mileage_format, update, save_original_as,
                                                        verbose)
            print(stanox_location)
        """

        METEX.StanoxLocation = 'StanoxLocation'
        path_to_pickle = self.cdd_tables(
            METEX.StanoxLocation + ("-mileage.pickle" if use_nr_mileage_format else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            stanox_location = load_pickle(path_to_pickle)

        else:
            try:
                # Read StanoxLocation table from the database
                stanox_location = self.read_table(METEX.StanoxLocation, index_col=None,
                                                  save_as=save_original_as, update=update)

                # Likely errors
                stanox_location.loc[stanox_location.Stanox == '52053', 'ELR':'LocationId'] = (
                    'BOK1', 6072, 534877)
                stanox_location.loc[stanox_location.Stanox == '52074', 'ELR':'LocationId'] = (
                    'ELL1', 440, 610096)

                def cleanse_stanox_location(sta_loc):
                    """
                    sta_loc = copy.deepcopy(stanox_location)
                    """
                    dat = copy.deepcopy(sta_loc)

                    # Use external data - Railway Codes
                    errata = load_json(cdd_network("Railway Codes", "METEX_errata.json"))
                    err_stanox, err_tiploc, err_stanme = errata.values()
                    # Note that {'CLAPS47': 'CLPHS47'} in err_tiploc is dubious.
                    dat.replace({'Stanox': err_stanox, 'Description': err_tiploc, 'Name': err_stanme},
                                inplace=True)

                    duplicated_stanox = dat[dat.Stanox.duplicated(keep=False)].sort_values('Stanox')
                    nan_idx = duplicated_stanox[['ELR', 'Yards', 'LocationId']].applymap(pd.isna).apply(
                        any, axis=1)
                    dat.drop(duplicated_stanox[nan_idx].index, inplace=True)

                    dat.drop_duplicates(subset=['Stanox'], keep='last', inplace=True)

                    lid = LocationIdentifiers()
                    location_codes = lid.fetch_location_codes()
                    location_codes = location_codes['Location codes']

                    #
                    for i, x in dat[dat.Description.isnull()].Stanox.items():
                        idx = location_codes[location_codes.STANOX == x].index
                        if len(idx) == 1:
                            idx = idx[0]
                            dat.loc[i, 'Description'] = \
                                location_codes[location_codes.STANOX == x].Location[idx]
                            dat.loc[i, 'Name'] = location_codes[location_codes.STANOX == x].STANME[idx]
                        else:
                            print("Errors occur at index \"{}\" where the corresponding STANOX is "
                                  "\"{}\"".format(i, x))
                            break
                    #
                    for i, x in dat[dat.Name.isnull()].Stanox.items():
                        temp = location_codes[location_codes.STANOX == x]
                        if temp.shape[0] > 1:
                            desc = dat[dat.Stanox == x].Description[i]
                            if desc in temp.TIPLOC.values:
                                idx = temp[temp.TIPLOC == desc].index
                            elif desc in temp.STANME.values:
                                idx = temp[location_codes.STANME == desc].index
                            else:
                                print("Errors occur at index \"{}\" where the corresponding STANOX is "
                                      "\"{}\"".format(i, x))
                                break
                        else:
                            idx = temp.index
                        if len(idx) > 1:
                            # Choose the first instance, and print a warning message
                            print("Warning: The STANOX \"{}\" at index \"{}\" is not unique. "
                                  "The first instance is chosen.".format(x, i))
                        idx = idx[0]
                        dat.loc[i, 'Description'] = temp.Location.loc[idx]
                        dat.loc[i, 'Name'] = temp.STANME.loc[idx]

                    location_stanme_dict = \
                        location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
                    dat.Name.replace(location_stanme_dict, inplace=True)

                    # Use manually-created dictionary of regular expressions
                    dat.replace(fetch_loc_names_repl_dict(k='Description'), inplace=True)
                    dat.replace(fetch_loc_names_repl_dict(k='Description', regex=True), inplace=True)

                    # Use STANOX dictionary
                    stanox_dict = lid.make_loc_id_dict('STANOX')
                    temp = dat.join(stanox_dict, on='Stanox')[['Description', 'Location']]
                    temp.loc[temp.Location.isnull(), 'Location'] = temp.loc[
                        temp.Location.isnull(), 'Description']
                    dat.Description = temp.apply(
                        lambda y: fuzzywuzzy.process.extractOne(y.Description, y.Location,
                                                                scorer=fuzzywuzzy.fuzz.ratio)[0]
                        if isinstance(y.Location, tuple) else y.Location, axis=1)

                    dat.Name = dat.Name.str.upper()

                    location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']].groupby(
                        ['STANOX', 'Location']).agg({'STANME': lambda y: list(y)[0]})
                    temp = dat.join(location_codes_cut, on=['Stanox', 'Description'])
                    dat.Name = temp.STANME

                    dat.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'},
                               inplace=True)

                    # dat.dropna(how='all', subset=['ELR', 'Yards', 'LocationId'], inplace=True)
                    return dat

                # Cleanse raw stanox_location
                stanox_location = cleanse_stanox_location(stanox_location)

                # For 'ELR', replace NaN with ''
                stanox_location.ELR.fillna('', inplace=True)

                # For 'LocationId'
                stanox_location.Yards = stanox_location.Yards.map(
                    lambda x: '' if np.isnan(x) else int(x))
                stanox_location.LocationId = stanox_location.LocationId.map(
                    lambda x: '' if np.isnan(x) else int(x))

                # For 'Mileages' - to convert yards to miles
                # (Note: Not the 'mileage' used by Network Rail)
                if use_nr_mileage_format:
                    stanox_location['Mileage'] = stanox_location.Yards.map(yards_to_nr_mileage)

                # Set index
                stanox_location.set_index('Stanox', inplace=True)

                save_pickle(stanox_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.StanoxLocation, e))
                stanox_location = None

        return stanox_location

    def get_stanox_section(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'StanoxSection'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'StanoxSection'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            stanox_section = metex.get_stanox_section(update, save_original_as, verbose)
            print(stanox_section)
        """

        METEX.StanoxSection = 'StanoxSection'
        path_to_pickle = self.cdd_tables(METEX.StanoxSection + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            stanox_section = load_pickle(path_to_pickle)

        else:
            try:
                # Read StanoxSection table from the database
                stanox_section = self.read_table(METEX.StanoxSection,
                                                 index_col=self.get_primary_key(METEX.StanoxSection),
                                                 save_as=save_original_as, update=update)
                stanox_section.index.name = METEX.StanoxSection + 'Id'
                stanox_section.LocationId = stanox_section.LocationId.apply(
                    lambda x: '' if np.isnan(x) else int(x))

                lid = LocationIdentifiers()
                stanox_dat = lid.make_loc_id_dict('STANOX')

                # Firstly, create a stanox-to-location dictionary,
                # and replace STANOX with location names
                for stanox_col_name in ['StartStanox', 'EndStanox']:
                    tmp_col = stanox_col_name + '_temp'
                    # Load stanox dictionary 1
                    stanox_dict = self.get_stanox_location(use_nr_mileage_format=True).Location.to_dict()
                    stanox_section[tmp_col] = stanox_section[stanox_col_name].replace(
                        stanox_dict)  # Create a temp column
                    tmp = stanox_section.join(stanox_dat, on=tmp_col).Location
                    tmp_idx = tmp[tmp.notnull()].index
                    stanox_section[tmp_col][tmp_idx] = tmp[tmp_idx]
                    stanox_section[tmp_col] = stanox_section[tmp_col].map(
                        lambda x: x[0] if isinstance(x, list) else x)

                stanme_dict = lid.make_loc_id_dict('STANME', as_dict=True)
                tiploc_dict = lid.make_loc_id_dict('TIPLOC', as_dict=True)

                # Secondly, process 'STANME' and 'TIPLOC'
                loc_name_replacement_dict = fetch_loc_names_repl_dict()
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict(regex=True)
                # Processing 'StartStanox_tmp'
                stanox_section.StartStanox_temp = stanox_section.StartStanox_temp. \
                    replace(stanme_dict).replace(tiploc_dict). \
                    replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
                # Processing 'EndStanox_tmp'
                stanox_section.EndStanox_temp = stanox_section.EndStanox_temp. \
                    replace(stanme_dict).replace(tiploc_dict). \
                    replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

                # Create 'STANOX' sections
                temp = stanox_section[stanox_section.StartStanox_temp.map(
                    lambda x: False if isinstance(x, str) else True)]
                temp['StartStanox_'] = temp.Description.str.split(' : ', expand=True)[0]
                stanox_section.loc[temp.index, 'StartStanox_temp'] = temp.apply(
                    lambda x: find_similar_str(x.StartStanox_, x.StartStanox_temp),
                    axis=1)  # Temporary!

                temp = stanox_section[
                    stanox_section.EndStanox_temp.map(lambda x: False if isinstance(x, str) else True)]
                temp['EndStanox_'] = temp.Description.str.split(' : ', expand=True)[1].fillna(
                    temp.Description)
                stanox_section.loc[temp.index, 'EndStanox_temp'] = temp.apply(
                    lambda x: find_similar_str(x.EndStanox_, x.EndStanox_temp), axis=1)  # Temporary!

                start_end = stanox_section.StartStanox_temp + ' - ' + stanox_section.EndStanox_temp
                point_idx = stanox_section.StartStanox_temp == stanox_section.EndStanox_temp
                start_end[point_idx] = stanox_section.StartStanox_temp[point_idx]
                stanox_section['StanoxSection'] = start_end

                # Finalising the cleaning process
                stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
                stanox_section.rename(
                    columns={'StartStanox_temp': 'StartLocation', 'EndStanox_temp': 'EndLocation'},
                    inplace=True)
                stanox_section = stanox_section[['LocationId', 'StanoxSection',
                                                 'StartLocation', 'StartStanox', 'EndLocation',
                                                 'EndStanox',
                                                 'ApproximateLocation']]

                save_pickle(stanox_section, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.StanoxSection, e))
                stanox_section = None

        return stanox_section

    def get_trust_incident(self, start_year=2006, end_year=None, update=False, save_original_as=None,
                           use_amendment_csv=True, verbose=False):
        """
        Get data of the table 'TrustIncident'.

        :param start_year: defaults to ``2006``
        :type start_year: int, None
        :param end_year: defaults to ``None``
        :type end_year: int, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'TrustIncident'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            start_year        = 2006
            end_year          = None
            update            = True
            save_original_as  = None
            verbose           = True

            use_amendment_csv = True
            trust_incident = metex.get_trust_incident(start_year, end_year, update, save_original_as,
                                                      use_amendment_csv, verbose)
            print(trust_incident)

            use_amendment_csv = False
            trust_incident = metex.get_trust_incident(start_year, end_year, update, save_original_as,
                                                      use_amendment_csv, verbose)
            print(trust_incident)
        """

        METEX.TrustIncident = 'TrustIncident'
        suffix_ext = "{}".format(
            "{}".format("-y{}".format(start_year) if start_year else "-up-to") +
            "{}".format("-y{}".format(2018 if not end_year or end_year >= 2019 else end_year)))
        filename_ = METEX.TrustIncident + suffix_ext
        path_to_pickle = self.cdd_tables(
            filename_ + "{}.pickle".format("-amended" if use_amendment_csv else ""))

        if os.path.isfile(path_to_pickle) and not update:
            trust_incident = load_pickle(path_to_pickle)

        else:
            try:
                trust_incident = self.read_table(METEX.TrustIncident,
                                                 index_col=self.get_primary_key(METEX.TrustIncident),
                                                 save_as=save_original_as, update=update)
                if use_amendment_csv:
                    zip_file = zipfile.ZipFile(cdd("updates", METEX.TrustIncident + ".zip"))
                    corrected_csv = pd.concat(
                        [pd.read_csv(zip_file.open(f), index_col='Id',
                                     parse_dates=['StartDate', 'EndDate'],
                                     infer_datetime_format=True, dayfirst=True)
                         for f in zip_file.infolist()])
                    zip_file.close()
                    # Remove raw data >= '2018-01-01', pd.to_datetime('2018-01-01')
                    trust_incident.drop(trust_incident[trust_incident.StartDate >= '2018-01-01'].index,
                                        inplace=True)
                    # Append corrected data
                    trust_incident = trust_incident.append(corrected_csv)

                trust_incident.index.name = 'TrustIncidentId'
                trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
                # Extract a subset of data,
                # in which the StartDateTime is between 'start_year' and 'end_year'?
                trust_incident = trust_incident[
                    (trust_incident.FinancialYear >= (start_year if start_year else 0)) &
                    (trust_incident.FinancialYear <= (
                        end_year if end_year else datetime.datetime.now().year))]
                # Convert float to int values for 'SourceLocationId'
                trust_incident.SourceLocationId = trust_incident.SourceLocationId.map(
                    lambda x: '' if pd.isna(x) else int(x))

                save_pickle(trust_incident, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.TrustIncident, e))
                trust_incident = None

        return trust_incident

    def get_weather(self, verbose=False):
        """
        Get data of the table 'Weather'.

        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'Weather'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            verbose = True

            weather = metex.get_weather(verbose)
            print(weather)
        """

        METEX.Weather = 'Weather'
        try:
            conn_db = establish_mssql_connection(database_name=self.DatabaseName)
            sql_query = "SELECT * FROM dbo.[{}]".format(METEX.Weather)
            #
            chunks = pd.read_sql_query(sql_query, conn_db, index_col=None, parse_dates=['DateTime'],
                                       chunksize=1000000)
            weather = pd.concat([pd.DataFrame(chunk) for chunk in chunks], ignore_index=True, sort=False)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(METEX.Weather, e)) if verbose else None
            weather = None
        return weather

    def query_weather_by_id_datetime(self, weather_cell_id, start_dt=None, end_dt=None, postulate=False,
                                     pickle_it=True, dat_dir=None, update=False, verbose=False):
        """
        Get weather data by ``'WeatherCell'`` and ``'DateTime'`` (Query from the database).

        :param weather_cell_id: weather cell ID
        :type weather_cell_id: int
        :param start_dt: start date and time, defaults to ``None``
        :type start_dt: datetime.datetime, str, None
        :param end_dt: end date and time, defaults to ``None``
        :type end_dt: datetime.datetime, str, None
        :param postulate: whether to add postulated data, defaults to ``False``
        :type postulate: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: weather data by ``'weather_cell_id'``, ``'start_dt'`` and ``'end_dt'``
        :rtype: pandas.DataFrame

        **Test**::

            import datetime
            from mssqlserver import metex

            weather_cell_id = 2367
            start_dt        = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            postulate       = False
            pickle_it       = False
            dat_dir         = None
            update          = True
            verbose         = True

            end_dt = datetime.datetime(2018, 6, 1, 13)  # '2018-06-01 13:00:00'
            weather_dat = metex.query_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                             postulate, pickle_it, dat_dir, update,
                                                             verbose)
            print(weather_dat)

            end_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            weather_dat = metex.query_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                             postulate, pickle_it, dat_dir, update,
                                                             verbose)
            print(weather_dat)

        """

        assert isinstance(weather_cell_id, (tuple, int, np.integer))

        # Make a pickle filename
        pickle_filename = "{}{}{}.pickle".format(
            "-".join(str(x) for x in list(weather_cell_id))
            if isinstance(weather_cell_id, tuple) else weather_cell_id,
            start_dt.strftime('_fr%Y%m%d%H%M') if start_dt else "",
            end_dt.strftime('_to%Y%m%d%H%M') if end_dt else "")

        # Specify a directory/path to store the pickle file (if appropriate)
        dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else self.cdd_views()
        path_to_pickle = cd(dat_dir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                # Establish a connection to the MSSQL server
                conn_metex = establish_mssql_connection(database_name=self.DatabaseName)
                # Specify database sql query
                sql_query = "SELECT * FROM dbo.[Weather] WHERE {}{}{} AND {} AND {};".format(
                    "[WeatherCell]", " IN " if isinstance(weather_cell_id, tuple) else " = ",
                    weather_cell_id,
                    "[DateTime] >= '{}'".format(start_dt) if start_dt else "",
                    "[DateTime] <= '{}'".format(end_dt) if end_dt else "")
                # Query the weather data
                weather_dat = pd.read_sql(sql_query, conn_metex)

                if postulate:
                    i = 0
                    snowfall = weather_dat.Snowfall.tolist()
                    precipitation = weather_dat.TotalPrecipitation.tolist()
                    while i + 3 < len(weather_dat):
                        snowfall[i + 1: i + 3] = np.linspace(snowfall[i], snowfall[i + 3], 4)[1:3]
                        precipitation[i + 1: i + 3] = np.linspace(
                            precipitation[i], precipitation[i + 3], 4)[1:3]
                        i += 3
                    if i + 2 == len(weather_dat):
                        snowfall[-1:], precipitation[-1:] = snowfall[-2], precipitation[-2]
                    elif i + 3 == len(weather_dat):
                        snowfall[-2:], precipitation[-2:] = [snowfall[-3]] * 2, [precipitation[-3]] * 2
                    weather_dat.Snowfall = snowfall
                    weather_dat.TotalPrecipitation = precipitation

                if pickle_it:
                    save_pickle(weather_dat, path_to_pickle, verbose=verbose)

                return weather_dat

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(
                    os.path.splitext(os.path.basename(path_to_pickle))[0], e))

    def get_weather_cell(self, route_name=None, update=False, save_original_as=None, show_map=False,
                         projection='tmerc', save_map_as=None, dpi=None, verbose=False):
        """
        Get data of the table 'WeatherCell'.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param show_map: whether to show a map of the weather cells, defaults to ``False``
        :type show_map: bool
        :param projection: defaults to ``'tmerc'``
        :type projection: str
        :param save_map_as: whether to save the created map or what format the created map is saved as,
            defaults to ``None``
        :type save_map_as: str, None
        :param dpi: defaults to ``None``
        :type dpi: int, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'WeatherCell'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            show_map         = True
            projection       = 'tmerc'
            save_map_as      = ".tif"
            dpi              = None
            verbose          = True

            route_name = None
            weather_cell = metex.get_weather_cell(route_name, update, save_original_as, show_map,
                                                  projection, save_map_as, dpi, verbose)
            print(weather_cell)

            route_name = 'Anglia'
            weather_cell = metex.get_weather_cell(route_name, update, save_original_as, show_map,
                                                  projection, save_map_as, dpi, verbose)
            print(weather_cell)
        """

        METEX.WeatherCell = 'WeatherCell'
        pickle_filename = make_filename(METEX.WeatherCell, route_name)
        path_to_pickle = self.cdd_tables(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            weather_cell = load_pickle(path_to_pickle)

        else:
            try:
                weather_cell = self.read_table(table_name=METEX.WeatherCell,
                                               index_col=self.get_primary_key(METEX.WeatherCell),
                                               save_as=save_original_as, update=update)
                id_name = METEX.WeatherCell + 'Id'
                weather_cell.index.rename(id_name, inplace=True)

                # Lower left corner:
                weather_cell['ll_Longitude'] = weather_cell.Longitude  # - weather_cell_map.width / 2
                weather_cell['ll_Latitude'] = weather_cell.Latitude  # - weather_cell_map.height / 2
                # Upper left corner:
                weather_cell['ul_Longitude'] = weather_cell.ll_Longitude  # - weather_cell_map.width / 2
                weather_cell['ul_Latitude'] = weather_cell.ll_Latitude + weather_cell.height  # / 2
                # Upper right corner:
                weather_cell['ur_Longitude'] = weather_cell.ul_Longitude + weather_cell.width  # / 2
                weather_cell['ur_Latitude'] = weather_cell.ul_Latitude  # + weather_cell_map.height / 2
                # Lower right corner:
                weather_cell[
                    'lr_Longitude'] = weather_cell.ur_Longitude  # + weather_cell_map.width  # / 2
                weather_cell['lr_Latitude'] = weather_cell.ur_Latitude - weather_cell.height  # / 2

                # Get IMDM Weather cell map
                imdm_weather_cell_map = self.get_imdm_weather_cell_map().reset_index()

                # Merge the acquired data set
                weather_cell = imdm_weather_cell_map.join(
                    weather_cell, on='WeatherCellId').sort_values('WeatherCellId')
                weather_cell.set_index('WeatherCellId', inplace=True)

                # Create polygons WGS84 (Longitude, Latitude)
                weather_cell['Polygon_WGS84'] = weather_cell.apply(
                    lambda x: shapely.geometry.Polygon(
                        zip([x.ll_Longitude, x.ul_Longitude, x.ur_Longitude, x.lr_Longitude],
                            [x.ll_Latitude, x.ul_Latitude, x.ur_Latitude, x.lr_Latitude])), axis=1)

                # Create polygons OSGB36 (Easting, Northing)
                weather_cell['ll_Easting'], weather_cell['ll_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ll_Longitude.values, weather_cell.ll_Latitude.values)
                weather_cell['ul_Easting'], weather_cell['ul_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ul_Longitude.values, weather_cell.ul_Latitude.values)
                weather_cell['ur_Easting'], weather_cell['ur_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ur_Longitude.values, weather_cell.ur_Latitude.values)
                weather_cell['lr_Easting'], weather_cell['lr_Northing'] = \
                    wgs84_to_osgb36(weather_cell.lr_Longitude.values, weather_cell.lr_Latitude.values)

                weather_cell['Polygon_OSGB36'] = weather_cell.apply(
                    lambda x: shapely.geometry.Polygon(
                        zip([x.ll_Easting, x.ul_Easting, x.ur_Easting, x.lr_Easting],
                            [x.ll_Northing, x.ul_Northing, x.ur_Northing, x.lr_Northing])), axis=1)

                regions_and_routes = load_json(cdd_network("Regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                weather_cell['Region'] = weather_cell.Route.replace(regions_and_routes_dict)

                weather_cell = get_subset(weather_cell, route_name)

                save_pickle(weather_cell, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.WeatherCell, e))
                weather_cell = None

        # Plot the Weather cells on the map?
        if show_map:
            weather_cell_wgs84 = shapely.geometry.MultiPolygon(list(weather_cell.Polygon_WGS84))
            minx, miny, maxx, maxy = weather_cell_wgs84.bounds

            print("Plotting weather cells ...", end="")
            fig, ax = plt.subplots(figsize=(5, 8))
            base_map = mpl_toolkits.basemap.Basemap(projection=projection,
                                                    # Transverse Mercator Projection
                                                    ellps='WGS84',
                                                    epsg=27700,
                                                    llcrnrlon=minx - 0.285,
                                                    llcrnrlat=miny - 0.255,
                                                    urcrnrlon=maxx + 1.185,
                                                    urcrnrlat=maxy + 0.255,
                                                    lat_ts=0,
                                                    resolution='l',
                                                    suppress_ticks=True)

            base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)

            weather_cell_map = weather_cell.drop_duplicates(
                subset=[s for s in weather_cell.columns if '_' in s and not s.startswith('Polygon')])

            for i in weather_cell_map.index:
                ll_x, ll_y = base_map(weather_cell_map.ll_Longitude[i], weather_cell_map.ll_Latitude[i])
                ul_x, ul_y = base_map(weather_cell_map.ul_Longitude[i], weather_cell_map.ul_Latitude[i])
                ur_x, ur_y = base_map(weather_cell_map.ur_Longitude[i], weather_cell_map.ur_Latitude[i])
                lr_x, lr_y = base_map(weather_cell_map.lr_Longitude[i], weather_cell_map.lr_Latitude[i])
                xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
                polygons = matplotlib.patches.Polygon(list(xy), fc='#D5EAFF', ec='#4b4747', alpha=0.5)
                ax.add_patch(polygons)
            plt.plot([], 's', label="Weather cell", ms=14, color='#D5EAFF', markeredgecolor='#4b4747')
            legend = plt.legend(numpoints=1, loc='best', fancybox=True, labelspacing=0.5)
            frame = legend.get_frame()
            frame.set_edgecolor('k')
            plt.tight_layout()

            print("Done.")

            if save_map_as:
                save_fig(self.cdd_figures(pickle_filename.replace(".pickle", save_map_as)), dpi=dpi,
                         verbose=verbose)

        return weather_cell

    def get_weather_cell_map_boundary(self, route_name=None, adjustment=(0.285, 0.255)):
        """
        Get the lower-left and upper-right corners for a weather cell map.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str, None
        :param adjustment: defaults to ``(0.285, 0.255)``
        :type adjustment: tuple
        :return: a boundary for a weather cell map
        :rtype: shapely.geometry.polygon.Polygon

        **Test**::

            from mssqlserver import metex

            route_name = None
            adjustment = (0.285, 0.255)

            boundary = metex.get_weather_cell_map_boundary(route_name, adjustment)
            print(boundary)
        """

        weather_cell = self.get_weather_cell()  # Get Weather cell

        if route_name:  # For a specific Route
            weather_cell = weather_cell[
                weather_cell.Route == find_similar_str(route_name, self.get_route().Route)]
        ll = tuple(weather_cell[['ll_Longitude', 'll_Latitude']].apply(min))
        lr = weather_cell.lr_Longitude.max(), weather_cell.lr_Latitude.min()
        ur = tuple(weather_cell[['ur_Longitude', 'ur_Latitude']].apply(max))
        ul = weather_cell.ul_Longitude.min(), weather_cell.ul_Latitude.max()

        if adjustment:  # Adjust the boundaries
            adj_values = np.array(adjustment)
            ll -= adj_values
            lr += (adj_values, -adj_values)
            ur += adj_values
            ul += (-adj_values, adj_values)

        boundary = shapely.geometry.Polygon((ll, lr, ur, ul))

        return boundary

    def get_track(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Track'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'Track'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            track = metex.get_track(update, save_original_as, verbose)
            print(track)
        """

        METEX.Track = 'Track'
        path_to_pickle = self.cdd_tables(METEX.Track + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            track = load_pickle(path_to_pickle)

        else:
            try:
                track = read_table_by_query(database_name=self.DatabaseName, table_name=METEX.Track,
                                            save_as=save_original_as).drop_duplicates()
                track.rename(columns={'S_MILEAGE': 'StartMileage', 'F_MILEAGE': 'EndMileage',
                                      'S_YARDAGE': 'StartYard', 'F_YARDAGE': 'EndYard',
                                      'MAINTAINER': 'Maintainer', 'ROUTE': 'Route',
                                      'DELIVERY_U': 'IMDM',
                                      'StartEasti': 'StartEasting', 'StartNorth': 'StartNorthing',
                                      'EndNorthin': 'EndNorthing'},
                             inplace=True)

                # Mileage and Yardage
                mileage_cols, yardage_cols = ['StartMileage', 'EndMileage'], ['StartYard', 'EndYard']
                track[mileage_cols] = track[mileage_cols].applymap(nr_mileage_num_to_str)
                track[yardage_cols] = track[yardage_cols].applymap(int)

                # Route
                update_nr_route_names(track, route_col_name='Route')

                # Delivery Unit and IMDM
                track.IMDM = track.IMDM.map(lambda x: 'IMDM ' + x)

                # Start and end longitude and latitude coordinates
                track['StartLongitude'], track['StartLatitude'] = osgb36_to_wgs84(
                    track.StartEasting.values, track.StartNorthing.values)
                track['EndLongitude'], track['EndLatitude'] = osgb36_to_wgs84(
                    track.EndEasting.values, track.EndNorthing.values)

                track[['StartMileage_num', 'EndMileage_num']] = track[
                    ['StartMileage', 'EndMileage']].applymap(nr_mileage_str_to_num)

                track.sort_values(['ELR', 'StartYard', 'EndYard'], inplace=True)
                track.index = range(len(track))

                save_pickle(track, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.Track, e))
                track = None

        return track

    @staticmethod
    def create_track_geometric_graph(geom_objs, rotate_labels=None):
        """
        Create a graph to illustrate track geometry.

        :param geom_objs: geometry objects
        :type geom_objs: iterable of [WKT str, shapely.geometry.LineString,
            or shapely.geometry.MultiLineString]
        :param rotate_labels: defaults to ``None``
        :type rotate_labels: numbers.Number, None
        :return: a graph demonstrating the tracks

        **Test**::

            from mssqlserver import metex

            track_data = metex.get_track()
            geom_objs = track_data.geom[list(range(len(track_data[track_data.ELR == 'AAV'])))]

            metex.create_track_geometric_graph(geom_objs)
        """

        g = nx.Graph()

        fig, ax = plt.subplots()

        max_node_id = 0
        for geom_obj in geom_objs:

            if isinstance(geom_obj, str):
                geom_obj = shapely.wkt.loads(geom_obj)

            geom_type, geom_pos = shapely.geometry.mapping(geom_obj).values()

            if geom_type == 'MultiLineString':

                # Sort line strings of the multi-line string
                geom_pos_idx = list(range(len(geom_pos)))
                pos_idx_sorted = []
                while geom_pos_idx:
                    y = geom_pos_idx[0]
                    p1 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                    if p1 and p1[0] not in pos_idx_sorted:
                        pos_idx_sorted = p1 + [y]
                    else:
                        pos_idx_sorted = [y]

                    p2 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                    if p2:
                        pos_idx_sorted += p2

                    geom_pos_idx = [a for a in geom_pos_idx if a not in pos_idx_sorted]

                    if len(geom_pos_idx) == 1:
                        y = geom_pos_idx[0]
                        p3 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                        if p3 and p3[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p3[0]) + 1, y)
                            break

                        p4 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                        if p4 and p4[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p4[0]), y)
                            break

                geom_pos = [geom_pos[i] for i in pos_idx_sorted]
                geom_pos = [x[:-1] for x in geom_pos[:-1]] + [geom_pos[-1]]
                geom_pos = [x for g_pos in geom_pos for x in g_pos]

            # j = 0
            # n = g.number_of_nodes()
            # while j < len(geom_pos):
            #     # Nodes
            #     g_pos = geom_pos[j]
            #     for i in range(len(g_pos)):
            #         g.add_node(i + n + 1, pos=g_pos[i])
            #     # Edges
            #     current_max_node_id = g.number_of_nodes()
            #     edges = [(x, x + 1) for x in range(n + 1, n + current_max_node_id)
            #              if x + 1 <= current_max_node_id]
            #     g.add_edges_from(edges)
            #     n = current_max_node_id
            #     j += 1

            # Nodes

            for i in range(len(geom_pos)):
                g.add_node(i + 1 + max_node_id, pos=geom_pos[i])

            # Edges
            number_of_nodes = g.number_of_nodes()
            edges = [(i, i + 1) for i in range(1 + max_node_id, number_of_nodes + 1) if
                     i + 1 <= number_of_nodes]
            # edges = [(i, i + 1) for i in range(1, number_of_nodes + 1) if i + 1 <= number_of_nodes]
            g.add_edges_from(edges)

            # Plot
            nx.draw_networkx(g, pos=nx.get_node_attributes(g, name='pos'), ax=ax, node_size=0,
                             with_labels=False)

            max_node_id = number_of_nodes

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, gridOn=True,
                       grid_linestyle='--')
        ax.ticklabel_format(useOffset=False)
        ax.set_aspect('equal')
        font_dict = {'fontsize': 13, 'fontname': 'Times New Roman'}
        ax.set_xlabel('Easting', fontname=font_dict['fontname'], fontsize=14)
        ax.set_xticklabels([int(x) for x in ax.get_xticks()], font_dict)
        ax.set_ylabel('Northing', fontname=font_dict['fontname'], fontsize=14)
        ax.set_yticklabels([int(y) for y in ax.get_yticks()], font_dict)
        if rotate_labels:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate_labels)
        plt.tight_layout()

    @staticmethod
    def cleanse_track_summary(dat):
        """
        Preprocess data of the table 'TrackSummary'.

        :param dat: data of the table 'TrackSummary'
        :type dat: pandas.DataFrame
        :return: preprocessed data frame of ``dat``
        :rtype: pandas.DataFrame

        **Test**::

            dat = track_summary.copy()

            cleanse_track_summary(dat)
        """

        # Change column names
        rename_cols = {'GEOGIS Switch ID': 'GeoGISSwitchID',
                       'TID': 'TrackID',
                       'StartYards': 'StartYard',
                       'EndYards': 'EndYard',
                       'Sub-route': 'SubRoute',
                       'CP6 criticality': 'CP6Criticality',
                       'CP5 Start Route': 'CP5StartRoute',
                       'Adjacent S&C': 'AdjacentS&C',
                       'Rail cumulative EMGT': 'RailCumulativeEMGT',
                       'Sleeper cumulative EMGT': 'SleeperCumulativeEMGT',
                       'Ballast cumulative EMGT': 'BallastCumulativeEMGT'}
        dat.rename(columns=rename_cols, inplace=True)
        renamed_cols = list(rename_cols.values())
        upper_columns = ['ID', 'SRS', 'ELR', 'IMDM', 'TME', 'TSM', 'MGTPA', 'EMGTPA', 'LTSF', 'IRJs']
        dat.columns = [
            string.capwords(x).replace(' ', '') if x not in upper_columns + renamed_cols else x
            for x in dat.columns]

        # IMDM
        dat.IMDM = dat.IMDM.map(lambda x: 'IMDM ' + x)

        # Route
        route_names_changes = load_json(cdd_network("Routes", "name-changes.json"))
        # noinspection PyTypeChecker
        temp1 = pd.DataFrame.from_dict(route_names_changes, orient='index', columns=['Route'])
        route_names_in_table = list(dat.SubRoute.unique())
        route_alt = [find_similar_str(x, temp1.index) for x in route_names_in_table]

        temp2 = pd.DataFrame.from_dict(dict(zip(route_names_in_table, route_alt)), 'index',
                                       columns=['RouteAlias'])
        temp = temp2.join(temp1, on='RouteAlias').dropna()
        route_names_changes_alt = dict(zip(temp.index, temp.Route))
        dat['Route'] = dat.SubRoute.replace(route_names_changes_alt)

        # Mileages
        mileage_colnames, yard_colnames = ['StartMileage', 'EndMileage'], ['StartYards', 'EndYards']
        dat[mileage_colnames] = dat[yard_colnames].applymap(yards_to_nr_mileage)

        return dat

    def get_track_summary(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Track Summary'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str, None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of the table 'Track Summary'
        :rtype: pandas.DataFrame, None

        **Test**::

            from mssqlserver import metex

            update           = True
            save_original_as = None
            verbose          = True

            track_summary = metex.get_track_summary(update, save_original_as, verbose)
            print(track_summary)
        """

        METEX.TrackSummary = 'Track Summary'
        path_to_pickle = self.cdd_tables(METEX.TrackSummary.replace(' ', '') + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            track_summary = load_pickle(path_to_pickle)

        else:
            try:
                track_summary_raw = self.read_table(
                    table_name=METEX.TrackSummary, save_as=save_original_as, update=update)

                track_summary = self.cleanse_track_summary(track_summary_raw)

                save_pickle(track_summary, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METEX.TrackSummary, e))
                track_summary = None

        return track_summary

    def query_track_summary(self, elr, track_id, start_yard=None, end_yard=None, pickle_it=True,
                            dat_dir=None, update=False, verbose=False):
        """
        Get track summary data by ``'Track ID'`` and ``'Yard'`` (Query from the database).

        :param elr: ELR
        :type elr: str
        :param track_id: TrackID
        :type track_id: tuple, int
        :param start_yard: start yard, defaults to ``None``
        :type start_yard: int, None
        :param end_yard: end yard, defaults to ``None``
        :type end_yard: int, None
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of track summary queried by
            ``'elr'``, ``'track_id'``, ``'start_yard'`` and ``'end_yard'``
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            elr        = 'AAV'
            track_id   = 1100
            start_yard = 51150
            end_yard   = 66220
            pickle_it  = False
            dat_dir    = None
            update     = False
            show       = False
            verbose    = False

            track_summary = metex.query_track_summary(elr, track_id, start_yard, end_yard, pickle_it,
                                                      dat_dir, update, verbose)
            print(track_summary)
        """

        assert isinstance(elr, str)
        assert isinstance(track_id, (tuple, int, np.integer))

        pickle_filename = "{}_{}_{}_{}.pickle".format(
            "-".join(str(x) for x in list(elr)) if isinstance(elr, tuple) else elr,
            "-".join(str(x) for x in list(track_id)) if isinstance(track_id, tuple) else track_id,
            start_yard, end_yard)

        dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else self.cdd_views()
        path_to_pickle = cd(dat_dir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                conn_metex = establish_mssql_connection(database_name=self.DatabaseName)
                sql_query = \
                    "SELECT * FROM dbo.[Track Summary] WHERE {}{}'{}' AND {}{}{} AND {} AND {};".format(
                        "[ELR]", " = " if isinstance(elr, str) else " IN ", elr,
                        "[TID]", " = " if isinstance(track_id, (int, np.integer)) else " IN ", track_id,
                        "[Start Yards] >= {}".format(start_yard) if start_yard else "",
                        "[End Yards] <= {}".format(end_yard) if end_yard else "")
                track_summary_raw = pd.read_sql(sql_query, conn_metex)

                track_summary = self.cleanse_track_summary(track_summary_raw)

                if pickle_it:
                    save_pickle(track_summary, path_to_pickle, verbose=verbose)

                return track_summary

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(
                    os.path.splitext(os.path.basename(path_to_pickle))[0], e))

    def update_metex_table_pickles(self, update=True, verbose=True):
        """
        Update the local pickle files for all tables.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool, int

        **Test**::

            update = True
            verbose = True

            update_metex_table_pickles(update, verbose)
        """

        if confirmed("To update the local pickles of the Table data of the NR_METEX database?"):

            _ = self.get_imdm(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_imdm(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_imdm_alias(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_imdm_alias(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_imdm_weather_cell_map(route_info=True, grouped=False, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=True, grouped=True, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=False, grouped=False, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=False, grouped=True, update=update,
                                               save_original_as=None, verbose=verbose)

            _ = self.get_incident_reason_info(plus=True, update=update, save_original_as=None,
                                              verbose=verbose)
            _ = self.get_incident_reason_info(plus=False, update=update, save_original_as=None,
                                              verbose=verbose)

            _ = self.get_weather_codes(as_dict=False, update=update, save_original_as=None,
                                       verbose=verbose)
            _ = self.get_weather_codes(as_dict=True, update=update, save_original_as=None,
                                       verbose=verbose)

            _ = self.get_incident_record(update=update, save_original_as=None, use_amendment_csv=True,
                                         verbose=verbose)

            _ = self.get_location(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_pfpi(plus=True, update=update, save_original_as=None, use_amendment_csv=True,
                              verbose=verbose)
            _ = self.get_pfpi(plus=False, update=update, save_original_as=None, use_amendment_csv=True,
                              verbose=verbose)

            _ = self.get_route(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_route(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_stanox_location(use_nr_mileage_format=True, update=update,
                                         save_original_as=None, verbose=verbose)
            _ = self.get_stanox_location(use_nr_mileage_format=False, update=update,
                                         save_original_as=None, verbose=verbose)

            _ = self.get_stanox_section(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_trust_incident(start_year=2006, end_year=None, update=update,
                                        save_original_as=None, use_amendment_csv=True, verbose=verbose)

            # _ = get_weather()

            _ = self.get_weather_cell(route_name=None, update=update, save_original_as=None,
                                      show_map=True, projection='tmerc', save_map_as=".tif", dpi=None,
                                      verbose=verbose)
            _ = self.get_weather_cell('Anglia', update=update, save_original_as=None, show_map=True,
                                      projection='tmerc', save_map_as=".tif", dpi=None, verbose=verbose)
            # _ = self.get_weather_cell_map_boundary(route=None, adjustment=(0.285, 0.255))

            _ = self.get_track(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_track_summary(update=update, save_original_as=None, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")

    # == Tools to make information integration easier =================================================

    @staticmethod
    def calculate_pfpi_stats(data_set, selected_features, sort_by=None):
        """
        Calculate the 'DelayMinutes' and 'DelayCosts' for grouped data.

        :param data_set: a given data frame
        :type data_set: pandas.DataFrame
        :param selected_features: a list of selected features (column names)
        :type selected_features: list
        :param sort_by: a column or a list of columns by which the selected data is sorted,
            defaults to ``None``
        :type sort_by: str, list, None
        :return: pandas.DataFrame

        **Test**::

            data_set = selected_data.copy()

            calculate_pfpi_stats(data_set, selected_features, sort_by=None)
        """

        pfpi_stats = data_set.groupby(selected_features[1:-2]).aggregate({
            # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
            'PfPIId': np.count_nonzero,
            'PfPIMinutes': np.sum,
            'PfPICosts': np.sum})

        pfpi_stats.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']
        pfpi_stats.reset_index(inplace=True)  # Reset the grouped indexes to columns

        if sort_by:
            pfpi_stats.sort_values(sort_by, inplace=True)

        return pfpi_stats

    # == Functions to create views ====================================================================

    def view_schedule8_data(self, route_name=None, weather_category=None, rearrange_index=False,
                            weather_attributed_only=False, update=False, pickle_it=False, verbose=False):
        """
        View Schedule 8 details (TRUST data).

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param rearrange_index: whether to rearrange the index of the queried data,
            defaults to ``False``
        :type rearrange_index: bool
        :param weather_attributed_only: defaults to ``False``
        :type weather_attributed_only: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of Schedule 8 details
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            route_name = None
            weather_category = None
            rearrange_index = True
            update = True
            pickle_it = True
            verbose = True

            weather_attributed_only = False
            schedule8_data = metex.view_schedule8_data(route_name, weather_category, rearrange_index,
                                                       weather_attributed_only, update, pickle_it,
                                                       verbose)
            print(schedule8_data)

            weather_attributed_only = True
            schedule8_data = metex.view_schedule8_data(route_name, weather_category, rearrange_index,
                                                       weather_attributed_only, update, pickle_it,
                                                       verbose)
            print(schedule8_data)
        """

        filename = "s8data" + ("-weather-attributed" if weather_attributed_only else "")
        pickle_filename = make_filename(filename, route_name, weather_category, save_as=".pickle")
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            schedule8_data = load_pickle(path_to_pickle)
            if rearrange_index and schedule8_data.index.name == 'PfPIId':
                schedule8_data.reset_index(inplace=True)

        else:
            path_to_merged = self.cdd_views("{}.pickle".format(filename))

            try:

                if os.path.isfile(path_to_merged) and not update:
                    schedule8_data = load_pickle(path_to_merged)

                else:
                    pfpi = self.get_pfpi(verbose=verbose)  # Get PfPI # (260645, 6)  # (5049003, 6)
                    incident_record = self.get_incident_record(
                        verbose=verbose)  # (233452, 4)  # (4704448, 5)
                    trust_incident = self.get_trust_incident(
                        verbose=verbose)  # (192054, 11)  # (4049984, 11)
                    location = self.get_location(verbose=verbose)  # (228851, 6)  # (653882, 7)
                    imdm = self.get_imdm(verbose=verbose)  # (42, 1)  # (42, 3)
                    incident_reason_info = self.get_incident_reason_info(
                        verbose=verbose)  # (393, 7)  # (174, 9)
                    stanox_location = self.get_stanox_location(verbose=verbose)  # (7560, 5)  # (7534, 6)
                    stanox_section = self.get_stanox_section(verbose=verbose)  # (9440, 7)  # (10601, 7)

                    if weather_attributed_only:
                        incident_record = incident_record[
                            incident_record.WeatherCategory != '']  # (320942, 5) ≈ 6.8%

                    # Merge the acquired data sets - starting with (5049003, 6)
                    schedule8_data = pfpi. \
                        join(incident_record,  # (260645, 10)  # (5049003, 11)
                             on='IncidentRecordId', how='inner'). \
                        join(trust_incident,  # (260483, 21)  # (5048710, 22)
                             on='TrustIncidentId', how='inner'). \
                        join(stanox_section,  # (260483, 28)  # (5048593, 29)
                             on='StanoxSectionId', how='inner'). \
                        join(location,  # (260470, 34)  # (5045204, 36)
                             on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
                        join(stanox_location,  # (260190, 39)  # (5029204, 42)
                             on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
                        join(stanox_location,  # (260140, 44)  # (5024725, 48)
                             on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
                        join(incident_reason_info,  # (260140, 51)  # (5024703, 57)
                             on='IncidentReasonCode', how='inner'). \
                        join(imdm, on='IMDM_Location', how='inner')  # (5024674, 60)

                    del pfpi, incident_record, trust_incident, stanox_section, location, \
                        stanox_location, incident_reason_info, imdm
                    gc.collect()

                    # Note: There may be errors in
                    # e.g. IMDM data/column, location id, of the TrustIncident table.

                    idx = schedule8_data[
                        ~schedule8_data.StartLocation.eq(schedule8_data.Location_Start)].index
                    for i in idx:
                        schedule8_data.loc[i, 'StartLocation'] = schedule8_data.loc[i, 'Location_Start']
                        schedule8_data.loc[i, 'EndLocation'] = schedule8_data.loc[i, 'Location_End']
                        if schedule8_data.loc[i, 'StartLocation'] == \
                                schedule8_data.loc[i, 'EndLocation']:
                            schedule8_data.loc[i, 'StanoxSection'] = schedule8_data.loc[
                                i, 'StartLocation']
                        else:
                            schedule8_data.loc[i, 'StanoxSection'] = \
                                schedule8_data.loc[i, 'StartLocation'] + ' - ' + schedule8_data.loc[
                                    i, 'EndLocation']

                    schedule8_data.drop(['IMDM', 'Location_Start', 'Location_End'], axis=1,
                                        inplace=True)  # (5024674, 57)

                    # (260140, 50)  # (5155014, 57)
                    schedule8_data.rename(columns={'LocationAlias_Start': 'StartLocationAlias',
                                                   'LocationAlias_End': 'EndLocationAlias',
                                                   'ELR_Start': 'StartELR', 'Yards_Start': 'StartYards',
                                                   'ELR_End': 'EndELR', 'Yards_End': 'EndYards',
                                                   'Mileage_Start': 'StartMileage',
                                                   'Mileage_End': 'EndMileage',
                                                   'LocationId_Start': 'StartLocationId',
                                                   'LocationId_End': 'EndLocationId',
                                                   'LocationId_Section': 'SectionLocationId',
                                                   'IMDM_Location': 'IMDM',
                                                   'StartDate': 'StartDateTime',
                                                   'EndDate': 'EndDateTime'},
                                          inplace=True)

                    # Use 'Station' data from Railway Codes website
                    stn = Stations()
                    station_locations = stn.fetch_station_data()['Railway station data']

                    station_locations = station_locations[
                        ['Station', 'Degrees Longitude', 'Degrees Latitude']]
                    station_locations = station_locations.dropna().drop_duplicates('Station',
                                                                                   keep='first')
                    station_locations.set_index('Station', inplace=True)
                    temp = schedule8_data[['StartLocation']].join(station_locations, on='StartLocation',
                                                                  how='left')
                    i = temp[temp['Degrees Longitude'].notna()].index
                    schedule8_data.loc[i, 'StartLongitude':'StartLatitude'] = \
                        temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()
                    temp = schedule8_data[['EndLocation']].join(station_locations, on='EndLocation',
                                                                how='left')
                    i = temp[temp['Degrees Longitude'].notna()].index
                    schedule8_data.loc[i, 'EndLongitude':'EndLatitude'] = \
                        temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()

                    # data.EndELR.replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)
                    i = schedule8_data.StartLocation == 'Highbury & Islington (North London Lines)'
                    schedule8_data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.1045, 51.5460]
                    i = schedule8_data.EndLocation == 'Highbury & Islington (North London Lines)'
                    schedule8_data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.1045, 51.5460]
                    i = schedule8_data.StartLocation == 'Dalston Junction (East London Line)'
                    schedule8_data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.0751, 51.5461]
                    i = schedule8_data.EndLocation == 'Dalston Junction (East London Line)'
                    schedule8_data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.0751, 51.5461]

                schedule8_data.reset_index(inplace=True)  # (5024674, 58)

                schedule8_data = get_subset(schedule8_data, route_name, weather_category,
                                            rearrange_index)

                if pickle_it:
                    if not os.path.isfile(path_to_merged) or path_to_pickle != path_to_merged:
                        save_pickle(schedule8_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to retrieve the data about Schedule 8 incidents. {}.".format(e))
                schedule8_data = None

        return schedule8_data

    def view_schedule8_data_pfpi(self, route_name=None, weather_category=None, update=False,
                                 pickle_it=False,
                                 verbose=False):
        """
        Get a view of essential details about Schedule 8 incidents.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: essential details about Schedule 8 incidents
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            route_name = None
            weather_category = None
            update = True
            pickle_it = True
            verbose = True

            data = metex.view_schedule8_data_pfpi(route_name, weather_category, update, pickle_it,
                                                  verbose)
            print(data)
        """

        filename = "s8data-pfpi"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))
                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    data = get_subset(temp_data, route_name, weather_category)

                else:
                    # Get the merged data
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    # Define the feature list
                    selected_features = [
                        'PfPIId',
                        'IncidentRecordId',
                        'TrustIncidentId',
                        'IncidentNumber',
                        'PerformanceEventCode', 'PerformanceEventGroup', 'PerformanceEventName',
                        'PfPIMinutes', 'PfPICosts', 'FinancialYear',
                        'IncidentRecordCreateDate',
                        'StartDateTime', 'EndDateTime',
                        'IncidentDescription', 'IncidentJPIPCategory',
                        'WeatherCategory',
                        'IncidentReasonCode', 'IncidentReasonDescription',
                        'IncidentCategory', 'IncidentCategoryDescription',
                        # 'IncidentCategoryGroupDescription',
                        'IncidentFMS', 'IncidentEquipment',
                        'WeatherCell',
                        'Route', 'IMDM', 'Region',
                        'StanoxSection', 'StartLocation', 'EndLocation',
                        'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                        'ApproximateLocation']
                    data = schedule8_data[selected_features]

                if pickle_it:
                    save_pickle(data, path_to_pickle, verbose=verbose)

                return data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_location(self, route_name=None, weather_category=None, update=False,
                                         pickle_it=True, verbose=False) -> pd.DataFrame:
        """
        Get Schedule 8 data by incident location and Weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 data by incident location and Weather category
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            update = True
            pickle_it = True
            verbose = True

            route_name = None
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_location(route_name, weather_category,
                                                                    update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_location(route_name, weather_category,
                                                                    update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Wind'
            extracted_data = metex.view_schedule8_costs_by_location(route_name, weather_category,
                                                                    update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Heat'
            extracted_data = metex.view_schedule8_costs_by_location(route_name, weather_category,
                                                                    update, pickle_it, verbose)
            print(extracted_data)
        """

        filename = "s8costs-by-location"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))
                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'WeatherCategory',
                        'Route', 'IMDM', 'Region', 'StanoxSection',
                        'StartLocation', 'EndLocation', 'StartELR', 'StartMileage', 'EndELR',
                        'EndMileage',
                        'StartStanox', 'EndStanox', 'StartLongitude', 'StartLatitude', 'EndLongitude',
                        'EndLatitude',
                        'PfPIMinutes', 'PfPICosts']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime_location(self, route_name=None, weather_category=None,
                                                  update=False,
                                                  pickle_it=True, verbose=False) -> pd.DataFrame:
        """
        Get Schedule 8 data by datetime and location.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 data by datetime and location
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            update = True
            pickle_it = True
            verbose = True

            route_name = None
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_datetime_location(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_datetime_location(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Wind'
            extracted_data = metex.view_schedule8_costs_by_datetime_location(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Heat'
            extracted_data = metex.view_schedule8_costs_by_datetime_location(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)
        """

        filename = "s8costs-by-datetime-location"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'FinancialYear',
                        'StartDateTime', 'EndDateTime',
                        'WeatherCategory',
                        'StanoxSection',
                        'Route', 'IMDM', 'Region',
                        'StartLocation', 'EndLocation',
                        'StartStanox', 'EndStanox',
                        'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                        'WeatherCell',
                        'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime_location_reason(self, route_name=None, weather_category=None,
                                                         update=False, pickle_it=True,
                                                         verbose=False) -> pd.DataFrame:
        """
        Get Schedule 8 costs by datetime, location and incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 costs by datetime, location and incident reason
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            update = True
            pickle_it = True
            verbose = True

            route_name = None
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_datetime_location_reason(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = None
            extracted_data = metex.view_schedule8_costs_by_datetime_location_reason(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Wind'
            extracted_data = metex.view_schedule8_costs_by_datetime_location_reason(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)

            route_name = 'Anglia'
            weather_category = 'Heat'
            extracted_data = metex.view_schedule8_costs_by_datetime_location_reason(
                route_name, weather_category, update, pickle_it, verbose)
            print(extracted_data)
        """

        filename = "s8costs-by-datetime-location-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'StartDateTime', 'EndDateTime',
                                         'WeatherCategory',
                                         'WeatherCell',
                                         'Route', 'IMDM', 'Region',
                                         'StanoxSection',
                                         'StartLocation', 'EndLocation',
                                         'StartStanox', 'EndStanox',
                                         'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                                         'StartLongitude', 'StartLatitude', 'EndLongitude',
                                         'EndLatitude',
                                         'IncidentDescription',
                                         'IncidentCategory',
                                         'IncidentCategoryDescription',
                                         'IncidentCategorySuperGroupCode',
                                         # 'IncidentCategoryGroupDescription',
                                         'IncidentReasonCode',
                                         'IncidentReasonName',
                                         'IncidentReasonDescription',
                                         'IncidentJPIPCategory',
                                         'PfPIMinutes', 'PfPICosts']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime(self, route_name=None, weather_category=None, update=False,
                                         pickle_it=False,
                                         verbose=False):
        """
        Get Schedule 8 data by datetime and Weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 data by datetime and Weather category
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-datetime"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))
                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'FinancialYear',
                        'StartDateTime', 'EndDateTime',
                        'WeatherCategory',
                        'Route', 'IMDM', 'Region',
                        'WeatherCell',
                        'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_reason(self, route_name=None, weather_category=None, update=False,
                                       pickle_it=False,
                                       verbose=False):
        """
        Get Schedule 8 costs by incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 costs by incident reason
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'Route', 'IMDM', 'Region',
                                         'WeatherCategory',
                                         'IncidentDescription',
                                         'IncidentCategory',
                                         'IncidentCategoryDescription',
                                         'IncidentCategorySuperGroupCode',
                                         # 'IncidentCategoryGroupDescription',
                                         'IncidentReasonCode',
                                         'IncidentReasonName',
                                         'IncidentReasonDescription',
                                         'IncidentJPIPCategory',
                                         'PfPIMinutes', 'PfPICosts']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=None)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_location_reason(self, route_name=None, weather_category=None,
                                                update=False,
                                                pickle_it=False,
                                                verbose=False):
        """
        Get Schedule 8 costs by location and incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 costs by location and incident reason
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-location-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'WeatherCategory',
                                         'Route', 'IMDM', 'Region',
                                         'StanoxSection',
                                         'StartStanox', 'EndStanox',
                                         'StartLocation', 'EndLocation',
                                         'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                                         'StartLongitude', 'StartLatitude', 'EndLongitude',
                                         'EndLatitude',
                                         'IncidentDescription',
                                         'IncidentCategory',
                                         'IncidentCategoryDescription',
                                         'IncidentCategorySuperGroupCode',
                                         # 'IncidentCategoryGroupDescription',
                                         'IncidentReasonCode',
                                         'IncidentReasonName',
                                         'IncidentReasonDescription',
                                         'IncidentJPIPCategory',
                                         'PfPIMinutes', 'PfPICosts']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=None)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_weather_category(self, route_name=None, weather_category=None,
                                                 update=False, pickle_it=False, verbose=False):
        """
        Get Schedule 8 costs by weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 costs by weather category
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-weather_category"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId', 'FinancialYear', 'Route', 'IMDM', 'Region',
                                         'WeatherCategory', 'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print(
                    "Failed to retrieve \"{}.\" \n{}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_metex_schedule8_incident_locations(self, route_name=None, weather_category=None,
                                                start_and_end_elr=None, update=False,
                                                verbose=False) -> pd.DataFrame:
        """
        Get Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param start_and_end_elr: indicating if start ELR and end ELR are the same or not,
            defaults to ``False``
        :type start_and_end_elr: str, bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section
        :rtype: pandas.DataFrame

        **Test**::

            from mssqlserver import metex

            weather_category = None
            update = True
            verbose = True

            route_name = None
            start_and_end_elr = None
            incident_locations = metex.view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr, update, verbose)
            print(incident_locations)

            route_name = None
            start_and_end_elr = 'same'
            incident_locations = metex.view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr, update, verbose)
            print(incident_locations)

            route_name = None
            start_and_end_elr = 'diff'
            incident_locations = metex.view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr, update, verbose)
            print(incident_locations)

            route_name = 'Anglia'
            start_and_end_elr = None
            incident_locations = metex.view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr, update, verbose)
            print(incident_locations)
        """

        assert start_and_end_elr in (None, 'same', 'diff')

        filename = "s8incident-locations"

        start_and_end_elr_ = start_and_end_elr + 'ELR' if start_and_end_elr else start_and_end_elr
        pickle_filename = make_filename(filename, route_name, weather_category, start_and_end_elr_)
        path_to_pickle = self.cdd_views(pickle_filename)

        try:
            if os.path.isfile(path_to_pickle) and not update:
                incident_locations = load_pickle(path_to_pickle)

            else:
                # All incident locations
                s8costs_by_location = self.view_schedule8_costs_by_location(route_name, weather_category,
                                                                            update=update)
                s8costs_by_location = s8costs_by_location.loc[:, 'Route':'EndLatitude']
                incident_locations = s8costs_by_location.drop_duplicates()

                # Remove records for which ELR information was missing
                incident_locations = incident_locations[
                    ~(incident_locations.StartELR.str.contains('^$')) & ~(
                        incident_locations.EndELR.str.contains('^$'))]

                # 'FJH'
                idx = (incident_locations.StartLocation == 'Halton Junction') & (
                        incident_locations.StartELR == 'FJH')
                incident_locations.loc[idx, 'StartMileage'] = '0.0000'
                idx = (incident_locations.EndLocation == 'Halton Junction') & (
                        incident_locations.EndELR == 'FJH')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'BNE'
                idx = (incident_locations.StartLocation == 'Benton North Junction') & (
                        incident_locations.StartELR == 'BNE')
                incident_locations.loc[idx, 'StartMileage'] = '0.0000'
                idx = (incident_locations.EndLocation == 'Benton North Junction') & (
                        incident_locations.EndELR == 'BNE')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'WCI'
                idx = (incident_locations.StartLocation == 'Grangetown (Cleveland)') & \
                      (incident_locations.StartELR == 'WCI')
                incident_locations.loc[
                    idx, ('StartELR', 'StartMileage')] = 'DSN2', mile_chain_to_nr_mileage('1.38')
                idx = (incident_locations.EndLocation == 'Grangetown (Cleveland)') & (
                        incident_locations.EndELR == 'WCI')
                incident_locations.loc[
                    idx, ('EndELR', 'EndMileage')] = 'DSN2', mile_chain_to_nr_mileage('1.38')
                # 'SJD'
                idx = (incident_locations.EndLocation == 'Skelton Junction [Manchester]') & \
                      (incident_locations.EndELR == 'SJD')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'HLK'
                idx = (incident_locations.EndLocation == 'High Level Bridge Junction') & \
                      (incident_locations.EndELR == 'HLK')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'

                # Create two additional columns about data of mileages (convert str to num)
                incident_locations[['StartMileage_num', 'EndMileage_num']] = \
                    incident_locations[['StartMileage', 'EndMileage']].applymap(nr_mileage_str_to_num)

                incident_locations['StartEasting'], incident_locations['StartNorthing'] = \
                    wgs84_to_osgb36(incident_locations.StartLongitude.values,
                                    incident_locations.StartLatitude.values)
                incident_locations['EndEasting'], incident_locations['EndNorthing'] = \
                    wgs84_to_osgb36(incident_locations.EndLongitude.values,
                                    incident_locations.EndLatitude.values)

                save_pickle(incident_locations, path_to_pickle, verbose=verbose)

            if start_and_end_elr is not None:
                if start_and_end_elr == 'same':
                    # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
                    incident_locations = incident_locations[
                        incident_locations.StartELR == incident_locations.EndELR]
                elif start_and_end_elr == 'diff':
                    # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
                    incident_locations = incident_locations[
                        incident_locations.StartELR != incident_locations.EndELR]

            return incident_locations

        except Exception as e:
            print("Failed to fetch \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    # (Unfinished)
    def view_schedule8_incident_location_tracks(self, shift_yards=220):
        incident_locations = self.view_metex_schedule8_incident_locations()

        track = self.get_track()

        # track_summary = self.get_track_summary()

        # Testing e.g.
        elr = incident_locations.StartELR.loc[24133]
        # start_location = shapely.geometry.Point(
        #     incident_locations[['StartEasting', 'StartNorthing']].iloc[0].values)
        # end_location = shapely.geometry.Point(
        #     incident_locations[['EndEasting', 'EndNorthing']].iloc[0].values)
        start_mileage_num = incident_locations.StartMileage_num.loc[24133]
        # start_yard = nr_mileage_to_yards(start_mileage_num)
        end_mileage_num = incident_locations.EndMileage_num.loc[24133]
        # end_yard = nr_mileage_to_yards(end_mileage_num)

        #
        track_elr_mileages = track[track.ELR == elr]

        # Call OSM railway data
        from pydriosm import GeofabrikReader

        geofabrik_reader = GeofabrikReader()

        # Download/Read GB OSM data
        geofabrik_reader.read_shp_zip('Great Britain', layer_names='railways', feature_names='rail',
                                      data_dir=cdd_network("OSM"), pickle_it=True, rm_extracts=True,
                                      rm_shp_zip=True)

        # Import it into PostgreSQL

        # Write a query to get track coordinates available in OSM data

        if start_mileage_num <= end_mileage_num:

            if start_mileage_num == end_mileage_num:
                start_mileage_num = shift_num_nr_mileage(start_mileage_num, -shift_yards)
                end_mileage_num = shift_num_nr_mileage(end_mileage_num, shift_yards)

            # Get adjusted mileages of start and end locations
            incident_track = track_elr_mileages[
                (start_mileage_num >= track_elr_mileages.StartMileage_num) &
                (end_mileage_num <= track_elr_mileages.EndMileage_num)]

        else:
            incident_track = track_elr_mileages[
                (start_mileage_num <= track_elr_mileages.EndMileage_num) &
                (end_mileage_num >= track_elr_mileages.StartMileage_num)]

        return incident_track

    def update_view_pickles(self, update=True, pickle_it=True, verbose=True):
        """
        Update the local pickle files for all essential views.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool, int

        **Test**::

            from mssqlserver import metex

            update = True
            pickle_it = True
            verbose = True

            metex.update_view_pickles(update, pickle_it, verbose)
        """

        if confirmed("To update the View pickles of the NR_METEX data?"):

            _ = self.view_schedule8_costs_by_location(None, None, update, pickle_it, verbose)
            _ = self.view_schedule8_costs_by_location('Anglia', None, update, pickle_it, verbose)
            _ = self.view_schedule8_costs_by_location('Anglia', 'Wind', update, pickle_it, verbose)
            _ = self.view_schedule8_costs_by_location('Anglia', 'Heat', update, pickle_it, verbose)

            _ = self.view_schedule8_costs_by_datetime_location(None, None, update, pickle_it, verbose)
            _ = self.view_schedule8_costs_by_datetime_location('Anglia', None, update, pickle_it,
                                                               verbose)
            _ = self.view_schedule8_costs_by_datetime_location('Anglia', 'Wind', update, pickle_it,
                                                               verbose)
            _ = self.view_schedule8_costs_by_datetime_location('Anglia', 'Heat', update, pickle_it,
                                                               verbose)

            _ = self.view_schedule8_costs_by_datetime_location_reason(None, None, update, pickle_it,
                                                                      verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason('Anglia', None, update, pickle_it,
                                                                      verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason('Anglia', 'Wind', update,
                                                                      pickle_it,
                                                                      verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason('Anglia', 'Heat', update,
                                                                      pickle_it,
                                                                      verbose)

            _ = self.view_metex_schedule8_incident_locations(None, None, start_and_end_elr=None,
                                                             update=update, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")


class Thresholds:
    """
    Weather-Thresholds_9306121.html

    Description:

      - "The following table defines Weather thresholds used to determine the classification
        of Weather as Normal, Alert, Adverse or Extreme. Note that the 'Alert' interval is
        inside the 'Normal' range."

      - "These are national thresholds. Route-specific thresholds may also be defined at
        some point."
    """

    def __init__(self):
        self.Name = 'Thresholds'

        self.SSDataDir = os.path.relpath(cdd_incidents("spreadsheets"))
        self.MTDataDir = os.path.relpath(cdd_metex("misc\\thresholds"))

        self.S8WeatherIncidentsFilename = "Schedule8WeatherIncidents-02062006-31032014"

    def get_schedule8_weather_thresholds(self, update=False, verbose=False):
        """
        Get threshold data available in ``workbook_filename``.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from weather import Thresholds

            >>> thr = Thresholds()

            >>> thresholds = thr.get_schedule8_weather_thresholds()
            >>> print(thresholds.head())
                     Description WeatherHazard  WeatherType     Period Condition  Threshold
            0   Wind Gust Normal        NORMAL    WIND_GUST  1 DAY MAX         <       60.0
            1    Wind Gust Alert         ALERT    WIND_GUST  1 DAY MAX        >=       50.0
            2  Wind Gust Adverse       ADVERSE    WIND_GUST  1 DAY MAX        >=       60.0
            3  Wind Gust Extreme       EXTREME    WIND_GUST  1 DAY MAX        >=       80.0
            4        Cold Normal        NORMAL  TEMPERATURE  1 DAY MIN         >       -4.0

            >>> thresholds = thr.get_schedule8_weather_thresholds(update=True, verbose=True)
            Updating "schedule8-weather-thresholds.pickle" at "\\data\\..." ... Done.
            >>> print(thresholds.head())
                     Description WeatherHazard  WeatherType     Period Condition  Threshold
            0   Wind Gust Normal        NORMAL    WIND_GUST  1 DAY MAX         <       60.0
            1    Wind Gust Alert         ALERT    WIND_GUST  1 DAY MAX        >=       50.0
            2  Wind Gust Adverse       ADVERSE    WIND_GUST  1 DAY MAX        >=       60.0
            3  Wind Gust Extreme       EXTREME    WIND_GUST  1 DAY MAX        >=       80.0
            4        Cold Normal        NORMAL  TEMPERATURE  1 DAY MIN         >       -4.0
        """

        pickle_filename = "schedule8-weather-thresholds.pickle"

        path_to_pickle = cd(self.SSDataDir, pickle_filename)
        if os.path.isfile(path_to_pickle) and not update:
            schedule8_weather_thresholds = load_pickle(path_to_pickle)

        else:
            path_to_spreadsheet = cd(self.SSDataDir, self.S8WeatherIncidentsFilename + ".xlsm")

            try:
                schedule8_weather_thresholds = pd.read_excel(
                    path_to_spreadsheet, sheet_name="Thresholds", usecols="A:F")
                schedule8_weather_thresholds.dropna(inplace=True)
                schedule8_weather_thresholds.columns = [
                    col.replace(' ', '') for col in schedule8_weather_thresholds.columns]
                schedule8_weather_thresholds.WeatherHazard = \
                    schedule8_weather_thresholds.WeatherHazard.str.strip().str.upper()
                schedule8_weather_thresholds.index = range(len(schedule8_weather_thresholds))
                save_pickle(schedule8_weather_thresholds, path_to_pickle, verbose=verbose)
            except Exception as e:
                print("Failed to get \"weather thresholds\" from the .xlsm file. {}.".format(e))
                schedule8_weather_thresholds = None

        return schedule8_weather_thresholds

    def get_metex_weather_thresholds(self, update=False, verbose=False):
        """
        Get threshold data available in ``html_filename``.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from weather import Thresholds

            >>> thr = Thresholds()

            >>> thresholds = thr.get_metex_weather_thresholds()
            >>> print(thresholds.head())
                                         Classification  ... ExtremeThreshold
            Temperature                            Cold  ...               -7
            Temperature                            Heat  ...               30
            Snow             3-Hourly (Wet or Dry Snow)  ...               15
            Rain         Hourly (normal and wet ground)  ...               40
            Rain                               3-Hourly  ...               60
            [5 rows x 13 columns]

            >>> thresholds = thr.get_metex_weather_thresholds(update=True, verbose=True)
            Updating "metex-weather-thresholds.pickle" at "\\data\\..." ... Done.
            >>> print(thresholds.head())
                                         Classification  ... ExtremeThreshold
            Temperature                            Cold  ...               -7
            Temperature                            Heat  ...               30
            Snow             3-Hourly (Wet or Dry Snow)  ...               15
            Rain         Hourly (normal and wet ground)  ...               40
            Rain                               3-Hourly  ...               60
            [5 rows x 13 columns]
        """

        path_to_pickle = cdd_metex(self.MTDataDir, "metex-weather-thresholds.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            metex_weather_thresholds = load_pickle(path_to_pickle)

        else:
            path_to_html = cdd_metex(self.MTDataDir, "Weather-Thresholds_9306121.html")

            try:
                metex_weather_thresholds = pd.read_html(path_to_html)[0]

                # Specify column names
                metex_weather_thresholds.columns = metex_weather_thresholds.loc[0].tolist()
                # Drop the first row, which has been used as the column names
                metex_weather_thresholds.drop(0, axis=0, inplace=True)

                # cls: classification
                cls = metex_weather_thresholds.Classification[
                    metex_weather_thresholds.eq(
                        metex_weather_thresholds.iloc[:, 0], axis=0).all(1)].tolist()

                cls_idx = []
                for i in range(len(cls)):
                    x = metex_weather_thresholds.index[
                        metex_weather_thresholds.Classification == cls[i]][0]
                    metex_weather_thresholds.drop(x, inplace=True)
                    if i + 1 < len(cls):
                        y = metex_weather_thresholds.index[
                            metex_weather_thresholds.Classification == cls[i + 1]][0]
                        to_rpt = y - x - 1
                    else:
                        to_rpt = metex_weather_thresholds.index[-1] - x
                    cls_idx += [cls[i]] * to_rpt
                metex_weather_thresholds.index = cls_idx

                # Add 'VariableName' and 'Unit'
                variables = ['T', 'x', 'r', 'w']
                units = ['degrees Celsius', 'cm', 'mm', 'mph']
                var_list, units_list = [], []
                for i in range(len(cls)):
                    var_temp = [variables[i]] * list(metex_weather_thresholds.index).count(cls[i])
                    units_temp = [units[i]] * list(metex_weather_thresholds.index).count(cls[i])
                    var_list += var_temp
                    units_list += units_temp
                metex_weather_thresholds.insert(1, 'VariableName', var_list)
                metex_weather_thresholds.insert(2, 'Unit', units_list)

                # Retain main description
                metex_weather_thresholds.Classification = \
                    metex_weather_thresholds.Classification.str.replace(
                        r'( \( oC \))|(,[(\xa0) ][xrw] \(((cm)|(mm)|(mph))\))', '')
                metex_weather_thresholds.Classification = \
                    metex_weather_thresholds.Classification.str.replace(
                        ' (mph)', ' ', regex=False)

                # Upper and lower boundaries
                def boundary(df, col, sep1=None, sep2=None):
                    if sep1:
                        lst_lb = [metex_weather_thresholds[col].iloc[0].split(sep1)[0]]
                        lst_lb += [v.split(sep2)[0] for v in metex_weather_thresholds[col].iloc[1:]]
                        df.insert(df.columns.get_loc(col) + 1, col + 'LowerBound', lst_lb)
                    if sep2:
                        lst_ub = [metex_weather_thresholds[col].iloc[0].split(sep2)[1]]
                        lst_ub += [v.split(sep1)[-1] for v in metex_weather_thresholds[col].iloc[1:]]
                        if sep1:
                            df.insert(df.columns.get_loc(col) + 2, col + 'UpperBound', lst_ub)
                        else:
                            df.insert(df.columns.get_loc(col) + 1, col + 'Threshold', lst_ub)

                # Normal
                boundary(metex_weather_thresholds, 'Normal', sep1=None, sep2='up to ')
                # Alert
                boundary(metex_weather_thresholds, 'Alert', sep1=' \u003C ', sep2=' \u2264 ')
                # Adverse
                boundary(metex_weather_thresholds, 'Adverse', sep1=' \u003C ', sep2=' \u2264 ')
                # Extreme
                extreme = [metex_weather_thresholds['Extreme'].iloc[0].split(' \u2264 ')[1]]
                extreme += [v.split(' \u2265 ')[1]
                            for v in metex_weather_thresholds['Extreme'].iloc[1:]]
                metex_weather_thresholds['ExtremeThreshold'] = extreme

                save_pickle(metex_weather_thresholds, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"weather thresholds\" from the HTML file. {}.".format(e))
                metex_weather_thresholds = None

        return metex_weather_thresholds

    def get_weather_thresholds(self, update=False, verbose=False):
        """
        Get data of weather thresholds.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from weather import Thresholds

            >>> thr = Thresholds()

            >>> thresholds = thr.get_weather_thresholds()
            >>> type(thresholds)
            dict

            >>> print(list(thresholds.keys()))
            ['METEX', 'Schedule8WeatherIncidents']
        """

        pickle_filename = "weather-thresholds.pickle"
        path_to_pickle = cdd_metex("weather\\thresholds", pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            weather_thresholds = load_pickle(path_to_pickle)
        else:

            try:
                metex_weather_thresholds = self.get_metex_weather_thresholds()
                schedule8_weather_thresholds = self.get_schedule8_weather_thresholds(update)
                weather_thresholds = {'METEX': metex_weather_thresholds,
                                      'Schedule8WeatherIncidents': schedule8_weather_thresholds}
                save_pickle(weather_thresholds, path_to_pickle, verbose=verbose)
            except Exception as e:
                print("Failed to get \"weather thresholds\". {}.".format(e))
                weather_thresholds = None

        return weather_thresholds


class MIDAS:

    def __init__(self):
        self.Name = 'Met Office RADTOB (Radiation values currently being reported).'
        self.Acronym = 'MIDAS'

        self.DataDir = cdd_weather(self.Acronym.lower())

        self.RadStnInfoFilename = "radiation-stations-information"
        self.RadtobFilename = "midas-radtob-2006-2019"
        self.HeadersFilename = "radiation-observation-data-headers"

        # Create an engine to the MSSQL server
        self.DatabaseEngine = create_mssql_connectable_engine(database_name='Weather')

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
            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> print(os.path.relpath(midas.cdd()))
            data\\weather\\midas
        """

        path = cd(self.DataDir, *sub_dir, mkdir=mkdir)

        return path

    def get_radiation_stations(self, update=False, verbose=False):
        """
        Get locations and relevant information of meteorological stations.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of meteorological stations
        :rtype: pandas.DataFrame

        **Test**::

            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> # dat = midas.get_radiation_stations(update=True, verbose=True)
            >>> dat = midas.get_radiation_stations()

            >>> print(dat)
                                    StationName  ...                                      EN_GEOM
            SRC_ID                               ...
            3                         FAIR ISLE  ...  POINT (421158.6189113986 1071232.475631491)
            9                           LERWICK  ...   POINT (445479.737983736 1139722.707449886)
            12                  BALTASOUND NO 2  ...  POINT (462590.2813043125 1207861.554269238)
            23                         KIRKWALL  ...  POINT (348329.9204614616 1007758.524571238)
            32                     WICK AIRPORT  ...  POINT (336581.3824620685 952272.3257549705)
                                         ...  ...                                          ...
            61986             TIBENHAM AIRFIELD  ...  POINT (615070.0334468307 288957.3690573045)
            62034   ROUDSEA WOOD AND MOSSES NNR  ...  POINT (333410.2674705166 482772.0753328541)
            62041           EXETER AIRPORT NO 2  ...  POINT (300979.5951061826 93939.67916976719)
            62122                   ALMONDSBURY  ...  POINT (361406.5017112559 183704.4311121872)
            62139                   WISLEY NO 2  ...   POINT (506466.6675712555 157848.166606131)
            [240 rows x 11 columns]
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
                    x.title().replace(' ', '') if x != 'src_id' else x.upper()
                    for x in rad_stn.columns
                ]
                rad_stn.StationName = rad_stn.StationName.str.replace(
                    r'(\xa0)+Locate', '', regex=True)

                rad_stn['Easting'], rad_stn['Northing'] = wgs84_to_osgb36(
                    rad_stn.Longitude.values, rad_stn.Latitude.values)
                rad_stn['EN_GEOM'] = [
                    shapely.geometry.Point(xy) for xy in zip(rad_stn.Easting, rad_stn.Northing)
                ]

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
        :type file: str
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
            >>> from weather import MIDAS

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
            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> pf = midas.make_radtob_pickle_path("midas-radtob-20060101-20141231", False, False)

            >>> print(os.path.relpath(pf))
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

            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> col_names = midas.get_radtob_headers()

            >>> print(col_names[:5])
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

            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> dat = midas.get_radtob(verbose=True)

            >>> print(dat)
                                       OB_END_DATE  ...  GLBL_IRAD_AMT
            SRC_ID OB_END_DATE_TIME                 ...
            9      2006-01-01 00:00:00  2006-01-01  ...            0.0
                   2006-01-01 01:00:00  2006-01-01  ...            0.0
                   2006-01-01 02:00:00  2006-01-01  ...            0.0
                   2006-01-01 03:00:00  2006-01-01  ...            0.0
                   2006-01-01 04:00:00  2006-01-01  ...            0.0
                                            ...  ...            ...
            62139  2019-06-30 20:00:00  2019-06-30  ...          379.0
                   2019-06-30 21:00:00  2019-06-30  ...           26.0
                   2019-06-30 22:00:00  2019-06-30  ...            0.0
                   2019-06-30 23:00:00  2019-06-30  ...            0.0
                   2019-06-30 23:59:00  2019-06-30  ...        20900.0
            [10638864 rows x 4 columns]
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

    def import_radtob(self, if_exists='append', chunk_size=100000, update=False, verbose=False):
        """
        See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

        :param if_exists:
        :param chunk_size:
        :param update:
        :param verbose:

        **Test**::

            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> midas.import_radtob(if_exists='append', chunk_size=100000, verbose=True)
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
            dtype_ = {
                'OB_END_DATE_TIME': sqlalchemy.types.DATETIME,
                'OB_END_DATE': sqlalchemy.types.DATE
            }

            chunk.to_sql(con=self.DatabaseEngine, schema='dbo', name=self.RadtobTblName,
                         if_exists=if_exists, index=False, dtype=dtype_, method='multi')

            del chunk
            gc.collect()

        temp_file.close()

        os.remove(csv_filename)

        print("Done. ")

    def process_suppl_dat(self, update=False, verbose=False):
        """

        :param update:
        :type update:
        :param verbose:
        :type verbose:
        :return:
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from weather import MIDAS

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

    def import_suppl_dat(self, if_exists='replace', update=False, verbose=False):
        """

        :param if_exists:
        :type if_exists:
        :param update:
        :type update:
        :param verbose:
        :type verbose:

        **Test**::

            >>> from weather import MIDAS

            >>> midas = MIDAS()

            >>> midas.import_suppl_dat(verbose=True)
        """

        supplement_data = self.process_suppl_dat(update=update, verbose=verbose)

        if supplement_data is not None and not supplement_data.empty:
            supplement_data.to_sql(con=self.DatabaseEngine,
                                   name=self.RadtobSupplTblName, schema='dbo',
                                   if_exists=if_exists,
                                   index=False,
                                   dtype={'OB_END_DATE': sqlalchemy.types.DATE})

    def query_radtob_by_grid_datetime(self, met_stn_id, period, route_name, use_suppl_dat=False,
                                      update=False, dat_dir=None, pickle_it=False, verbose=False):
        """
        Get MIDAS RADTOB (Radiation data) by met station ID (Query from the database)
        for the given ``period``.

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
        :type verbose: bool, int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from weather import MIDAS

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

        p_start, p_end = period.left.min().strftime('%Y%m%d%H'), period.right.max().strftime('%Y%m%d%H')

        # Make a pickle file
        pickle_filename = "{}-{}.pickle".format(str(met_stn_id[0]), "-".join([p_start, p_end]))

        # Specify a directory/path to store the pickle file (if appropriate)
        if os.path.isabs(dat_dir):
            dat_dir_ = dat_dir
        else:
            dat_dir_ = self.cdd("dat")
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

            midas_radtob = pd.read_sql(sql=sql_query, con=self.DatabaseEngine)  # Query the weather data

            if midas_radtob.empty and use_suppl_dat:
                dates = tuple(
                    x.strftime('%Y-%m-%d') for x in [period.left.min(), period.right.max()])

                sql_query = \
                    f"SELECT * FROM dbo.[MIDAS_RADTOB_suppl] " \
                    f"WHERE [Route] = '{route_name}' " \
                    f"AND [OB_END_DATE] BETWEEN '{dates[0]}' AND '{dates[1]}';"

                midas_radtob = pd.read_sql(sql=sql_query, con=self.DatabaseEngine)

            if pickle_it:
                save_pickle(midas_radtob, path_to_pickle, verbose=verbose)

        return midas_radtob


class UKCP09:
    """

    :param start_date: start date on which the observation data was collected, formatted as 'yyyy-mm-dd'
    :type start_date: str

    **Test**::

        >>> from weather import UKCP09

        >>> ukcp = UKCP09()

        >>> print(ukcp.Name)
        UK Climate Projections
    """

    def __init__(self, start_date='2006-01-01'):
        self.Name = 'UK Climate Projections'
        self.Acronym = 'UKCP09'
        self.Description = 'UKCP09 gridded weather observations: ' \
                           'maximum temperature, minimum temperature and precipitation.'

        self.DataDir = cdd_weather(self.Acronym.lower())
        self.StartDate = start_date

        # Create an engine to the MSSQL server
        self.DatabaseEngine = create_mssql_connectable_engine(database_name='Weather')

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\weather\\ukcp" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a file)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to ``"data\\weather\\ukcp"`` and sub-directories / a file
        :rtype: str
        """

        path = cd(self.DataDir, *sub_dir, mkdir=mkdir)

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
        :type filename: str
        :return: parsed data of the observation grids
        :rtype: pandas.DataFrame

        **Example**::

            file = "ukcp09_gridded-land-obs-daily_timeseries_maximum-" \
                       "temperature_000000E_450000N_19600101-20161231.csv"

            obs_grids = parse_observation_grids(file)
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
        :rtype: pandas.DataFrame

        **Example**::

            from weather.ukcp import get_observation_grids

            update = True
            verbose = True

            zip_filename = "daily-precipitation.zip"
            observation_grids = get_observation_grids(zip_filename, update, verbose)
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
                observation_grids.index = pd.Index(
                    range(len(observation_grids)), name='Pseudo_Grid_ID')

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
        :type filename: str
        :param var_name: variable name,
            e.g. 'Maximum_Temperature', 'Minimum_Temperature', 'Precipitation'
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

        filename_suffix = "" if self.StartDate is None \
            else "-{}".format(self.StartDate.replace("-", ""))
        pickle_filename = filename + filename_suffix + ".pickle"
        path_to_pickle = cdd_weather("ukcp", pickle_filename)
        return path_to_pickle

    def get_var_obs(self, zip_filename, var_name, use_pseudo_grid_id=False, update=False, verbose=False):
        """
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
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of daily gridded weather observations
        :rtype: pandas.DataFrame

        **Example**::

            from weather.ukcp import get_var_obs

            zip_filename = "daily-maximum-temperature"
            var_name = 'Maximum_Temperature'
            start_date = '2006-01-01'
            use_pseudo_grid_id = False
            update = False
            verbose = True

            gridded_obs = get_var_obs(zip_filename, var_name, start_date, use_pseudo_grid_id,
                                             update, verbose)
        """

        assert isinstance(pd.to_datetime(self.StartDate), pd.Timestamp) or self.StartDate is None

        filename = os.path.splitext(zip_filename)[0]
        path_to_pickle = self.make_pickle_path(filename)

        if os.path.isfile(path_to_pickle) and not update:
            gridded_obs = load_pickle(path_to_pickle)

        else:
            try:
                path_to_zip = cdd_weather("ukcp", zip_filename + ".zip")

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

                save_pickle(gridded_obs, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(filename.replace("-", " "), e))
                gridded_obs = None

        return gridded_obs

    def get_data(self, use_pseudo_grid_id=True, update=False, verbose=False):
        """
        Fetch integrated weather observations of different variables from local pickle.

        :param use_pseudo_grid_id: defaults to ``False``
        :type use_pseudo_grid_id: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool, int
        :return: data of integrated daily gridded weather observations
        :rtype: pandas.DataFrame
        :return: data of integrated daily gridded weather observations
        :rtype: pandas.DataFrame

        **Example**::

            from weather.ukcp import get_data

            start_date = '2006-01-01'
            use_pseudo_grid_id = False
            update = False
            verbose = True

            ukcp09_data = get_data(start_date, use_pseudo_grid_id, update, verbose)
        """

        filename = "ukcp-daily-gridded-weather"
        path_to_pickle = self.make_pickle_path(filename)

        if os.path.isfile(path_to_pickle) and not update:
            ukcp09_data = load_pickle(path_to_pickle)

        else:
            try:
                d_max_temp = self.get_var_obs(
                    zip_filename="daily-maximum-temperature", var_name='Maximum_Temperature',
                    use_pseudo_grid_id=False, update=update, verbose=verbose)
                d_min_temp = self.get_var_obs(
                    zip_filename="daily-minimum-temperature", var_name='Minimum_Temperature',
                    use_pseudo_grid_id=False, update=update, verbose=verbose)
                d_precipitation = self.get_var_obs(
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

                path_to_pickle = self.make_pickle_path(filename)
                save_pickle(ukcp09_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to integrate the UKCP09 gridded weather observations. {}".format(e))
                ukcp09_data = None

        return ukcp09_data

    def import_data(self, table_name='UKCP091', if_exists='append', chunk_size=100000,
                    update=False, verbose=False):
        """
        See also [`DUDTM <https://stackoverflow.com/questions/50689082>`_].

        :param table_name:
        :param if_exists:
        :param chunk_size:
        :param update:
        :param verbose:
        """

        ukcp09_data = self.get_data(update=update, verbose=verbose)
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
                chunk.to_sql(table_name, ukcp09_engine, schema='dbo', if_exists=if_exists,
                             index=False, dtype={'Date': sqlalchemy.types.DATE}, method='multi')

                del chunk
                gc.collect()

            temp_file.close()

        os.remove(temp_file.name)

        print("Done. ")

    def query_by_grid_datetime(self, grids, period,
                               update=False, dat_dir=None, pickle_it=False, verbose=False):
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
        :type verbose: bool, int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Examples**::

            from weather.ukcp import query_by_grid_datetime

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
            ukcp09_dat = pd.read_sql(sql=sql_query, con=self.DatabaseEngine)

            if pickle_it:
                save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

        return ukcp09_dat

    def query_by_grid_datetime_(self, grids, period,
                                update=False, dat_dir=None, pickle_it=False, verbose=False):
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
        :type verbose: bool, int
        :return: UKCP09 data by ``grids`` and ``period``
        :rtype: pandas.DataFrame

        **Examples**::

            from weather.ukcp import query_by_grid_datetime

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
            ukcp09_dat = pd.read_sql(sql=sql_query, con=self.DatabaseEngine)

            if pickle_it:
                save_pickle(ukcp09_dat, path_to_pickle, verbose=verbose)

        return ukcp09_dat
