""" Read and cleanse data of the database 'NR_METEX'

    Schedule 4 compensates train operators for the impact of planned service disruption, and
    Schedule 8 compensates train operators for the impact of unplanned service disruption.

"""

import copy
import gc
import os
import string
import zipfile

import fuzzywuzzy.fuzz
import fuzzywuzzy.process
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import numpy as np
import pandas as pd
import shapely.geometry
from pyhelpers.dir import cd
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.misc import confirmed
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_json, load_pickle, save, save_fig, save_pickle
from pyhelpers.text import find_similar_str
from pyrcs.line_data import LineData
from pyrcs.other_assets import OtherAssets
from pyrcs.utils import fetch_location_names_repl_dict
from pyrcs.utils import nr_mileage_num_to_str, str_to_num_mileage, yards_to_nr_mileage

from misc.delay_attribution_glossary import get_incident_reason_metadata, get_performance_event_code
from mssqlserver.tools import establish_mssql_connection, get_table_primary_keys, read_table_by_query
from utils import cdd_metex, cdd_network, update_nr_route_names

pd_preferences()

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\METEX\\Database"
def cdd_metex_db(*sub_dir):
    """
    Testing e.g.
        cdd_metex_db()
        cdd_metex_db("test")
    """
    path = cdd_metex("Database")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\METEX\\Database\\Tables"
def cdd_metex_db_tables(*sub_dir):
    """
    Testing e.g.
        cdd_metex_db_tables()
        cdd_metex_db_tables("test")
    """
    path = cdd_metex_db("Tables")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\METEX\\Database\\Views"
def cdd_metex_db_views(*sub_dir):
    """
    Testing e.g.
        cdd_metex_db_views()
        cdd_metex_db_views("test")
    """
    path = cdd_metex_db("Views")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "METEX\\Figures"
def cdd_metex_db_fig(*sub_dir):
    """
    Testing e.g.
        cdd_metex_db_fig()
        cdd_metex_db_fig("test")
    """
    path = cdd_metex_db("Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Read table data from the database """


# Read tables available in Database
def read_metex_table(table_name, index_col=None, route_name=None, weather_category=None, coerce_float=True,
                     parse_dates=None, chunk_size=None, params=None, schema_name='dbo', save_as=None, update=False):
    """
    :param table_name: [str] name of a queried table from the database
    :param index_col: [str; None (default)] name of a column that is set to be the index
    :param route_name: [str; None (default)] name of the specific route
    :param weather_category: [str; None (default)] name of the specific weather category
    :param coerce_float: [bool] (default: True)
    :param parse_dates: [list; None (default)]
    :param chunk_size: [str; None (default)]
    :param params: [list or tuple; dict; None (default)]
    :param schema_name: [str] (default: 'dbo')
    :param save_as: [str; None (default)]
    :param update: [bool] (default: False)
    :return: [pd.DataFrame] the queried data as a pd.DataFrame

    Testing e.g.
        table_name = 'IMDM'
        index_col = None
        route_name = None
        weather_category = None
        coerce_float = True
        parse_dates = None
        chunk_size = None
        params = None
        schema_name = 'dbo'
        save_as = None
        update = True
    """
    table = '{}."{}"'.format(schema_name, table_name)
    # Connect to the queried database
    conn_metex = establish_mssql_connection('NR_METEX_20190203')
    # Specify possible scenarios:
    if not route_name and not weather_category:
        sql_query = "SELECT * FROM {}".format(table)  # Get all data of a given table
    elif route_name and not weather_category:
        sql_query = "SELECT * FROM {} WHERE Route = '{}'".format(table, route_name)  # given Route
    elif route_name is None and weather_category is not None:
        sql_query = "SELECT * FROM {} WHERE WeatherCategory = '{}'".format(table, weather_category)  # given Weather
    else:
        # Get all data of a table, given Route and Weather category e.g. data about wind-related events on Anglia Route
        sql_query = "SELECT * FROM {} WHERE Route = '{}' AND WeatherCategory = '{}'".format(
            table, route_name, weather_category)
    # Create a pd.DataFrame of the queried table
    table_data = pd.read_sql(sql_query, conn_metex, index_col=index_col, coerce_float=coerce_float,
                             parse_dates=parse_dates, chunksize=chunk_size, params=params)
    # Disconnect the database
    conn_metex.close()
    if save_as:
        path_to_file = cdd_metex_db_tables(table_name + save_as)
        if not os.path.isfile(path_to_file) or update:
            save(table_data, path_to_file, index=True if index_col else False)
    return table_data


# Get primary keys of a table in database
def get_metex_table_pk(table_name):
    """
    Testing e.g.
        table_name = 'IMDM'
    """
    pri_key = get_table_primary_keys('NR_METEX_20190203', table_name=table_name)
    return pri_key


# ====================================================================================================================
""" Get table data """


# Get IMDM
def get_imdm(as_dict=False, update=False, save_original_as=None, verbose=False):
    """
    :param as_dict: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)] e.g. ".csv"
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        as_dict = False
        update = True
        save_original_as = None
        verbose = True

        get_imdm(as_dict, update, save_original_as, verbose)
    """
    table_name = 'IMDM'
    path_to_file = cdd_metex_db_tables("".join([table_name, ".json" if as_dict else ".pickle"]))

    if os.path.isfile(path_to_file) and not update:
        imdm = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

    else:
        try:
            imdm = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                    save_as=save_original_as, update=update)
            imdm.index.rename(name='IMDM', inplace=True)  # Rename index

            # Update route names
            update_nr_route_names(imdm)

            # Add regions
            regions_and_routes = load_json(cdd_network("Regions", "routes.json"))
            regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
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
            print("Failed to get \"{}\"{}. {}.".format(table_name, " as a dictionary" if as_dict else "", e))
            imdm = None

    return imdm


# Get ImdmAlias
def get_imdm_alias(as_dict=False, update=False, save_original_as=None, verbose=False):
    """
    :param as_dict: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)] e.g. ".csv"
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        as_dict = False
        update = True
        save_original_as = None
        verbose = True

        get_imdm_alias(as_dict, update, save_original_as, verbose)
    """
    table_name = 'ImdmAlias'
    path_to_file = cdd_metex_db_tables(table_name + (".json" if as_dict else ".pickle"))
    if os.path.isfile(path_to_file) and not update:
        imdm_alias = load_json(path_to_file) if as_dict else load_pickle(path_to_file)
    else:
        try:
            imdm_alias = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                          save_as=save_original_as, update=update)
            imdm_alias.index.rename(name='ImdmAlias', inplace=True)  # Rename index
            imdm_alias.rename(columns={'Imdm': 'IMDM'}, inplace=True)  # Rename a column
            if as_dict:
                imdm_alias = imdm_alias.to_dict()
                # imdm_alias = imdm_alias['IMDM']
            save(imdm_alias, path_to_file, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " as a dictionary" if as_dict else "", e))
            imdm_alias = None
    return imdm_alias


# Get IMDMWeatherCellMap
def get_imdm_weather_cell_map(route_info=True, grouped=False, update=False, save_original_as=None, verbose=False):
    """
    :param route_info: [bool] (default: True)
    :param grouped: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        route_info = True
        grouped = False
        update = True
        save_original_as = None
        verbose = True

        get_imdm_weather_cell_map(route_info, grouped, update, save_original_as, verbose)
    """
    table_name = 'IMDMWeatherCellMap_pc' if route_info else 'IMDMWeatherCellMap'
    path_to_pickle = cdd_metex_db_tables(table_name + ("-grouped.pickle" if grouped else ".pickle"))

    if os.path.isfile(path_to_pickle) and not update:
        weather_cell_map = load_pickle(path_to_pickle)

    else:
        try:
            # Read IMDMWeatherCellMap table
            weather_cell_map = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                                coerce_float=False, save_as=save_original_as, update=update)

            if route_info:
                update_nr_route_names(weather_cell_map)
                weather_cell_map[['Id', 'WeatherCell']] = weather_cell_map[['Id', 'WeatherCell']].applymap(int)
                weather_cell_map.set_index('Id', inplace=True)

            weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index
            weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'}, inplace=True)  # Rename a column

            if grouped:  # To find out how many IMDMs each 'WeatherCellId' is associated with
                weather_cell_map = weather_cell_map.groupby('Route' if route_info else 'WeatherCellId').aggregate(
                    lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))

            save_pickle(weather_cell_map, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " (being grouped)" if grouped else "", e))
            weather_cell_map = None

    return weather_cell_map


# Get IncidentReasonInfo
def get_incident_reason_info(plus=True, update=False, save_original_as=None, verbose=False):
    """
    :param plus: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        plus = True
        update = True
        save_original_as = None
        verbose = True

        get_incident_reason_info(plus, update, save_original_as, verbose)
    """
    table_name = 'IncidentReasonInfo'
    path_to_pickle = cdd_metex_db_tables(table_name + ("-plus.pickle" if plus else ".pickle"))
    if os.path.isfile(path_to_pickle) and not update:
        incident_reason_info = load_pickle(path_to_pickle)
    else:
        try:
            # Get data from the database
            incident_reason_info = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                                    save_as=save_original_as, update=update)
            incident_reason_info.index.rename('IncidentReasonCode', inplace=True)  # Rename index label
            incident_reason_info.rename(columns={'Description': 'IncidentReasonDescription',
                                                 'Category': 'IncidentCategory',
                                                 'CategoryDescription': 'IncidentCategoryDescription'},
                                        inplace=True)
            if plus:  # To include data of more detailed description about incident reasons
                incident_reason_metadata = get_incident_reason_metadata()
                incident_reason_metadata.index.name = 'IncidentReasonCode'
                incident_reason_metadata.columns = [x.replace('_', '') for x in incident_reason_metadata.columns]
                incident_reason_info = incident_reason_info.join(incident_reason_metadata, rsuffix='_plus')
                # incident_reason_info.dropna(axis=1, inplace=True)
            save_pickle(incident_reason_info, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " with extra information" if plus else "", e))
            incident_reason_info = None
    return incident_reason_info


# Get WeatherCategoryLookup
def get_weather_codes(as_dict=False, update=False, save_original_as=None, verbose=False):
    """
    :param as_dict: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        as_dict = False
        update = True
        save_original_as = None
        verbose = True

        get_weather_codes(as_dict, update, save_original_as, verbose)
    """
    table_name = 'WeatherCodes'  # WeatherCodes
    path_to_file = cdd_metex_db_tables(table_name + (".json" if as_dict else ".pickle"))
    if os.path.isfile(path_to_file) and not update:
        weather_codes = load_json(path_to_file) if as_dict else load_pickle(path_to_file)
    else:
        try:
            weather_codes = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                             save_as=save_original_as, update=update)
            weather_codes.rename(columns={'Code': 'WeatherCategoryCode',
                                          'Weather Category': 'WeatherCategory'}, inplace=True)
            if as_dict:
                weather_codes.set_index('WeatherCategoryCode', inplace=True)
                weather_codes = weather_codes.to_dict()
            save(weather_codes, path_to_file, verbose=verbose)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_codes = None
    return weather_codes


# Get IncidentRecord and fill 'None' value with NaN
def get_incident_record(update=False, save_original_as=None, use_corrected_csv=True, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param use_corrected_csv: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        use_corrected_csv = True
        verbose = True

        get_incident_record(update, save_original_as, use_corrected_csv, verbose)
        get_incident_record(True, save_original_as, use_corrected_csv, verbose)
    """
    table_name = 'IncidentRecord'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        incident_record = load_pickle(path_to_pickle)

    else:
        try:
            incident_record = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                               save_as=save_original_as, update=update)

            if use_corrected_csv:
                corrected_csv = pd.read_csv(cdd_metex_db("Updates", table_name + ".zip"), index_col='Id',
                                            parse_dates=['CreateDate'], infer_datetime_format=True, dayfirst=True)
                corrected_csv.columns = incident_record.columns
                corrected_csv.loc[corrected_csv[corrected_csv.WeatherCategory.isna()].index, 'WeatherCategory'] = None
                incident_record.drop(incident_record[incident_record.CreateDate >= pd.to_datetime('2018-01-01')].index,
                                     inplace=True)
                incident_record = incident_record.append(corrected_csv)

            incident_record.index.rename(table_name + 'Id', inplace=True)  # Rename index name
            incident_record.rename(columns={'CreateDate': table_name + 'CreateDate', 'Reason': 'IncidentReasonCode'},
                                   inplace=True)  # Rename column names

            weather_codes = get_weather_codes(as_dict=True)  # Get a weather category lookup dictionary
            # Replace each weather category code with its full name
            incident_record.replace(weather_codes, inplace=True)
            incident_record.WeatherCategory.fillna(value='', inplace=True)

            save_pickle(incident_record, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            incident_record = None

    return incident_record


# Get Location
def get_location(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_location(update, save_original_as, verbose)
    """
    table_name = 'Location'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        location = load_pickle(path_to_pickle)

    else:
        try:
            location = read_metex_table(table_name, index_col=get_metex_table_pk(table_name), coerce_float=False,
                                        save_as=save_original_as, update=update)
            location.index.rename('LocationId', inplace=True)
            location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
            location[['WeatherCell', 'SMDCell']] = location[['WeatherCell', 'SMDCell']].applymap(
                lambda x: 0 if pd.np.isnan(x) else int(x))
            # location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]

            save_pickle(location, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            location = None

    return location


# Get PfPI (Process for Performance Improvement)
def get_pfpi(plus=True, update=False, save_original_as=None, use_corrected_csv=True, verbose=False):
    """
    :param plus: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param use_corrected_csv: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        plus = True
        update = True
        save_original_as = None
        use_corrected_csv = True
        verbose = True

        get_pfpi(plus, update, save_original_as, use_corrected_csv, verbose)
        get_pfpi(False, True, save_original_as, use_corrected_csv, verbose)
        get_pfpi(plus, True, save_original_as, use_corrected_csv, verbose)
    """
    table_name = 'PfPI'
    path_to_pickle = cdd_metex_db_tables(table_name + ("-plus.pickle" if plus else ".pickle"))

    if os.path.isfile(path_to_pickle) and not update:
        pfpi = load_pickle(path_to_pickle)

    else:
        try:
            pfpi = read_metex_table(table_name, index_col=get_metex_table_pk(table_name), save_as=save_original_as,
                                    update=update)

            if use_corrected_csv:
                incident_record = read_metex_table('IncidentRecord', index_col=get_metex_table_pk('IncidentRecord'))
                min_id = incident_record[incident_record.CreateDate >= pd.to_datetime('2018-01-01')].index.min()
                pfpi.drop(pfpi[pfpi.IncidentRecordId >= min_id].index, inplace=True)
                pfpi = pfpi.append(pd.read_csv(cdd_metex_db("Updates", table_name + ".zip", ), index_col='Id'))

            pfpi.index.rename(table_name + pfpi.index.name, inplace=True)

            if plus:  # To include more information for 'PerformanceEventCode'
                performance_event_code = get_performance_event_code()
                performance_event_code.index.rename('PerformanceEventCode', inplace=True)
                performance_event_code.columns = [x.replace('_', '') for x in performance_event_code.columns]
                # Merge pfpi and pe_code
                pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')

            save_pickle(pfpi, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " with performance event name" if plus else "", e))
            pfpi = None

    return pfpi


# Get Route (Note that there is only one column in the original table)
def get_route(as_dict=False, update=False, save_original_as=None, verbose=False):
    """
    :param as_dict: [bool] (default: False)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        as_dict = False
        update = True
        save_original_as = None
        verbose = True

        get_route(as_dict, update, save_original_as, verbose)
    """
    table_name = "Route"
    path_to_pickle = cdd_metex_db_tables(table_name + (".json" if as_dict else ".pickle"))

    if os.path.isfile(path_to_pickle) and not update:
        route = load_pickle(path_to_pickle)

    else:
        try:
            route = read_metex_table(table_name, save_as=save_original_as, update=update)
            route.rename(columns={'Name': 'Route'}, inplace=True)
            update_nr_route_names(route)

            # Add regions
            regions_and_routes = load_json(cdd_network("Regions", "routes.json"))
            regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
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


# Get StanoxLocation
def get_stanox_location(use_nr_mileage_format=True, update=False, save_original_as=None, verbose=False):
    """
    :param use_nr_mileage_format: [bool] (default: True)
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        use_nr_mileage_format = True
        update = True
        save_original_as = None
        verbose = True

        get_stanox_location(use_nr_mileage_format, update, save_original_as, verbose)
        get_stanox_location(False, update, save_original_as, verbose)
    """
    table_name = 'StanoxLocation'
    path_to_pickle = cdd_metex_db_tables(table_name + ("-mileage.pickle" if use_nr_mileage_format else ".pickle"))

    if os.path.isfile(path_to_pickle) and not update:
        stanox_location = load_pickle(path_to_pickle)

    else:
        try:
            # Read StanoxLocation table from the database
            stanox_location = read_metex_table(table_name, index_col=None, save_as=save_original_as, update=update)

            # Likely errors
            stanox_location.loc[stanox_location.Stanox == '52053', 'ELR':'LocationId'] = ('BOK1', 6072, 534877)
            stanox_location.loc[stanox_location.Stanox == '52074', 'ELR':'LocationId'] = ('ELL1', 440, 610096)

            def cleanse_stanox_location(sta_loc):
                """
                sta_loc = copy.deepcopy(stanox_location)
                """
                dat = copy.deepcopy(sta_loc)

                # Use external data - Railway Codes
                errata = load_json(cdd_network("Railway Codes", "METEX_errata.json"))
                err_stanox, err_tiploc, err_stanme = errata.values()
                # Note that {'CLAPS47': 'CLPHS47'} in err_tiploc is dubious.
                dat.replace({'Stanox': err_stanox, 'Description': err_tiploc, 'Name': err_stanme}, inplace=True)

                duplicated_stanox = dat[dat.Stanox.duplicated(keep=False)].sort_values('Stanox')
                nan_idx = duplicated_stanox[['ELR', 'Yards', 'LocationId']].applymap(pd.isna).apply(any, axis=1)
                dat.drop(duplicated_stanox[nan_idx].index, inplace=True)

                dat.drop_duplicates(subset=['Stanox'], keep='last', inplace=True)

                line_data = LineData()
                location_codes = line_data.LocationIdentifiers.fetch_location_codes()
                location_codes = location_codes['Location_codes']

                #
                for i, x in dat[dat.Description.isnull()].Stanox.items():
                    idx = location_codes[location_codes.STANOX == x].index
                    if len(idx) == 1:
                        idx = idx[0]
                        dat.loc[i, 'Description'] = location_codes[location_codes.STANOX == x].Location[idx]
                        dat.loc[i, 'Name'] = location_codes[location_codes.STANOX == x].STANME[idx]
                    else:
                        print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, x))
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
                            print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, x))
                            break
                    else:
                        idx = temp.index
                    if len(idx) > 1:
                        # Choose the first instance, and print a warning message
                        print("Warning: The STANOX \"{}\" at index \"{}\" is not unique.".format(x, i))
                    idx = idx[0]
                    dat.loc[i, 'Description'] = temp.Location.loc[idx]
                    dat.loc[i, 'Name'] = temp.STANME.loc[idx]

                location_stanme_dict = location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
                dat.Name.replace(location_stanme_dict, inplace=True)

                # Use manually-created dictionary of regular expressions
                dat.replace(fetch_location_names_repl_dict(k='Description'), inplace=True)
                dat.replace(fetch_location_names_repl_dict(k='Description', regex=True), inplace=True)

                # Use STANOX dictionary
                stanox_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('STANOX')
                temp = dat.join(stanox_dict, on='Stanox')[['Description', 'Location']]
                temp.loc[temp.Location.isnull(), 'Location'] = temp.loc[temp.Location.isnull(), 'Description']
                dat.Description = temp.apply(
                    lambda y: fuzzywuzzy.process.extractOne(y.Description, y.Location, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    if isinstance(y.Location, tuple) else y.Location, axis=1)

                dat.Name = dat.Name.str.upper()

                location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']].groupby(
                    ['STANOX', 'Location']).agg({'STANME': lambda y: list(y)[0]})
                temp = dat.join(location_codes_cut, on=['Stanox', 'Description'])
                dat.Name = temp.STANME

                dat.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'}, inplace=True)

                # dat.dropna(how='all', subset=['ELR', 'Yards', 'LocationId'], inplace=True)
                return dat

            # Cleanse raw stanox_location
            stanox_location = cleanse_stanox_location(stanox_location)

            # For 'ELR', replace NaN with ''
            stanox_location.ELR.fillna('', inplace=True)

            # For 'LocationId'
            stanox_location.Yards = stanox_location.Yards.map(lambda x: '' if np.isnan(x) else int(x))
            stanox_location.LocationId = stanox_location.LocationId.map(lambda x: '' if np.isnan(x) else int(x))

            # For 'Mileages' - to convert yards to miles (Note: Not the 'mileage' used by Network Rail)
            if use_nr_mileage_format:
                stanox_location['Mileage'] = stanox_location.Yards.map(yards_to_nr_mileage)

            # Set index
            stanox_location.set_index('Stanox', inplace=True)

            save_pickle(stanox_location, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_location = None

    return stanox_location


# Get StanoxSection
def get_stanox_section(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_stanox_section(update, save_original_as, verbose)
    """
    table_name = 'StanoxSection'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        stanox_section = load_pickle(path_to_pickle)

    else:
        try:
            # Read StanoxSection table from the database
            stanox_section = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                              save_as=save_original_as, update=update)
            stanox_section.index.name = table_name + 'Id'
            stanox_section.LocationId = stanox_section.LocationId.apply(lambda x: '' if pd.np.isnan(x) else int(x))

            line_data = LineData()
            stanox_dat = line_data.LocationIdentifiers.make_location_codes_dictionary('STANOX')

            # Firstly, create a stanox-to-location dictionary, and replace STANOX with location names
            for stanox_col_name in ['StartStanox', 'EndStanox']:
                tmp_col = stanox_col_name + '_temp'
                # Load stanox dictionary 1
                stanox_dict = get_stanox_location(use_nr_mileage_format=True).Location.to_dict()
                stanox_section[tmp_col] = stanox_section[stanox_col_name].replace(stanox_dict)  # Create a temp column
                tmp = stanox_section.join(stanox_dat, on=tmp_col).Location
                tmp_idx = tmp[tmp.notnull()].index
                stanox_section[tmp_col][tmp_idx] = tmp[tmp_idx]
                stanox_section[tmp_col] = stanox_section[tmp_col].map(lambda x: x[0] if isinstance(x, list) else x)

            stanme_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('STANME', as_dict=True)
            tiploc_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('TIPLOC', as_dict=True)

            # Secondly, process 'STANME' and 'TIPLOC'
            loc_name_replacement_dict = fetch_location_names_repl_dict()
            loc_name_regexp_replacement_dict = fetch_location_names_repl_dict(regex=True)
            # Processing 'StartStanox_tmp'
            stanox_section.StartStanox_temp = stanox_section.StartStanox_temp. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_tmp'
            stanox_section.EndStanox_temp = stanox_section.EndStanox_temp. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

            # Create 'STANOX' sections
            temp = stanox_section[stanox_section.StartStanox_temp.map(lambda x: False if isinstance(x, str) else True)]
            temp['StartStanox_'] = temp.Description.str.split(' : ', expand=True)[0]
            stanox_section.loc[temp.index, 'StartStanox_temp'] = temp.apply(
                lambda x: find_similar_str(x.StartStanox_, x.StartStanox_temp), axis=1)  # Temporary!

            temp = stanox_section[stanox_section.EndStanox_temp.map(lambda x: False if isinstance(x, str) else True)]
            temp['EndStanox_'] = temp.Description.str.split(' : ', expand=True)[1].fillna(temp.Description)
            stanox_section.loc[temp.index, 'EndStanox_temp'] = temp.apply(
                lambda x: find_similar_str(x.EndStanox_, x.EndStanox_temp), axis=1)  # Temporary!

            start_end = stanox_section.StartStanox_temp + ' - ' + stanox_section.EndStanox_temp
            point_idx = stanox_section.StartStanox_temp == stanox_section.EndStanox_temp
            start_end[point_idx] = stanox_section.StartStanox_temp[point_idx]
            stanox_section['StanoxSection'] = start_end

            # Finalising the cleaning process
            stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
            stanox_section.rename(columns={'StartStanox_temp': 'StartLocation', 'EndStanox_temp': 'EndLocation'},
                                  inplace=True)
            stanox_section = stanox_section[['LocationId', 'StanoxSection',
                                             'StartLocation', 'StartStanox', 'EndLocation', 'EndStanox',
                                             'ApproximateLocation']]

            save_pickle(stanox_section, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_section = None

    return stanox_section


# Get TrustIncident
def get_trust_incident(start_year=2006, end_year=None, update=False, save_original_as=None, use_corrected_csv=True,
                       verbose=False):
    """
    :param start_year: [int; None] (default: 2006)
    :param end_year: [int; None (default)]
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param use_corrected_csv: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        start_year = 2006
        end_year = None
        update = True
        save_original_as = None
        use_corrected_csv = True
        verbose = True

        get_trust_incident(start_year, end_year, update, save_original_as, use_corrected_csv, verbose)
    """
    table_name = 'TrustIncident'
    suffix_ext = "{}.pickle".format(
        "{}".format("-y{}".format(start_year) if start_year else "_up_to") +
        "{}".format("-y{}".format(2018 if not end_year or end_year >= 2019 else end_year)))
    path_to_pickle = cdd_metex_db_tables(table_name + suffix_ext)

    if os.path.isfile(path_to_pickle) and not update:
        trust_incident = load_pickle(path_to_pickle)

    else:
        try:
            trust_incident = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                              save_as=save_original_as, update=update)
            if use_corrected_csv:
                zip_file = zipfile.ZipFile(cdd_metex_db("Updates", table_name + ".zip"))
                corrected_csv = pd.concat(
                    [pd.read_csv(zip_file.open(f), index_col='Id', parse_dates=['StartDate', 'EndDate'],
                                 infer_datetime_format=True, dayfirst=True)
                     for f in zip_file.infolist()])
                zip_file.close()
                # Remove raw data >= '2018-01-01', pd.to_datetime('2018-01-01')
                trust_incident.drop(trust_incident[trust_incident.StartDate >= '2018-01-01'].index, inplace=True)
                # Append corrected data
                trust_incident = trust_incident.append(corrected_csv)

            trust_incident.index.name = 'TrustIncidentId'
            trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
            # Extract a subset of data, in which the StartDateTime is between 'start_year' and 'end_year'?
            trust_incident = trust_incident[
                (trust_incident.FinancialYear >= (start_year if start_year else 0)) &
                (trust_incident.FinancialYear <= (end_year if end_year else pd.datetime.now().year))]
            # Convert float to int values for 'SourceLocationId'
            trust_incident.SourceLocationId = trust_incident.SourceLocationId.map(
                lambda x: '' if pd.isna(x) else int(x))

            save_pickle(trust_incident, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            trust_incident = None

    return trust_incident


# Get Weather
def get_weather():
    conn_db = establish_mssql_connection('NR_METEX_20190203')
    sql_query = "SELECT * FROM dbo.Weather"
    #
    chunks = pd.read_sql_query(sql_query, conn_db, index_col=None, parse_dates=['DateTime'], chunksize=1000000)
    weather = pd.concat([pd.DataFrame(chunk) for chunk in chunks], ignore_index=True)
    return weather


# Get Weather data by 'WeatherCell' and 'DateTime' (Query from the database)
def fetch_weather_by_id_datetime(weather_cell_id, start_dt=None, end_dt=None, postulate=False, pickle_it=True,
                                 dat_dir=None, update=False, verbose=False):
    """
    :param weather_cell_id: [int]
    :param start_dt: [datetime.datetime; str; None (default)]
    :param end_dt: [datetime.datetime; None (default)]
    :param postulate: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param dat_dir: [str; None (default)]
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        weather_cell_id = 2367
        start_dt = pd.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
        end_dt = pd.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
        postulate = False
        pickle_it = False
        dat_dir = None
        update = True
        verbose = True

        fetch_weather_by_id_datetime(weather_cell_id, start_dt, end_dt, postulate, pickle_it, dat_dir, update, verbose)
    """
    # assert all(isinstance(x, pd.np.int64) for x in weather_cell_id)
    assert isinstance(weather_cell_id, tuple) or isinstance(weather_cell_id, (int, pd.np.integer))
    #
    pickle_filename = "{}{}{}.pickle".format(
        "_".join(str(x) for x in list(weather_cell_id)) if isinstance(weather_cell_id, tuple) else weather_cell_id,
        start_dt.strftime('_fr%Y%m%d%H%M') if start_dt else "",
        end_dt.strftime('_to%Y%m%d%H%M') if end_dt else "")

    dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else cdd_metex_db_views()
    #
    path_to_pickle = cd(dat_dir, pickle_filename)
    #
    if not os.path.isfile(path_to_pickle) or update:
        try:
            conn_metex = establish_mssql_connection('NR_METEX_20190203')
            sql_query = \
                "SELECT * FROM dbo.Weather WHERE WeatherCell {} {}{}{};".format(
                    "=" if isinstance(weather_cell_id, (int, pd.np.integer)) else "IN",
                    weather_cell_id,
                    " AND DateTime >= '{}'".format(start_dt) if start_dt else "",
                    " AND DateTime <= '{}'".format(end_dt) if end_dt else "")
            weather = pd.read_sql(sql_query, conn_metex)

            if postulate:
                def postulate_missing_hourly_precipitation(dat):
                    i = 0
                    snowfall, precipitation = dat.Snowfall.tolist(), dat.TotalPrecipitation.tolist()
                    while i + 3 < len(dat):
                        snowfall[i + 1: i + 3] = pd.np.linspace(snowfall[i], snowfall[i + 3], 4)[1:3]
                        precipitation[i + 1: i + 3] = pd.np.linspace(precipitation[i], precipitation[i + 3], 4)[1:3]
                        i += 3
                    if i + 2 == len(dat):
                        snowfall[-1:], precipitation[-1:] = snowfall[-2], precipitation[-2]
                    elif i + 3 == len(dat):
                        snowfall[-2:], precipitation[-2:] = [snowfall[-3]] * 2, [precipitation[-3]] * 2
                    dat.Snowfall = snowfall
                    dat.TotalPrecipitation = precipitation

                postulate_missing_hourly_precipitation(weather)

            if pickle_it:
                save_pickle(weather, path_to_pickle, verbose=verbose)

            return weather

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(os.path.basename(path_to_pickle))[0], e))


# Get WeatherCell
def get_weather_cell(update=False, save_original_as=None, show_map=False, projection='tmerc', save_map_as=None,
                     dpi=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param show_map: [bool] (default: False)
    :param projection: [str] (default: 'tmerc')
    :param save_map_as: [str; None (default)]
    :param dpi: [int; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        show_map = True
        projection = 'tmerc'
        save_map_as = None
        dpi = None
        verbose = True

        get_weather_cell(update, save_original_as, show_map, projection, save_map_as, dpi, verbose)
    """
    table_name = 'WeatherCell'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        weather_cell = load_pickle(path_to_pickle)

    else:
        try:
            weather_cell = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                            save_as=save_original_as, update=update)
            id_name = table_name + 'Id'
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
            weather_cell['lr_Longitude'] = weather_cell.ur_Longitude  # + weather_cell_map.width  # / 2
            weather_cell['lr_Latitude'] = weather_cell.ur_Latitude - weather_cell.height  # / 2

            # Get IMDM Weather cell map
            imdm_weather_cell_map = get_imdm_weather_cell_map().reset_index()

            # Merge the acquired data set
            weather_cell = imdm_weather_cell_map.join(weather_cell, on='WeatherCellId').sort_values('WeatherCellId')
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
            regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
            weather_cell['Region'] = weather_cell.Route.replace(regions_and_routes_dict)

            save_pickle(weather_cell, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_cell = None

    # Plot the Weather cells on the map?
    if show_map:
        weather_cell_wgs84 = shapely.geometry.MultiPolygon(list(weather_cell.Polygon_WGS84))
        minx, miny, maxx, maxy = weather_cell_wgs84.bounds

        print("Plotting weather cells ...", end="")
        fig, ax = plt.subplots(figsize=(5, 8))
        base_map = mpl_toolkits.basemap.Basemap(projection=projection,  # Transverse Mercator Projection
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
            save_fig(cdd_metex_db_fig(table_name + save_map_as), dpi=dpi, verbose=verbose)

    return weather_cell


# Get the lower-left and upper-right boundaries of weather cells
def get_weather_cell_map_boundary(route_name=None, adjustment=(0.285, 0.255)):
    """
    :param route_name: [str; None (default)]
    :param adjustment: [tuple] (numbers.Number, numbers.Number) (default: (0.285, 0.255))
    :return: [shapely.geometry.polygon.Polygon]

    Testing e.g.
        route_name = None
        adjustment = (0.285, 0.255)

        get_weather_cell_map_boundary(route_name, adjustment)
    """
    weather_cell = get_weather_cell()  # Get Weather cell

    if route_name:  # For a specific Route
        weather_cell = weather_cell[weather_cell.Route == find_similar_str(route_name, get_route().Route)]
    ll = tuple(weather_cell[['ll_Longitude', 'll_Latitude']].apply(min))
    lr = weather_cell.lr_Longitude.max(), weather_cell.lr_Latitude.min()
    ur = tuple(weather_cell[['ur_Longitude', 'ur_Latitude']].apply(max))
    ul = weather_cell.ul_Longitude.min(), weather_cell.ul_Latitude.max()

    if adjustment:  # Adjust the boundaries
        adj_values = pd.np.array(adjustment)
        ll -= adj_values
        lr += (adj_values, -adj_values)
        ur += adj_values
        ul += (-adj_values, adj_values)

    boundary = shapely.geometry.Polygon((ll, lr, ur, ul))

    return boundary


# Track
def get_track(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_track(update, save_original_as, verbose)
    """
    table_name = 'Track'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        track = load_pickle(path_to_pickle)

    else:
        try:
            track = read_table_by_query('NR_METEX_20190203', table_name, save_as=save_original_as)
            track.rename(columns={'S_MILEAGE': 'StartMileage', 'F_MILEAGE': 'EndMileage',
                                  'S_YARDAGE': 'StartYardage', 'F_YARDAGE': 'EndYardage',
                                  'MAINTAINER': 'Maintainer', 'ROUTE': 'Route', 'DELIVERY_U': 'IMDM',
                                  'StartEasti': 'StartEasting', 'StartNorth': 'StartNorthing',
                                  'EndNorthin': 'EndNorthing'},
                         inplace=True)

            # Mileage and Yardage
            mileage_cols, yardage_cols = ['StartMileage', 'EndMileage'], ['StartYardage', 'EndYardage']
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

            save_pickle(track, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            track = None

    return track


# Track Summary
def get_track_summary(update=False, save_original_as=None, verbose=False):
    """
    :param update: [bool] (default: False)
    :param save_original_as: [str; None (default)]
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        update = True
        save_original_as = None
        verbose = True

        get_track_summary(update, save_original_as, verbose)
    """
    table_name = 'Track Summary'
    path_to_pickle = cdd_metex_db_tables(table_name.replace(' ', '') + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        track_summary = load_pickle(path_to_pickle)

    else:
        try:
            track_summary = read_metex_table(table_name, save_as=save_original_as, update=update)

            # Column names
            rename_cols = {'Id': 'TrackId',
                           'Sub-route': 'SubRoute',
                           'CP6 criticality': 'CP6Criticality',
                           'CP5 Start Route': 'CP5StartRoute',
                           'Adjacent S&C': 'AdjacentS&C',
                           'Rail cumulative EMGT': 'RailCumulativeEMGT',
                           'Sleeper cumulative EMGT': 'SleeperCumulativeEMGT',
                           'Ballast cumulative EMGT': 'BallastCumulativeEMGT'}
            track_summary.rename(columns=rename_cols, inplace=True)
            renamed_cols = list(rename_cols.values())
            upper_columns = ['SRS', 'ELR', 'TID', 'IMDM', 'TME', 'TSM', 'MGTPA', 'EMGTPA', 'LTSF', 'IRJs']
            track_summary.columns = [string.capwords(x).replace(' ', '') if x not in upper_columns + renamed_cols else x
                                     for x in track_summary.columns]

            # IMDM
            track_summary.IMDM = track_summary.IMDM.map(lambda x: 'IMDM ' + x)

            # Route
            route_names_changes = load_json(cdd_network("Routes", "name-changes.json"))
            temp1 = pd.DataFrame.from_dict(route_names_changes, orient='index', columns=['Route'])
            route_names_in_table = list(track_summary.SubRoute.unique())
            route_alt = [find_similar_str(x, temp1.index) for x in route_names_in_table]

            temp2 = pd.DataFrame.from_dict(dict(zip(route_names_in_table, route_alt)), 'index', columns=['RouteAlias'])
            temp = temp2.join(temp1, on='RouteAlias').dropna()
            route_names_changes_alt = dict(zip(temp.index, temp.Route))

            track_summary['Route'] = track_summary.SubRoute.replace(route_names_changes_alt)

            save_pickle(track_summary, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            track_summary = None

    return track_summary


# Update the local pickle files for all tables
def update_metex_table_pickles(update=True, verbose=True):
    """
    :param update: [bool] (default: True)
    :param verbose: [bool] (default: True)

    Testing e.g.
        update = True
        verbose = True
        update_metex_table_pickles(update, verbose)
    """
    if confirmed("To update the local pickles of the Table data of the NR_METEX database?"):

        _ = get_imdm(as_dict=False, update=update, save_original_as=None, verbose=verbose)
        _ = get_imdm(as_dict=True, update=update, save_original_as=None, verbose=verbose)

        _ = get_imdm_alias(as_dict=False, update=update, save_original_as=None, verbose=verbose)
        _ = get_imdm_alias(as_dict=True, update=update, save_original_as=None, verbose=verbose)

        _ = get_imdm_weather_cell_map(route_info=True, grouped=False, update=update, save_original_as=None,
                                      verbose=verbose)
        _ = get_imdm_weather_cell_map(route_info=True, grouped=True, update=update, save_original_as=None,
                                      verbose=verbose)
        _ = get_imdm_weather_cell_map(route_info=False, grouped=False, update=update, save_original_as=None,
                                      verbose=verbose)
        _ = get_imdm_weather_cell_map(route_info=False, grouped=True, update=update, save_original_as=None,
                                      verbose=verbose)

        _ = get_incident_reason_info(plus=True, update=update, save_original_as=None, verbose=verbose)
        _ = get_incident_reason_info(plus=False, update=update, save_original_as=None, verbose=verbose)

        _ = get_weather_codes(as_dict=False, update=update, save_original_as=None, verbose=verbose)
        _ = get_weather_codes(as_dict=True, update=update, save_original_as=None, verbose=verbose)

        _ = get_incident_record(update=update, save_original_as=None, use_corrected_csv=True, verbose=verbose)

        _ = get_location(update=update, save_original_as=None, verbose=verbose)

        _ = get_pfpi(plus=True, update=update, save_original_as=None, use_corrected_csv=True, verbose=verbose)
        _ = get_pfpi(plus=False, update=update, save_original_as=None, use_corrected_csv=True, verbose=verbose)

        _ = get_route(as_dict=False, update=update, save_original_as=None, verbose=verbose)
        _ = get_route(as_dict=True, update=update, save_original_as=None, verbose=verbose)

        _ = get_stanox_location(use_nr_mileage_format=True, update=update, save_original_as=None, verbose=verbose)
        _ = get_stanox_location(use_nr_mileage_format=False, update=update, save_original_as=None, verbose=verbose)

        _ = get_stanox_section(update=update, save_original_as=None, verbose=verbose)

        _ = get_trust_incident(start_year=2006, end_year=None, update=update, save_original_as=None,
                               use_corrected_csv=True, verbose=verbose)
        # _ = get_weather()

        _ = get_weather_cell(update=update, save_original_as=None, show_map=True, projection='tmerc',
                             save_map_as=".png", dpi=600, verbose=verbose)
        # _ = get_weather_cell_map_boundary(route=None, adjustment=(0.285, 0.255))

        _ = get_track(update=update, save_original_as=None, verbose=verbose)

        _ = get_track_summary(update=update, save_original_as=None, verbose=verbose)

        if verbose:
            print("\nUpdate finished.")


# ====================================================================================================================
""" Tools """


# Create a filename
def make_filename(base_name=None, route_name=None, weather_category=None, *extra_suffixes, sep="-", save_as=".pickle"):
    """
    :param base_name: [str; None (default)]
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param extra_suffixes: [str; None (default)]
    :param sep: [str] (default: "-")
    :param save_as: [str] (default: ".pickle")
    :return: [str]

    Testing e.g.
        base_name = "test"  # None
        route_name = None
        weather_category = None
        sep = "-"
        save_as = ".pickle"

        make_filename(base_name, route_name, weather_category)
        make_filename(None, route_name, weather_category, "test1", "test2")
        make_filename(base_name, route_name, weather_category, "test1", "test2")
        make_filename(base_name, 'Anglia', weather_category, "test1", "test2")
        make_filename(base_name, 'North and East', 'Heat', "test1", "test2")
    """
    base_name_ = "" if base_name is None else base_name
    route_name_ = "" if route_name is None \
        else (sep if base_name_ else "") + find_similar_str(route_name, get_route().Route).replace(" ", "_")
    weather_category_ = "" if weather_category is None \
        else (sep if route_name_ else "") + find_similar_str(weather_category,
                                                             get_weather_codes().WeatherCategory).replace(" ", "_")
    if extra_suffixes:
        suffix = ["{}".format(s) for s in extra_suffixes if s]
        suffix = (sep if any(x for x in (base_name_, route_name_, weather_category_)) else "") + sep.join(suffix) \
            if len(suffix) > 1 else sep + suffix[0]
        filename = base_name_ + route_name_ + weather_category_ + suffix + save_as
    else:
        filename = base_name_ + route_name_ + weather_category_ + save_as
    return filename


# Subset the required data given 'route' and 'weather category'
def get_subset(data_set, route_name=None, weather_category=None, rearrange_index=False):
    """
    :param data_set: [pd.DataFrame; None]
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param rearrange_index: [bool] (default: False)
    :return: [pd.DataFrame; None]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = None
        rearrange_index = False
    """
    if data_set is not None:
        assert isinstance(data_set, pd.DataFrame) and not data_set.empty
        data_subset = data_set.copy(deep=True)

        if route_name:
            try:  # assert 'Route' in data_subset.columns
                data_subset.Route = data_subset.Route.astype(str)
                route_names = get_route()
                route_lookup = list(set(route_names.Route)) + list(set(route_names.RouteAlias))
                route_name_ = [
                    fuzzywuzzy.process.extractOne(x, route_lookup, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    for x in ([route_name] if isinstance(route_name, str) else list(route_name))]
                data_subset = data_subset[data_subset.Route.isin(route_name_)]
            except AttributeError:
                print("Couldn't slice the data by \"Route\". The attribute may not exist in the 'data_set'.")
                pass

        if weather_category:
            try:  # assert 'WeatherCategory' in data_subset.columns
                data_subset.WeatherCategory = data_subset.WeatherCategory.astype(str)
                weather_category_code = get_weather_codes()
                weather_category_lookup = list(set(weather_category_code.WeatherCategory))
                weather_category_ = [
                    fuzzywuzzy.process.extractOne(x, weather_category_lookup, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    for x in ([weather_category] if isinstance(weather_category, str) else list(weather_category))]
                data_subset = data_subset[data_subset.WeatherCategory.isin(weather_category_)]
            except AttributeError:
                print("Couldn't slice the data by \"WeatherCategory\". The attribute may not exist in the 'data_set'.")
                pass

        if rearrange_index:
            data_subset.index = range(len(data_subset))  # NOT dat.reset_index(inplace=True)
    else:
        data_subset = None
    return data_subset


# Calculate the DelayMinutes and DelayCosts for grouped data
def calculate_pfpi_stats(data_set, selected_features, sort_by=None):
    """
    :param data_set: [pd.DataFrame]
    :param selected_features: [list]
    :param sort_by: [str; list; None (default)]
    :return: [pd.DataFrame]

    Testing e.g.
        data_set = selected_data.copy()
    """
    pfpi_stats = data_set.groupby(selected_features[1:-2]).aggregate({
        # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
        'PfPIId': pd.np.count_nonzero,
        'PfPIMinutes': pd.np.sum,
        'PfPICosts': pd.np.sum})

    pfpi_stats.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']
    pfpi_stats.reset_index(inplace=True)  # Reset the grouped indexes to columns

    if sort_by:
        pfpi_stats.sort_values(sort_by, inplace=True)

    return pfpi_stats


# ====================================================================================================================
""" Get views """


# View Schedule 8 details (TRUST data)
def view_schedule8_data(route_name=None, weather_category=None, rearrange_index=False, weather_attributed_only=False,
                        update=False, pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param rearrange_index: [bool] (default: False)
    :param weather_attributed_only: [bool] (default: False)
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        rearrange_index = True
        weather_attributed_only = False
        update = True
        pickle_it = False
        verbose = True

        view_schedule8_data(route_name, weather_category, reset_index, weather_attributed_only,
                            update, pickle_it, verbose)
    """
    filename = "Schedule8-data" + ("-weather-attributed" if weather_attributed_only else "")
    pickle_filename = make_filename(filename, route_name, weather_category, save_as=".pickle")
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        schedule8_data = load_pickle(path_to_pickle)
        if rearrange_index and schedule8_data.index.name == 'PfPIId':
            schedule8_data.reset_index(inplace=True)

    else:
        path_to_merged = cdd_metex_db_views("{}.pickle".format(filename))

        try:

            if os.path.isfile(path_to_merged) and not update:
                schedule8_data = load_pickle(path_to_merged)

            else:
                pfpi = get_pfpi(verbose=verbose)  # Get PfPI # (260645, 6)  # (5049003, 6)
                incident_record = get_incident_record(verbose=verbose)  # (233452, 4)  # (4704448, 5)
                trust_incident = get_trust_incident(verbose=verbose)  # (192054, 11)  # (4049984, 11)
                location = get_location(verbose=verbose)  # (228851, 6)  # (653882, 7)
                imdm = get_imdm(verbose=verbose)  # (42, 1)  # (42, 3)
                incident_reason_info = get_incident_reason_info(verbose=verbose)  # (393, 7)  # (174, 9)
                stanox_location = get_stanox_location(verbose=verbose)  # (7560, 5)  # (7534, 6)
                stanox_section = get_stanox_section(verbose=verbose)  # (9440, 7)  # (10601, 7)

                if weather_attributed_only:
                    incident_record = incident_record[incident_record.WeatherCategory != '']  # (320942, 5) ≈ 6.8%

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

                gc.collect()

                # Note: There may be errors in e.g. IMDM data/column, location id, of the TrustIncident table.

                idx = schedule8_data[~schedule8_data.StartLocation.eq(schedule8_data.Location_Start)].index
                for i in idx:
                    schedule8_data.loc[i, 'StartLocation'] = schedule8_data.loc[i, 'Location_Start']
                    schedule8_data.loc[i, 'EndLocation'] = schedule8_data.loc[i, 'Location_End']
                    if schedule8_data.loc[i, 'StartLocation'] == schedule8_data.loc[i, 'EndLocation']:
                        schedule8_data.loc[i, 'StanoxSection'] = schedule8_data.loc[i, 'StartLocation']
                    else:
                        schedule8_data.loc[i, 'StanoxSection'] = \
                            schedule8_data.loc[i, 'StartLocation'] + ' - ' + schedule8_data.loc[i, 'EndLocation']

                schedule8_data.drop(['IMDM', 'Location_Start', 'Location_End'], axis=1, inplace=True)  # (5024674, 57)

                # (260140, 50)  # (5155014, 57)
                schedule8_data.rename(columns={'LocationAlias_Start': 'StartLocationAlias',
                                               'LocationAlias_End': 'EndLocationAlias',
                                               'ELR_Start': 'StartELR', 'Yards_Start': 'StartYards',
                                               'ELR_End': 'EndELR', 'Yards_End': 'EndYards',
                                               'Mileage_Start': 'StartMileage', 'Mileage_End': 'EndMileage',
                                               'LocationId_Start': 'StartLocationId',
                                               'LocationId_End': 'EndLocationId',
                                               'LocationId_Section': 'SectionLocationId', 'IMDM_Location': 'IMDM',
                                               'StartDate': 'StartDateTime', 'EndDate': 'EndDateTime'},
                                      inplace=True)

                # Use 'Station' data from Railway Codes website
                other_assets = OtherAssets()
                station_locations = other_assets.Stations.fetch_station_locations()

                station_locations = station_locations['Stations'][['Station', 'Degrees Longitude', 'Degrees Latitude']]
                station_locations = station_locations.dropna().drop_duplicates('Station', keep='first')
                station_locations.set_index('Station', inplace=True)
                temp = schedule8_data[['StartLocation']].join(station_locations, on='StartLocation', how='left')
                i = temp[temp['Degrees Longitude'].notna()].index
                schedule8_data.loc[i, 'StartLongitude':'StartLatitude'] = \
                    temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()
                temp = schedule8_data[['EndLocation']].join(station_locations, on='EndLocation', how='left')
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

            schedule8_data = get_subset(schedule8_data, route_name, weather_category, rearrange_index)

            if pickle_it:
                if path_to_pickle != path_to_merged:
                    save_pickle(schedule8_data, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to retrieve the data about Schedule 8 incidents. {}.".format(e))
            schedule8_data = None

    return schedule8_data


# Essential details about Incidents
def view_schedule8_data_pfpi(route_name=None, weather_category=None, update=False, pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = False
        pickle_it = False
        verbose = True

        view_schedule8_data_pfpi(route_name, weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-details-pfpi"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))
            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                data = get_subset(temp_data, route_name, weather_category)

            else:
                # Get the merged data
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
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


# Get Schedule 8 data by incident location and Weather category
def view_schedule8_costs_by_location(route_name=None, weather_category=None, update=False,
                                     pickle_it=True, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = True
        verbose = True

        view_schedule8_costs_by_location(route_name, weather_category, update, pickle_it, verbose)
        view_schedule8_costs_by_location('Anglia', weather_category, update, pickle_it, verbose)
        view_schedule8_costs_by_location('Anglia', 'Wind', update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-location"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))
            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
                selected_features = [
                    'PfPIId',
                    # 'TrustIncidentId', 'IncidentRecordCreateDate',
                    'WeatherCategory',
                    'Route', 'IMDM', 'Region', 'StanoxSection',
                    'StartLocation', 'EndLocation', 'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                    'StartStanox', 'EndStanox', 'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                    'PfPIMinutes', 'PfPICosts']
                selected_data = schedule8_data[selected_features]
                extracted_data = calculate_pfpi_stats(selected_data, selected_features)

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 data by datetime and location
def view_schedule8_costs_by_datetime_location(route_name=None, weather_category=None, update=False,
                                              pickle_it=True, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = True
        verbose = True

        view_schedule8_costs_by_datetime_location(route_name, weather_category, update, pickle_it, verbose)
        view_schedule8_costs_by_datetime_location('Anglia', weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-datetime-location"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
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
                extracted_data = calculate_pfpi_stats(selected_data, selected_features,
                                                      sort_by=['StartDateTime', 'EndDateTime'])

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 costs by datetime, location and incident reason
def view_schedule8_costs_by_datetime_location_reason(route_name=None, weather_category=None, update=False,
                                                     pickle_it=True, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = True
        verbose = True

        view_schedule8_costs_by_datetime_location_reason(route_name, weather_category, update, pickle_it, verbose)
        view_schedule8_costs_by_datetime_location_reason('Anglia', weather_category, update, pickle_it, verbose)
        view_schedule8_costs_by_datetime_location_reason('Anglia', 'Wind', update, pickle_it, verbose)
        view_schedule8_costs_by_datetime_location_reason('Anglia', 'Heat', update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-datetime-location-reason"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
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
                                     'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
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
                extracted_data = calculate_pfpi_stats(selected_data, selected_features,
                                                      sort_by=['StartDateTime', 'EndDateTime'])

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 data by datetime and Weather category
def view_schedule8_costs_by_datetime(route_name=None, weather_category=None, update=False,
                                     pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = False
        verbose = True

        view_schedule8_costs_by_datetime(route_name, weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-datetime"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))
            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
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
                extracted_data = calculate_pfpi_stats(selected_data, selected_features,
                                                      sort_by=['StartDateTime', 'EndDateTime'])

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 costs by incident reason
def view_schedule8_costs_by_reason(route_name=None, weather_category=None, update=False,
                                   pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = False
        verbose = True

        view_schedule8_costs_by_reason(route_name, weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-reason"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
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
                extracted_data = calculate_pfpi_stats(selected_data, selected_features, sort_by=None)

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 costs by location and incident reason
def view_schedule8_costs_by_location_reason(route_name=None, weather_category=None, update=False,
                                            pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = False
        verbose = True

        view_schedule8_costs_by_location_reason(route_name, weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-location-reason"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
                selected_features = ['PfPIId',
                                     'FinancialYear',
                                     'WeatherCategory',
                                     'Route', 'IMDM', 'Region',
                                     'StanoxSection',
                                     'StartStanox', 'EndStanox',
                                     'StartLocation', 'EndLocation',
                                     'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                                     'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
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
                extracted_data = calculate_pfpi_stats(selected_data, selected_features, sort_by=None)

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 costs by Weather category
def view_schedule8_costs_by_weather_category(route_name=None, weather_category=None, update=False,
                                             pickle_it=False, verbose=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param update: [bool] (default: False)
    :param pickle_it: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        update = True
        pickle_it = False
        verbose = True

        view_schedule8_costs_by_weather_category(route_name, weather_category, update, pickle_it, verbose)
    """
    filename = "Schedule8-costs-by-weather_category"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            path_to_pickle_temp = cdd_metex_db_views(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                temp_data = load_pickle(path_to_pickle_temp)
                extracted_data = get_subset(temp_data, route_name, weather_category)

            else:
                schedule8_data = view_schedule8_data(route_name, weather_category, rearrange_index=True)
                selected_features = ['PfPIId', 'FinancialYear', 'Route', 'IMDM', 'Region',
                                     'WeatherCategory', 'PfPICosts', 'PfPIMinutes']
                selected_data = schedule8_data[selected_features]
                extracted_data = calculate_pfpi_stats(selected_data, selected_features)

            if pickle_it:
                save_pickle(extracted_data, path_to_pickle, verbose=verbose)

            return extracted_data

        except Exception as e:
            print("Failed to retrieve \"{}.\" \n{}.".format(os.path.splitext(pickle_filename)[0], e))


# Get Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section
def fetch_incident_locations_from_nr_metex(route_name=None, weather_category=None, start_and_end_elr=None,
                                           update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param start_and_end_elr: [str; None (default)] 'same', 'diff'
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return:

    Testing e.g.
        route_name = None
        weather_category = None
        start_and_end_elr = None
        update = True
        verbose = True

        fetch_incident_locations_from_nr_metex('Anglia', weather_category, start_and_end_elr, update, verbose)
    """
    assert start_and_end_elr in (None, 'same', 'diff')

    filename = "Schedule8-incident-locations"
    pickle_filename = make_filename(filename, route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(pickle_filename)

    try:
        if os.path.isfile(path_to_pickle) and not update:
            incident_locations = load_pickle(path_to_pickle)

        else:
            # All incident locations
            s8costs_by_location = view_schedule8_costs_by_location(route_name, weather_category, update=update)
            s8costs_by_location = s8costs_by_location.loc[:, 'Route':'EndLatitude']
            incident_locations = s8costs_by_location.drop_duplicates()

            # Create two additional columns about data of mileages (convert str to num)
            incident_locations[['StartMileage_num', 'EndMileage_num']] = \
                incident_locations[['StartMileage', 'EndMileage']].applymap(str_to_num_mileage)

            # Remove records for which ELR information was missing
            incident_locations = incident_locations[
                ~(incident_locations.StartELR.str.contains('^$')) &
                ~(incident_locations.EndELR.str.contains('^$'))]

            save_pickle(incident_locations, path_to_pickle, verbose=verbose)

        if start_and_end_elr is not None:
            if start_and_end_elr == 'same':
                # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
                incident_locations = incident_locations[incident_locations.StartELR == incident_locations.EndELR]
            elif start_and_end_elr == 'diff':
                # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
                incident_locations = incident_locations[incident_locations.StartELR != incident_locations.EndELR]

        return incident_locations

    except Exception as e:
        print("Failed to fetch \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# Update the local pickle files for all essential views
def update_metex_view_pickles(update=True, pickle_it=True, verbose=True):
    """
    :param update: [bool] (default: True)
    :param pickle_it: [bool] (default: True)
    :param verbose: [bool] (default: True)

    Testing e.g.
        update = True
        pickle_it = True
        verbose = True

        update_metex_view_pickles(update, pickle_it, verbose)
    """
    if confirmed("To update the View pickles of the NR_METEX data?"):

        _ = view_schedule8_costs_by_location(None, None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_location('Anglia', None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_location('Anglia', 'Wind', update, pickle_it, verbose)
        _ = view_schedule8_costs_by_location('Anglia', 'Heat', update, pickle_it, verbose)

        _ = view_schedule8_costs_by_datetime_location(None, None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location('Anglia', None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location('Anglia', 'Wind', update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location('Anglia', 'Heat', update, pickle_it, verbose)

        _ = view_schedule8_costs_by_datetime_location_reason(None, None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location_reason('Anglia', None, update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location_reason('Anglia', 'Wind', update, pickle_it, verbose)
        _ = view_schedule8_costs_by_datetime_location_reason('Anglia', 'Heat', update, pickle_it, verbose)

        _ = fetch_incident_locations_from_nr_metex(None, None, start_and_end_elr=None, update=update, verbose=verbose)

        if verbose:
            print("\nUpdate finished.")
