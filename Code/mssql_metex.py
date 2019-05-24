""" Read and cleanse data of NR_METEX database """

import copy
import os
import string

import fuzzywuzzy.fuzz
import fuzzywuzzy.process
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import pandas as pd
import shapely.geometry

from converters import nr_mileage_num_to_str, osgb36_to_wgs84, wgs84_to_osgb36, yards_to_nr_mileage
from delay_attr_glossary import get_incident_reason_metadata, get_performance_event_code
from loc_code_dict import location_names_regexp_replacement_dict, location_names_replacement_dict
from mssql_utils import establish_mssql_connection, get_table_primary_keys, read_table_by_query
from railwaycodes_utils import get_location_codes, get_station_locations
from railwaycodes_utils import get_location_codes_dictionary, get_location_codes_dictionary_v2
from utils import cd, cdd, cdd_metex, cdd_rc, load_json, load_pickle, save, save_fig, save_pickle

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\METEX\\Database"
def cdd_metex_db(*sub_dir):
    path = cdd_metex("Database")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\METEX\\Database\\Tables" and sub-directories
def cdd_metex_db_tables(*sub_dir):
    path = cdd_metex_db("Tables")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\METEX\\Database\\Views" and sub-directories
def cdd_metex_db_views(*sub_dir):
    path = cdd_metex_db("Views")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "METEX\\Figures" and sub-directories
def cdd_metex_fig_db(*sub_dir):
    path = cdd_metex_db("Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Publications\\...\\Figures" and sub-directories
def cdd_metex_fig_pub(pid, *sub_dir):
    path = cd("Publications", "{} - ".format(pid), "Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Read tables available in Database
def read_metex_table(table_name, index_col=None, route_name=None, weather_category=None, coerce_float=True,
                     parse_dates=None, chunk_size=None, params=None, schema_name='dbo', save_as=None, update=False):
    """
    :param table_name: [str] name of a queried table from the Database
    :param index_col: [str] name of a column that is set to be the index
    :param route_name: [str] name of the specific Route
    :param weather_category: [str] name of the specific Weather category
    :param coerce_float: [bool]
    :param parse_dates: [list; None]
    :param chunk_size: [str; None]
    :param params: [list, tuple or dict, optional, default: None]
    :param schema_name: [str] 'dbo', as default
    :param save_as: [str]
    :param update: [bool]
    :return: [pandas.DataFrame] the queried data as a DataFrame
    """
    table = '{}."{}"'.format(schema_name, table_name)
    # Connect to the queried database
    conn_metex = establish_mssql_connection(database_name='NR_METEX_20190203')
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
    # Create a pandas.DataFrame of the queried table
    table_data = pd.read_sql(sql_query, conn_metex, index_col=index_col, coerce_float=coerce_float,
                             parse_dates=parse_dates, chunksize=chunk_size, params=params)
    # Disconnect the database
    conn_metex.close()
    if save_as:
        path_to_file = cdd_metex_db_tables(table_name + save_as)
        if not os.path.isfile(path_to_file) or update:
            save(table_data, path_to_file, index=True if index_col else False)
    return table_data


# Get primary keys of a table in database NR_METEX
def get_metex_table_pk(table_name):
    pri_key = get_table_primary_keys(database_name="NR_METEX_20190203", table_name=table_name)
    return pri_key


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Get IMDM
def get_imdm(as_dict=False, update=False, save_original_as=None):
    """
    :param as_dict: [bool]
    :param update: [bool]
    :param save_original_as: [str; None (default)] e.g. ".csv"
    :return: [pandas.DataFrame; None]
    """
    table_name = 'IMDM'
    path_to_file = cdd_metex_db_tables("".join([table_name, ".json" if as_dict else ".pickle"]))
    if os.path.isfile(path_to_file) and not update:
        imdm = load_json(path_to_file) if as_dict else load_pickle(path_to_file)
    else:
        try:
            imdm = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                    save_as=save_original_as, update=update)
            imdm.index.rename(name='IMDM', inplace=True)  # Rename a column and index
            imdm.rename(columns={'Route': 'RouteAlias'}, inplace=True)
            # Update route names
            route_names_changes = load_json(cdd("Network\\Routes", "route-names-changes.json"))
            imdm['Route'] = imdm.RouteAlias.replace(route_names_changes)
            if as_dict:
                imdm_dict = imdm.to_dict()
                imdm = imdm_dict['Route']
                imdm.pop('None')
            save(imdm, path_to_file)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " as a dictionary" if as_dict else "", e))
            imdm = None
    return imdm


# Get ImdmAlias
def get_imdm_alias(as_dict=False, update=False, save_original_as=None):
    """
    :param as_dict: [bool]
    :param update: [bool]
    :param save_original_as: [str; None (default)] e.g. ".csv"
    :return: [pandas.DataFrame; None]
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
            save(imdm_alias, path_to_file)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " as a dictionary" if as_dict else "", e))
            imdm_alias = None
    return imdm_alias


# Get IMDMWeatherCellMap
def get_imdm_weather_cell_map(route_info=True, grouped=False, update=False, save_original_as=None):
    """
    :param route_info: [bool]
    :param grouped: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'IMDMWeatherCellMap_pc' if route_info else 'IMDMWeatherCellMap'
    path_to_pickle = cdd_metex_db_tables(table_name + ("_grouped.pickle" if grouped else ".pickle"))
    if os.path.isfile(path_to_pickle) and not update:
        weather_cell_map = load_pickle(path_to_pickle)
    else:
        try:
            # Read IMDMWeatherCellMap table
            weather_cell_map = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                                coerce_float=False, save_as=save_original_as, update=update)
            if route_info:
                weather_cell_map.rename(columns={'Route': 'RouteAlias'}, inplace=True)
                route_names_changes = load_json(cdd("Network\\Routes", "route-names-changes.json"))
                weather_cell_map['Route'] = weather_cell_map.RouteAlias.replace(route_names_changes)
                weather_cell_map[['Id', 'WeatherCell']] = weather_cell_map[['Id', 'WeatherCell']].applymap(int)
                weather_cell_map.set_index('Id', inplace=True)
            weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index
            weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'}, inplace=True)  # Rename a column
            if grouped:  # To find out how many IMDMs each 'WeatherCellId' is associated with
                weather_cell_map = weather_cell_map.groupby('Route' if route_info else 'WeatherCellId').aggregate(
                    lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))
            save_pickle(weather_cell_map, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " (being grouped)" if grouped else "", e))
            weather_cell_map = None
    return weather_cell_map


# Get IncidentReasonInfo
def get_incident_reason_info(plus=True, update=False, save_original_as=None):
    """
    :param plus: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'IncidentReasonInfo'
    path_to_pickle = cdd_metex_db_tables(table_name + ("_plus.pickle" if plus else ".pickle"))
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
            save_pickle(incident_reason_info, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " with extra information" if plus else "", e))
            incident_reason_info = None
    return incident_reason_info


# Get WeatherCategoryLookup
def get_weather_codes(as_dict=False, update=False, save_original_as=None):
    """
    :param as_dict: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'WeatherCategoryLookup'  # WeatherCodes
    path_to_file = cdd_metex_db_tables(table_name + (".json" if as_dict else ".pickle"))
    if os.path.isfile(path_to_file) and not update:
        weather_codes = load_json(path_to_file) if as_dict else load_pickle(path_to_file)
    else:
        try:
            weather_codes = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                             save_as=save_original_as, update=update)
            weather_codes.rename(columns={'Name': 'WeatherCategory'}, inplace=True)
            weather_codes.index.rename(name='WeatherCategoryCode', inplace=True)
            save((weather_codes.to_dict() if as_dict else weather_codes), path_to_file)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_codes = None
    return weather_codes


# Get IncidentRecord and fill 'None' value with NaN
def get_incident_record(update=False, save_original_as=None):
    """
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'IncidentRecord'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        incident_record = load_pickle(path_to_pickle)
    else:
        try:
            incident_record = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                               save_as=save_original_as, update=update)
            incident_record.index.rename('IncidentRecordId', inplace=True)  # Rename index name
            incident_record.rename(columns={'CreateDate': 'IncidentRecordCreateDate', 'Reason': 'IncidentReasonCode'},
                                   inplace=True)  # Rename column names
            weather_codes = get_weather_codes(as_dict=True)  # Get a weather category lookup dictionary
            # Replace each weather category code with its full name
            incident_record.replace(weather_codes, inplace=True)
            incident_record.WeatherCategory.fillna(value='', inplace=True)
            save_pickle(incident_record, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            incident_record = None
    return incident_record


# Get Location
def get_location(update=False, save_original_as=None):
    """
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'Location'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        location = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'Location' table
            location = read_metex_table(table_name, index_col=get_metex_table_pk(table_name), coerce_float=False,
                                        save_as=save_original_as, update=update)
            location.index.rename('LocationId', inplace=True)
            location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
            location[['WeatherCell', 'SMDCell']] = location[['WeatherCell', 'SMDCell']].applymap(
                lambda x: 0 if pd.np.isnan(x) else int(x))
            # location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]
            save_pickle(location, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            location = None
    return location


# Get PfPI (Process for Performance Improvement)
def get_pfpi(plus=True, update=False, save_original_as=None):
    """
    :param plus: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'PfPI'
    path_to_pickle = cdd_metex_db_tables(table_name + ("_plus.pickle" if plus else ".pickle"))
    if os.path.isfile(path_to_pickle) and not update:
        pfpi = load_pickle(path_to_pickle)
    else:
        try:
            # Read the 'PfPI' table
            pfpi = read_metex_table(table_name, index_col=get_metex_table_pk(table_name), save_as=save_original_as,
                                    update=update)
            pfpi.index.rename('PfPIId', inplace=True)
            if plus:  # To include more information for 'PerformanceEventCode'
                performance_event_code = get_performance_event_code()
                performance_event_code.index.rename('PerformanceEventCode', inplace=True)
                performance_event_code.columns = [x.replace('_', '') for x in performance_event_code.columns]
                # Merge pfpi and pe_code
                pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')
            # Save the data
            save_pickle(pfpi, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\"{}. {}.".format(table_name, " with extra information" if plus else "", e))
            pfpi = None
    return pfpi


# Get Route (Note that there is only one column in the original table)
def get_route(as_dict=False, update=False, save_original_as=None):
    """
    :param as_dict: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = "Route"
    path_to_pickle = cdd_metex_db_tables(table_name + (".json" if as_dict else ".pickle"))
    if os.path.isfile(path_to_pickle) and not update:
        route = load_pickle(path_to_pickle)
    else:
        try:
            route = read_metex_table(table_name, save_as=save_original_as, update=update)
            route_names_changes = load_json(cdd("Network\\Routes", "route-names-changes.json"))
            route['Route'] = route.Name.replace(route_names_changes)
            route.rename(columns={'Name': 'RouteAlias'}, inplace=True)
            if as_dict:
                route.drop_duplicates('Route', inplace=True)
                route = dict(zip(range(len(route)), route.Route))
            save(route, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            route = None
    return route


# Get StanoxLocation
def get_stanox_location(use_nr_mileage_format=True, update=False, save_original_as=None):
    """
    :param use_nr_mileage_format: [bool]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'StanoxLocation'
    path_to_pickle = cdd_metex_db_tables(table_name + (".pickle" if use_nr_mileage_format else "_miles.pickle"))

    if os.path.isfile(path_to_pickle) and not update:
        stanox_location = load_pickle(path_to_pickle)
    else:
        try:
            # Read StanoxLocation table from the database
            stanox_location = read_metex_table(table_name, index_col=None, save_as=save_original_as, update=update)

            # stanox_location.rename(columns={'Stanox': 'STANOX'}, inplace=True)

            def cleanse_stanox_location(sta_loc):
                # sta_loc = stanox_location
                dat = copy.deepcopy(sta_loc)

                # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                errata = load_json(cdd_rc("errata.json"))
                errata_stanox, errata_tiploc, errata_stanme = errata.values()
                dat.Stanox = dat.Stanox.replace(errata_stanox)
                dat.Description = dat.Description.replace(errata_tiploc)
                dat.Name = dat.Name.replace(errata_stanme)

                # Use external data - Railway Codes
                location_codes = get_location_codes()['Locations']

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
                    if location_codes[location_codes.STANOX == x].shape[0] > 1:
                        desc = dat[dat.Stanox == x].Description[i]
                        if desc in location_codes[location_codes.STANOX == x].TIPLOC.tolist():
                            idx = location_codes[(location_codes.STANOX == x) & (location_codes.TIPLOC == desc)].index
                        elif desc in location_codes[location_codes.STANOX == x].STANME.tolist():
                            idx = location_codes[(location_codes.STANOX == x) & (location_codes.STANME == desc)].index
                        else:
                            print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, x))
                            break
                    else:
                        idx = location_codes[location_codes.STANOX == x].index
                    if len(idx) > 1:
                        print("Warning: The STANOX \"{}\" is not unique. Check at index \"{}\"".format(i, x))
                    idx = idx[0]  # Temporary solution!
                    dat.loc[i, 'Description'] = location_codes[location_codes.STANOX == x].Location.loc[idx]
                    dat.loc[i, 'Name'] = location_codes[location_codes.STANOX == x].STANME.loc[idx]

                location_stanme_dict = location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
                dat.Name.replace(location_stanme_dict, inplace=True)

                # Use manually-created dictionary of regular expressions
                dat.replace(location_names_replacement_dict('Description'), inplace=True)
                dat.replace(location_names_regexp_replacement_dict('Description'), inplace=True)

                # Use STANOX dictionary
                stanox_dict = get_location_codes_dictionary_v2(['STANOX'], update=update)
                temp = dat.join(stanox_dict, on='Stanox')[['Description', 'Location']]
                na_loc = temp.Location.isnull()
                temp.loc[na_loc, 'Location'] = temp.loc[na_loc, 'Description']
                dat.Description = temp.apply(
                    lambda y: fuzzywuzzy.process.extractOne(y.Description, y.Location, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    if isinstance(y.Location, list) else y.Location, axis=1)

                dat.Name = dat.Name.str.upper()

                location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']]
                location_codes_cut = location_codes_cut.groupby(['STANOX', 'Location']).agg({'STANME': list})
                location_codes_cut.STANME = location_codes_cut.STANME.map(
                    lambda y: x if isinstance(y, list) and len(y) > 1 else x[0])
                temp = dat.join(location_codes_cut, on=['Stanox', 'Description'])
                dat.Name = temp.STANME

                return dat

            # Cleanse raw stanox_location
            stanox_location = cleanse_stanox_location(stanox_location)

            # Rename names
            stanox_location.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'}, inplace=True)

            # For 'ELR', replace NaN with ''
            stanox_location.ELR.fillna('', inplace=True)

            # For 'LocationId'
            stanox_location.LocationId = stanox_location.LocationId.map(lambda x: '' if pd.np.isnan(x) else int(x))

            # For 'Mileages' - to convert yards to miles (Note: Not the 'mileage' used by Network Rail)
            stanox_location['Mileage'] = stanox_location.Yards.map(
                lambda x: yards_to_nr_mileage(x) if use_nr_mileage_format else (None if pd.isnull(x) else x / 1760))

            # Set index
            stanox_location.set_index('Stanox', inplace=True)

            # Most likely errors
            stanox_location.loc['52053', 'ELR':'LocationId'] = ['BOK1', '3.0792', 534877]
            stanox_location.loc['52074', 'ELR':'LocationId'] = ['ELL1', '0.0440', 610096]

            save_pickle(stanox_location, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_location = None

    return stanox_location


# Get StanoxSection
def get_stanox_section(update=False, save_original_as=None):
    """
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
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
            stanox_section.index.name = 'StanoxSectionId'
            stanox_section.LocationId = stanox_section.LocationId.apply(lambda x: int(x) if not pd.np.isnan(x) else '')

            # Firstly, create a stanox-to-location dictionary, and replace STANOX with location names
            def cleanse_upon_stanox(dat, *stanox_col_names):
                for stanox_col_name in stanox_col_names:
                    temp_col = stanox_col_name + '_temp'
                    # Load stanox dictionary 1
                    stanox_dict = get_stanox_location(use_nr_mileage_format=True).Location.to_dict()
                    dat[temp_col] = dat[stanox_col_name].replace(stanox_dict)  # Create a temp column
                    # Load stanox dictionary 2
                    stanox_dat = get_location_codes_dictionary_v2(['STANOX'], update=update)
                    temp = dat.join(stanox_dat, on=temp_col).Location
                    temp_idx = temp[temp.notnull()].index
                    dat[temp_col][temp_idx] = temp[temp_idx]
                    dat[temp_col] = dat[temp_col].map(lambda x: x[0] if isinstance(x, list) else x)

            cleanse_upon_stanox(stanox_section, 'StartStanox')
            cleanse_upon_stanox(stanox_section, 'EndStanox')

            # Secondly, process 'STANME' and 'TIPLOC'
            stanme_dict = get_location_codes_dictionary(keyword='STANME')
            tiploc_dict = get_location_codes_dictionary(keyword='TIPLOC')
            loc_name_replacement_dict = location_names_replacement_dict()
            loc_name_regexp_replacement_dict = location_names_regexp_replacement_dict()
            # Processing 'StartStanox_tmp'
            stanox_section.StartStanox_temp = stanox_section.StartStanox_temp. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_tmp'
            stanox_section.EndStanox_temp = stanox_section.EndStanox_temp. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Create 'STANOX' sections
            start_end = stanox_section.StartStanox_temp + ' - ' + stanox_section.EndStanox_temp
            point_idx = stanox_section.StartStanox_temp == stanox_section.EndStanox_temp
            start_end[point_idx] = stanox_section.StartStanox_temp[point_idx]
            stanox_section['StanoxSection'] = start_end

            # Finalising the cleaning process
            stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
            stanox_section.rename(columns={'StartStanox_temp': 'StartLocation', 'EndStanox_temp': 'EndLocation'},
                                  inplace=True)
            stanox_section = stanox_section[['StanoxSection', 'StartLocation', 'StartStanox', 'EndLocation',
                                             'EndStanox', 'LocationId', 'ApproximateLocation']]

            save_pickle(stanox_section, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_section = None

    return stanox_section


# Get TrustIncident
def get_trust_incident(start_year=2006, end_year=None, update=False, save_original_as=None):
    """
    :param start_year: [int; None]
    :param end_year: [int; None]
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
    """
    table_name = 'TrustIncident'
    path_to_pickle = cdd_metex_db_tables(
        table_name + "{}.pickle".format(
            "{}".format("_y{}".format(start_year) if start_year else "_up_to") +
            "{}".format("_y{}".format(2018 if not end_year or end_year >= 2019 else end_year))))
    if os.path.isfile(path_to_pickle) and not update:
        trust_incident = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'TrustIncident' table
            trust_incident = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                              save_as=save_original_as, update=update)
            trust_incident.index.name = 'TrustIncidentId'
            trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
            # Extract a subset of data, in which the StartDateTime is between 'start_year' and 'end_year'?
            trust_incident = trust_incident[
                (trust_incident.FinancialYear >= (start_year if start_year else 0)) &
                (trust_incident.FinancialYear <= (end_year if end_year else pd.datetime.now().year))]
            # Convert float to int values for 'SourceLocationId'
            trust_incident.SourceLocationId = trust_incident.SourceLocationId.map(lambda x: 0 if pd.isna(x) else int(x))
            save_pickle(trust_incident, path_to_pickle)
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


# Get WeatherCell
def get_weather_cell(update=False, save_original_as=None,
                     show_map=False, projection='tmerc', save_map_as=".png", dpi=600):
    """
    :param update: [bool]
    :param save_original_as: [str; None]
    :param show_map: [bool]
    :param projection: [str]
    :param save_map_as: [str; None]
    :param dpi: [int; None]
    :return: [pandas.DataFrame]
    """
    table_name = 'WeatherCell'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        weather_cell_map = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'WeatherCell' table
            weather_cell_map = read_metex_table(table_name, index_col=get_metex_table_pk(table_name),
                                                save_as=save_original_as, update=update)
            weather_cell_map.index.rename('WeatherCellId', inplace=True)  # Rename index

            # Lower left corner:
            weather_cell_map['ll_Longitude'] = weather_cell_map.Longitude  # - weather_cell_map.width / 2
            weather_cell_map['ll_Latitude'] = weather_cell_map.Latitude  # - weather_cell_map.height / 2
            # Upper left corner:
            weather_cell_map['ul_Longitude'] = weather_cell_map.ll_Longitude  # - cell['width'] / 2
            weather_cell_map['ul_Latitude'] = weather_cell_map.ll_Latitude + weather_cell_map.height  # / 2
            # Upper right corner:
            weather_cell_map['ur_Longitude'] = weather_cell_map.ul_Longitude + weather_cell_map.width  # / 2
            weather_cell_map['ur_Latitude'] = weather_cell_map.ul_Latitude  # + weather_cell_map.height / 2
            # Lower right corner:
            weather_cell_map['lr_Longitude'] = weather_cell_map.ur_Longitude  # + weather_cell_map.width  # / 2
            weather_cell_map['lr_Latitude'] = weather_cell_map.ur_Latitude - weather_cell_map.height  # / 2

            # Get IMDM Weather cell map
            imdm_weather_cell_map = get_imdm_weather_cell_map().reset_index()[['WeatherCellId', 'IMDM', 'Route']]
            imdm_weather_cell_map = imdm_weather_cell_map.groupby('WeatherCellId').agg(
                lambda x: x if len(list(set(x))) == 1 else list(set(x)))

            # Merge the acquired data set
            weather_cell_map = weather_cell_map.join(imdm_weather_cell_map)

            # Create polygons (Longitude, Latitude)
            weather_cell_map['Polygon_WGS84'] = weather_cell_map.apply(
                lambda x: shapely.geometry.Polygon(
                    zip([x.ll_Longitude, x.ul_Longitude, x.ur_Longitude, x.lr_Longitude],
                        [x.ll_Latitude, x.ul_Latitude, x.ur_Latitude, x.lr_Latitude])), axis=1)

            weather_cell_map[['ll_Easting', 'll_Northing']] = weather_cell_map[['ll_Longitude', 'll_Latitude']].apply(
                lambda x: pd.Series(wgs84_to_osgb36(x.ll_Longitude, x.ll_Latitude)), axis=1)
            weather_cell_map[['ul_Easting', 'ul_Northing']] = weather_cell_map[['ul_Longitude', 'ul_Latitude']].apply(
                lambda x: pd.Series(wgs84_to_osgb36(x.ul_Longitude, x.ul_Latitude)), axis=1)
            weather_cell_map[['ur_Easting', 'ur_Northing']] = weather_cell_map[['ur_Longitude', 'ur_Latitude']].apply(
                lambda x: pd.Series(wgs84_to_osgb36(x.ur_Longitude, x.ur_Latitude)), axis=1)
            weather_cell_map[['lr_Easting', 'lr_Northing']] = weather_cell_map[['lr_Longitude', 'lr_Latitude']].apply(
                lambda x: pd.Series(wgs84_to_osgb36(x.lr_Longitude, x.lr_Latitude)), axis=1)

            weather_cell_map['Polygon_OSGB36'] = weather_cell_map.apply(
                lambda x: shapely.geometry.Polygon(
                    zip([x.ll_Easting, x.ul_Easting, x.ur_Easting, x.lr_Easting],
                        [x.ll_Northing, x.ul_Northing, x.ur_Northing, x.lr_Northing])), axis=1)

            save_pickle(weather_cell_map, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_cell_map = None

    # Plot the Weather cells on the map?
    if show_map:
        print("Plotting the map ...", end="")

        plt.figure(figsize=(5, 8))

        min_val, max_val = weather_cell_map.min(), weather_cell_map.max()
        boundary_box = ((min(min_val.ll_Longitude, min_val.ul_Longitude),
                         min(min_val.ll_Latitude, min_val.ul_Latitude)),
                        (max(max_val.lr_Longitude, max_val.ur_Longitude),
                         max(max_val.lr_Latitude, max_val.ur_Latitude)))

        base_map = mpl_toolkits.basemap.Basemap(projection=projection,  # Transverse Mercator Projection
                                                ellps='WGS84',
                                                epsg=27700,
                                                llcrnrlon=boundary_box[0][0] - 0.285,
                                                llcrnrlat=boundary_box[0][1] - 0.255,
                                                urcrnrlon=boundary_box[1][0] + 1.185,
                                                urcrnrlat=boundary_box[1][1] + 0.255,
                                                lat_ts=0,
                                                resolution='h',
                                                suppress_ticks=True)
        # base_map.drawmapboundary(color='none', fill_color='white')
        # base_map.drawcoastlines()
        # base_map.fillcontinents()
        base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)

        weather_cell_map_plot = weather_cell_map.drop_duplicates([s for s in weather_cell_map.columns if '_' in s and
                                                                  not s.startswith('Polygon')])

        for i in weather_cell_map_plot.index:
            ll_x, ll_y = base_map(weather_cell_map_plot.ll_Longitude[i], weather_cell_map_plot.ll_Latitude[i])
            ul_x, ul_y = base_map(weather_cell_map_plot.ul_Longitude[i], weather_cell_map_plot.ul_Latitude[i])
            ur_x, ur_y = base_map(weather_cell_map_plot.ur_Longitude[i], weather_cell_map_plot.ur_Latitude[i])
            lr_x, lr_y = base_map(weather_cell_map_plot.lr_Longitude[i], weather_cell_map_plot.lr_Latitude[i])
            xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
            polygons = matplotlib.patches.Polygon(list(xy), fc='#D5EAFF', ec='#4b4747', alpha=0.5)
            plt.gca().add_patch(polygons)
        plt.plot([], 's', label="Weather cell", ms=14, color='#D5EAFF', markeredgecolor='#4b4747')
        legend = plt.legend(numpoints=1, loc='best', fancybox=True, labelspacing=0.5)
        frame = legend.get_frame()
        frame.set_edgecolor('k')
        plt.tight_layout()

        # # Plot points
        # from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
        # ll_lon_lat = [Point(base_map(lon, lat))
        #               for lon, lat in zip(weather_cell_map_plot.ll_Longitude, weather_cell_map_plot.ll_Latitude)]
        # ur_lon_lat = [Point(base_map(lon, lat))
        #               for lon, lat in zip(weather_cell_map_plot.ur_Longitude, weather_cell_map_plot.ur_Latitude)]
        # map_points = MultiPoint(ll_lon_lat + ur_lon_lat)
        # base_map.scatter([geom.x for geom in map_points], [geom.y for geom in map_points],
        #                  marker='x', s=16, lw=1, facecolor='#5a7b6c', edgecolor='w', label='Hazardous tress',
        #                  alpha=0.6, antialiased=True, zorder=3)
        #
        # # Plot squares
        # for i in range(len(cell)):
        #     ll_x, ll_y = base_map(weather_cell_map_plot.ll_Longitude[i], weather_cell_map_plot.ll_Latitude[i])
        #     base_map.plot([ll_x, ul_x], [ll_y, ul_y], color='#5a7b6c')
        #     ul_x, ul_y = base_map(weather_cell_map_plot.ul_Longitude[i], weather_cell_map_plot.ul_Latitude[i])
        #     base_map.plot([ul_x, ur_x], [ul_y, ur_y], color='#5a7b6c')
        #     ur_x, ur_y = base_map(weather_cell_map_plot.ur_Longitude[i], weather_cell_map_plot.ur_Latitude[i])
        #     base_map.plot([ur_x, lr_x], [ur_y, lr_y], color='#5a7b6c')
        #     lr_x, lr_y = base_map(weather_cell_map_plot.lr_Longitude[i], weather_cell_map_plot.lr_Latitude[i])
        #     base_map.plot([lr_x, ll_x], [lr_y, ll_y], color='#5a7b6c')

        print("Done.")

        if save_map_as:
            save_fig(cdd_metex_fig_db(table_name + save_map_as), dpi=dpi)

    return weather_cell_map


# Get the lower-left and upper-right boundaries of Weather cells
def get_weather_cell_map_boundary(route=None, adjustment=(0.285, 0.255)):
    weather_cell = get_weather_cell()  # Get Weather cell
    if route:  # For a specific Route
        weather_cell = weather_cell[weather_cell.Route == fuzzywuzzy.process.extractOne(route, get_route().Route)[0]]
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
    return shapely.geometry.Polygon((ll, lr, ur, ul))


# Track
def get_track(update=False, save_original_as=None):
    """
    :param update: [bool]
    :param save_original_as: [str; None]
    :return: [pandas.DataFrame; None]
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
                                  'MAINTAINER': 'Maintainer', 'ROUTE': 'RouteAlias', 'DELIVERY_U': 'IMDM',
                                  'StartEasti': 'StartEasting', 'StartNorth': 'StartNorthing',
                                  'EndNorthin': 'EndNorthing'},
                         inplace=True)
            # Mileage and Yardage
            track[['StartMileage', 'EndMileage']] = track[['StartMileage', 'EndMileage']].applymap(
                nr_mileage_num_to_str)
            track[['StartYardage', 'EndYardage']] = track[['StartYardage', 'EndYardage']].applymap(int)
            # Route
            route_names_changes = load_json(cdd("Network\\Routes", "route-names-changes.json"))
            track['Route'] = track.RouteAlias.replace(route_names_changes)
            # Delivery Unit and IMDM
            track.IMDM = track.IMDM.map(lambda x: 'IMDM ' + x)
            # Coordinates
            track[['StartLongitude', 'StartLatitude']] = track[['StartEasting', 'StartNorthing']].apply(
                lambda x: pd.Series(osgb36_to_wgs84(x.StartEasting, x.StartNorthing)), axis=1)
            track[['EndLongitude', 'EndLatitude']] = track[['EndEasting', 'EndNorthing']].apply(
                lambda x: pd.Series(osgb36_to_wgs84(x.EndEasting, x.EndNorthing)), axis=1)
            save_pickle(track, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            track = None
    return track


# Track Summary
def get_track_summary(update=False, save_original_as=None):
    table_name = 'Track Summary'
    path_to_pickle = cdd_metex_db_tables(table_name.replace(' ', '') + ".pickle")
    if os.path.isfile(path_to_pickle) and not update:
        track_summary = load_pickle(path_to_pickle)
    else:
        try:
            track_summary = read_metex_table(table_name, save_as=save_original_as, update=update)
            # Column names
            rename_cols = {'Sub-route': 'SubRoute',
                           'CP6 criticality': 'CP6Criticality', 'CP5 Start Route': 'CP5StartRoute',
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
            route_names_changes = load_json(cdd("Network\\Routes", "route-names-changes.json"))
            temp1 = pd.DataFrame.from_dict(route_names_changes, orient='index', columns=['Route'])
            route_names_in_table = list(track_summary.SubRoute.unique())
            route_alt = [fuzzywuzzy.process.extractOne(x, temp1.index)[0] for x in route_names_in_table]
            temp2 = pd.DataFrame.from_dict(dict(zip(route_alt, route_names_in_table)), 'index', columns=['RouteAlias'])
            temp = temp1.join(temp2).dropna()
            route_names_changes_alt = dict(zip(temp.RouteAlias, temp.Route))
            track_summary['Route'] = track_summary.SubRoute.replace(route_names_changes_alt)
            save_pickle(track_summary, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            track_summary = None
    return track_summary


# ====================================================================================================================
""" Utils for creating views """


# Form a file name in terms of specific 'Route' and 'Weather' category
def make_filename(base_name, route_name, weather_category, *extra_suffixes, sep="_", save_as=".pickle"):
    if route_name:
        route_name = fuzzywuzzy.process.extractOne(route_name, get_route().Route, scorer=fuzzywuzzy.fuzz.ratio)[0]
    if weather_category:
        weather_category = fuzzywuzzy.process.extractOne(weather_category, get_weather_codes().WeatherCategory,
                                                         scorer=fuzzywuzzy.fuzz.ratio)[0]
    filename_suffix = [s for s in (route_name, weather_category) if s]  # "s" stands for "suffix"
    filename = sep.join([base_name] + filename_suffix + [str(s) for s in extra_suffixes if s]) + save_as
    return filename


# Subset the required data given 'route' and 'Weather'
def subset(data, route_name=None, weather_category=None, reset_index=False):
    if data is not None:
        assert 'Route' in data.columns and 'WeatherCategory' in data.columns
        route_name = fuzzywuzzy.process.extractOne(route_name, list(set(data.Route)), scorer=fuzzywuzzy.fuzz.ratio)[0] \
            if route_name else None
        weather_category = fuzzywuzzy.process.extractOne(weather_category, list(set(data.WeatherCategory)),
                                                         scorer=fuzzywuzzy.fuzz.ratio)[0] \
            if weather_category else None
        # Select data for a specific route and Weather category
        if route_name and weather_category:
            data_subset = data[(data.Route == route_name) & (data.WeatherCategory == weather_category)]
        elif route_name and not weather_category:
            data_subset = data[data.Route == route_name]
        elif not route_name and weather_category:
            data_subset = data[data.WeatherCategory == weather_category]
        else:
            data_subset = data
        if reset_index:
            data_subset.reset_index(inplace=True)  # dat.index = range(len(dat))
    else:
        data_subset = None
    return data_subset


# Calculate the DelayMinutes and DelayCosts for grouped data
def pfpi_stats(dat, selected_features, sort_by=None):
    data = dat.groupby(selected_features[1:-2]).aggregate({
        # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
        'PfPIId': pd.np.count_nonzero,
        'PfPIMinutes': pd.np.sum,
        'PfPICosts': pd.np.sum})
    data.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']
    # data = dat.groupby(selected_features[1:-2]).aggregate({
    #     # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
    #     'PfPIId': {'IncidentCount': np.count_nonzero},
    #     'PfPIMinutes': {'DelayMinutes': np.sum},
    #     'PfPICosts': {'DelayCost': np.sum}})
    # data.columns = data.columns.droplevel(0)
    data.reset_index(inplace=True)  # Reset the grouped indexes to columns
    if sort_by:
        data.sort_values(sort_by, inplace=True)
    return data


# ====================================================================================================================
""" Get views based on the NR_METEX data """


# Get Weather data by 'WeatherCell' and 'DateTime'
def view_weather_by_id_datetime(weather_cell_id, start_dt=None, end_dt=None, postulate=False,
                                pickle_it=True, dat_dir=None, update=False):
    """
    :param weather_cell_id: [int]
    :param start_dt: [datetime.datetime; str; None] e.g. pd.datetime(2019, 5, 1, 12), '2019-05-01 12:00:00'
    :param end_dt: [datetime.datetime; None] e.g. pd.datetime(2019, 5, 1, 12), '2019-05-01 12:00:00'
    :param postulate: [bool]
    :param pickle_it: [bool]
    :param dat_dir: [str; None]
    :param update: [bool]
    :return: [pandas.DataFrame; None]
    """
    # assert all(isinstance(x, pd.np.int64) for x in weather_cell_id)
    assert isinstance(weather_cell_id, tuple) or isinstance(weather_cell_id, (int, pd.np.integer))
    pickle_filename = "{}{}{}.pickle".format(
        "_".join(str(x) for x in list(weather_cell_id)) if isinstance(weather_cell_id, tuple) else weather_cell_id,
        start_dt.strftime('_fr%Y%m%d%H%M') if start_dt else "", end_dt.strftime('_to%Y%m%d%H%M') if end_dt else "")
    dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else cdd_metex_db_views()
    path_to_pickle = cd(dat_dir, pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        weather = load_pickle(path_to_pickle)

    else:
        try:
            conn_metex = establish_mssql_connection(database_name='NR_METEX_20190203')
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

            # Save the processed data
            if pickle_it:
                save_pickle(weather, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(os.path.basename(path_to_pickle))[0], e))
            weather = None

    return weather


# Retrieve the TRUST
def merge_schedule8_data(weather_attributed=False, save_as=".pickle"):
    """
    :param weather_attributed: [bool]
    :param save_as: [str]
    :return: [pandas.DataFrame]
    """
    try:
        pfpi = get_pfpi(plus=True)  # Get PfPI # (260645, 6)  # (5165562, 6)
        incident_record = get_incident_record()  # (233452, 4)  # (4661505, 5)
        trust_incident = get_trust_incident(start_year=2006)  # (192054, 11)  # (3988006, 11)
        location = get_location()  # (228851, 6)  # (653882, 7)
        imdm = get_imdm(as_dict=False)  # (42, 1)  # (42, 2)
        incident_reason_info = get_incident_reason_info(plus=True)  # (393, 7)  # (174, 9)
        stanox_location = get_stanox_location(use_nr_mileage_format=True)  # (7560, 5)  # (7560, 6)
        stanox_section = get_stanox_section()  # (9440, 7)  # (10601, 7)

        # Merge the acquired data sets
        data = pfpi. \
            join(incident_record,  # (260645, 10)  # (5165562, 11)
                 on='IncidentRecordId', how='inner'). \
            join(trust_incident,  # (260483, 21)  # (5165538, 22)
                 on='TrustIncidentId', how='inner'). \
            join(stanox_section,  # (260483, 28)  # (5165538, 29)
                 on='StanoxSectionId', how='inner'). \
            join(location,  # (260470, 34)  # (5162033, 36)
                 on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
            join(stanox_location,  # (260190, 39)  # (5154847, 42)
                 on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
            join(stanox_location,  # (260140, 44)  # (5155047, 48)
                 on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
            join(incident_reason_info,  # (260140, 51)  # (5155047, 57)
                 on='IncidentReasonCode', how='inner'). \
            join(imdm, on='IMDM_Location', how='inner')

        if weather_attributed:
            data = data[data.WeatherCategory != '']
            data.sort_index(inplace=True)
            filename = "Schedule8_details_weather_attributed"
        else:
            filename = "Schedule8_details"

        # Note: There may be errors in e.g. IMDM data/column, location id, of the TrustIncident table.

        # # Get a ELR-IMDM-Route "dictionary" from Vegetation database
        # route_du_elr = get_furlong_location(useful_columns_only=True)[['Route', 'ELR', 'DU']].drop_duplicates()
        # route_du_elr.index = range(len(route_du_elr))  # (1276, 3)
        #
        # # Further cleaning the data
        # data.reset_index(inplace=True)
        # temp = data.merge(route_du_elr, how='left', left_on=['ELR_Start', 'IMDM'], right_on=['ELR', 'DU'])
        # temp = temp.merge(route_du_elr, how='left', left_on=['ELR_End', 'IMDM'], right_on=['ELR', 'DU'])
        #
        # temp[['Route_', 'IMDM_']] = temp[['Route_x', 'DU_x']]
        # idx_x = (temp.Route_x.isnull()) & (~temp.Route_y.isnull())
        # temp.loc[idx_x, 'Route_'], temp.loc[idx_x, 'IMDM_'] = temp.Route_y.loc[idx_x], temp.DU_y.loc[idx_x]
        #
        # idx_y = (temp.Route_x.isnull()) & (temp.Route_y.isnull())
        # temp.loc[idx_y, 'IMDM_'] = temp.IMDM_Location.loc[idx_y]
        # temp.loc[idx_y, 'Route_'] = temp.loc[idx_y, 'IMDM_'].replace(imdm.to_dict()['Route'])
        #
        # temp.drop(labels=['Route_x', 'ELR_x', 'DU_x', 'Route_y', 'ELR_y', 'DU_y',
        #                   'StanoxSection_Start', 'StanoxSection_End',
        #                   'IMDM', 'IMDM_Location'], axis=1, inplace=True)

        i = data[~data.StartLocation.eq(data.Location_Start)].index
        for i in i:
            data.loc[i, 'StartLocation'] = data.loc[i, 'Location_Start']
            data.loc[i, 'EndLocation'] = data.loc[i, 'Location_End']
            if data.loc[i, 'StartLocation'] == data.loc[i, 'EndLocation']:
                data.loc[i, 'StanoxSection'] = data.loc[i, 'StartLocation']
            else:
                data.loc[i, 'StanoxSection'] = data.loc[i, 'StartLocation'] + ' - ' + data.loc[i, 'EndLocation']

        data.drop(['IMDM', 'Location_Start', 'Location_End'], axis=1, inplace=True)

        # (260140, 50)  # (5155014, 56)
        data.rename(columns={'LocationAlias_Start': 'StartLocationAlias', 'LocationAlias_End': 'EndLocationAlias',
                             'ELR_Start': 'StartELR', 'Yards_Start': 'StartYards',
                             'ELR_End': 'EndELR', 'Yards_End': 'EndYards',
                             'Mileage_Start': 'StartMileage', 'Mileage_End': 'EndMileage',
                             'LocationId_Start': 'StartLocationId', 'LocationId_End': 'EndLocationId',
                             'LocationId_Section': 'SectionLocationId', 'IMDM_Location': 'IMDM',
                             'StartDate': 'StartDateTime', 'EndDate': 'EndDateTime'}, inplace=True)

        # Use 'Station' data from Railway Codes website
        station_locations = get_station_locations()['Station'][['Station', 'Degrees Longitude', 'Degrees Latitude']]
        station_locations = station_locations.dropna().drop_duplicates('Station', keep='first')
        station_locations.set_index('Station', inplace=True)
        temp = data[['StartLocation']].join(station_locations, on='StartLocation', how='left')
        i = temp[temp['Degrees Longitude'].notna()].index
        data.loc[i, 'StartLongitude':'StartLatitude'] = \
            temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()
        temp = data[['EndLocation']].join(station_locations, on='EndLocation', how='left')
        i = temp[temp['Degrees Longitude'].notna()].index
        data.loc[i, 'EndLongitude':'EndLatitude'] = temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()

        # data.EndELR.replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)
        i = data.StartLocation == 'Highbury & Islington (North London Lines)'
        data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.1045, 51.5460]
        i = data.EndLocation == 'Highbury & Islington (North London Lines)'
        data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.1045, 51.5460]
        i = data.StartLocation == 'Dalston Junction (East London Line)'
        data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.0751, 51.5461]
        i = data.EndLocation == 'Dalston Junction (East London Line)'
        data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.0751, 51.5461]

        # # Sort the merged data frame by index 'PfPIId'
        # data = data.set_index('PfPIId').sort_index()  # (260140, 49)
        #
        # # Further cleaning the 'IMDM' and 'Route'
        # for section in data.StanoxSection.unique():
        #     temp = data[data.StanoxSection == section]
        #     # IMDM
        #     if len(temp.IMDM.unique()) >= 2:
        #         imdm_temp = data.loc[temp.index].IMDM.value_counts()
        #         data.loc[temp.index, 'IMDM'] = imdm_temp[imdm_temp == imdm_temp.max()].index[0]
        #     # Route
        #     if len(temp.Route.unique()) >= 2:
        #         route_temp = data.loc[temp.index].Route.value_counts()
        #         data.loc[temp.index, 'Route'] = route_temp[route_temp == route_temp.max()].index[0]

        save_pickle(data, cdd_metex_db_views(filename + save_as))

    except Exception as e:
        print("Failed to merge Schedule 8 details. {}".format(e))
        data = None

    return data


# Get the TRUST data
def view_schedule8_details(route_name=None, weather_category=None, reset_index=False, weather_attributed=False,
                           update=False, pickle_it=True):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param reset_index: [bool]
    :param weather_attributed: [bool]
    :param update: [bool]
    :param pickle_it: [bool]
    :return: [pandas.DataFrame]
    """
    filename = "Schedule8_details" + ("_weather_attributed" if weather_attributed else "")
    pickle_filename = make_filename(filename, route_name, weather_category, save_as=".pickle")
    path_to_pickle = cdd_metex_db_views(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        schedule8_details = load_pickle(path_to_pickle)
        if reset_index and schedule8_details.index.name == 'PfPIId':
            schedule8_details.reset_index(inplace=True)
    else:
        try:
            path_to_merged = cdd_metex_db_views("{}.pickle".format(filename))
            if not os.path.isfile(path_to_merged) or update:
                schedule8_details = merge_schedule8_data(weather_attributed, save_as=".pickle")
            else:
                schedule8_details = load_pickle(path_to_merged)
            schedule8_details = subset(schedule8_details, route_name, weather_category, reset_index)
            if pickle_it:
                if path_to_pickle != path_to_merged:
                    save_pickle(schedule8_details, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve details about Schedule 8 incidents. {}".format(e))
            schedule8_details = None
    return schedule8_details


# Essential details about Incidents
def view_schedule8_details_pfpi(route_name=None, weather_category=None, update=False, pickle_it=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it:
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_details_pfpi", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get merged data sets
            schedule8_data = view_schedule8_details(route_name, weather_category, reset_index=True)
            # Select a list of columns
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
                'IncidentReasonCode', 'IncidentReasonDescription', 'IncidentCategory', 'IncidentCategoryDescription',
                # 'IncidentCategoryGroupDescription',
                'IncidentFMS', 'IncidentEquipment',
                'WeatherCell',
                'Route', 'IMDM',
                'StanoxSection', 'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'ApproximateLocation']
            data = schedule8_data[selected_features]
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 data by incident location and Weather category
def view_schedule8_cost_by_location(route_name=None, weather_category=None, update=False, pickle_it=True):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it: [bool]
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_location", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'WeatherCategory',
                'Route', 'IMDM',
                'StanoxSection',
                'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features)
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 data by datetime and location
def view_schedule8_cost_by_datetime_location(route_name=None, weather_category=None, update=False, pickle_it=True):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it:
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_datetime_location", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDateTime', 'EndDateTime',
                'WeatherCategory',
                'StanoxSection',
                'Route', 'IMDM',
                'StartLocation', 'EndLocation',
                'StartStanox', 'EndStanox',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features, sort_by=['StartDateTime', 'EndDateTime'])
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 cost by datetime, location and incident reason
def view_schedule8_cost_by_datetime_location_reason(route_name=None, weather_category=None, update=False,
                                                    pickle_it=True):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it: [bool]
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_datetime_location_reason", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True,
                                                       weather_attributed=False)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'StartDateTime', 'EndDateTime',
                                 'WeatherCategory',
                                 'WeatherCell',
                                 'Route', 'IMDM',
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
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features, sort_by=['StartDateTime', 'EndDateTime'])
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 data by datetime and Weather category
def view_schedule8_cost_by_datetime(route_name=None, weather_category=None, update=False, pickle_it=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it:
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_datetime", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDateTime', 'EndDateTime',
                'WeatherCategory',
                'Route', 'IMDM',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features, sort_by=['StartDateTime', 'EndDateTime'])
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 cost by incident reason
def view_schedule8_cost_by_reason(route_name=None, weather_category=None, update=False, pickle_it=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it: [bool]
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_reason", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'Route',
                                 # 'IMDM',
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
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features)
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 cost by location and incident reason
def view_schedule8_cost_by_location_reason(route_name=None, weather_category=None, update=False, pickle_it=False):
    filename = make_filename("Schedule8_costs_by_location_reason", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'WeatherCategory',
                                 'Route', 'IMDM',
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
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features)
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data


# Get Schedule 8 cost by Weather category
def view_schedule8_cost_by_weather_category(route_name=None, weather_category=None, update=False, pickle_it=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param update: [bool]
    :param pickle_it: [bool]
    :return: [pandas.DataFrame]
    """
    filename = make_filename("Schedule8_costs_by_weather_category", route_name, weather_category)
    path_to_pickle = cdd_metex_db_views(filename)
    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = view_schedule8_details(route_name, weather_category, reset_index=True)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'Route',
                                 'IMDM',
                                 'WeatherCategory',
                                 'PfPICosts',
                                 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = pfpi_stats(schedule8_data, selected_features)
            if pickle_it:
                save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to retrieve \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None
    return data
