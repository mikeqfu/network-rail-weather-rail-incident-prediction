""" Read and clean data of NR_METEX database """

import os

import datetime_truncate
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import pandas as pd
import shapely.geometry
from fuzzywuzzy.process import extractOne

import database_utils as db
import database_veg as dbv
import railwaycodes_utils as rc
from converters import yards_to_mileage
from delay_attr_glossary import get_incident_reason_metadata, get_performance_event_code
from loc_code_dict import create_location_names_regexp_replacement_dict, create_location_names_replacement_dict
from utils import cd, cdd, cdd_rc, find_match, load_json, load_pickle, save, save_pickle

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\METEX\\Database\\Tables" and sub-directories
def cdd_metex_db_tables(*directories):
    path = db.cdd_metex_db("Tables")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Data\\METEX\\Database\\Views" and sub-directories
def cdd_metex_db_views(*directories):
    path = db.cdd_metex_db("Views")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "METEX\\Database\\Figures" and sub-directories
def cdd_metex_db_fig(*directories):
    path = db.cdd_metex_db("Figures")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Publications\\Journals\\Figures" and sub-directories
def cdd_metex_db_fig_pub(pid, *directories):
    path = cd("Publications", "Journals", "{}".format(pid), "Figures")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Get primary keys of a table in database NR_METEX
def metex_pk(table_name):
    pri_key = db.get_pri_keys(db_name="NR_METEX", table_name=table_name)
    return pri_key


# Transform a DataFrame to dictionary
def group_items(data_frame, by, to_group, group_name, level=None, as_dict=False):
    # Create a dictionary
    temp_obj = data_frame.groupby(by, level=level)[to_group]
    d = {group_name: {k: list(v) for k, v in temp_obj}}
    if as_dict:
        return d
    else:
        d_df = pd.DataFrame(d)
        d_df.index.name = by
        return d_df


# ====================================================================================================================
""" Get table data from the NR_METEX database """


# Get IMDM
def get_imdm(as_dict=False, update=False):
    table_name = 'IMDM'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_pickle) and not update:
        imdm = load_pickle(path_to_pickle)
    else:
        try:
            imdm = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            imdm.index.rename(name='IMDM', inplace=True)  # Rename a column and index
            imdm.rename(columns={'Name': 'IMDM'}, inplace=True)
            if as_dict:
                imdm_dict = imdm.to_dict()
                imdm = imdm_dict['Route']
                imdm.pop('None', None)
            save_pickle(imdm, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            imdm = pd.DataFrame()

    return imdm


# Get ImdmAlias
def get_imdm_alias(as_dict=False, update=False):
    table_name = 'ImdmAlias'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_pickle) and not update:
        imdm_alias = load_pickle(path_to_pickle)
    else:
        try:
            imdm_alias = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            imdm_alias.rename(columns={'Imdm': 'IMDM'}, inplace=True)  # Rename a column
            imdm_alias.index.rename(name='ImdmAlias', inplace=True)  # Rename index
            if as_dict:
                imdm_alias_dict = imdm_alias.to_dict()
                imdm_alias = imdm_alias_dict['IMDM']
            save_pickle(imdm_alias, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            imdm_alias = None

    return imdm_alias


# Get IMDMWeatherCellMap
def get_imdm_weather_cell_map(grouped=False, update=False):
    table_name = 'IMDMWeatherCellMap'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if grouped:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_grouped")

    if os.path.isfile(path_to_pickle) and not update:
        weather_cell_map = load_pickle(path_to_pickle)
    else:
        try:
            # Read IMDMWeatherCellMap table
            weather_cell_map = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)

            weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'}, inplace=True)  # Rename a column
            weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index

            if grouped:  # Transform the dataframe into a dictionary-like form
                weather_cell_map = group_items(weather_cell_map, by='WeatherCellId', to_group='IMDM', group_name='IMDM')

            save_pickle(weather_cell_map, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_cell_map = None

    return weather_cell_map


# Get IncidentReasonInfo
def get_incident_reason_info(database_plus=True, update=False):
    table_name = 'IncidentReasonInfo'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")
    if database_plus:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_plus")

    if os.path.isfile(path_to_pickle) and not update:
        incident_reason_info = load_pickle(path_to_pickle)
    else:
        try:
            # Get data from the database
            incident_reason_info = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename columns
            incident_reason_info.rename(columns={'Description': 'IncidentReasonDescription',
                                                 'Category': 'IncidentCategory',
                                                 'CategoryDescription': 'IncidentCategoryDescription'}, inplace=True)
            # Rename index label
            incident_reason_info.index.rename('IncidentReason', inplace=True)

            if database_plus:
                incident_reason_metadata = get_incident_reason_metadata()
                incident_reason_metadata.index.name = 'IncidentReason'
                incident_reason_metadata.columns = [x.replace('_', '') for x in incident_reason_metadata.columns]
                incident_reason_info = incident_reason_metadata.join(incident_reason_info, rsuffix='_orig')
                incident_reason_info.dropna(axis=1, inplace=True)

            save_pickle(incident_reason_info, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            incident_reason_info = None

    return incident_reason_info


# Get WeatherCategoryLookup
def get_weather_category_lookup(as_dict=False, update=False):
    table_name = 'WeatherCategoryLookup'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if as_dict:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_dict")

    if os.path.isfile(path_to_pickle) and not update:
        weather_category_lookup = load_pickle(path_to_pickle)
    else:
        try:
            weather_category_lookup = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename a column and index label
            weather_category_lookup.rename(columns={'Name': 'WeatherCategory'}, inplace=True)
            weather_category_lookup.index.rename(name='WeatherCategoryCode', inplace=True)
            # Transform the DataFrame to a dictionary?
            if as_dict:
                weather_category_lookup = weather_category_lookup.to_dict()
            save_pickle(weather_category_lookup, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_category_lookup = None

    return weather_category_lookup


# Get IncidentRecord and fill 'None' value with NaN
def get_incident_record(update=False):
    table_name = 'IncidentRecord'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        incident_record = load_pickle(path_to_pickle)
    else:
        try:
            # Read the 'IncidentRecord' table
            incident_record = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename column names
            incident_record.rename(columns={'CreateDate': 'IncidentRecordCreateDate',
                                            'Reason': 'IncidentReason'}, inplace=True)
            # Rename index name
            incident_record.index.rename('IncidentRecordId', inplace=True)
            # Get a weather category lookup dictionary
            weather_category_lookup = get_weather_category_lookup(as_dict=True)
            # Replace the weather category code with the corresponding full name
            incident_record.replace(weather_category_lookup, inplace=True)
            incident_record.fillna(value='', inplace=True)
            # Save the data
            save_pickle(incident_record, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            incident_record = None

    return incident_record


# Get Location
def get_location(update=False):
    table_name = 'Location'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        location = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'Location' table
            location = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename a column and index label
            location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
            location.index.rename('LocationId', inplace=True)
            # location['WeatherCell'].fillna(value='', inplace=True)
            location.WeatherCell = location.WeatherCell.apply(lambda x: '' if pd.np.isnan(x) else int(x))
            location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]
            # Save the data
            save_pickle(location, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            location = None

    return location


# Get PfPI (Process for Performance Improvement)
def get_pfpi(update=False):
    table_name = 'PfPI'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        pfpi = load_pickle(path_to_pickle)
    else:
        try:
            # Read the 'PfPI' table
            pfpi = db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename a column name
            pfpi.index.rename('PfPIId', inplace=True)
            # To replace Performance Event Code
            performance_event_code = get_performance_event_code()
            performance_event_code.index.name = 'PerformanceEventCode'
            performance_event_code.columns = [x.replace('_', '') for x in performance_event_code.columns]
            # Merge pfpi and pe_code
            pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')
            # Change columns' order
            cols = pfpi.columns.tolist()
            pfpi = pfpi[cols[0:2] + cols[-2:] + cols[2:4]]
            # Save the data
            save_pickle(pfpi, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            pfpi = None

    return pfpi


# Get Route (Note that there is only one column in the original table)
def get_route(update=False):
    table_name = "Route"
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        route = load_pickle(path_to_pickle)
    else:
        try:
            route = db.read_metex_table(table_name, save_as=".csv", update=update)
            # Rename a column
            route.rename(columns={'Name': 'Route'}, inplace=True)
            # Save the processed data
            save_pickle(route, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            route = None

    return route


# Get StanoxLocation
def get_stanox_location(nr_mileage_format=True, update=False):
    table_name = 'StanoxLocation'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if not nr_mileage_format:
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_miles")

    if os.path.isfile(path_to_pickle) and not update:
        stanox_location = load_pickle(path_to_pickle)
    else:
        try:
            # Read StanoxLocation table from the database
            stanox_location = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)

            # Cleanse stanox_location ---------------------------
            stanox_location.reset_index(inplace=True)
            location_codes = rc.get_location_codes()['Locations']

            errata = load_json(cdd_rc("errata.json"))  # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
            errata_stanox, errata_tiploc, errata_stanme = errata.values()
            stanox_location.Stanox = stanox_location.Stanox.replace(errata_stanox)
            stanox_location.Description = stanox_location.Description.replace(errata_tiploc)
            stanox_location.Name = stanox_location.Name.replace(errata_stanme)

            #
            na_desc = stanox_location.Description.isnull()
            for i, v in stanox_location[na_desc].Stanox.iteritems():
                idx = location_codes[location_codes.STANOX == v].index
                if len(idx) != 1:
                    print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, v))
                    break
                else:
                    idx = idx[0]
                    stanox_location.loc[i, 'Description'] = location_codes[location_codes.STANOX == v].Location[idx]
                    stanox_location.loc[i, 'Name'] = location_codes[location_codes.STANOX == v].STANME[idx]

            #
            na_name = stanox_location.Name.isnull()
            for i, v in stanox_location[na_name].Stanox.iteritems():
                if location_codes[location_codes.STANOX == v].shape[0] > 1:
                    desc = stanox_location[stanox_location.Stanox == v].Description[i]
                    if desc in list(location_codes[location_codes.STANOX == v].TIPLOC):
                        idx = location_codes[(location_codes.STANOX == v) & (location_codes.TIPLOC == desc)].index
                    elif desc in list(location_codes[location_codes.STANOX == v].STANME):
                        idx = location_codes[(location_codes.STANOX == v) & (location_codes.STANME == desc)].index
                    else:
                        print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, v))
                        break
                else:
                    idx = location_codes[location_codes.STANOX == v].index
                if len(idx) != 1:
                    print("Errors occur at index \"{}\" where the corresponding STANOX is \"{}\"".format(i, v))
                    break
                else:
                    idx = idx[0]
                stanox_location.loc[i, 'Description'] = location_codes[location_codes.STANOX == v].loc[idx, 'Location']
                stanox_location.loc[i, 'Name'] = location_codes[location_codes.STANOX == v].loc[idx, 'STANME']

            location_stanme_dict = location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
            stanox_location.Name = stanox_location.Name.replace(location_stanme_dict)

            loc_name_replacement_dict = create_location_names_replacement_dict('Description')
            stanox_location = stanox_location.replace(loc_name_replacement_dict)
            loc_name_regexp_replacement_dict = create_location_names_regexp_replacement_dict('Description')
            stanox_location = stanox_location.replace(loc_name_regexp_replacement_dict)

            # STANOX dictionary
            stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'], update=update)
            temp = stanox_location.join(stanox_dict, on='Stanox')[['Description', 'Location']]
            na_loc = temp.Location.isnull()
            temp.loc[na_loc, 'Location'] = temp.loc[na_loc, 'Description']
            stanox_location.Description = temp.apply(
                lambda x: extractOne(x[0], x[1], score_cutoff=10)[0] if isinstance(x[1], list) else x[1], axis=1)

            stanox_location.Name = stanox_location.Name.str.upper()

            location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']]
            location_codes_cut = location_codes_cut.groupby(['STANOX', 'Location']).agg({'STANME': list})
            location_codes_cut.STANME = location_codes_cut.STANME.map(
                lambda x: x if isinstance(x, list) and len(x) > 1 else x[0])
            temp = stanox_location.join(location_codes_cut, on=['Stanox', 'Description'])
            stanox_location.Name = temp.STANME

            # Change location names
            stanox_location.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'}, inplace=True)

            # For 'ELR', replace NaN with ''
            stanox_location.ELR.fillna('', inplace=True)

            # For 'LocationId'
            stanox_location.LocationId = stanox_location.LocationId.map(lambda x: int(x) if not pd.np.isnan(x) else '')

            # For 'Mileages'
            if nr_mileage_format:
                yards = stanox_location.Yards.map(lambda x: yards_to_mileage(x) if not pd.isnull(x) else '')
            else:  # to convert yards to miles (Note: Not the 'mileage' used by Network Rail)
                yards = stanox_location.Yards.map(lambda x: x / 1760 if not pd.isnull(x) else '')
            stanox_location.Yards = yards
            stanox_location.rename(columns={'Yards': 'Mileage'}, inplace=True)

            stanox_location.set_index('Stanox', inplace=True)

            stanox_location.loc['52053', 'ELR':'LocationId'] = ['BOK1', '3.0792', 534877]  # Revise '52053'
            stanox_location.loc['52074', 'ELR':'LocationId'] = ['ELL1', '0.0440', 610096]  # Revise '52074'

            save_pickle(stanox_location, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_location = None

    return stanox_location


# Get StanoxSection
def get_stanox_section(update=False):
    table_name = 'StanoxSection'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        stanox_section = load_pickle(path_to_pickle)
    else:
        try:
            # Read StanoxSection table from the database
            stanox_section = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Pre-cleaning the original data
            stanox_section.LocationId = stanox_section.LocationId.apply(lambda x: int(x) if not pd.np.isnan(x) else '')
            stanox_section.index.name = 'StanoxSectionId'
            # Firstly, create a stanox-to-location dictionary, and replace STANOX with location names
            stanox_loc = get_stanox_location(nr_mileage_format=True)
            stanox_dict_1 = stanox_loc.Location.to_dict()
            stanox_dict_2 = rc.get_location_codes_dictionary(keyword='STANOX', drop_duplicates=False)
            # Processing 'StartStanox'
            stanox_section['StartStanox_loc'] = stanox_section.StartStanox.replace(stanox_dict_1).replace(stanox_dict_2)
            # Processing 'EndStanox'
            stanox_section['EndStanox_loc'] = stanox_section.EndStanox.replace(stanox_dict_1).replace(stanox_dict_2)
            # Secondly, process 'STANME' and 'TIPLOC'
            stanme_dict = rc.get_location_codes_dictionary(keyword='STANME')
            tiploc_dict = rc.get_location_codes_dictionary(keyword='TIPLOC')
            loc_name_replacement_dict = create_location_names_replacement_dict()
            loc_name_regexp_replacement_dict = create_location_names_regexp_replacement_dict()
            # Processing 'StartStanox_loc'
            stanox_section.StartStanox_loc = stanox_section.StartStanox_loc. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_loc'
            stanox_section.EndStanox_loc = stanox_section.EndStanox_loc. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Create 'STANOX' sections
            start_end = stanox_section.StartStanox_loc + ' - ' + stanox_section.EndStanox_loc
            point_idx = stanox_section.StartStanox_loc == stanox_section.EndStanox_loc
            start_end[point_idx] = stanox_section.StartStanox_loc[point_idx]
            stanox_section['StanoxSection'] = start_end
            # Finalising the cleaning process
            stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
            # Rename the columns of the start and end locations
            stanox_section.rename(columns={'StartStanox_loc': 'StanoxSection_Start',
                                           'EndStanox_loc': 'StanoxSection_End'}, inplace=True)
            # Reorder columns
            stanox_section = stanox_section[[
                'StanoxSection', 'StanoxSection_Start', 'StartStanox', 'StanoxSection_End', 'EndStanox',
                'LocationId', 'ApproximateLocation']]
            # Save the data
            save_pickle(stanox_section, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            stanox_section = None

    return stanox_section


# Get TrustIncident
def get_trust_incident(financial_years_06_14=True, update=False):
    table_name = 'TrustIncident'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if financial_years_06_14:  # StartDate is between 01/04/2006 and 31/03/2014
        path_to_pickle = path_to_pickle.replace(table_name, table_name + "_06_14")

    if os.path.isfile(path_to_pickle) and not update:
        trust_incident = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'TrustIncident' table
            trust_incident = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            # Rename column names
            trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
            # Rename index label
            trust_incident.index.name = 'TrustIncidentId'
            # Convert float to int values for 'SourceLocationId'
            trust_incident.SourceLocationId = \
                trust_incident.SourceLocationId.apply(lambda x: '' if pd.isnull(x) else int(x))
            # Retain data of which the StartDate is between 01/04/2006 and 31/03/2014?
            if financial_years_06_14:
                trust_incident = trust_incident[(trust_incident.FinancialYear >= 2006) &
                                                (trust_incident.FinancialYear <= 2014)]
            # Save the processed data
            save_pickle(trust_incident, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            trust_incident = None

    return trust_incident


# Get Weather
def get_weather(update=False):
    table_name = 'Weather'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        weather_data = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'Weather' table
            weather_data = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), update=update)
            # Save original data read from the database (the file is too big)
            if not os.path.isfile(db.cdd_metex_db("Tables_original", table_name + ".csv")):
                save(weather_data, db.cdd_metex_db("Tables_original", table_name + ".csv"))
            # Firstly,
            i = 0
            snowfall, precipitation = weather_data.Snowfall.tolist(), weather_data.TotalPrecipitation.tolist()
            while i + 3 < len(weather_data):
                snowfall[i + 1: i + 3] = pd.np.linspace(snowfall[i], snowfall[i + 3], 4)[1:3]
                precipitation[i + 1: i + 3] = pd.np.linspace(precipitation[i], precipitation[i + 3], 4)[1:3]
                i += 3
            # Secondly,
            if i + 2 == len(weather_data):
                snowfall[-1:], precipitation[-1:] = snowfall[-2], precipitation[-2]
            elif i + 3 == len(weather_data):
                snowfall[-2:], precipitation[-2:] = [snowfall[-3]] * 2, [precipitation[-3]] * 2
            else:
                pass
            # Finally,
            weather_data.Snowfall = snowfall
            weather_data.TotalPrecipitation = precipitation
            # Save the processed data
            save_pickle(weather_data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_data = None

    return weather_data


# Get Weather in a chunk-wise way
def get_weather_by_part(chunk_size=100000, index=True, save_as=None, save_by_chunk=False, save_by_value=False):
    """
    Note that it might be too large for pd.read_sql to read with low memory. Instead, we may read the 'Weather' table
    chunk-wise and assemble the full data set from individual pieces afterwards, especially when we'd like to save
    the data locally.
    """

    weather = db.read_table_by_part(db_name="NR_METEX",
                                    table_name="Weather",
                                    index_col=metex_pk("Weather") if index is True else None,
                                    parse_dates=None,
                                    chunk_size=chunk_size,
                                    save_as=save_as,
                                    save_by_chunk=save_by_chunk,
                                    save_by_value=save_by_value)

    return weather


# Get WeatherCell
def get_weather_cell(update=False, show_map=False, projection='tmerc', save_map_as=".png", dpi=600):
    table_name = 'WeatherCell'
    path_to_pickle = cdd_metex_db_tables(table_name + ".pickle")

    if os.path.isfile(path_to_pickle) and not update:
        weather_cell_map = load_pickle(path_to_pickle)
    else:
        try:
            # Read 'WeatherCell' table
            weather_cell_map = \
                db.read_metex_table(table_name, index_col=metex_pk(table_name), save_as=".csv", update=update)
            weather_cell_map.index.rename('WeatherCellId', inplace=True)  # Rename index
            # Lower left corner:
            ll_longitude = weather_cell_map.Longitude  # - weather_cell_map.width / 2
            weather_cell_map['ll_Longitude'] = ll_longitude
            ll_latitude = weather_cell_map.Latitude  # - weather_cell_map.height / 2
            weather_cell_map['ll_Latitude'] = ll_latitude
            # Upper left corner:
            ul_lon = weather_cell_map.Longitude  # - cell['width'] / 2
            weather_cell_map['ul_lon'] = ul_lon
            ul_lat = weather_cell_map.Latitude + weather_cell_map.height  # / 2
            weather_cell_map['ul_lat'] = ul_lat
            # Upper right corner:
            ur_longitude = weather_cell_map.Longitude + weather_cell_map.width  # / 2
            weather_cell_map['ur_Longitude'] = ur_longitude
            ur_latitude = weather_cell_map.Latitude + weather_cell_map.height  # / 2
            weather_cell_map['ur_Latitude'] = ur_latitude
            # Lower right corner:
            lr_lon = weather_cell_map.Longitude + weather_cell_map.width  # / 2
            weather_cell_map['lr_lon'] = lr_lon
            lr_lat = weather_cell_map.Latitude  # - weather_cell_map.height / 2
            weather_cell_map['lr_lat'] = lr_lat
            # Get weather cell map
            imdm_weather_cell_map = get_imdm_weather_cell_map()
            # Get IMDM info
            imdm = get_imdm(as_dict=False)
            # Merge the acquired data set
            weather_cell_map = imdm_weather_cell_map.join(weather_cell_map, on='WeatherCellId').join(imdm, on='IMDM')
            # Save the processed data
            save_pickle(weather_cell_map, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(table_name, e))
            weather_cell_map = None

    # Plot the weather cells on the map?
    if show_map:
        print("Plotting the map ...", end="")

        plt.figure(figsize=(7, 8))

        ll = weather_cell_map[['ll_Longitude', 'll_Latitude']].apply(min).values
        ur = weather_cell_map[['ur_Longitude', 'ur_Latitude']].apply(max).values

        base_map = mpl_toolkits.basemap.Basemap(projection=projection,  # Transverse Mercator Projection
                                                lon_0=-2.,
                                                lat_0=49.,
                                                ellps='WGS84',
                                                epsg=27700,
                                                llcrnrlon=ll[0] - 0.285,  # -0.570409,  #
                                                llcrnrlat=ll[1] - 0.255,  # 51.23622,  #
                                                urcrnrlon=ur[0] + 0.285,  # 1.915975,  #
                                                urcrnrlat=ur[1] + 0.255,  # 53.062591,  #
                                                lat_ts=0,
                                                resolution='h',
                                                suppress_ticks=True)
        # base_map.drawmapboundary(color='none', fill_color='white')
        # base_map.drawcoastlines()
        # base_map.fillcontinents()
        base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)

        weather_cell_map_plot = weather_cell_map.drop_duplicates([s for s in weather_cell_map.columns if '_' in s])

        for i in weather_cell_map_plot.index:
            ll_x, ll_y = base_map(weather_cell_map_plot['ll_Longitude'][i], weather_cell_map_plot['ll_Latitude'][i])
            ul_x, ul_y = base_map(weather_cell_map_plot['ul_lon'][i], weather_cell_map_plot['ul_lat'][i])
            ur_x, ur_y = base_map(weather_cell_map_plot['ur_Longitude'][i], weather_cell_map_plot['ur_Latitude'][i])
            lr_x, lr_y = base_map(weather_cell_map_plot['lr_lon'][i], weather_cell_map_plot['lr_lat'][i])
            xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
            poly = matplotlib.patches.Polygon(list(xy), fc='#fff68f', ec='b', alpha=0.5)
            plt.gca().add_patch(poly)
        plt.tight_layout()

        # # Plot points
        # from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
        # ll_lon_lat = [Point(m(lon, lat)) for lon, lat in zip(cell['ll_Longitude'], cell['ll_Latitude'])]
        # ur_lon_lat = [Point(m(lon, lat)) for lon, lat in zip(cell['ur_Longitude'], cell['ur_Latitude'])]
        # map_points = MultiPoint(ll_lon_lat + ur_lon_lat)
        # base_map.scatter([geom.x for geom in map_points], [geom.y for geom in map_points],
        #                  marker='x', s=16, lw=1, facecolor='#5a7b6c', edgecolor='w', label='Hazardous tress',
        #                  alpha=0.6, antialiased=True, zorder=3)
        #
        # # Plot squares
        # for i in range(len(cell)):
        #     ll_x, ll_y = base_map(cell['ll_Longitude'].iloc[i], cell['ll_Latitude'].iloc[i])
        #     ur_x, ur_y = base_map(cell['ur_Longitude'].iloc[i], cell['ur_Latitude'].iloc[i])
        #     ul_x, ul_y = base_map(cell['ul_lon'].iloc[i], cell['ul_lat'].iloc[i])
        #     lr_x, lr_y = base_map(cell['lr_lon'].iloc[i], cell['lr_lat'].iloc[i])
        #     base_map.plot([ll_x, ul_x], [ll_y, ul_y], color='#5a7b6c')
        #     base_map.plot([ul_x, ur_x], [ul_y, ur_y], color='#5a7b6c')
        #     base_map.plot([ur_x, lr_x], [ur_y, lr_y], color='#5a7b6c')
        #     base_map.plot([lr_x, ll_x], [lr_y, ll_y], color='#5a7b6c')

        print("Done.")
        # Save the map
        if save_map_as:
            plt.savefig(cdd_metex_db_fig(table_name + save_map_as), dpi=dpi)

    return weather_cell_map


# Get the lower-left and upper-right boundaries of weather cells
def get_weather_cell_map_boundary(route=None, adjusted=(0.285, 0.255)):
    # Get weather cell
    weather_cell = get_weather_cell()
    # For a certain Route?
    if route is not None:
        rte = find_match(route, get_route().Route.tolist())
        weather_cell = weather_cell[weather_cell.Route == rte]  # Select data for the specified route only
    ll = tuple(weather_cell[['ll_Longitude', 'll_Latitude']].apply(pd.np.min))
    lr = weather_cell.lr_lon.max(), weather_cell.lr_lat.min()
    ur = tuple(weather_cell[['ur_Longitude', 'ur_Latitude']].apply(pd.np.max))
    ul = weather_cell.ul_lon.min(), weather_cell.ul_lat.max()
    # Adjust (broaden) the boundaries?
    if adjusted:
        adj_values = pd.np.array(adjusted)
        ll = ll - adj_values
        lr = lr + (adj_values, -adj_values)
        ur = ur + adj_values
        ul = ul + (-adj_values, adj_values)
    return shapely.geometry.Polygon((ll, lr, ur, ul))


"""
update = True

get_performance_event_code(update=update)
get_incident_reason_info_ref(update=update)
get_imdm(as_dict=False, update=update)
get_imdm_alias(as_dict=False, update=update)
get_imdm_weather_cell_map(grouped=False, update=update)
get_incident_reason_info(database_plus=True, update=update)
get_incident_record(update=update)
get_location(update=update)
get_pfpi(update=update)
get_route(update=update)
get_stanox_location(nr_mileage_format=True, update=update)
get_stanox_section(update=update)
get_trust_incident(financial_years_06_14=True, update=update)
get_weather(update=update)
get_weather_category_lookup(as_dict=False, update=update)
get_weather_cell(update=update, show_map=True, projection='tmerc', save_map_as=".png", dpi=600)
"""

# ====================================================================================================================
""" Utils for creating views """


# Form a file name in terms of specific 'Route' and 'weather' category
def make_filename(base_name, route, weather, *extra_suffixes, save_as=".pickle"):
    if route is not None:
        route_lookup = load_json(cdd("Network\\Routes", "route-names.json"))
        route = find_match(route, route_lookup['Route'])
    if weather is not None:
        weather_category_lookup = load_json(cdd("Weather", "weather-categories.json"))
        weather = find_match(weather, weather_category_lookup['WeatherCategory'])
    filename_suffix = [s for s in (route, weather) if s is not None]  # "s" stands for "suffix"
    filename = "-".join([base_name] + filename_suffix + [str(s) for s in extra_suffixes]) + save_as
    return filename


# Subset the required data given 'route' and 'weather'
def subset(data, route=None, weather=None, reset_index=False):
    if data is None:
        data_subset = None
    else:
        route_lookup = list(set(data.Route))
        weather_category_lookup = list(set(data.WeatherCategory))
        # Select data for a specific route and weather category
        if not route and not weather:
            data_subset = data.copy()
        elif route and not weather:
            data_subset = data[data.Route == find_match(route, route_lookup)]
        elif not route and weather:
            data_subset = data[data.WeatherCategory == find_match(weather, weather_category_lookup)]
        else:
            data_subset = data[(data.Route == find_match(route, route_lookup)) &
                               (data.WeatherCategory == find_match(weather, weather_category_lookup))]
        # Reset index
        if reset_index:
            data_subset.reset_index(inplace=True)  # dat.index = range(len(dat))
    return data_subset


# Calculate the DelayMinutes and DelayCosts for grouped data
def agg_pfpi_stats(dat, selected_features, sort_by=None):
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
    if sort_by is not None:
        data.sort_values(sort_by, inplace=True)

    return data


# ====================================================================================================================
""" Get views based on the NR_METEX data """


# Retrieve the TRUST
def merge_schedule8_data(save_as=".pickle"):
    pfpi = get_pfpi()  # Get PfPI (260645, 6)
    incident_record = get_incident_record()  # (233452, 4)
    trust_incident = get_trust_incident(financial_years_06_14=True)  # (192054, 11)
    location = get_location()  # (228851, 6)
    imdm = get_imdm()  # (42, 1)
    incident_reason_info = get_incident_reason_info()  # (393, 7)
    stanox_location = get_stanox_location()  # (7560, 5)
    stanox_section = get_stanox_section()  # (9440, 7)

    # Merge the acquired data sets
    data = pfpi. \
        join(incident_record,  # (260645, 10)
             on='IncidentRecordId', how='inner'). \
        join(trust_incident,  # (260483, 21)
             on='TrustIncidentId', how='inner'). \
        join(stanox_section,  # (260483, 28)
             on='StanoxSectionId', how='inner'). \
        join(location,  # (260470, 34)
             on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
        join(stanox_location,  # (260190, 39)
             on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
        join(stanox_location,  # (260140, 44)
             on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
        join(incident_reason_info,  # (260140, 51)
             on='IncidentReason', how='inner')  # .\
    # join(imdm, on='IMDM_Location', how='inner')  # (260140, 52)

    """
    There are "errors" in the IMDM data/column of the TrustIncident table.
    Not sure if the information about location id number is correct.
    """

    # Get a ELR-IMDM-Route "dictionary" from vegetation database
    route_du_elr = dbv.get_furlong_location(useful_columns_only=True)[['Route', 'ELR', 'DU']].drop_duplicates()
    route_du_elr.index = range(len(route_du_elr))  # (1276, 3)

    # Further cleaning the data
    data.reset_index(inplace=True)
    temp = data.merge(route_du_elr, how='left', left_on=['ELR_Start', 'IMDM'], right_on=['ELR', 'DU'])  # (260140, 55)
    temp = temp.merge(route_du_elr, how='left', left_on=['ELR_End', 'IMDM'], right_on=['ELR', 'DU'])  # (260140, 58)

    temp[['Route_', 'IMDM_']] = temp[['Route_x', 'DU_x']]
    idx_x = (temp.Route_x.isnull()) & (~temp.Route_y.isnull())
    temp.loc[idx_x, 'Route_'], temp.loc[idx_x, 'IMDM_'] = temp.Route_y.loc[idx_x], temp.DU_y.loc[idx_x]

    idx_y = (temp.Route_x.isnull()) & (temp.Route_y.isnull())
    temp.loc[idx_y, 'IMDM_'] = temp.IMDM_Location.loc[idx_y]
    temp.loc[idx_y, 'Route_'] = temp.loc[idx_y, 'IMDM_'].replace(imdm.to_dict()['Route'])

    temp.drop(labels=['Route_x', 'ELR_x', 'DU_x', 'Route_y', 'ELR_y', 'DU_y',
                      'StanoxSection_Start', 'StanoxSection_End',
                      'IMDM', 'IMDM_Location'], axis=1, inplace=True)  # (260140, 50)

    data = temp.rename(columns={'Location_Start': 'StartLocation',
                                'Location_End': 'EndLocation',
                                'LocationAlias_Start': 'StartLocationAlias',
                                'LocationAlias_End': 'EndLocationAlias',
                                'ELR_Start': 'StartELR',
                                'ELR_End': 'EndELR',
                                'Mileage_Start': 'StartMileage',
                                'Mileage_End': 'EndMileage',
                                'LocationId_Start': 'StartLocationId',
                                'LocationId_End': 'EndLocationId',
                                'LocationId_Section': 'SectionLocationId',
                                'Route_': 'Route',
                                'IMDM_': 'IMDM'})  # (260140, 50)

    idx = data.StartLocation == 'Highbury & Islington (North London Lines)'
    data.loc[idx, ['StartLongitude', 'StartLatitude']] = [-0.1045, 51.5460]
    idx = data.EndLocation == 'Highbury & Islington (North London Lines)'
    data.loc[idx, ['EndLongitude', 'EndLatitude']] = [-0.1045, 51.5460]
    idx = data.StartLocation == 'Dalston Junction (East London Line)'
    data.loc[idx, ['StartLongitude', 'StartLatitude']] = [-0.0751, 51.5461]
    idx = data.EndLocation == 'Dalston Junction (East London Line)'
    data.loc[idx, ['EndLongitude', 'EndLatitude']] = [-0.0751, 51.5461]

    data.EndELR.replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)

    # Sort the merged data frame by index 'PfPIId'
    data = data.set_index('PfPIId').sort_index()  # (260140, 49)

    # Further cleaning the 'IMDM' and 'Route'
    for section in data.StanoxSection.unique():
        temp = data[data.StanoxSection == section]
        # IMDM
        if len(temp.IMDM.unique()) >= 2:
            imdm_temp = data.loc[temp.index].IMDM.value_counts()
            data.loc[temp.index, 'IMDM'] = imdm_temp[imdm_temp == imdm_temp.max()].index[0]
        # Route
        if len(temp.Route.unique()) >= 2:
            route_temp = data.loc[temp.index].Route.value_counts()
            data.loc[temp.index, 'Route'] = route_temp[route_temp == route_temp.max()].index[0]

    if save_as:
        filename = make_filename("Schedule8_details", route=None, weather=None, save_as=save_as)
        save_pickle(data, cdd_metex_db_views(filename))

    # Return the DataFrame
    return data


# Get the TRUST data
def get_schedule8_details(route=None, weather=None, reset_index=False, update=False):
    filename = make_filename("Schedule8_details", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        schedule8_details = load_pickle(path_to_pickle)
        if reset_index:
            schedule8_details.reset_index(inplace=True)
    else:
        try:
            schedule8_details = merge_schedule8_data(save_as=".pickle")
            schedule8_details = subset(schedule8_details, route, weather, reset_index)
            save_pickle(schedule8_details, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            schedule8_details = None

    return schedule8_details


# Essential details about incidents
def get_schedule8_details_pfpi(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_details_pfpi", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get merged data sets
            schedule8_data = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',  # 260140
                'IncidentRecordId',  # 232978
                'TrustIncidentId',  # 191759
                'IncidentNumber',  # 176370
                'PerformanceEventCode', 'PerformanceEventGroup', 'PerformanceEventName',
                'PfPIMinutes', 'PfPICosts', 'FinancialYear',  # 9
                'IncidentRecordCreateDate',  # 3287
                'StartDate', 'EndDate',
                'IncidentDescription', 'IncidentJPIPCategory',
                'WeatherCategory',  # 10
                'IncidentReason', 'IncidentReasonDescription', 'IncidentCategory', 'IncidentCategoryDescription',
                'IncidentCategoryGroupDescription', 'IncidentFMS', 'IncidentEquipment',
                'WeatherCell',  # 106
                'Route', 'IMDM',
                'StanoxSection', 'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'ApproximateLocation']
            # Acquire the subset (260140, 40)
            data = schedule8_data[selected_features]
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Schedule 8 details combined with weather data
def get_schedule8_details_and_weather(route=None, weather=None, ip_start_hrs=-12, ip_end_hrs=12, update=False):
    """
    :param route:
    :param weather:
    :param ip_start_hrs: [numeric] incident period start time, i.e. hours before the recorded incident start
    :param ip_end_hrs: [numeric] incident period end time, i.e. hours after the recorded incident end time
    :param update: [bool] default, False
    :return:
    """

    filename = make_filename("Schedule8_details_and_weather", route, weather)
    add_suffix = [str(s) for s in (ip_start_hrs, ip_end_hrs)]
    filename = "_".join([filename] + add_suffix) + ".pickle"
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Getting Schedule 8 details (i.e. 'Schedule8_details')
            schedule8_details = get_schedule8_details(route, weather, reset_index=False)
            # Truncates "month" and "time" parts from datetime
            schedule8_details['incident_duration'] = \
                schedule8_details.EndDate - schedule8_details.StartDate
            schedule8_details['critical_start'] = \
                schedule8_details.StartDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_start_hrs)
            schedule8_details['critical_end'] = \
                schedule8_details.EndDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_end_hrs)
            schedule8_details['critical_period'] = \
                schedule8_details.critical_end - schedule8_details.critical_start
            # Get weather data
            weather_data = get_weather()
            # Merge the two data sets
            data = schedule8_details.join(weather_data, on=['WeatherCell', 'critical_start'], how='inner')
            data.sort_index(inplace=True)  # (257608, 61)
            # Save the merged data
            save_pickle(data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by incident location and weather category
def get_schedule8_cost_by_location(route=None, weather=None, update=False):
    """
    :param route: 
    :param weather: 
    :param update: 
    :return: 
    """

    filename = make_filename("Schedule8_costs_by_location", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get Schedule8_details
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select columns
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
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Getting '{}' ... Failed due to {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime and weather category
def get_schedule8_cost_by_datetime(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get Schedule8_details
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDate', 'EndDate',
                'WeatherCategory',
                'Route', 'IMDM',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features, sort_by=['StartDate', 'EndDate'])
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime and location
def get_schedule8_cost_by_datetime_location(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get merged data sets
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select a list of columns
            selected_features = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDate', 'EndDate',
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
            data = agg_pfpi_stats(schedule8_data, selected_features, sort_by=['StartDate', 'EndDate'])
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 data by datetime, location and weather
def get_schedule8_cost_by_datetime_location_weather(route=None, weather=None, ip_start=-12, ip_end=12, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location_weather", route, weather)
    add_suffix = [str(s) for s in (ip_start, ip_end)]
    filename = "_".join([filename] + add_suffix) + ".pickle"
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get Schedule8_costs_by_datetime_location
            schedule8_data = get_schedule8_cost_by_datetime_location(route, weather)
            # Create critical start and end datetimes (truncating "month" and "time" parts from datetime)
            schedule8_data['incident_duration'] = \
                schedule8_data.EndDate - schedule8_data.StartDate
            schedule8_data['critical_start'] = \
                schedule8_data.StartDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_start)
            schedule8_data['critical_end'] = \
                schedule8_data.EndDate.apply(datetime_truncate.truncate_hour) + \
                pd.DateOffset(hours=ip_end)
            schedule8_data['critical_period'] = \
                schedule8_data.critical_end - schedule8_data.critical_start
            # Get weather data
            weather_data = get_weather()
            # Merge the two data sets
            data = schedule8_data.join(weather_data, on=['WeatherCell', 'critical_start'], how='inner')
            # Save the merged data
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by incident reason
def get_schedule8_cost_by_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_reason", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Get merged data sets
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            # Select columns
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'Route',
                                 # 'IMDM',
                                 'WeatherCategory',
                                 'IncidentDescription',
                                 'IncidentCategory',
                                 'IncidentCategoryDescription',
                                 'IncidentCategorySuperGroupCode',
                                 'IncidentCategoryGroupDescription',
                                 'IncidentReason',
                                 'IncidentReasonName',
                                 'IncidentReasonDescription',
                                 'IncidentJPIPCategory',
                                 'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by location and incident reason
def get_schedule8_cost_by_location_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_location_reason", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather).reset_index()
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
                                 'IncidentCategoryGroupDescription',
                                 'IncidentReason',
                                 'IncidentReasonName',
                                 'IncidentReasonDescription',
                                 'IncidentJPIPCategory',
                                 'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by datetime, location and incident reason
def get_schedule8_cost_by_datetime_location_reason(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_datetime_location_reason", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'StartDate', 'EndDate',
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
                                 'IncidentCategoryGroupDescription',
                                 'IncidentReason',
                                 'IncidentReasonName',
                                 'IncidentReasonDescription',
                                 'IncidentJPIPCategory',
                                 'PfPIMinutes', 'PfPICosts']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


# Get Schedule 8 cost by weather category
def get_schedule8_cost_by_weather_category(route=None, weather=None, update=False):
    filename = make_filename("Schedule8_costs_by_weather_category", route, weather)
    path_to_pickle = cdd_metex_db_views(filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            schedule8_details = get_schedule8_details(route, weather, reset_index=True)
            selected_features = ['PfPIId',
                                 'FinancialYear',
                                 'Route',
                                 'IMDM',
                                 'WeatherCategory',
                                 'PfPICosts',
                                 'PfPIMinutes']
            schedule8_data = schedule8_details[selected_features]
            data = agg_pfpi_stats(schedule8_data, selected_features)
            save_pickle(data, path_to_pickle)
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(filename)[0], e))
            data = None

    return data


"""
route = None
weather = None
update = True

get_schedule8_details(route, weather, reset_index=False, update=update)
get_schedule8_details_pfpi(route, weather, update)
get_schedule8_details_and_weather(route, weather, -12, 12, update=update)
get_schedule8_cost_by_location(route, weather, update=update)
get_schedule8_cost_by_datetime(route, weather, update=update)
get_schedule8_cost_by_datetime_location(route, weather, update=update)
get_schedule8_cost_by_datetime_location_weather(route, weather, -12, 12, update=update)
get_schedule8_cost_by_reason(route, weather, update=update)
get_schedule8_cost_by_location_reason(route, weather, update=update)
get_schedule8_cost_by_datetime_location_reason(route, weather, update=update)
get_schedule8_cost_by_weather_category(route, weather, update=update)
"""
