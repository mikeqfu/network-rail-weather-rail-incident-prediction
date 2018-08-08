""" Schedule 8 incidents """

import itertools
import os
import re

import pandas as pd
from fuzzywuzzy.process import extractOne

from delay_attr_glossary import get_incident_reason_metadata
from loc_code_dict import create_loc_name_regexp_replacement_dict, create_loc_name_replacement_dict
from railwaycodes_utils import get_location_codes, get_location_codes_dictionary_v2, get_station_locations
from utils import cdd, cdd_rc, load_json, load_pickle, save_pickle


# Change directory to "Incidents"
def cdd_incidents(*directories):
    path = cdd("Incidents")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Data\\METEX\\Database\\Tables_original" and sub-directories
def cdd_metex_db_tables_original(*directories):
    path = cdd("METEX", "Database", "Tables_original")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Utilities """


# Get rid of duplicated records
def tidy_duplicated_codes(code_dict, raw_loc):
    temp_loc = raw_loc.replace(code_dict)
    if not temp_loc.equals(raw_loc):
        temp_loc.columns = [x + '_temp' for x in temp_loc.columns]
        temp_loc = pd.concat([raw_loc, temp_loc], axis=1)
        if temp_loc.shape[1] == 4:
            temp_loc = temp_loc.apply(lambda x: extractOne(x[0], x[2])[0] if isinstance(x[2], list) else x, axis=1)
            temp_loc = temp_loc.apply(lambda x: extractOne(x[1], x[3])[0] if isinstance(x[3], list) else x, axis=1)
        elif temp_loc.shape[1] == 2:
            temp_loc = temp_loc.apply(lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x, axis=1)
        temp_loc.drop(list(raw_loc.columns), axis=1, inplace=True)
        temp_loc.columns = [x.replace('_temp', '') for x in temp_loc.columns]
    return temp_loc


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
            data_subset = data[data.Route == extractOne(route, route_lookup, score_cutoff=10)[0]]
        elif not route and weather:
            data_subset = data[data.WeatherCategory == extractOne(weather, weather_category_lookup, score_cutoff=10)[0]]
        else:
            data_subset = data[
                (data.Route == extractOne(route, route_lookup, score_cutoff=10)[0]) &
                (data.WeatherCategory == extractOne(weather, weather_category_lookup, score_cutoff=10)[0])]
        # Reset index
        if reset_index:
            data_subset.reset_index(inplace=True)  # dat.index = range(len(dat))
    return data_subset


# ====================================================================================================================
""" Location metadata """


# Cleanse STANOX locations data
def cleanse_location_metadata(stanox):
    """
    :param stanox:
    :return:
    """
    stanox.columns = [x.upper().replace(' ', '_') for x in stanox.columns]
    stanox['LOOKUP_NAME_Raw'] = stanox.LOOKUP_NAME

    print("Starting to cleanse \"stanox\" ... ")
    # Clean 'STANOX'
    stanox.STANOX = stanox.STANOX.fillna('')
    stanox.STANOX = stanox.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)

    errata_stanox = load_json(cdd_rc("Errata.json"))['STANOX']
    stanox.STANOX = stanox.STANOX.replace(errata_stanox)

    loc_name_replacement_dict = create_loc_name_replacement_dict('LOOKUP_NAME')
    stanox = stanox.replace(loc_name_replacement_dict)

    loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('LOOKUP_NAME')
    stanox = stanox.replace(loc_name_regexp_replacement_dict)

    # The 'stanox' dataframe has many 'N/A's in both the 'LOOKUP_NAME' and 'LINE_DESCRIPTION' columns
    stanox_dict = get_location_codes_dictionary_v2(['STANOX'])
    na_name = stanox[stanox.LOOKUP_NAME.isnull()]
    temp = na_name.join(stanox_dict, on='STANOX')[['STANME', 'Location']]
    stanox.loc[na_name.index, 'LOOKUP_NAME'] = temp.apply(
        lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

    temp = stanox.join(stanox_dict, on='STANOX')[['LOOKUP_NAME', 'Location']]
    stanox.LOOKUP_NAME = temp.apply(lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

    stanox.dropna(subset=['LOOKUP_NAME'], inplace=True)
    stanox.fillna('', inplace=True)
    stanox.LINE_DESCRIPTION = stanox.LINE_DESCRIPTION.replace({re.compile('[Ss]tn'): 'Station',
                                                               re.compile('[Ll]oc'): 'Location',
                                                               re.compile('[Jj]n'): 'Junction',
                                                               re.compile('[Ss]dg'): 'Siding',
                                                               re.compile('[Ss]dgs'): 'Sidings'})
    stanox.drop_duplicates(inplace=True)

    # Get reference metadata from RailwayCodes
    station_data = get_station_locations()['Station']
    station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].drop_duplicates('Station')
    station_data.set_index('Station', inplace=True)
    temp = stanox.join(station_data, on='LOOKUP_NAME')
    idx = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
    stanox.loc[idx, 'DEG_LAT':'DEG_LONG'] = temp.loc[idx, ['Degrees Latitude', 'Degrees Longitude']].values

    stanox.sort_values(['STANOX', 'SHAPE_LENG'], ascending=[True, False], inplace=True)
    stanox.index = range(len(stanox))

    print("Done.")

    return stanox


# STANOX locations data
def get_location_metadata(update=False):

    pickle_filename = "STANOXLocationsData.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        stanox_locations_data = load_pickle(path_to_pickle)
    else:
        try:
            workbook = pd.ExcelFile(cdd_rc(pickle_filename.replace(".pickle", ".xlsx")))

            # 'TIPLOC_LocationsLyr' -----------------------------------------------
            tiploc_locations_lyr = workbook.parse(sheet_name='TIPLOC_LocationsLyr',
                                                  parse_dates=['LASTEDITED', 'LAST_UPD_1'], dayfirst=True,
                                                  converters={'STANOX': str})
            fillna_columns_str = ['STANOX', 'STANME', 'TIPLOC', 'LOOKUP_NAME', 'STATUS',
                                  'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
            tiploc_locations_lyr.fillna({x: '' for x in fillna_columns_str}, inplace=True)
            tiploc_locations_lyr.STANOX = \
                tiploc_locations_lyr.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)

            # 'STANOX' -----------------------------------------------------------------------------
            stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})

            stanox = cleanse_location_metadata(stanox)

            # Collect the above two dataframes and store them in a dictionary ---------------------
            stanox_locations_data = dict(zip(workbook.sheet_names, [tiploc_locations_lyr, stanox]))

            workbook.close()

            save_pickle(stanox_locations_data, path_to_pickle)

        except Exception as e:
            print("Failed to fetch \"STANOX locations data\" from {}.".format(os.path.dirname(path_to_pickle)))
            print(e)
            stanox_locations_data = None

    return stanox_locations_data


#
def get_location_metadata_plus(update=False):

    pickle_filename = "location-metadata-plus.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        location_metadata_plus = load_pickle(path_to_pickle)
    else:
        try:

            # Location
            location = pd.read_csv(cdd_metex_db_tables_original("Location.csv"), index_col='Id')
            location.index.rename('LocationId', inplace=True)
            location.loc[610096, 'StartLongitude':'EndLatitude'] = [-0.0751, 51.5461, -0.0751, 51.5461]

            # STANOX location --------------------------------------------------------------------------------------
            stanox_location = pd.read_csv(cdd_metex_db_tables_original("StanoxLocation.csv"), dtype={'Stanox': str})

            # Cleanse stanox_location
            location_codes = get_location_codes()['Locations']

            errata_stanox, errata_tiploc, errata_stanme = load_json(cdd_rc("Errata.json")).values()
            # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
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

            loc_name_replacement_dict = create_loc_name_replacement_dict('Description')
            stanox_location = stanox_location.replace(loc_name_replacement_dict)
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('Description')
            stanox_location = stanox_location.replace(loc_name_regexp_replacement_dict)

            # STANME dictionary
            location_stanme_dict = location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
            stanox_location.Name = stanox_location.Name.replace(location_stanme_dict)

            # STANOX dictionary
            stanox_dict = get_location_codes_dictionary_v2(['STANOX'])
            temp = stanox_location.join(stanox_dict, on='Stanox')[['Description', 'Location']]
            na_loc = temp.Location.isnull()
            temp.loc[na_loc, 'Location'] = temp.loc[na_loc, 'Description']
            stanox_location.Description = temp.apply(
                lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

            stanox_location.Name = stanox_location.Name.str.upper()

            location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']]
            location_codes_cut = location_codes_cut.groupby(['STANOX', 'Location']).agg({'STANME': list})
            location_codes_cut.STANME = location_codes_cut.STANME.map(lambda x: x[0] if isinstance(x, list) else x)
            temp = stanox_location.join(location_codes_cut, on=['Stanox', 'Description'])
            stanox_location.Name = temp.STANME

            stanox_duplicated = stanox_location[stanox_location.duplicated('Stanox', keep=False)]
            idx_duplicated = stanox_duplicated.index
            stanox_duplicated.dropna(subset=['LocationId'], inplace=True)
            stanox_duplicated.drop_duplicates(['Description', 'Name'], inplace=True)
            idx_drop = [x for x in idx_duplicated if x not in stanox_duplicated.index]
            stanox_location.drop(idx_drop, axis=0, inplace=True)

            stanox_location.ELR.fillna('', inplace=True)

            stanox_location.set_index('Stanox', inplace=True)

            # STANOX section --------------------------------------------------------------
            stanox_section = pd.read_csv(cdd_metex_db_tables_original("StanoxSection.csv"),
                                         dtype={'StartStanox': str, 'EndStanox': str})
            stanox_section.rename(columns={'Id': 'StanoxSectionId',
                                           'Description': 'StanoxSection',
                                           'LocationId': 'SectionLocationId'}, inplace=True)
            stanox_section[['StartStanox', 'EndStanox']] = \
                stanox_section[['StartStanox', 'EndStanox']].replace(errata_stanox)

            # -----------------------------------------------------------------------------------------
            stanox_location.rename(columns={'Description': 'Location', 'Name': 'Stanme'}, inplace=True)
            location_metadata = stanox_section.join(stanox_location, on='StartStanox')
            start_cols = ['Start' + x for x in stanox_location.columns]
            location_metadata.rename(columns=dict(zip(list(stanox_location.columns), start_cols)), inplace=True)

            location_metadata = location_metadata.join(stanox_location, on='EndStanox')
            end_cols = ['End' + x for x in stanox_location.columns]
            location_metadata.rename(columns=dict(zip(list(stanox_location.columns), end_cols)), inplace=True)

            temp_start = location_metadata.join(location, on='SectionLocationId')
            na_start = temp_start.StartLocation.isnull()
            temp = temp_start[na_start].join(stanox_dict, on='StartStanox')
            location_metadata.loc[na_start, 'StartLocation'] = \
                temp.Location.map(lambda x: x[0] if isinstance(x, list) else x)

            temp_end = location_metadata.join(location, on='SectionLocationId')
            na_end = temp_end.EndLocation.isnull()
            temp = temp_end[na_end].join(stanox_dict, on='EndStanox')
            location_metadata.loc[na_end, 'EndLocation'] = \
                temp.Location.map(lambda x: x[0] if isinstance(x, list) else x)

            # --------------------------------------------------------------------------
            location_metadata = location_metadata.join(location, on='SectionLocationId')
            location_metadata.StanoxSection = location_metadata.StartLocation + ' : ' + location_metadata.EndLocation

            start_columns = [x for x in location_metadata.columns if 'Start' in x]
            start_locations = location_metadata[start_columns]
            start_locations.rename(columns=dict(zip(start_columns, [x.replace('Start', '') for x in start_columns])),
                                   inplace=True)

            end_columns = [x for x in location_metadata.columns if 'End' in x]
            end_locations = location_metadata[end_columns]
            end_locations.rename(columns=dict(zip(end_columns, [x.replace('End', '') for x in end_columns])),
                                 inplace=True)

            loc_metadata = pd.concat([start_locations, end_locations], axis=0, ignore_index=True).drop_duplicates()

            # Get reference metadata from RailwayCodes
            station_data = get_station_locations()['Station']
            station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].drop_duplicates('Station')
            station_data.set_index('Station', inplace=True)
            temp = loc_metadata.join(station_data, on='Location')
            idx = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
            loc_metadata.loc[idx, 'Longitude':'Latitude'] = \
                temp.loc[idx, ['Degrees Longitude', 'Degrees Latitude']].values

            loc_metadata.sort_values(['Stanox', 'Location', 'Longitude', 'Latitude'], inplace=True)
            loc_metadata.index = range(len(loc_metadata))

            location_metadata_plus = {'STANOX_section': location_metadata, 'STANOX_location': loc_metadata}

            save_pickle(location_metadata_plus, path_to_pickle)

        except Exception as e:
            print("Failed to get \"stanox_location_metadata\". {}.".format(e))
            location_metadata_plus = None

    return location_metadata_plus


# ====================================================================================================================
""" Incidents """


# Cleanse location data
def cleanse_location_data(data, loc_col_names='StanoxSection', sep=' : '):
    """

    :param data:
    :param sep:
    :param loc_col_names:
    :return:
    """
    #
    old_column_name = loc_col_names + '_Raw'
    data.rename(columns={loc_col_names: old_column_name}, inplace=True)
    if sep is not None and isinstance(sep, str):
        location_data = data[old_column_name].str.split(sep, expand=True)
        #
        location_data.columns = ['StartLocation_Raw', 'EndLocation_Raw']
        location_data.EndLocation_Raw.fillna(location_data.StartLocation_Raw, inplace=True)
    else:
        location_data = data[[old_column_name]]

    #
    stanox_dict = get_location_codes_dictionary_v2(['STANOX'], as_dict=True)

    print("Cleansing duplicates by \"STANOX\" ... ", end="")

    errata_stanox, errata_tiploc, errata_stanme = load_json(cdd_rc("Errata.json")).values()
    # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
    data.Stanox = data.Stanox.replace(errata_stanox)
    data.Description = data.Description.replace(errata_tiploc)
    data.Name = data.Name.replace(errata_stanme)

    location_data_temp = tidy_duplicated_codes(stanox_dict, location_data)
    print("Successfully.")
    #
    stanme_dict = get_location_codes_dictionary_v2(['STANME'], as_dict=True)
    print("Cleansing duplicates by \"STANME\" ... ", end="")
    location_data_temp = tidy_duplicated_codes(stanme_dict, location_data_temp)
    print("Successfully.")
    #
    tiploc_dict = get_location_codes_dictionary_v2(['TIPLOC'], as_dict=True)
    print("Cleansing duplicates by \"TIPLOC\" ... ", end="")
    location_data_temp = tidy_duplicated_codes(tiploc_dict, location_data_temp)
    print("Successfully.")

    #
    loc_name_replacement_dict = create_loc_name_replacement_dict()
    location_data_temp = location_data_temp.replace(loc_name_replacement_dict)
    loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict()
    location_data_temp = location_data_temp.replace(loc_name_regexp_replacement_dict)

    location_data_temp.columns = [x.replace('_Raw', '') for x in location_data_temp.columns]

    # Form new StanoxSection column
    location_data_temp[loc_col_names] = location_data_temp.StartLocation + sep + location_data_temp.EndLocation
    mask_single = location_data_temp.StartLocation == location_data_temp.EndLocation
    location_data_temp[loc_col_names][mask_single] = location_data_temp.StartLocation[mask_single]

    # Resort column order
    col_names = [list(v) for k, v in itertools.groupby(list(data.columns), lambda x: x == old_column_name) if not k]
    add_names = [old_column_name, 'StartLocation_Raw', 'EndLocation_Raw', loc_col_names, 'StartLocation', 'EndLocation']
    col_names = col_names[0] + add_names + col_names[1]

    cleansed_data = data.join(pd.concat([location_data, location_data_temp], axis=1))[col_names]

    return cleansed_data


# Schedule 8 weather incidents
def read_schedule8_weather_incidents(update=False):

    pickle_filename = "Schedule8WeatherIncidents.pickle"
    path_to_pickle = cdd_incidents("Spreadsheets", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Load data from the raw file
            data = pd.read_excel(path_to_pickle.replace(".pickle", ".xlsx"),
                                 parse_dates=['StartDate', 'EndDate'], day_first=True,
                                 converters={'stanoxSection': str})
            data.rename(columns={'stanoxSection': 'StanoxSection',
                                 'imdm': 'IMDM',
                                 'WeatherCategory': 'WeatherCategoryCode',
                                 'WeatherCategory.1': 'WeatherCategory',
                                 'Reason': 'IncidentReason',
                                 # 'Minutes': 'DelayMinutes',
                                 'Description': 'IncidentReasonDescription',
                                 'Category': 'IncidentCategory',
                                 'CategoryDescription': 'IncidentCategoryDescription'}, inplace=True)

            # Add information about incident reason
            incident_reason_metadata = get_incident_reason_metadata()
            data = data.join(incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
            data.drop([x for x in data.columns if '_meta' in x], axis=1, inplace=True)

            # Cleanse the location data ------------------------------------------------
            data = cleanse_location_data(data, loc_col_names='StanoxSection', sep=' : ')

            # Get location metadata for reference ---------------------------------
            location_metadata, loc_metadata = get_location_metadata_plus().values()

            ref_metadata_0 = location_metadata.drop_duplicates('StanoxSection').set_index('StanoxSection')
            ref_metadata_0 = ref_metadata_0[['StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']]
            data = data.join(ref_metadata_0, on='StanoxSection')

            # ref 1 ------------------------------------------------------------
            temp0 = data[data.StartLongitude.isnull()]

            ref_metadata_1 = loc_metadata[['Location', 'Longitude', 'Latitude']]
            ref_metadata_1 = ref_metadata_1.dropna().drop_duplicates('Location')
            ref_metadata_1.set_index('Location', inplace=True)

            temp1 = temp0.join(ref_metadata_1, on='StartLocation')
            data.loc[temp1.index, 'StartLongitude':'StartLatitude'] = temp1[['Longitude', 'Latitude']].values

            temp1 = temp0.join(ref_metadata_1, on='EndLocation')
            data.loc[temp1.index, 'EndLongitude':'EndLatitude'] = temp1[['Longitude', 'Latitude']].values

            # ref 2 -------------------------------------------
            ref_metadata_2 = get_station_locations()['Station']
            ref_metadata_2 = ref_metadata_2[['Station', 'Degrees Longitude', 'Degrees Latitude']]
            ref_metadata_2 = ref_metadata_2.dropna().drop_duplicates('Station')
            ref_metadata_2.columns = [x.replace('Degrees ', '') for x in ref_metadata_2.columns]
            ref_metadata_2.set_index('Station', inplace=True)

            temp0 = data[data.StartLongitude.isnull()]
            temp1 = temp0.join(ref_metadata_2, on='StartLocation')
            data.loc[temp1.index, 'StartLongitude':'StartLatitude'] = temp1[['Longitude', 'Latitude']].values

            temp0 = data[data.EndLongitude.isnull()]
            temp1 = temp0.join(ref_metadata_2, on='EndLocation')
            data.loc[temp1.index, 'EndLongitude':'EndLatitude'] = temp1[['Longitude', 'Latitude']].values

            # ref 3 -------------------------------------------
            ref_metadata_3 = get_location_metadata()['STANOX']
            ref_metadata_3 = ref_metadata_3[['LOOKUP_NAME', 'DEG_LONG', 'DEG_LAT']]
            ref_metadata_3 = ref_metadata_3.dropna().drop_duplicates('LOOKUP_NAME')
            ref_metadata_3.set_index('LOOKUP_NAME', inplace=True)

            temp0 = data[data.StartLongitude.isnull()]
            temp1 = temp0.join(ref_metadata_3, on='StartLocation')
            data.loc[temp1.index, 'StartLongitude':'StartLatitude'] = temp1[['DEG_LONG', 'DEG_LAT']].values

            temp0 = data[data.EndLongitude.isnull()]
            temp1 = temp0.join(ref_metadata_3, on='EndLocation')
            data.loc[temp1.index, 'EndLongitude':'EndLatitude'] = temp1[['DEG_LONG', 'DEG_LAT']].values




            save_pickle(data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Schedule 8 weather incidents\". {}".format(e))
            data = pd.DataFrame()

    return data


# Schedule8WeatherIncidents_02062006_31032014.xlsm
def get_schedule8_weather_incidents_02062006_31032014(route=None, weather=None, update=False):
    """
    Description:
    "Details of schedule 8 incidents together with weather leading up to the incident. Although this file contains
    other weather categories, the main focus of this prototype is adhesion."

    "* WORK IN PROGRESS *  MET-9 - Report of Schedule 8 adhesion incidents vs weather conditions Done."

    """
    # Path to the file
    pickle_filename = "Schedule8WeatherIncidents-02062006-31032014.pickle"
    path_to_pickle = cdd_incidents("Spreadsheets", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        workbook_data = load_pickle(path_to_pickle)
    else:
        try:
            # Open the original file
            workbook = pd.ExcelFile(path_to_pickle.replace(".pickle", ".xlsm"))

            # 'Thresholds' =============================================================
            thresholds = workbook.parse(sheet_name='Thresholds', usecols='A:F').dropna()
            thresholds.index = range(len(thresholds))
            thresholds.columns = [col.replace(' ', '_') for col in thresholds.columns]
            thresholds.Weather_Hazard = thresholds.Weather_Hazard.map(lambda x: x.upper().strip())

            """
            period = workbook.parse(sheet_name='Thresholds', usecols='O').applymap(lambda x: x.replace(' ', '_'))
            weather_type = workbook.parse(sheet_name='Thresholds', usecols='P')
            condition_type = workbook.parse(sheet_name='Thresholds', usecols='Q')

            import itertools
            combination = itertools.product(period.values.flatten(),
                                            weather_type.values.flatten(),
                                            condition_type.values.flatten())
            """

            # 'Data' ====================================================================================
            data = workbook.parse(sheet_name='Data', parse_dates=['StartDate', 'EndDate'], dayfirst=True,
                                  converters={'stanoxSection': str})
            data.rename(columns={'Year': 'FinancialYear',
                                 'stanoxSection': 'StanoxSection',
                                 'imdm': 'IMDM',
                                 'Reason': 'IncidentReason',
                                 # 'Minutes': 'DelayMinutes',
                                 # 'Cost': 'DelayCost',
                                 'CategoryDescription': 'IncidentCategoryDescription'},
                        inplace=True)
            hazard_cols = [x for x in enumerate(data.columns) if 'Weather Hazard' in x[1]]
            obs_cols = [(i - 1, re.search('(?<= \()\w+', x).group().upper()) for i, x in hazard_cols]
            hazard_cols = [(i + 1, x + '_WeatherHazard') for i, x in obs_cols]
            for i, x in obs_cols + hazard_cols:
                data.rename(columns={data.columns[i]: x}, inplace=True)
            # data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')
            data = cleanse_location_data(data, loc_col_names='StanoxSection', sep=' : ')

            #
            incident_reason_metadata = get_incident_reason_metadata()
            data = data.join(incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
            data.drop([x for x in data.columns if '_meta' in x], axis=1, inplace=True)

            # Retain data for specific Route and weather category
            data = subset(data, route, weather)

            # Weather'CategoryLookup' ===========================================
            weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
            weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

            # Make a dictionary
            workbook_data = dict(zip(workbook.sheet_names, [thresholds, data, weather_category_lookup]))

            workbook.close()

            # Save the workbook data
            save_pickle(workbook_data, path_to_pickle)

        except Exception as e:
            print('Failed to get \"Schedule8WeatherIncidents_02062006_31032014.xlsm\" due to {}.'.format(e))
            workbook_data = None

    return workbook_data
