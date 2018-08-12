""" Schedule 8 incidents """

import itertools
import os
import re

import pandas as pd
from fuzzywuzzy.process import extractOne

import railwaycodes_utils as rc
from delay_attr_glossary import get_incident_reason_metadata
from loc_code_dict import create_loc_name_regexp_replacement_dict, create_loc_name_replacement_dict
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
def cleanse_location_metadata(metadata, ref_cols):
    """
    :param metadata:
    :param ref_cols:
    :return:
    """
    metadata.columns = [x.upper().replace(' ', '_') for x in metadata.columns]
    metadata = metadata.replace({'\xa0\xa0': ' '}, regex=True).replace({'\xa0': ''}, regex=True)
    metadata['LOOKUP_NAME_Raw'] = metadata.LOOKUP_NAME

    print("Starting to cleanse ... ")

    # Clean 'STANOX' ('N/A's exist in both the 'LOOKUP_NAME' and 'LINE_DESCRIPTION' columns)
    metadata.STANOX = metadata.STANOX.fillna('')
    metadata.STANOX = metadata.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)

    # Correct errors in STANOX
    errata = load_json(cdd_rc("Errata.json"))
    errata_stanox, errata_tiploc, errata_stanme = [{k: v} for k, v in errata.items()]
    metadata.replace(errata_stanox, inplace=True)
    metadata.replace(errata_stanme, inplace=True)
    if 'TIPLOC' in ref_cols:
        metadata.replace(errata_tiploc, inplace=True)

    # Correct known issues for the location names in the data set
    loc_name_replacement_dict = create_loc_name_replacement_dict('LOOKUP_NAME')
    metadata = metadata.replace(loc_name_replacement_dict)
    loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('LOOKUP_NAME')
    metadata = metadata.replace(loc_name_regexp_replacement_dict)

    # Fill missing location names
    na_name = metadata[metadata.LOOKUP_NAME.isnull()]
    stanox_stanme_dict = rc.get_location_codes_dictionary_v2(ref_cols)
    comparable_col = ['TIPLOC', 'Location'] if 'TIPLOC' in ref_cols else ['STANME', 'Location']
    temp = na_name.join(stanox_stanme_dict, on=ref_cols)[comparable_col]
    metadata.loc[na_name.index, 'LOOKUP_NAME'] = temp.apply(
        lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

    temp = metadata.join(stanox_stanme_dict, on=ref_cols)[['LOOKUP_NAME', 'Location']]
    temp = temp[temp.Location.notnull()]
    metadata.loc[temp.index, 'LOOKUP_NAME'] = \
        temp.apply(lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

    # metadata.dropna(subset=['LOOKUP_NAME'], inplace=True)  # if there is NaN values (though none actually remains)

    metadata.replace({'LINE_DESCRIPTION': {re.compile('[Ss]tn'): 'Station',
                                           re.compile('[Ll]oc'): 'Location',
                                           re.compile('[Jj]n'): 'Junction',
                                           re.compile('[Ss]dg'): 'Siding',
                                           re.compile('[Ss]dgs'): 'Sidings'}}, inplace=True)

    # metadata.fillna('', inplace=True)
    metadata.drop_duplicates(inplace=True)

    # Get reference metadata from RailwayCodes
    station_data = rc.get_station_locations()['Station']
    station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
    station_data.drop_duplicates(subset=['Station'], inplace=True)
    station_data.set_index('Station', inplace=True)
    temp = metadata.join(station_data, on='LOOKUP_NAME')
    na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
    metadata.loc[na_i, 'DEG_LAT':'DEG_LONG'] = temp.loc[na_i, ['Degrees Latitude', 'Degrees Longitude']].values

    metadata.fillna('', inplace=True)
    metadata.sort_values(ref_cols + ['SHAPE_LENG'], ascending=[True] * len(ref_cols) + [False], inplace=True)
    metadata.index = range(len(metadata))

    print("Done.")

    return metadata


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
            tiploc_loc_lyr = workbook.parse(sheet_name='TIPLOC_LocationsLyr',
                                            parse_dates=['LASTEDITED', 'LAST_UPD_1'], dayfirst=True,
                                            converters={'STANOX': str})
            tiploc_loc_lyr.fillna({x: '' for x in ['STANOX', 'STANME', 'TIPLOC', 'LOOKUP_NAME', 'STATUS',
                                  'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']}, inplace=True)
            tiploc_loc_lyr = cleanse_location_metadata(tiploc_loc_lyr, ref_cols=['STANOX', 'STANME', 'TIPLOC'])

            # 'STANOX' -----------------------------------------------------------------------------
            stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})
            stanox = cleanse_location_metadata(stanox, ref_cols=['STANOX', 'STANME'])

            # Collect the above two dataframes and store them in a dictionary ---------------------
            stanox_locations_data = dict(zip(workbook.sheet_names, [tiploc_loc_lyr, stanox]))

            workbook.close()

            save_pickle(stanox_locations_data, path_to_pickle)

        except Exception as e:
            print("Failed to fetch \"STANOX locations data\" from {}.".format(os.path.dirname(path_to_pickle)))
            print(e)
            stanox_locations_data = None

    return stanox_locations_data


# Location data with LocationId and WeatherCell, available from NR_METEX
def get_metex_db_location(update=False):

    pickle_filename = "db-metex-location.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        location = load_pickle(path_to_pickle)
    else:
        try:
            filename = "Location.csv"
            location = pd.read_csv(cdd_metex_db_tables_original(filename), dtype={'Id': str})
            location.rename(columns={'Id': 'LocationId'}, inplace=True)
            location.set_index('LocationId', inplace=True)
            location.WeatherCell = location.WeatherCell.map(lambda x: '' if pd.isna(x) else str(int(x)))
            # # Correct a known error
            # location.loc['610096', 'StartLongitude':'EndLatitude'] = [-0.0751, 51.5461, -0.0751, 51.5461]
            save_pickle(location, path_to_pickle)
        except Exception as e:
            print("Failed to get NR_METEX location data. {}".format(e))
            location = pd.DataFrame()

    return location


# STANOX location data available from NR_METEX
def get_metex_db_stanox_location(update=False):

    pickle_filename = "db-metex-stanox-location.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        stanox_location = load_pickle(path_to_pickle)
    else:
        try:
            filename = "StanoxLocation.csv"
            stanox_location = pd.read_csv(cdd_metex_db_tables_original(filename), dtype={'Stanox': str})

            # Cleanse "stanox_location"
            errata = load_json(cdd_rc("Errata.json"))
            errata_stanox, errata_tiploc, errata_stanme = errata.values()
            # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
            stanox_location.replace({'Stanox': errata_stanox}, inplace=True)
            stanox_location.replace({'Description': errata_tiploc}, inplace=True)
            stanox_location.replace({'Name': errata_stanme}, inplace=True)

            # Get reference data from the Railway Codes website
            rc_codes = rc.get_location_codes()['Locations']
            rc_codes = rc_codes[['Location', 'TIPLOC', 'STANME', 'STANOX']].drop_duplicates()

            # Fill in NA 'Description's (i.e. Location names)
            na_desc = stanox_location[stanox_location.Description.isnull()]
            temp = na_desc.join(rc_codes.set_index('STANOX'), on='Stanox')
            stanox_location.loc[na_desc.index, 'Description':'Name'] = temp[['Location', 'STANME']].values

            # Fill in NA 'Name's (i.e. STANME)
            na_name = stanox_location[stanox_location.Name.isnull()]
            # Some 'Description's are recorded by 'TIPLOC's instead
            rc_tiploc_dict = rc.get_location_codes_dictionary_v2(['TIPLOC'])
            temp = na_name.join(rc_tiploc_dict, on='Description')
            temp = temp.join(rc_codes.set_index(['STANOX', 'TIPLOC', 'Location']),
                             on=['Stanox', 'Description', 'Location'])
            temp = temp[temp.Location.notnull() & temp.STANME.notnull()]
            stanox_location.loc[temp.index, 'Description':'Name'] = temp[['Location', 'STANME']].values
            # Still, there are some NA 'Name's remaining due to incorrect spelling of 'TIPLOC' ('Description')
            na_name = stanox_location[stanox_location.Name.isnull()]
            temp = na_name.join(rc_codes.set_index('STANOX'), on='Stanox')
            stanox_location.loc[temp.index, 'Description':'Name'] = temp[['Location', 'STANME']].values

            # Apply manually-created dictionaries
            loc_name_replacement_dict = create_loc_name_replacement_dict('Description')
            stanox_location.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('Description')
            stanox_location.replace(loc_name_regexp_replacement_dict, inplace=True)

            # Check if 'Description' has STANOX code instead of location name using STANOX-dictionary
            rc_stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'])
            temp = stanox_location.join(rc_stanox_dict, on='Description')
            valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
            if not valid_loc.empty:
                stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                    lambda x: extractOne(x[1], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Check if 'Description' has TIPLOC code instead of location name using STANOX-TIPLOC-dictionary
            rc_stanox_tiploc_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'TIPLOC'])
            temp = stanox_location.join(rc_stanox_tiploc_dict, on=['Stanox', 'Description'])
            valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
            if not valid_loc.empty:
                stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                    lambda x: extractOne(x[1], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Check if 'Description' has STANME code instead of location name using STANOX-STANME-dictionary
            rc_stanox_stanme_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'STANME'])
            temp = stanox_location.join(rc_stanox_stanme_dict, on=['Stanox', 'Description'])
            valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
            if not valid_loc.empty:
                stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                    lambda x: extractOne(x[1], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Finalise cleansing 'Description' (i.e. location names)
            temp = stanox_location.join(rc_stanox_dict, on='Stanox')
            temp = temp[['Description', 'Name', 'Location']]
            stanox_location.Description = temp.apply(
                lambda x: extractOne(x[0], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Cleanse 'Name' (i.e. STANME)
            stanox_location.Name = stanox_location.Name.str.upper()

            loc_stanme_dict = rc_codes.groupby(['STANOX', 'Location']).agg({'STANME': list})
            loc_stanme_dict.STANME = loc_stanme_dict.STANME.map(lambda x: x[0] if len(x) == 1 else x)
            temp = stanox_location.join(loc_stanme_dict, on=['Stanox', 'Description'])
            stanox_location.Name = temp.apply(
                lambda x: extractOne(x[2], x[6])[0] if isinstance(x[6], list) else x[6], axis=1)

            # Below is not available in any reference data
            stanox_location.loc[298, 'Stanox':'Name'] = ['03330', 'Inverkeithing PPM Point', '']

            # Cleanse remaining NA values in 'ELR', 'Yards' and 'LocationId'
            stanox_location.ELR.fillna('', inplace=True)
            stanox_location.Yards = stanox_location.Yards.map(lambda x: '' if pd.isna(x) else str(int(x)))

            # Clear duplicates by 'STANOX'
            temp = stanox_location[stanox_location.duplicated(subset=['Stanox'], keep=False)]
            # temp.sort_values(['Description', 'LocationId'], ascending=[True, False], inplace=True)
            stanox_location.drop(index=temp[temp.LocationId.isna()].index, inplace=True)

            stanox_location.LocationId = stanox_location.LocationId.map(lambda x: '' if pd.isna(x) else str(int(x)))

            # Finish
            stanox_location.index = range(len(stanox_location))

            save_pickle(stanox_location, path_to_pickle)

        except Exception as e:
            print("Failed to get NR_METEX STANOX location data. {}.".format(e))
            stanox_location = pd.DataFrame()

    return stanox_location


# STANOX section data available from NR_METEX
def get_metex_db_stanox_section(update=False):

    pickle_filename = "db-metex-stanox-section.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        stanox_section = load_pickle(path_to_pickle)
    else:
        try:
            filename = "StanoxSection.csv"
            str_cols = {'StartStanox': str, 'EndStanox': str}
            stanox_section = pd.read_csv(cdd_metex_db_tables_original(filename), dtype=str_cols)

            errata = load_json(cdd_rc("Errata.json"))
            errata_stanox = errata['STANOX']

            stanox_section.rename(columns={'Id': 'StanoxSectionId', 'Description': 'StanoxSection'}, inplace=True)
            stanox_section.replace({'StartStanox': errata_stanox}, inplace=True)
            stanox_section.replace({'EndStanox': errata_stanox}, inplace=True)
            stanox_section.LocationId = stanox_section.LocationId.map(lambda x: '' if pd.isna(x) else str(int(x)))

            stanox_sec = stanox_section.copy(deep=True)

            stanox_sec[['Start_Raw', 'End_Raw']] = stanox_sec.StanoxSection.str.split(' : ').apply(pd.Series)
            stanox_sec[['Start', 'End']] = stanox_sec[['Start_Raw', 'End_Raw']]
            unknown_stanox_loc = load_json(cdd_rc("Problematic-STANOX-locations.json"))  # To solve duplicated STANOX
            stanox_sec.replace({'Start': unknown_stanox_loc}, inplace=True)
            stanox_sec.replace({'End': unknown_stanox_loc}, inplace=True)

            #
            stanox_location = get_metex_db_stanox_location()
            stanox_loc = stanox_location.set_index(['Stanox', 'LocationId'])
            temp = stanox_sec.join(stanox_loc, on=['StartStanox', 'LocationId'])
            temp = temp[temp.Description.notnull()]
            stanox_sec.loc[temp.index, 'Start'] = temp.Description
            temp = stanox_sec.join(stanox_loc, on=['EndStanox', 'LocationId'])
            temp = temp[temp.Description.notnull()]
            stanox_sec.loc[temp.index, 'End'] = temp.Description

            # Check if 'Start' and 'End' have STANOX codes instead of location names using STANOX-dictionary
            rc_stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'])
            temp = stanox_sec.join(rc_stanox_dict, on='Start')
            valid_loc = temp[temp.Location.notnull()]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.Location
            temp = stanox_sec.join(rc_stanox_dict, on='End')
            valid_loc = temp[temp.Location.notnull()]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.Location

            # Check if 'Start' and 'End' have STANOX/TIPLOC codes using STANOX-TIPLOC-dictionary
            rc_stanox_tiploc_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'TIPLOC'])
            temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['StartStanox', 'Start'])
            valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                    lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)
            temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['EndStanox', 'End'])
            valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                    lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

            # Check if 'Start' and 'End' have STANOX/STANME codes using STANOX-STANME-dictionary
            rc_stanox_stanme_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'STANME'])
            temp = stanox_sec.join(rc_stanox_stanme_dict, on=['StartStanox', 'Start'])
            valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                    lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)
            temp = stanox_sec.join(rc_stanox_stanme_dict, on=['EndStanox', 'End'])
            valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                    lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

            # Apply manually-created dictionaries
            loc_name_replacement_dict = create_loc_name_replacement_dict('Start')
            stanox_sec.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('Start')
            stanox_sec.replace(loc_name_regexp_replacement_dict, inplace=True)
            loc_name_replacement_dict = create_loc_name_replacement_dict('End')
            stanox_sec.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict('End')
            stanox_sec.replace(loc_name_regexp_replacement_dict, inplace=True)

            # Finalise cleansing
            stanox_sec.End.fillna(stanox_sec.Start, inplace=True)
            temp = stanox_sec.join(rc_stanox_dict, on='StartStanox')
            temp = temp[temp.Location.notnull()][['Start', 'Location']]
            stanox_sec.loc[temp.index, 'Start'] = temp.apply(
                lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)
            temp = stanox_sec.join(rc_stanox_dict, on='EndStanox')
            temp = temp[temp.Location.notnull()][['End', 'Location']]
            stanox_sec.loc[temp.index, 'End'] = temp.apply(
                lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

            #
            section = stanox_sec.Start + ' : ' + stanox_sec.End
            non_sec = stanox_sec.Start.eq(stanox_sec.End)
            section[non_sec] = stanox_sec[non_sec].Start
            stanox_section.StanoxSection = section
            stanox_section.insert(2, 'StartLocation', stanox_sec.Start.values)
            stanox_section.insert(3, 'EndLocation', stanox_sec.End.values)
            stanox_section[['StartLocation', 'EndLocation']] = stanox_sec[['Start', 'End']]

            # Add raw data to the original dataframe
            raw_col_names = ['StanoxSection_Raw', 'StartLocation_Raw', 'EndLocation_Raw']
            stanox_section[raw_col_names] = stanox_sec[['StanoxSection', 'Start_Raw', 'End_Raw']]

            save_pickle(stanox_section, path_to_pickle)

        except Exception as e:
            print("Failed to get NR_METEX STANOX section data. {}.".format(e))
            stanox_section = pd.DataFrame()

    return stanox_section


# Assemble data from NR_METEX to serve as metadata for location codes
def get_location_metadata_plus(update=False):

    pickle_filename = "db-metex-location-metadata.pickle"
    path_to_pickle = cdd_rc(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        metadata = load_pickle(path_to_pickle)
    else:
        try:
            # Location
            location = get_metex_db_location()

            # # STANOX location
            stanox_location = get_metex_db_stanox_location()
            stanox_location = stanox_location[['Stanox', 'Description', 'Name', 'ELR', 'Yards']]
            stanox_location.drop([2964, 6049, 2258, 1068, 2361, 4390], inplace=True)
            stanox_location.set_index(['Stanox', 'Description'], inplace=True)

            # STANOX section
            stanox_section = get_metex_db_stanox_section()

            # Get metadata by 'LocationId'
            metadata = stanox_section.join(location, on='LocationId')

            # --------------------------------------------------------------------------
            ref = get_location_metadata()['STANOX']
            ref.drop_duplicates(subset=['STANOX', 'LOOKUP_NAME'], inplace=True)
            ref.set_index(['STANOX', 'LOOKUP_NAME'], inplace=True)

            # Replace metadata's coordinates data with ref coordinates if available
            temp = metadata.join(ref, on=['StartStanox', 'StartLocation'])
            temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
            metadata.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[['DEG_LONG', 'DEG_LAT']].values
            metadata.loc[temp.index, 'ApproximateStartLocation'] = False
            metadata.loc[metadata.index.difference(temp.index), 'ApproximateStartLocation'] = True

            temp = metadata.join(ref, on=['EndStanox', 'EndLocation'])
            temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
            metadata.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[['DEG_LONG', 'DEG_LAT']].values
            metadata.loc[temp.index, 'ApproximateEndLocation'] = False
            metadata.loc[metadata.index.difference(temp.index), 'ApproximateEndLocation'] = True

            # Get reference metadata from RailwayCodes
            station_data = rc.get_station_locations()['Station']
            station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
            station_data.drop_duplicates(subset=['Station'], inplace=True)
            station_data.set_index('Station', inplace=True)

            temp = metadata.join(station_data, on='StartLocation')
            na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
            metadata.loc[na_i, 'StartLongitude':'StartLatitude'] = \
                temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
            metadata.loc[na_i, 'ApproximateStartLocation'] = False
            metadata.loc[~na_i, 'ApproximateStartLocation'] = True

            temp = metadata.join(station_data, on='EndLocation')
            na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
            metadata.loc[na_i, 'EndLongitude':'EndLatitude'] = \
                temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
            metadata.loc[na_i, 'ApproximateEndLocation'] = False
            metadata.loc[~na_i, 'ApproximateEndLocation'] = True

            metadata.ApproximateLocation = metadata.ApproximateStartLocation | metadata.ApproximateEndLocation

            # Finalise
            location_cols = stanox_location.columns
            start_cols = ['Start' + x for x in location_cols]
            metadata = metadata.join(stanox_location.drop_duplicates('Name'), on=['StartStanox', 'StartLocation'])
            metadata.rename(columns=dict(zip(location_cols, start_cols)), inplace=True)
            end_cols = ['End' + x for x in location_cols]
            metadata = metadata.join(stanox_location.drop_duplicates('Name'), on=['EndStanox', 'EndLocation'])
            metadata.rename(columns=dict(zip(location_cols, end_cols)), inplace=True)
            metadata[start_cols + end_cols].fillna('', inplace=True)

            save_pickle(metadata, path_to_pickle)

        except Exception as e:
            print("Failed to get \"db-metex-location-metadata.\" {}.".format(e))
            metadata = pd.DataFrame()

    return metadata


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
    stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'], as_dict=True)

    print("Cleansing duplicates by \"STANOX\" ... ", end="")

    errata_stanox, errata_tiploc, errata_stanme = load_json(cdd_rc("Errata.json")).values()
    # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
    data.Stanox = data.Stanox.replace(errata_stanox)
    data.Description = data.Description.replace(errata_tiploc)
    data.Name = data.Name.replace(errata_stanme)

    location_data_temp = tidy_duplicated_codes(stanox_dict, location_data)
    print("Successfully.")
    #
    stanme_dict = rc.get_location_codes_dictionary_v2(['STANME'], as_dict=True)
    print("Cleansing duplicates by \"STANME\" ... ", end="")
    location_data_temp = tidy_duplicated_codes(stanme_dict, location_data_temp)
    print("Successfully.")
    #
    tiploc_dict = rc.get_location_codes_dictionary_v2(['TIPLOC'], as_dict=True)
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
            ref_metadata_2 = rc.get_station_locations()['Station']
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
