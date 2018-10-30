""" Schedule 8 incidents """

import itertools
import os
import re

import pandas as pd
from fuzzywuzzy.process import extractOne
from shapely.geometry import Point

import railwaycodes_utils as rc
from converters import osgb36_to_wgs84, wgs84_to_osgb36
from delay_attr_glossary import get_incident_reason_metadata
from loc_code_dict import create_location_names_regexp_replacement_dict, create_location_names_replacement_dict
from utils import cdd, cdd_rc, load_json, load_pickle, make_filename, save_pickle, subset


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
""" Location metadata """


# Pre-cleanse location metadata spreadsheet
def pre_cleanse_location_metadata(metadata):
    meta_dat = metadata.copy(deep=True)
    meta_dat.columns = [x.upper().replace(' ', '_') for x in metadata.columns]
    meta_dat.replace({'\xa0\xa0': ' '}, regex=True, inplace=True)
    meta_dat.replace({'\xa0': ''}, regex=True, inplace=True)
    meta_dat.replace({'LINE_DESCRIPTION': {re.compile('[Ss]tn'): 'Station',
                                           re.compile('[Ll]oc'): 'Location',
                                           re.compile('[Jj]n'): 'Junction',
                                           re.compile('[Ss]dg'): 'Siding',
                                           re.compile('[Ss]dgs'): 'Sidings'}}, inplace=True)
    meta_dat['LOOKUP_NAME_Raw'] = meta_dat.LOOKUP_NAME
    meta_dat.fillna({'STANOX': '', 'STANME': ''}, inplace=True)
    meta_dat.STANOX = meta_dat.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)
    return meta_dat


# Cleanse 'TIPLOC_LocationsLyr' sheet
def cleanse_location_metadata_tiploc_sheet(metadata, update_dict=False):
    """
    :param metadata:
    :param update_dict:
    :return:
    """

    meta_dat = pre_cleanse_location_metadata(metadata)

    meta_dat.TIPLOC = meta_dat.TIPLOC.fillna('').str.upper()

    ref_cols = ['STANOX', 'STANME', 'TIPLOC']
    dat = meta_dat[ref_cols + ['LOOKUP_NAME', 'DEG_LONG', 'DEG_LAT', 'LOOKUP_NAME_Raw']]

    # Rectify errors in STANOX
    errata = load_json(cdd_rc("errata.json"))  # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
    errata_stanox, errata_tiploc, errata_stanme = [{k: v} for k, v in errata.items()]
    dat.replace(errata_stanox, inplace=True)
    dat.replace(errata_stanme, inplace=True)
    dat.replace(errata_tiploc, inplace=True)

    # Rectify known issues for the location names in the data set
    location_names_replacement_dict = create_location_names_replacement_dict('LOOKUP_NAME')
    dat.replace(location_names_replacement_dict, inplace=True)
    location_names_regexp_replacement_dict = create_location_names_regexp_replacement_dict('LOOKUP_NAME')
    dat.replace(location_names_regexp_replacement_dict, inplace=True)

    # Fill in missing location names
    na_name = dat[dat.LOOKUP_NAME.isnull()]
    ref_dict = rc.get_location_codes_dictionary_v2(ref_cols, update=update_dict)
    temp = na_name.join(ref_dict, on=ref_cols)
    temp = temp[['TIPLOC', 'Location']]
    dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.apply(
        lambda x: extractOne(x[0], x[1])[0] if isinstance(x[1], list) else x[1], axis=1)

    # Rectify 'LOOKUP_NAME' according to 'TIPLOC'
    na_name = dat[dat.LOOKUP_NAME.isnull()]
    ref_dict = rc.get_location_codes_dictionary_v2(['TIPLOC'], update=update_dict)
    temp = na_name.join(ref_dict, on='TIPLOC')
    dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.Location.values

    not_na_name = dat[dat.LOOKUP_NAME.notnull()]
    temp = not_na_name.join(ref_dict, on='TIPLOC')

    def extract_one(lookup_name, ref_loc):
        if isinstance(ref_loc, list):
            n = extractOne(lookup_name.replace(' ', ''), ref_loc)[0]
        elif pd.isnull(ref_loc):
            n = lookup_name
        else:
            n = ref_loc
        return n

    dat.loc[not_na_name.index, 'LOOKUP_NAME'] = temp.apply(lambda x: extract_one(x[3], x[7]), axis=1)

    # Rectify 'STANOX'+'STANME'
    location_codes = rc.get_location_codes()['Locations']
    location_codes = location_codes.drop_duplicates(['TIPLOC', 'Location']).set_index(['TIPLOC', 'Location'])
    temp = dat.join(location_codes, on=['TIPLOC', 'LOOKUP_NAME'], rsuffix='_Ref').fillna('')
    dat.loc[temp.index, 'STANOX':'STANME'] = temp[['STANOX_Ref', 'STANME_Ref']].values

    # Update coordinates with reference data from RailwayCodes
    station_data = rc.get_station_locations()['Station']
    station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
    station_data = station_data.drop_duplicates(subset=['Station']).set_index('Station')
    temp = dat.join(station_data, on='LOOKUP_NAME')
    na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
    dat.loc[na_i, 'DEG_LONG':'DEG_LAT'] = temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values

    # Finalising...
    meta_dat.update(dat)
    meta_dat.dropna(subset=['LOOKUP_NAME'] + ref_cols, inplace=True)
    meta_dat.fillna('', inplace=True)
    meta_dat.sort_values(['LOOKUP_NAME', 'SHAPE_LENG'], ascending=[True, False], inplace=True)
    meta_dat.index = range(len(meta_dat))

    return meta_dat


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

            cols_with_na = ['STATUS', 'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
            tiploc_loc_lyr.fillna({x: '' for x in cols_with_na}, inplace=True)

            tiploc_loc_lyr = cleanse_location_metadata_tiploc_sheet(tiploc_loc_lyr, update_dict=update)

            # 'STANOX' -----------------------------------------------------------------------------
            stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})

            stanox = pre_cleanse_location_metadata(stanox)
            stanox.fillna({'LOOKUP_NAME_Raw': ''}, inplace=True)

            ref_cols = ['SHAPE_LENG', 'EASTING', 'NORTHING', 'GRIDREF']
            ref_data = tiploc_loc_lyr.set_index(ref_cols)
            stanox.drop_duplicates(ref_cols, inplace=True)
            temp = stanox.join(ref_data, on=ref_cols, rsuffix='_Ref').drop_duplicates(ref_cols)

            ref_cols_ok = [c for c in temp.columns if '_Ref' in c]
            stanox.loc[:, [c.replace('_Ref', '') for c in ref_cols_ok]] = temp[ref_cols_ok].values

            # Collect the above two dataframes and store them in a dictionary ---------------------
            stanox_locations_data = dict(zip(workbook.sheet_names, [tiploc_loc_lyr, stanox]))

            workbook.close()

            save_pickle(stanox_locations_data, path_to_pickle)

        except Exception as e:
            print("Failed to fetch \"STANOX locations data\" from {}.".format(os.path.dirname(path_to_pickle)))
            print(e)
            stanox_locations_data = None

    return stanox_locations_data


# ====================================================================================================================
""" Location metadata available from NR_METEX """


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
            errata = load_json(cdd_rc("errata.json"))  # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
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
            rc_tiploc_dict = rc.get_location_codes_dictionary_v2(['TIPLOC'], update=update)
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
            loc_name_replacement_dict = create_location_names_replacement_dict('Description')
            stanox_location.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_location_names_regexp_replacement_dict('Description')
            stanox_location.replace(loc_name_regexp_replacement_dict, inplace=True)

            # Check if 'Description' has STANOX code instead of location name using STANOX-dictionary
            rc_stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'], update=update)
            temp = stanox_location.join(rc_stanox_dict, on='Description')
            valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
            if not valid_loc.empty:
                stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                    lambda x: extractOne(x[1], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Check if 'Description' has TIPLOC code instead of location name using STANOX-TIPLOC-dictionary
            rc_stanox_tiploc_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'TIPLOC'], update=update)
            temp = stanox_location.join(rc_stanox_tiploc_dict, on=['Stanox', 'Description'])
            valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
            if not valid_loc.empty:
                stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                    lambda x: extractOne(x[1], x[2])[0] if isinstance(x[2], list) else x[2], axis=1)

            # Check if 'Description' has STANME code instead of location name using STANOX-STANME-dictionary
            rc_stanox_stanme_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'STANME'], update=update)
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

            errata = load_json(cdd_rc("errata.json"))  # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
            errata_stanox = errata['STANOX']

            stanox_section.rename(columns={'Id': 'StanoxSectionId', 'Description': 'StanoxSection'}, inplace=True)
            stanox_section.replace({'StartStanox': errata_stanox}, inplace=True)
            stanox_section.replace({'EndStanox': errata_stanox}, inplace=True)
            stanox_section.LocationId = stanox_section.LocationId.map(lambda x: '' if pd.isna(x) else str(int(x)))

            stanox_sec = stanox_section.copy(deep=True)

            stanox_sec[['Start_Raw', 'End_Raw']] = stanox_sec.StanoxSection.str.split(' : ').apply(pd.Series)
            stanox_sec[['Start', 'End']] = stanox_sec[['Start_Raw', 'End_Raw']]
            unknown_stanox_loc = load_json(cdd_rc("problematic-STANOX-locations.json"))  # To solve duplicated STANOX
            stanox_sec.replace({'Start': unknown_stanox_loc}, inplace=True)
            stanox_sec.replace({'End': unknown_stanox_loc}, inplace=True)

            #
            stanox_location = get_metex_db_stanox_location(update)
            stanox_loc = stanox_location.set_index(['Stanox', 'LocationId'])
            temp = stanox_sec.join(stanox_loc, on=['StartStanox', 'LocationId'])
            temp = temp[temp.Description.notnull()]
            stanox_sec.loc[temp.index, 'Start'] = temp.Description
            temp = stanox_sec.join(stanox_loc, on=['EndStanox', 'LocationId'])
            temp = temp[temp.Description.notnull()]
            stanox_sec.loc[temp.index, 'End'] = temp.Description

            # Check if 'Start' and 'End' have STANOX codes instead of location names using STANOX-dictionary
            rc_stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'], update=update)
            temp = stanox_sec.join(rc_stanox_dict, on='Start')
            valid_loc = temp[temp.Location.notnull()]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.Location
            temp = stanox_sec.join(rc_stanox_dict, on='End')
            valid_loc = temp[temp.Location.notnull()]
            if not valid_loc.empty:
                stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.Location

            # Check if 'Start' and 'End' have STANOX/TIPLOC codes using STANOX-TIPLOC-dictionary
            rc_stanox_tiploc_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'TIPLOC'], update=update)
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
            rc_stanox_stanme_dict = rc.get_location_codes_dictionary_v2(['STANOX', 'STANME'], update=update)
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
            loc_name_replacement_dict = create_location_names_replacement_dict('Start')
            stanox_sec.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_location_names_regexp_replacement_dict('Start')
            stanox_sec.replace(loc_name_regexp_replacement_dict, inplace=True)
            loc_name_replacement_dict = create_location_names_replacement_dict('End')
            stanox_sec.replace(loc_name_replacement_dict, inplace=True)
            loc_name_regexp_replacement_dict = create_location_names_regexp_replacement_dict('End')
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
            location = get_metex_db_location(update)

            # # STANOX location
            stanox_location = get_metex_db_stanox_location(update)
            stanox_location = stanox_location[['Stanox', 'Description', 'Name', 'ELR', 'Yards']]
            stanox_location.drop([2964, 6049, 2258, 1068, 2361, 4390], inplace=True)
            stanox_location.set_index(['Stanox', 'Description'], inplace=True)

            # STANOX section
            stanox_section = get_metex_db_stanox_section(update)

            # Get metadata by 'LocationId'
            metadata = stanox_section.join(location, on='LocationId')

            # --------------------------------------------------------------------------
            ref = get_location_metadata(update)['STANOX']
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
            metadata[start_cols + end_cols] = metadata[start_cols + end_cols].fillna('')

            save_pickle(metadata, path_to_pickle)

        except Exception as e:
            print("Failed to get \"db-metex-location-metadata.\" {}.".format(e))
            metadata = pd.DataFrame()

    return metadata


# ====================================================================================================================
""" Incidents """


# Cleanse location data
def cleanse_stanox_section_column(data, col_name='StanoxSection', sep=' : ', update_dict=False):
    """
    :param data:
    :param sep:
    :param col_name:
    :param update_dict:
    :return:
    """
    #
    dat = data.copy(deep=True)
    old_column_name = col_name + '_Raw'
    dat.rename(columns={col_name: old_column_name}, inplace=True)
    if sep is not None and isinstance(sep, str):
        start_end_raw = dat[old_column_name].str.split(sep, expand=True)
        #
        start_end_raw.columns = ['StartLocation_Raw', 'EndLocation_Raw']
        start_end_raw.EndLocation_Raw.fillna(start_end_raw.StartLocation_Raw, inplace=True)
        dat = dat.join(start_end_raw)
    else:
        start_end_raw = dat[[old_column_name]]

    errata = load_json(cdd_rc("errata.json"))  # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
    errata_stanox, errata_tiploc, errata_stanme = errata.values()
    start_end = start_end_raw.replace(errata_stanox)
    start_end.replace(errata_tiploc, inplace=True)
    start_end.replace(errata_stanme, inplace=True)

    # Get rid of duplicated records
    def tidy_alt_codes(code_dict_df, raw_loc):
        raw_loc.columns = [x.replace('_Raw', '') for x in raw_loc.columns]
        tmp = raw_loc.join(code_dict_df, on='StartLocation')
        tmp = tmp[tmp.Location.notnull()]
        raw_loc.loc[tmp.index, 'StartLocation'] = tmp.Location.values
        tmp = raw_loc.join(code_dict_df, on='EndLocation')
        tmp = tmp[tmp.Location.notnull()]
        raw_loc.loc[tmp.index, 'EndLocation'] = tmp.Location.values
        return raw_loc

    #
    stanox_dict = rc.get_location_codes_dictionary_v2(['STANOX'], update=update_dict)
    start_end = tidy_alt_codes(stanox_dict, start_end)
    #
    stanme_dict = rc.get_location_codes_dictionary_v2(['STANME'], update=update_dict)
    start_end = tidy_alt_codes(stanme_dict, start_end)
    #
    tiploc_dict = rc.get_location_codes_dictionary_v2(['TIPLOC'], update=update_dict)
    start_end = tidy_alt_codes(tiploc_dict, start_end)

    #
    location_names_replacement_dict = create_location_names_replacement_dict()
    start_end.replace(location_names_replacement_dict, inplace=True)
    location_names_regexp_replacement_dict = create_location_names_regexp_replacement_dict()
    start_end.replace(location_names_regexp_replacement_dict, regex=True, inplace=True)

    # ref = rc.get_location_codes()['Locations']
    # ref.drop_duplicates(subset=['Location'], inplace=True)
    # ref_loc = list(set(ref.Location))
    #
    # temp = start_end_raw.join(ref.set_index('Location'), on='StartLocation')
    # temp_loc = list(set(temp[temp.STANOX.isnull()].StartLocation))
    #
    # temp = start_end_raw.join(ref.set_index('Location'), on='EndLocation')
    # temp_loc = list(set(temp[temp.STANOX.isnull()].EndLocation))

    # Create new StanoxSection column
    dat[col_name] = start_end.StartLocation + sep + start_end.EndLocation
    mask_single = start_end.StartLocation == start_end.EndLocation
    dat[col_name][mask_single] = start_end.StartLocation[mask_single]

    # Resort column order
    col_names = [list(v) for k, v in itertools.groupby(list(dat.columns), lambda x: x == old_column_name) if not k]
    add_names = [old_column_name] + col_names[1][-3:] + ['StartLocation', 'EndLocation']
    col_names = col_names[0] + add_names + col_names[1][:-3]

    cleansed_data = dat.join(start_end)[col_names]

    return cleansed_data


# Look up geographical coordinates for each incident location
def cleanse_geographical_coordinates(data, update_metadata=False):

    dat = data.copy(deep=True)

    # Find geographical coordinates for each incident location
    ref_loc_dat_1, _ = get_location_metadata(update=update_metadata).values()
    coords_cols = ['EASTING', 'NORTHING', 'DEG_LONG', 'DEG_LAT']
    coords_cols_alt = ['Easting', 'Northing', 'Longitude', 'Latitude']
    ref_loc_dat_1.rename(columns=dict(zip(coords_cols, coords_cols_alt)), inplace=True)

    ref_loc_dat_1 = ref_loc_dat_1.drop_duplicates('LOOKUP_NAME').set_index('LOOKUP_NAME')
    dat = dat.join(ref_loc_dat_1[coords_cols_alt], on='StartLocation')
    dat.rename(columns=dict(zip(coords_cols_alt, ['Start' + c for c in coords_cols_alt])), inplace=True)
    dat = dat.join(ref_loc_dat_1[coords_cols_alt], on='EndLocation')
    dat.rename(columns=dict(zip(coords_cols_alt, ['End' + c for c in coords_cols_alt])), inplace=True)

    # Get location metadata for reference --------------------------------
    location_metadata = get_location_metadata_plus(update=update_metadata)
    start_locations = location_metadata[['StartLocation', 'StartLongitude', 'StartLatitude']]
    start_locations.columns = [c.replace('Start', '') for c in start_locations.columns]
    end_locations = location_metadata[['EndLocation', 'EndLongitude', 'EndLatitude']]
    end_locations.columns = [c.replace('End', '') for c in end_locations.columns]
    loc_metadata = pd.concat([start_locations, end_locations], ignore_index=True)
    loc_metadata = loc_metadata.drop_duplicates('Location').set_index('Location')

    # Fill in NA coordinates
    temp = dat[dat.StartEasting.isnull() | dat.StartLongitude.isnull()]
    temp = temp.join(loc_metadata, on='StartLocation')
    dat.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[['Longitude', 'Latitude']].values
    dat.loc[temp.index, 'StartEasting':'StartNorthing'] = [
        list(wgs84_to_osgb36(x[0], x[1])) for x in temp[['Longitude', 'Latitude']].values]

    temp = dat[dat.EndEasting.isnull() | dat.EndLongitude.isnull()]
    temp = temp.join(loc_metadata, on='EndLocation')
    dat.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[['Longitude', 'Latitude']].values

    # Dalston Junction (East London Line)     --> Dalston Junction [-0.0751, 51.5461]
    # Ashford West Junction (CTRL)            --> Ashford West Junction [0.86601557, 51.146927]
    # Southfleet Junction                     --> ? [0.34262910, 51.419354]
    # Channel Tunnel Eurotunnel Boundary CTRL --> ? [1.1310482, 51.094808]
    na_loc = ['Dalston Junction (East London Line)', 'Ashford West Junction (CTRL)',
              'Southfleet Junction', 'Channel Tunnel Eurotunnel Boundary CTRL']
    na_loc_longlat = [[-0.0751, 51.5461], [0.86601557, 51.146927], [0.34262910, 51.419354], [1.1310482, 51.094808]]
    for x, longlat in zip(na_loc, na_loc_longlat):
        if x in list(temp.EndLocation):
            idx = temp[temp.EndLocation == x].index
            temp.loc[idx, 'EndLongitude':'Latitude'] = longlat * 2
            dat.loc[idx, 'EndLongitude':'EndLatitude'] = longlat

    dat.loc[temp.index, 'EndEasting':'EndNorthing'] = [
        list(wgs84_to_osgb36(x[0], x[1])) for x in temp[['Longitude', 'Latitude']].values]

    # ref 2 ----------------------------------------------
    ref_metadata_2 = rc.get_station_locations()['Station']
    ref_metadata_2 = ref_metadata_2[['Station', 'Degrees Longitude', 'Degrees Latitude']]
    ref_metadata_2 = ref_metadata_2.dropna().drop_duplicates('Station')
    ref_metadata_2.columns = [x.replace('Degrees ', '') for x in ref_metadata_2.columns]
    ref_metadata_2.set_index('Station', inplace=True)

    temp = dat.join(ref_metadata_2, on='StartLocation')
    temp_start = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
    dat.loc[temp_start.index, 'StartLongitude':'StartLatitude'] = temp_start[['Longitude', 'Latitude']].values

    temp = dat.join(ref_metadata_2, on='EndLocation')
    temp_end = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
    dat.loc[temp_end.index, 'EndLongitude':'EndLatitude'] = temp_end[['Longitude', 'Latitude']].values

    # Let (Longitude, Latitude) be almost equivalent to (Easting, Northing)
    dat.loc[:, 'StartLongitude':'StartLatitude'] = dat.loc[:, 'StartEasting':'StartNorthing'].apply(
        lambda x: osgb36_to_wgs84(x['StartEasting'], x['StartNorthing']), axis=1).values.tolist()
    dat.loc[:, 'EndLongitude':'EndLatitude'] = dat.loc[:, 'EndEasting':'EndNorthing'].apply(
        lambda x: osgb36_to_wgs84(x['EndEasting'], x['EndNorthing']), axis=1).values.tolist()

    #
    def convert_to_point(x, h_col, v_col):
        if pd.np.isnan(x[h_col]) or pd.np.isnan(x[v_col]):
            p = Point()
        else:
            p = Point((x[h_col], x[v_col]))
        return p

    # Convert coordinates to shapely.geometry.Point
    dat['StartLongLat'] = dat.apply(lambda x: convert_to_point(x, 'StartLongitude', 'StartLatitude'), axis=1)
    dat['StartNE'] = dat.apply(lambda x: convert_to_point(x, 'StartEasting', 'StartNorthing'), axis=1)
    dat['EndLongLat'] = dat.apply(lambda x: convert_to_point(x, 'EndLongitude', 'EndLatitude'), axis=1)
    dat['EndNE'] = dat.apply(lambda x: convert_to_point(x, 'EndEasting', 'EndNorthing'), axis=1)

    # temp = dat[[x for x in dat.columns if x.startswith('Start')]]  # For the start locations
    # temp['StartLongLat_temp'] = temp.apply(
    #     lambda x: Point(osgb36_to_wgs84(x['StartEasting'], x['StartNorthing']))
    #     if not x['StartNE'].is_empty else x['StartNE'], axis=1)
    # temp['distance'] = temp.apply(lambda x: x['StartLongLat'].distance(x['StartLongLat_temp']), axis=1)
    # idx = temp[temp.distance.map(lambda x: True if x > 0.00005 else False)].index
    # dat.loc[idx, 'StartLongLat'] = temp.loc[idx, 'StartLongLat_temp']
    # dat.loc[idx, 'StartLongitude':'StartLatitude'] = temp.loc[idx, 'StartLongLat_temp'].apply(
    #     lambda x: [x.x, x.y]).values.tolist()
    #
    # temp = dat[[x for x in dat.columns if x.startswith('End')]]  # For the end locations
    # temp['EndLongLat_temp'] = temp.apply(
    #     lambda x: Point(osgb36_to_wgs84(x['EndEasting'], x['EndNorthing']))
    #     if not x['EndNE'].is_empty else x['EndNE'], axis=1)
    # temp['distance'] = temp.apply(lambda x: x['EndLongLat'].distance(x['EndLongLat_temp']), axis=1)
    # idx = temp[temp.distance.map(lambda x: True if x > 0.00005 else False)].index
    # dat.loc[idx, 'EndLongLat'] = temp.loc[idx, 'EndLongLat_temp']
    # dat.loc[idx, 'EndLongitude':'EndLatitude'] = temp.loc[idx, 'EndLongLat_temp'].apply(
    #     lambda x: [x.x, x.y]).values.tolist()

    return dat


# Schedule 8 weather incidents
def get_schedule8_weather_incidents(route_name=None, weather_category=None, update=False):

    pickle_filename = make_filename("Schedule8WeatherIncidents", route_name, weather_category)
    path_to_pickle = cdd_incidents("Spreadsheets", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        data = load_pickle(path_to_pickle)
    else:
        try:
            # Load data from the raw file
            data = pd.read_excel(path_to_pickle.replace(".pickle", ".xlsx"),
                                 parse_dates=['StartDate', 'EndDate'], day_first=True,
                                 converters={'stanoxSection': str})
            data.rename(columns={'StartDate': 'StartDateTime',
                                 'EndDate': 'EndDateTime',
                                 'stanoxSection': 'StanoxSection',
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
            incident_reason_metadata.columns = [c.replace('_', '') for c in incident_reason_metadata.columns]
            data = data.join(incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
            data.drop([x for x in data.columns if '_meta' in x], axis=1, inplace=True)

            # Cleanse the location data
            data = cleanse_stanox_section_column(data, col_name='StanoxSection', sep=' : ', update_dict=update)

            # Look up geographical coordinates for each incident location
            data = cleanse_geographical_coordinates(data)

            # Retain data for specific Route and weather category
            data = subset(data, route_name, weather_category)

            save_pickle(data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Schedule 8 weather incidents\". {}".format(e))
            data = pd.DataFrame()

    return data


# Schedule8WeatherIncidents-02062006-31032014.xlsm
def get_schedule8_weather_incidents_02062006_31032014(route_name=None, weather_category=None, update=False):
    """
    Description:
    "Details of schedule 8 incidents together with weather leading up to the incident. Although this file contains
    other weather categories, the main focus of this prototype is adhesion."

    "* WORK IN PROGRESS *  MET-9 - Report of Schedule 8 adhesion incidents vs weather conditions Done."

    """
    # Path to the file
    pickle_filename = make_filename("Schedule8WeatherIncidents-02062006-31032014", route_name, weather_category)
    path_to_pickle = cdd_incidents("Spreadsheets", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        workbook_data = load_pickle(path_to_pickle)
    else:
        try:
            # Open the original file
            path_to_xlsm = path_to_pickle.replace(".pickle", ".xlsm")
            workbook = pd.ExcelFile(path_to_xlsm)

            # 'Thresholds' -------------------------------------------------------------
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

            # 'Data' ------------------------------------------------------------------------------------
            data = workbook.parse(sheet_name='Data', parse_dates=['StartDate', 'EndDate'], dayfirst=True,
                                  converters={'stanoxSection': str})
            data.rename(columns={'StartDate': 'StartDateTime',
                                 'EndDate': 'EndDateTime',
                                 'Year': 'FinancialYear',
                                 'stanoxSection': 'StanoxSection',
                                 'imdm': 'IMDM',
                                 'Reason': 'IncidentReason',
                                 # 'Minutes': 'DelayMinutes',
                                 # 'Cost': 'DelayCost',
                                 'CategoryDescription': 'IncidentCategoryDescription'}, inplace=True)
            hazard_cols = [x for x in enumerate(data.columns) if 'Weather Hazard' in x[1]]
            obs_cols = [(i - 1, re.search('(?<= \()\w+', x).group().upper()) for i, x in hazard_cols]
            hazard_cols = [(i + 1, x + '_WeatherHazard') for i, x in obs_cols]
            for i, x in obs_cols + hazard_cols:
                data.rename(columns={data.columns[i]: x}, inplace=True)

            # data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')

            incident_reason_metadata = get_incident_reason_metadata()
            incident_reason_metadata.columns = [c.replace('_', '') for c in incident_reason_metadata.columns]
            data = data.join(incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
            data.drop([x for x in data.columns if '_meta' in x], axis=1, inplace=True)

            # Cleanse the location data
            data = cleanse_stanox_section_column(data, col_name='StanoxSection', sep=' : ', update_dict=update)

            # Look up geographical coordinates for each incident location
            data = cleanse_geographical_coordinates(data)

            # Retain data for specific Route and weather category
            data = subset(data, route_name, weather_category)

            # Weather'CategoryLookup' -------------------------------------------
            weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
            weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

            # Make a dictionary
            workbook_data = dict(zip(workbook.sheet_names, [thresholds, data, weather_category_lookup]))

            workbook.close()

            # Save the workbook data
            save_pickle(workbook_data, path_to_pickle)

        except Exception as e:
            print('Failed to get \"Schedule8WeatherIncidents-02062006-31032014.xlsm\" due to {}.'.format(e))
            workbook_data = None

    return workbook_data
