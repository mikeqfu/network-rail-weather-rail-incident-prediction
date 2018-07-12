import os
import itertools
import requests
import urllib.request

import pandas as pd

from utils import cdd, cdd_rc, load_pickle, save_pickle
from loc_code_dict import *

import railwaycodes_utils as rc


# Change directory to "Schedule 8 incidents"
def cdd_schedule8(*directories):
    path = cdd("Schedule 8 incidents")
    for directory in directories:
        path = os.path.join(path, directory)
    return path



# STANOX locations data
def get_stanox_locations_data(update=False):
    filename = "STANOXLocationsData"
    path_to_file = cdd_rc(filename + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        stanox_locations_data = load_pickle(path_to_file)
    else:
        try:
            workbook = pd.ExcelFile(cdd_rc(filename + ".xlsx"))

            # 'TIPLOC_LocationsLyr' -----------------------------------------------
            tiploc_locations_lyr = workbook.parse(sheet_name='TIPLOC_LocationsLyr',
                                                  parse_dates=['LASTEDITED', 'LAST_UPD_1'], dayfirst=True,
                                                  converters={'STANOX': str})
            fillna_columns_str = ['STANOX', 'STANME', 'TIPLOC', 'LOOKUP_NAME', 'STATUS',
                                  'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
            tiploc_locations_lyr.fillna({x: '' for x in fillna_columns_str}, inplace=True)
            tiploc_locations_lyr.STANOX = tiploc_locations_lyr.STANOX.map(
                lambda x: '0' * (5 - len(x)) + x if x != '' else x)

            # 'STANOX' -------------------------------------------------------------
            stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})
            stanox.STANOX = stanox.STANOX.map(lambda x: '0' * (5 - len(x)) + x if x != '' else x)

            stanox_locations_data = dict(zip(workbook.sheet_names, [tiploc_locations_lyr, stanox]))

            save_pickle(stanox_locations_data, path_to_file)

        except Exception as e:
            print("Failed to fetch {} from {} due to {}".format(filename, os.path.dirname(path_to_file), e))
            stanox_locations_data = None

    return stanox_locations_data


# Download delay attribution glossary
def download_historic_delay_attribution_glossary():
    filename = "Historic-Delay-Attribution-Glossary.xlsx"

    years = [str(x) for x in range(2018, 2030)]
    months = ['%.2d' % x for x in range(1, 13)]

    for y, m in list(itertools.product(years, months)):
        url = 'https://cdn.networkrail.co.uk/wp-content/uploads/{}/{}'.format(y + '/' + m, filename)
        response = requests.get(url)
        if response.ok:
            path_to_file = cdd_schedule8("Delay attribution", filename.capitalize().replace("-", " "))
            print("Downloading ... ", end="")
            try:
                urllib.request.urlretrieve(url, path_to_file)
                print("Successfully.")
            except Exception as e:
                print("Failed due to {}".format(e))
            break


# Get metadata about incident reasons
def get_incident_reason_metadata(update=False):
    filename = "incident_reason_metadata"
    path_to_file = cdd_schedule8("Delay attribution", filename + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        incident_reason_metadata = load_pickle(path_to_file)
    else:
        path_to_original_file = cdd_schedule8("Delay attribution", "Historic delay attribution glossary.xlsx")
        if not os.path.isfile(path_to_original_file) or update:
            download_historic_delay_attribution_glossary()
        else:
            pass
        try:
            # Get data from the original glossary file
            incident_reason_metadata = pd.read_excel(path_to_original_file, sheet_name="Incident Reason")
            incident_reason_metadata.columns = [x.replace(' ', '') for x in incident_reason_metadata.columns]
            incident_reason_metadata.drop(incident_reason_metadata.tail(1).index, inplace=True)
            incident_reason_metadata.set_index('IncidentReason', inplace=True)
            # Save the data
            save_pickle(incident_reason_metadata, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(path_to_file, e))
            incident_reason_metadata = None
    return incident_reason_metadata


# ====================================================================================================================


# Cleanse location data
def cleanse_location_data(data, loc_col_names='StanoxSection', sep=' : '):
    """

    :param data:
    :param sep:
    :param loc_col_names:
    :return:
    """
    old_column_name = loc_col_names + '_Raw'
    data.rename(columns={loc_col_names: old_column_name}, inplace=True)
    stanox_section = data[old_column_name].str.split(sep, expand=True)

    stanox_section.columns = ['StartLocation_Raw', 'EndLocation_Raw']
    stanox_section.EndLocation_Raw.fillna(stanox_section.StartLocation_Raw, inplace=True)

    stanox_dict = rc.get_location_dictionary('STANOX', drop_duplicates=False)

    stanox_section.StartLocation_Raw = stanox_section.StartLocation_Raw.replace(stanox_dict)
    stanox_section.EndLocation_Raw = stanox_section.EndLocation_Raw.replace(stanox_dict)

    stanme_dict = rc.get_location_dictionary('STANME')
    tiploc_dict = rc.get_location_dictionary('TIPLOC')
    loc_name_replacement_dict = create_loc_name_replacement_dict()
    loc_name_regexp_replacement_dict = create_loc_name_regexp_replacement_dict()
    # Processing 'StartStanox'
    stanox_section.StartLocation_Raw = stanox_section.StartLocation_Raw. \
        replace(stanme_dict).replace(tiploc_dict). \
        replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
    # Processing 'EndStanox_loc'
    stanox_section.EndLocation_Raw = stanox_section.EndLocation_Raw. \
        replace(stanme_dict).replace(tiploc_dict). \
        replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

    # Form new STANOX sections
    stanox_section[loc_col_names] = stanox_section.StartLocation_Raw + ' - ' + stanox_section.EndLocation_Raw
    point_idx = stanox_section.StartLocation_Raw == stanox_section.EndLocation_Raw
    stanox_section[loc_col_names][point_idx] = stanox_section.StartLocation_Raw[point_idx]

    # Resort column order
    col_names = list(data.columns)
    col_names.insert(col_names.index(loc_col_names) + 1, 'StartLocation')
    col_names.insert(col_names.index('StartLocation') + 1, 'EndLocation')
    data = stanox_section.join(data.drop(loc_col_names, axis=1))[col_names]

    return data


# Schedule 8 weather incidents
def read_schedule8_weather_incidents(update=False):
    filename = "Schedule8WeatherIncidents.xlsx"
    path_to_file = cdd_schedule8("Spreadsheets", os.path.splitext(filename)[0] + ".pickle")
    if os.path.isfile(path_to_file) and not update:
        data = load_pickle(path_to_file)
    else:
        try:
            data = pd.read_excel(cdd_schedule8("Spreadsheets", filename),
                                 parse_dates=['StartDate', 'EndDate'], day_first=True)
            data.rename(columns={'stanoxSection': 'StanoxSection', 'imdm': 'IMDM',
                                 'WeatherCategory': 'WeatherCategoryCode',
                                 'WeatherCategory.1': 'WeatherCategory',
                                 'Reason': 'IncidentReason',
                                 # 'Minutes': 'DelayMinutes',
                                 'Description': 'IncidentReasonDescription',
                                 'Category': 'IncidentCategory',
                                 'CategoryDescription': 'IncidentCategoryDescription'}, inplace=True)
            incident_reason_metadata = get_incident_reason_metadata(update=False)
            data = data.join(incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
            data.drop([x for x in data.columns if '_meta' in x], axis=1, inplace=True)

            data[['StartLocation_LookUp', 'EndLocation_LookUp']] = data.StanoxSection.str.split(' : ', expand=True)
            data.EndLocation_LookUp.fillna(data.StartLocation_LookUp, inplace=True)

            stanox_locations_data = get_stanox_locations_data()['STANOX']
            stanox_locations_data.set_index('LOOKUP_NAME', inplace=True)

            data.join(stanox_locations_data, on=['StartLocation_LookUp'], lsuffix='_Start')






            data = pd.concat([data, stanox_section], axis=1)

            save_pickle(data, path_to_file)
        except Exception as e:
            print("Failed to get {} due to {}".format(filename, e))
            data = None
    return data


# ====================================================================================================================
""" S8_Weather 02_06_2006 - 31-03-2014.xlsm """


def get_schedule8_weather_incidents_02062006_31032014(route=None, weather=None, update=False):
    """
    Description:
    "Details of schedule 8 incidents together with weather leading up to the incident. Although this file contains
    other weather categories, the main focus of this prototype is adhesion.

    * WORK IN PROGRESS *  MET-9 - Report of Schedule 8 adhesion incidents vs weather conditions Done."

    """
    # Path to the file
    filename = "S8_Weather 02_06_2006 - 31-03-2014"
    filename_modified = "_".join(filter(None, [re.sub('[_-]', '', x) for x in filename.split(" ")]))
    path_to_file = cdd_schedule8("Spreadsheets", filename_modified + ".pickle")

    if os.path.isfile(path_to_file) and not update:
        workbook_data = load_pickle(path_to_file)
    else:
        try:
            # Open the original file
            workbook = pd.ExcelFile(cdd_schedule8("Spreadsheets", filename + ".xlsm"))

            # 'Thresholds' ================================================================
            thresholds = workbook.parse(sheet_name='Thresholds', parse_cols='A:F').dropna()
            thresholds.index = range(len(thresholds))
            thresholds.columns = [col.replace(' ', '') for col in thresholds.columns]
            thresholds.WeatherHazard = thresholds.WeatherHazard.map(lambda x: x.upper().strip())

            # 'Data' =========================================================================================
            data = workbook.parse('Data', parse_dates=False, dayfirst=True, converters={'stanoxSection': str})
            data.columns = [c.replace('(C)', '(degrees Celcius)').replace(' ', '') for c in data.columns]
            data.rename(columns={'imdm': 'IMDM',
                                 'stanoxSection': 'StanoxSection',
                                 'Minutes': 'DelayMinutes',
                                 'Cost': 'DelayCost',
                                 'Reason': 'IncidentReason',
                                 'CategoryDescription': 'IncidentCategoryDescription',
                                 'WeatherHazard(pdmint)': 'PreviousDayMinTemperature_WeatherHazard',
                                 'WeatherHazard(pdmaxt)': 'PreviousDayMaxTemperature_WeatherHazard',
                                 'WeatherHazard(pddt)': 'PreviousDayDeltaTemperature_WeatherHazard',
                                 'WeatherHazard(pdrh)': 'PreviousDayRelativeHumidity_WeatherHazard',
                                 '3-HourRain(mm)': 'PreviousThreeHourRain(mm)',
                                 'WeatherHazard(thr)': 'PreviousThreeHourRain_WeatherHazard',
                                 'DailyRain(mm)': 'PreviousDailyRain(mm)',
                                 'WeatherHazard(pdr)': 'PreviousDailyRain_WeatherHazard',
                                 '15-DayRain(mm)': 'PreviousFifteenDayRain(mm)',
                                 'WeatherHazard(pfdr)': 'PreviousFifteenDayRain_WeatherHazard',
                                 'DailySnow(cm)': 'PreviousDailySnow(cm)',
                                 'WeatherHazard(pds)': 'PreviousDailySnow_WeatherHazard'}, inplace=True)
            data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')

            stanox_section = data.StanoxSection.str.split(' : ', expand=True)
            stanox_section.columns = ['StartLocation', 'EndLocation']
            stanox_section.EndLocation.fillna(stanox_section.StartLocation, inplace=True)

            stanox_dict_1 = dbm.get_stanox_location().Location.to_dict()
            stanox_dict_2 = rc.get_location_dictionary('STANOX', drop_duplicates=False)

            stanox_section.StartLocation = stanox_section.StartLocation.replace(stanox_dict_1).replace(stanox_dict_2)
            stanox_section.EndLocation = stanox_section.EndLocation.replace(stanox_dict_1).replace(stanox_dict_2)

            stanme_dict = rc.get_location_dictionary('STANME')
            tiploc_dict = rc.get_location_dictionary('TIPLOC')
            loc_name_replacement_dict = dbm.create_loc_name_replacement_dict()
            loc_name_regexp_replacement_dict = dbm.create_loc_name_regexp_replacement_dict()
            # Processing 'StartStanox'
            stanox_section.StartLocation = stanox_section.StartLocation. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_loc'
            stanox_section.EndLocation = stanox_section.EndLocation. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

            # Form new STANOX sections
            stanox_section['StanoxSection'] = stanox_section.StartLocation + ' - ' + stanox_section.EndLocation
            point_idx = stanox_section.StartLocation == stanox_section.EndLocation
            stanox_section.StanoxSection[point_idx] = stanox_section.StartLocation[point_idx]

            # Resort column order
            col_names = list(data.columns)
            col_names.insert(col_names.index('StanoxSection') + 1, 'StartLocation')
            col_names.insert(col_names.index('StartLocation') + 1, 'EndLocation')
            data = stanox_section.join(data.drop('StanoxSection', axis=1))[col_names]

            # Add incident reason metadata --------------------------
            incident_reason_metadata = get_incident_reason_metadata()
            data = pd.merge(data, incident_reason_metadata.reset_index(),
                            on=['IncidentReason', 'IncidentCategoryDescription'], how='inner')

            # Weather'CategoryLookup' ===========================================
            weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
            weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

            # Make a dictionary
            workbook_data = dict(zip(workbook.sheet_names, [thresholds, data, weather_category_lookup]))
            workbook.close()
            # Save the workbook data
            save_pickle(workbook_data, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(filename, e))
            workbook_data = None

        # Retain data for specific Route and weather category
        workbook_data['Data'] = dbm.subset(workbook_data['Data'], route, weather)

    return workbook_data




