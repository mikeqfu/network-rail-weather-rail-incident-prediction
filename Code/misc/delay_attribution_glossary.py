"""

Historic delay attribution glossary

https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/our-information-and-data/

"""

import datetime
import itertools
import os
import urllib.request

import numpy as np
import pandas as pd
import requests
from pyhelpers.dir import cdd
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

pd_preferences()


# Change directory to "Incidents\\Delay attribution\\Glossary\\Current\\..."
def cdd_delay_attr_glossary(*directories):
    path = cdd("Incidents\\Delay attribution\\Glossary\\Current")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Path to the original .xlsx file
def path_to_original_file():
    path_to_file = cdd_delay_attr_glossary("Delay attribution glossary.xlsx")
    return path_to_file


# ====================================================================================================================
""" """


# Download delay attribution glossary
def download_delay_attribution_glossary():
    spreadsheet_filename = "Historic-Delay-Attribution-Glossary.xlsx"

    current_year = datetime.datetime.now().year
    years = [str(x) for x in range(current_year, current_year + 1)]
    months = ['%.2d' % x for x in range(1, 13)]

    for y, m in list(itertools.product(years, months)):
        url = 'https://cdn.networkrail.co.uk/wp-content/uploads/{}/{}'.format(y + '/' + m, spreadsheet_filename)
        response = requests.get(url)
        if response.ok:
            path_to_file = cdd_delay_attr_glossary(
                spreadsheet_filename.replace("-", " ").replace("Historic ", "").capitalize())
            directory = cdd_delay_attr_glossary().replace(cdd(), '.\\Data')
            print("Downloading \"{}\" to \"{}\" ... ".format(spreadsheet_filename, directory), end="")
            try:
                urllib.request.urlretrieve(url, path_to_file)
                print("Successfully.")
            except Exception as e:
                print("Failed. {}".format(e))
            break


# ====================================================================================================================
""" """


# Stanox Codes
def get_stanox_codes(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_stanox_codes(update, hard_update, verbose)
    """
    pickle_filename = "stanox-codes.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        stanox_codes = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            stanox_codes = pd.read_excel(path_to_original_file(), sheet_name="Stanox Codes", dtype={'STANOX NO.': str})
            stanox_codes.columns = [x.strip('.').replace(' ', '_') for x in stanox_codes.columns]
            stanox_codes.STANOX_NO = stanox_codes.STANOX_NO.map(lambda x: '0' * (5 - len(x)) + x)
            save_pickle(stanox_codes, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Stanox Codes\". {}.".format(e))
            stanox_codes = None

    return stanox_codes


# Period Dates
def get_period_dates(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_period_dates(update, hard_update, verbose)
    """
    pickle_filename = "period-dates.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        period_dates = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            raw = pd.read_excel(path_to_original_file(), sheet_name="Period Dates", skiprows=3)
            raw.dropna(axis=1, how='all', inplace=True)

            periods = raw[['Unnamed: 0']].dropna()
            raw.drop('Unnamed: 0', axis='columns', inplace=True)
            periods.columns = ['Period']

            financial_years = [x.replace('YEAR ', '').replace('/', '-20') for x in raw.columns if 'Unnamed' not in x]

            raw.columns = raw.iloc[0]
            raw.drop(0, axis=0, inplace=True)

            no = int(raw.shape[1] / 3)
            period_dates_data = pd.concat(np.split(raw.dropna(axis='index'), no, axis=1), ignore_index=True, sort=False)

            period_dates_data.columns = [x.title().replace(' ', '_') for x in period_dates_data.columns]
            period_dates_data.rename(columns={'Day': 'Period_End_Day'}, inplace=True)
            period_dates_data.Period_End_Day = period_dates_data.Period_End_Date.dt.day_name()
            period_dates_data['Period_Start_Date'] = \
                period_dates_data.Period_End_Date - period_dates_data.No_Of_Days.map(lambda x: pd.Timedelta(days=x - 1))
            period_dates_data['Period_Start_Day'] = period_dates_data.Period_Start_Date.dt.day_name()
            period_dates_data['Period'] = periods.Period.to_list() * no
            period_dates_data['Financial_Year'] = np.repeat(financial_years, len(periods))

            period_dates = period_dates_data[['Financial_Year', 'Period', 'Period_Start_Date', 'Period_Start_Day',
                                              'Period_End_Date', 'Period_End_Day', 'No_Of_Days']]

            save_pickle(period_dates, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Period Dates\". {}.".format(e))
            period_dates = None

    return period_dates


# Incident Reason
def get_incident_reason_metadata(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_incident_reason_metadata(update, hard_update, verbose)
    """
    pickle_filename = "incident-reason-metadata.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_reason_metadata = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            incident_reason_metadata = pd.read_excel(path_to_original_file(), sheet_name="Incident Reason")
            incident_reason_metadata.columns = [x.replace(' ', '_') for x in incident_reason_metadata.columns]
            incident_reason_metadata.drop(incident_reason_metadata.tail(1).index, inplace=True)
            incident_reason_metadata.set_index('Incident_Reason', inplace=True)

            save_pickle(incident_reason_metadata, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Incident Reason\". {}.".format(e))
            incident_reason_metadata = None

    return incident_reason_metadata


# Responsible Manager
def get_responsible_manager(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_responsible_manager(update, hard_update, verbose)
    """
    pickle_filename = "responsible-manager.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        responsible_manager = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            responsible_manager = pd.read_excel(path_to_original_file(), sheet_name="Responsible Manager")
            responsible_manager.columns = [x.replace(' ', '_') for x in responsible_manager.columns]
            responsible_manager.Responsible_Manager_Name = responsible_manager.Responsible_Manager_Name.str.strip()
            save_pickle(responsible_manager, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Responsible Manager\". {}.".format(e))
            responsible_manager = None

    return responsible_manager


# Reactionary Reason Code
def get_reactionary_reason_code(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_reactionary_reason_code(update, hard_update, verbose)
    """
    pickle_filename = "reactionary-reason-code.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        reactionary_reason_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            reactionary_reason_code = pd.read_excel(path_to_original_file(), sheet_name="Reactionary Reason Code")
            reactionary_reason_code.columns = [x.replace(' ', '_') for x in reactionary_reason_code.columns]
            save_pickle(reactionary_reason_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Reactionary Reason Code\". {}.".format(e))
            reactionary_reason_code = None

    return reactionary_reason_code


# Performance Event Code
def get_performance_event_code(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_performance_event_code(update, hard_update, verbose)
    """
    pickle_filename = "performance-event-code.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        performance_event_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            performance_event_code = pd.read_excel(path_to_original_file(), sheet_name="Performance Event Code")
            # Rename columns
            performance_event_code.columns = [x.replace(' ', '_') for x in performance_event_code.columns]
            # Set an index
            performance_event_code.set_index('Performance_Event_Code', inplace=True)
            # Save the data as .pickle
            save_pickle(performance_event_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Performance Event Code\". {}.".format(e))
            performance_event_code = None

    return performance_event_code


# Train Service Code
def get_train_service_code(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_train_service_code(update, hard_update, verbose)
    """
    pickle_filename = "train-service-code.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        train_service_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            train_service_code = pd.read_excel(path_to_original_file(), sheet_name="Train Service Code")
            train_service_code.columns = [x.replace(' ', '_') for x in train_service_code.columns]
            save_pickle(train_service_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Train Service Code\". {}.".format(e))
            train_service_code = None

    return train_service_code


# Operator Name
def get_operator_name(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_operator_name(update, hard_update, verbose)
    """
    pickle_filename = "operator-name.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        operator_name = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            operator_name = pd.read_excel(path_to_original_file(), sheet_name="Operator Name")
            operator_name.columns = [x.replace(' ', '_') for x in operator_name.columns]
            save_pickle(operator_name, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Operator Name\". {}.".format(e))
            operator_name = None

    return operator_name


# Service Group Code
def get_service_group_code(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = False
        hard_update = False
        verbose = False

        get_service_group_code(update, hard_update, verbose)
    """
    pickle_filename = "service-group-code.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        service_group_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            service_group_code = pd.read_excel(path_to_original_file(), sheet_name="Service Group Code")
            service_group_code.columns = [x.replace(' ', '_') for x in service_group_code.columns]
            save_pickle(service_group_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Service Group Code\". {}.".format(e))
            service_group_code = None

    return service_group_code


# Historic delay attribution glossary
def get_delay_attr_glossary(update=False, hard_update=False, verbose=False):
    """
    :param update: [bool] (default: False)
    :param hard_update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        update = True
        hard_update = True
        verbose = True

        get_delay_attr_glossary(update, hard_update, verbose)
    """
    pickle_filename = "delay-attribution-glossary.pickle"
    path_to_pickle = cdd_delay_attr_glossary(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        delay_attr_glossary = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()

        try:
            glossary = [get_stanox_codes(update, hard_update=False, verbose=verbose),
                        get_period_dates(update, hard_update=False, verbose=verbose),
                        get_incident_reason_metadata(update, hard_update=False, verbose=verbose),
                        get_responsible_manager(update, hard_update=False, verbose=verbose),
                        get_reactionary_reason_code(update, hard_update=False, verbose=verbose),
                        get_performance_event_code(update, hard_update=False, verbose=verbose),
                        get_train_service_code(update, hard_update=False, verbose=verbose),
                        get_operator_name(update, hard_update=False, verbose=verbose),
                        get_service_group_code(update, hard_update=False, verbose=verbose)]

            workbook = pd.ExcelFile(path_to_original_file())
            delay_attr_glossary = dict(zip([x.replace(' ', '_') for x in workbook.sheet_names], glossary))
            workbook.close()

            save_pickle(glossary, path_to_pickle, verbose=True)

        except Exception as e:
            print("Failed to get \"Historic delay attribution glossary\". {}.".format(e))
            delay_attr_glossary = {None: None}

    return delay_attr_glossary
