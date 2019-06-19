"""

Historic delay attribution glossary

Source: https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/datasets/

"""

import itertools
import os
import urllib.request

import pandas as pd
import requests
from pyhelpers.dir import cdd
from pyhelpers.store import load_pickle, save_pickle


# Change directory to "Delay attribution"
def cdd_delay_attr(*directories):
    path = cdd("Incidents", "Delay attribution")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


def path_to_original_file():
    path_to_file = cdd_delay_attr("Delay attribution glossary.xlsx")
    return path_to_file


# ====================================================================================================================
""" """


# Download delay attribution glossary
def download_delay_attribution_glossary():
    spreadsheet_filename = "Historic-Delay-Attribution-Glossary.xlsx"

    years = [str(x) for x in range(2018, 2030)]
    months = ['%.2d' % x for x in range(1, 13)]

    for y, m in list(itertools.product(years, months)):
        url = 'https://cdn.networkrail.co.uk/wp-content/uploads/{}/{}'.format(y + '/' + m, spreadsheet_filename)
        response = requests.get(url)
        if response.ok:
            path_to_file = cdd_delay_attr(spreadsheet_filename.replace("-", " ").replace("Historic ", "").capitalize())
            directory = cdd_delay_attr().replace(cdd(), '.\\Data')
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
def get_stanox_codes(update=False, hard_update=False):
    pickle_filename = "stanox-codes.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        stanox_codes = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            stanox_codes = pd.read_excel(path_to_original_file(), sheet_name="Stanox Codes", dtype={'STANOX NO.': str})
            stanox_codes.columns = [x.strip('.').replace(' ', '_') for x in stanox_codes.columns]
            stanox_codes.STANOX_NO = stanox_codes.STANOX_NO.map(lambda x: '0' * (5 - len(x)) + x)
            save_pickle(stanox_codes, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Stanox Codes\". {}.".format(e))
            stanox_codes = None
    return stanox_codes


# Period Dates
def get_period_dates(update=False, hard_update=False):
    pickle_filename = "period-dates.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
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
            period_dates_data = raw.dropna(axis='index')
            period_dates_data['Period End Date'] = period_dates_data['Period End Date'].applymap(lambda x: x.date())
            period_dates_data.columns = [x.replace(' ', '_') for x in period_dates_data.columns]
            period_dates_data.index = periods.Period.values
            period_dates_list = [period_dates_data.iloc[:, i:i + 3] for i in range(0, period_dates_data.shape[1], 3)]
            period_dates = dict(zip(financial_years, period_dates_list))

            save_pickle(period_dates, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Period Dates\". {}.".format(e))
            period_dates = None

    return period_dates


# Incident Reason
def get_incident_reason_metadata(update=False, hard_update=False):
    pickle_filename = "incident-reason-metadata.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        incident_reason_metadata = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            # Get data from the original glossary file
            incident_reason_metadata = pd.read_excel(path_to_original_file(), sheet_name="Incident Reason")
            incident_reason_metadata.columns = [x.replace(' ', '_') for x in incident_reason_metadata.columns]
            incident_reason_metadata.drop(incident_reason_metadata.tail(1).index, inplace=True)
            incident_reason_metadata.set_index('Incident_Reason', inplace=True)
            # Save the data
            save_pickle(incident_reason_metadata, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Incident Reason\". {}.".format(e))
            incident_reason_metadata = None
    return incident_reason_metadata


# Responsible Manager
def get_responsible_manager(update=False, hard_update=False):
    pickle_filename = "responsible-manager.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        responsible_manager = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            responsible_manager = pd.read_excel(path_to_original_file(), sheet_name="Responsible Manager")
            responsible_manager.columns = [x.replace(' ', '_') for x in responsible_manager.columns]
            responsible_manager.Responsible_Manager_Name = responsible_manager.Responsible_Manager_Name.str.strip()
            save_pickle(responsible_manager, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Responsible Manager\". {}.".format(e))
            responsible_manager = None
    return responsible_manager


# Reactionary Reason Code
def get_reactionary_reason_code(update=False, hard_update=False):
    pickle_filename = "reactionary-reason-code.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        reactionary_reason_code = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            reactionary_reason_code = pd.read_excel(path_to_original_file(), sheet_name="Reactionary Reason Code")
            reactionary_reason_code.columns = [x.replace(' ', '_') for x in reactionary_reason_code.columns]
            save_pickle(reactionary_reason_code, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Reactionary Reason Code\". {}.".format(e))
            reactionary_reason_code = None
    return reactionary_reason_code


# Performance Event Code
def get_performance_event_code(update=False, hard_update=False):
    pickle_filename = "performance-event-code.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
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
            save_pickle(performance_event_code, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Performance Event Code\". {}.".format(e))
            performance_event_code = None
    return performance_event_code


# Train Service Code
def get_train_service_code(update=False, hard_update=False):
    pickle_filename = "train-service-code.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        train_service_code = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            train_service_code = pd.read_excel(path_to_original_file(), sheet_name="Train Service Code")
            train_service_code.columns = [x.replace(' ', '_') for x in train_service_code.columns]
            save_pickle(train_service_code, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Train Service Code\". {}.".format(e))
            train_service_code = None
    return train_service_code


# Operator Name
def get_operator_name(update=False, hard_update=False):
    pickle_filename = "operator-name.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        operator_name = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            operator_name = pd.read_excel(path_to_original_file(), sheet_name="Operator Name")
            operator_name.columns = [x.replace(' ', '_') for x in operator_name.columns]
            save_pickle(operator_name, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Operator Name\". {}.".format(e))
            operator_name = None
    return operator_name


# Service Group Code
def get_service_group_code(update=False, hard_update=False):
    pickle_filename = "service-group-code.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        service_group_code = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            service_group_code = pd.read_excel(path_to_original_file(), sheet_name="Service Group Code")
            service_group_code.columns = [x.replace(' ', '_') for x in service_group_code.columns]
            save_pickle(service_group_code, path_to_pickle)
        except Exception as e:
            print("Failed to get \"Service Group Code\". {}.".format(e))
            service_group_code = None
    return service_group_code


# Historic delay attribution glossary
def get_delay_attr_glossary(update=False, hard_update=False):
    pickle_filename = "delay-attribution-glossary.pickle"
    path_to_pickle = cdd_delay_attr(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        delay_attr_glossary = load_pickle(path_to_pickle)
    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary()
        try:
            glossary = [get_stanox_codes(update),
                        get_period_dates(update),
                        get_incident_reason_metadata(update),
                        get_responsible_manager(update),
                        get_reactionary_reason_code(update),
                        get_performance_event_code(update),
                        get_train_service_code(update),
                        get_operator_name(update),
                        get_service_group_code(update)]

            workbook = pd.ExcelFile(path_to_original_file())
            delay_attr_glossary = dict(zip([x.replace(' ', '_') for x in workbook.sheet_names], glossary))
            workbook.close()

            save_pickle(glossary, path_to_pickle)

        except Exception as e:
            print("Failed to get \"Historic delay attribution glossary\". {}.".format(e))
            delay_attr_glossary = {None: None}

    return delay_attr_glossary
