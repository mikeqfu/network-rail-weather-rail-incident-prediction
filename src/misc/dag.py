""" Historic delay attribution glossary

Information provided by Network Rail – https://www.networkrail.co.uk/transparency/datasets/
"""

import datetime
import itertools
import os
import urllib.request

import fake_useragent
import numpy as np
import pandas as pd
import requests
from pyhelpers.ops import confirmed
from pyhelpers.settings import pd_preferences
from pyhelpers.store import load_pickle, save_pickle

from utils import cdd_incidents

pd_preferences()


def cdd_dag(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\incidents\\delay attribution\\glossary\\current\\" and sub-directories (or files).

    :param sub_dir: sub-directory name(s) or filename(s)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: "..\\data\\incidents\\delay attribution\\glossary\\current\\..."
    :rtype: str

    **Testing**::

        import os
        from misc.dag import cdd_dag

        path_to_dag = cdd_dag()

        os.path.isdir(path_to_dag)  # True
    """

    path = cdd_incidents("delay attribution\\glossary\\current", *sub_dir, mkdir=mkdir)
    return path


def path_to_original_file():
    """
    Path to the original .xlsx file

    :return: "..\\data\\incidents\\delay attribution\\glossary\\current\\delay-attribution-glossary.xlsx"
    :rtype: str

    Test::

        import os
        from misc.dag import path_to_original_file

        path_to_file = path_to_original_file()
        os.path.isfile(path_to_file)  # True
    """

    path_to_file = cdd_dag("delay-attribution-glossary.xlsx")
    return path_to_file


def download_delay_attribution_glossary(confirmation_required=True, verbose=False):
    """
    Download delay attribution glossary.

    :param confirmation_required: defaults to ``True``
    :param verbose: defaults to ``False``

    **Testing**::

        from misc.dag import download_delay_attribution_glossary

        verbose = True

        confirmation_required = True
        download_delay_attribution_glossary(confirmation_required, verbose)

        confirmation_required = False
        download_delay_attribution_glossary(confirmation_required, verbose)
    """

    def _download(user_agent_, filename_, file_dir_, verbose_):
        if verbose_:
            print("Downloading \"{}\" to \"{}\" ... ".format(filename_, file_dir_), end="")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', user_agent_)]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, path_to_file)
            print("Successfully.") if verbose else ""
        except Exception as e:
            print("Failed to download \"{}\". {}".format(filename_, e))

    spreadsheet_filename = "Historic-Delay-Attribution-Glossary.xlsx"

    current_year = datetime.datetime.now().year
    years = [str(x) for x in range(current_year - 1, current_year + 1)]
    months = ['%.2d' % x for x in range(1, 13)]

    for y, m in list(itertools.product(years, months)):
        url = 'https://www.networkrail.co.uk/wp-content/uploads/{}/{}/{}'.format(y, m, spreadsheet_filename)
        user_agent = fake_useragent.UserAgent().random
        response = requests.get(url, headers={'User-Agent': user_agent})
        if response.ok:
            filename = spreadsheet_filename.replace("Historic-", "").lower()
            path_to_file = cdd_dag(filename)
            file_dir = "..\\" + os.path.relpath(cdd_dag())
            if os.path.isfile(path_to_file):
                if confirmed("Replace the current version?", confirmation_required=confirmation_required):
                    _download(user_agent, filename, file_dir, verbose)
                    break
            else:
                _download(user_agent, filename, file_dir, verbose)
                break


def get_stanox_codes(update=False, hard_update=False, verbose=False):
    """
    Get STANOX codes.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_stanox_codes

        verbose = True

        update = False
        hard_update = False
        stanox_codes = get_stanox_codes(update, hard_update, verbose)

        update = True
        hard_update = True
        stanox_codes = get_stanox_codes(update, hard_update, verbose)
    """

    pickle_filename = "stanox-codes.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        stanox_codes = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            stanox_codes = pd.read_excel(path_to_original_file(), sheet_name="Stanox Codes", dtype={'STANOX NO.': str})
            stanox_codes.columns = [x.strip('.').replace(' ', '_') for x in stanox_codes.columns]
            stanox_codes.STANOX_NO = stanox_codes.STANOX_NO.map(lambda x: '0' * (5 - len(x)) + x)
            save_pickle(stanox_codes, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Stanox Codes\". {}.".format(e))
            stanox_codes = None

    return stanox_codes


def get_period_dates(update=False, hard_update=False, verbose=False):
    """
    Get period dates.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_period_dates

        verbose = True

        update = False
        hard_update = False
        period_dates = get_period_dates(update, hard_update, verbose)

        update = True
        hard_update = True
        period_dates = get_period_dates(update, hard_update, verbose)
    """

    pickle_filename = "period-dates.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        period_dates = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

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


def get_incident_reason_metadata(update=False, hard_update=False, verbose=False):
    """
    Get incident reasons.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_incident_reason_metadata

        verbose = True

        update = False
        hard_update = False
        incident_reason_metadata = get_incident_reason_metadata(update, hard_update, verbose)

        update = True
        hard_update = True
        incident_reason_metadata = get_incident_reason_metadata(update, hard_update, verbose)
    """

    pickle_filename = "incident-reason-metadata.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_reason_metadata = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

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


def get_responsible_manager(update=False, hard_update=False, verbose=False):
    """
    Get responsible manager.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_responsible_manager

        verbose = True

        update = False
        hard_update = False
        responsible_manager = get_responsible_manager(update, hard_update, verbose)

        update = True
        hard_update = True
        responsible_manager = get_responsible_manager(update, hard_update, verbose)
    """

    pickle_filename = "responsible-manager.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        responsible_manager = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            responsible_manager = pd.read_excel(path_to_original_file(), sheet_name="Responsible Manager")
            responsible_manager.columns = [x.replace(' ', '_') for x in responsible_manager.columns]
            responsible_manager.Responsible_Manager_Name = responsible_manager.Responsible_Manager_Name.str.strip()
            save_pickle(responsible_manager, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Responsible Manager\". {}.".format(e))
            responsible_manager = None

    return responsible_manager


def get_reactionary_reason_code(update=False, hard_update=False, verbose=False):
    """
    Get reactionary reason code.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_reactionary_reason_code

        verbose = True

        update = False
        hard_update = False
        reactionary_reason_code = get_reactionary_reason_code(update, hard_update, verbose)

        update = True
        hard_update = True
        reactionary_reason_code = get_reactionary_reason_code(update, hard_update, verbose)
    """

    pickle_filename = "reactionary-reason-code.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        reactionary_reason_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            reactionary_reason_code = pd.read_excel(path_to_original_file(), sheet_name="Reactionary Reason Code")
            reactionary_reason_code.columns = [x.replace(' ', '_') for x in reactionary_reason_code.columns]
            save_pickle(reactionary_reason_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Reactionary Reason Code\". {}.".format(e))
            reactionary_reason_code = None

    return reactionary_reason_code


def get_performance_event_code(update=False, hard_update=False, verbose=False):
    """
    Get performance event code.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_performance_event_code

        verbose = True

        update = False
        hard_update = False
        performance_event_code = get_performance_event_code(update, hard_update, verbose)

        update = True
        hard_update = True
        performance_event_code = get_performance_event_code(update, hard_update, verbose)
    """

    pickle_filename = "performance-event-code.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        performance_event_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

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


def get_train_service_code(update=False, hard_update=False, verbose=False):
    """
    Get train service code.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_train_service_code

        verbose = True

        update = False
        hard_update = False
        train_service_code = get_train_service_code(update, hard_update, verbose)

        update = True
        hard_update = True
        train_service_code = get_train_service_code(update, hard_update, verbose)
    """

    pickle_filename = "train-service-code.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        train_service_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            train_service_code = pd.read_excel(path_to_original_file(), sheet_name="Train Service Code")
            train_service_code.columns = [x.replace(' ', '_') for x in train_service_code.columns]
            save_pickle(train_service_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Train Service Code\". {}.".format(e))
            train_service_code = None

    return train_service_code


def get_operator_name(update=False, hard_update=False, verbose=False):
    """
    Get operator name.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_operator_name

        verbose = True

        update = False
        hard_update = False
        operator_name = get_operator_name(update, hard_update, verbose)

        update = True
        hard_update = True
        operator_name = get_operator_name(update, hard_update, verbose)
    """

    pickle_filename = "operator-name.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        operator_name = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            operator_name = pd.read_excel(path_to_original_file(), sheet_name="Operator Name")
            operator_name.columns = [x.replace(' ', '_') for x in operator_name.columns]
            save_pickle(operator_name, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Operator Name\". {}.".format(e))
            operator_name = None

    return operator_name


def get_service_group_code(update=False, hard_update=False, verbose=False):
    """
    Get service group code.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_service_group_code

        verbose = True

        update = False
        hard_update = False
        service_group_code = get_service_group_code(update, hard_update, verbose)

        update = True
        hard_update = True
        service_group_code = get_service_group_code(update, hard_update, verbose)
    """

    pickle_filename = "service-group-code.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        service_group_code = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

        try:
            service_group_code = pd.read_excel(path_to_original_file(), sheet_name="Service Group Code")
            service_group_code.columns = [x.replace(' ', '_') for x in service_group_code.columns]
            save_pickle(service_group_code, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get \"Service Group Code\". {}.".format(e))
            service_group_code = None

    return service_group_code


def get_delay_attr_glossary(update=False, hard_update=False, verbose=False):
    """
    Get historic delay attribution glossary.

    :param update: defaults to ``False``
    :type update: bool
    :param hard_update: defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool
    :return: pandas.DataFrame

    **Testing**::

        from misc.dag import get_delay_attr_glossary

        verbose = True

        update = False
        hard_update = False
        delay_attr_glossary = get_delay_attr_glossary(update, hard_update, verbose)

        update = True
        hard_update = True
        delay_attr_glossary = get_delay_attr_glossary(update, hard_update, verbose)
    """

    pickle_filename = "delay-attribution-glossary.pickle"
    path_to_pickle = cdd_dag(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        delay_attr_glossary = load_pickle(path_to_pickle)

    else:
        if not os.path.isfile(path_to_original_file()) or hard_update:
            download_delay_attribution_glossary(confirmation_required=False, verbose=False)

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
            print("Failed to get \"historic delay attribution glossary\". {}.".format(e))
            delay_attr_glossary = {None: None}

    return delay_attr_glossary
