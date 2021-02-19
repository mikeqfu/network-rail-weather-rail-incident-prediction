"""
Historic delay attribution glossary.

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
from pyhelpers import confirmed, load_pickle, save_pickle

from utils import cdd_incidents


class DelayAttributionGlossary:

    def __init__(self):
        self.DataDir = cdd_incidents("delay attribution\\glossary\\current")
        self.RelPath = os.path.relpath(self.DataDir, cdd_incidents())
        self.Filename = "delay-attribution-glossary.xlsx"

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "\\data\\incidents\\delay attribution\\glossary\\current\\"
        and sub-directories (or files).

        :param sub_dir: sub-directory name(s) or filename(s)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: "\\data\\incidents\\delay attribution\\glossary\\current\\..."
        :rtype: str

        **Example**::

            >>> import os
            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> path_to_dag = dag.cdd()

            >>> print(os.path.relpath(path_to_dag))
            data\\incidents\\delay attribution\\glossary\\current
        """

        path = cdd_incidents(self.RelPath, *sub_dir, mkdir=mkdir)

        return path

    def path_to_original_file(self):
        """
        Path to the original data file of "delay-attribution-glossary.xlsx".

        :return: an absolute path to the original data file
        :rtype: str

        **Example**::

            >>> import os
            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> print(os.path.isfile(dag.path_to_original_file()))
            True
        """

        path_to_file = self.cdd(self.Filename)

        return path_to_file

    def download_dag(self, confirmation_required=True, verbose=False):
        """
        Download delay attribution glossary.

        :param confirmation_required: defaults to ``True``
        :param verbose: defaults to ``False``

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> dag.download_dag(confirmation_required=True, verbose=True)
            Replace the current version? [No]|Yes: yes
            Downloading "delay-attribution-glossary.xlsx" to "\\data\\..." ... Successfully.
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
            url = 'https://www.networkrail.co.uk/wp-content/uploads/{}/{}/{}'.format(
                y, m, spreadsheet_filename)
            user_agent = fake_useragent.UserAgent().random
            response = requests.get(url, headers={'User-Agent': user_agent})
            if response.ok:
                filename = spreadsheet_filename.replace("Historic-", "").lower()
                path_to_file = self.cdd(filename)
                file_dir = "\\" + os.path.relpath(self.cdd())
                if os.path.isfile(path_to_file):
                    if confirmed("Replace the current version?",
                                 confirmation_required=confirmation_required):
                        _download(user_agent, filename, file_dir, verbose)
                        break
                else:
                    _download(user_agent, filename, file_dir, verbose)
                    break

    def read_stanox_codes(self, update=False, hard_update=False, verbose=False):
        """
        Get STANOX codes.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of STANOX codes
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> stanox = dag.read_stanox_codes()
            >>> print(stanox.head())
              STANOX_NO                   FULL_NAME CRS_CODE     NR_ROUTE
            0     00005  AACHEN                               Non Britain
            1     88601  ABBEY WOOD                      ABW         Kent
            2     04309  ABBEYHILL JN                            Scotland
            3     45185  ABBOTS RIPTON                                LNE
            4     67371  ABBOTSWOOD JN                            Western

            >>> stanox = dag.read_stanox_codes(update=True, hard_update=True, verbose=True)
            Updating "stanox-codes.pickle" at "\\data\\..." ... Done.
            >>> print(stanox.head())
              STANOX_NO                   FULL_NAME CRS_CODE     NR_ROUTE
            0     00005  AACHEN                               Non Britain
            1     88601  ABBEY WOOD                      ABW         Kent
            2     04309  ABBEYHILL JN                            Scotland
            3     45185  ABBOTS RIPTON                                LNE
            4     67371  ABBOTSWOOD JN                            Western
        """

        pickle_filename = "stanox-codes.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            stanox_codes = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                stanox_codes = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Stanox Codes",
                    dtype={'STANOX NO.': str})
                stanox_codes.columns = [
                    x.strip('.').replace(' ', '_') for x in stanox_codes.columns]
                stanox_codes.STANOX_NO = stanox_codes.STANOX_NO.map(
                    lambda x: '0' * (5 - len(x)) + x)
                save_pickle(stanox_codes, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Stanox Codes\". {}.".format(e))
                stanox_codes = None

        return stanox_codes

    def read_period_dates(self, update=False, hard_update=False, verbose=False):
        """
        Get period dates.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of period dates
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> dates = dag.read_period_dates(verbose=True)
            >>> print(dates.head())
              Financial_Year    Period  ...  Period_End_Day No_Of_Days
            0      2016-2017  Period 1  ...        Saturday         30
            1      2016-2017  Period 2  ...        Saturday         28
            2      2016-2017  Period 3  ...        Saturday         28
            3      2016-2017  Period 4  ...        Saturday         28
            4      2016-2017  Period 5  ...        Saturday         28

            >>> dates = dag.read_period_dates(update=True, hard_update=True, verbose=True)
            Updating "period-dates.pickle" at "\\data\\..." ... Done.
            >>> print(dates.head())
              Financial_Year    Period  ...  Period_End_Day No_Of_Days
            0      2016-2017  Period 1  ...        Saturday         30
            1      2016-2017  Period 2  ...        Saturday         28
            2      2016-2017  Period 3  ...        Saturday         28
            3      2016-2017  Period 4  ...        Saturday         28
            4      2016-2017  Period 5  ...        Saturday         28
        """

        pickle_filename = "period-dates.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            period_dates = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Period Dates", skiprows=3)
                raw.dropna(axis=1, how='all', inplace=True)

                periods = raw[['Unnamed: 0']].dropna()
                raw.drop('Unnamed: 0', axis='columns', inplace=True)
                periods.columns = ['Period']

                financial_years = [
                    x.replace('YEAR ', '').replace('/', '-20') for x in raw.columns
                    if 'Unnamed' not in x]

                raw.columns = raw.iloc[0]
                raw.drop(0, axis=0, inplace=True)

                no = int(raw.shape[1] / 3)
                period_dates_data = pd.concat(np.split(raw.dropna(axis='index'), no, axis=1),
                                              ignore_index=True, sort=False)

                period_dates_data.columns = [x.title().replace(' ', '_') for x in
                                             period_dates_data.columns]
                period_dates_data.rename(columns={'Day': 'Period_End_Day'}, inplace=True)
                period_dates_data.Period_End_Day = period_dates_data.Period_End_Date.dt.day_name()
                period_dates_data['Period_Start_Date'] = \
                    period_dates_data.Period_End_Date - period_dates_data.No_Of_Days.map(
                        lambda x: pd.Timedelta(days=x - 1))
                period_dates_data[
                    'Period_Start_Day'] = period_dates_data.Period_Start_Date.dt.day_name()
                period_dates_data['Period'] = periods.Period.to_list() * no
                period_dates_data['Financial_Year'] = np.repeat(financial_years, len(periods))

                period_dates = period_dates_data[
                    ['Financial_Year', 'Period', 'Period_Start_Date', 'Period_Start_Day',
                     'Period_End_Date', 'Period_End_Day', 'No_Of_Days']]

                save_pickle(period_dates, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Period Dates\". {}.".format(e))
                period_dates = None

        return period_dates

    def read_incident_reason_metadata(self, update=False, hard_update=False, verbose=False):
        """
        Get incident reasons.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: metadata of incident reasons
        :rtype: pandas.DataFrame

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> ir_metadata = dag.read_incident_reason_metadata(verbose=True)
            >>> print(ir_metadata.head())
                            Incident_Category  ...  Incident_Category_Super_Group_Code
            Incident_Reason
            IB                            101  ...                                NTAG
            IP                            101  ...                                NTAG
            JT                            101  ...                                NTAG
            IQ                            102  ...                                 MCG
            ID                            103  ...                                NTAG

            >>> ir_metadata = dag.read_incident_reason_metadata(update=True,
            ...                                                 hard_update=True,
            ...                                                 verbose=True)
            Updating "incident-reason-metadata.pickle" at "\\data\\..." ... Done.
            >>> print(ir_metadata.head())
                            Incident_Category  ...  Incident_Category_Super_Group_Code
            Incident_Reason
            IB                            101  ...                                NTAG
            IP                            101  ...                                NTAG
            JT                            101  ...                                NTAG
            IQ                            102  ...                                 MCG
            ID                            103  ...                                NTAG
        """

        pickle_filename = "incident-reason-metadata.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_reason_metadata = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                incident_reason_metadata = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Incident Reason")
                incident_reason_metadata.columns = [
                    x.replace(' ', '_') for x in incident_reason_metadata.columns]
                incident_reason_metadata.drop(
                    incident_reason_metadata.tail(1).index, inplace=True)
                incident_reason_metadata.set_index('Incident_Reason', inplace=True)

                save_pickle(incident_reason_metadata, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Incident Reason\". {}.".format(e))
                incident_reason_metadata = None

        return incident_reason_metadata

    def read_responsible_manager(self, update=False, hard_update=False, verbose=False):
        """
        Get responsible manager.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of responsible manager
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> rm = dag.read_responsible_manager()
            >>> print(rm.head())
              Responsible_Manager  ... Responsible_Org_NR-TOC/FOC_Others
            0                   0  ...                            OTHERS
            1                ACCQ  ...                           TOC/FOC
            2                ACDA  ...                           TOC/FOC
            3                ACQA  ...                           TOC/FOC
            4                ADAA  ...                           TOC/FOC
            [5 rows x 6 columns]

            >>> rm = dag.read_responsible_manager(update=True, hard_update=True,
            ...                                   verbose=True)
            Updating "responsible-manager.pickle" at "\\data\\..." ... Done.
            >>> print(rm.head())
              Responsible_Manager  ... Responsible_Org_NR-TOC/FOC_Others
            0                   0  ...                            OTHERS
            1                ACCQ  ...                           TOC/FOC
            2                ACDA  ...                           TOC/FOC
            3                ACQA  ...                           TOC/FOC
            4                ADAA  ...                           TOC/FOC
            [5 rows x 6 columns]
        """

        pickle_filename = "responsible-manager.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            responsible_manager = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                responsible_manager = pd.read_excel(self.path_to_original_file(),
                                                    sheet_name="Responsible Manager")
                responsible_manager.columns = [x.replace(' ', '_')
                                               for x in responsible_manager.columns]
                responsible_manager.Responsible_Manager_Name = \
                    responsible_manager.Responsible_Manager_Name.str.strip()
                save_pickle(responsible_manager, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Responsible Manager\". {}.".format(e))
                responsible_manager = None

        return responsible_manager

    def read_reactionary_reason_code(self, update=False, hard_update=False, verbose=False):
        """
        Get reactionary reason code.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of reactionary reason code
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> rrc = dag.read_reactionary_reason_code()
            >>> print(rrc.head())
              Reactionary_Reason_Code  ... Reactionary_Reason_Name
            0                       0  ...                 UNKNOWN
            1                       A  ...                 UNKNOWN
            2                      AA  ...              WTG ACCEPT
            3                      AB  ...               DOCUMENTS
            4                      AC  ...              TRAIN PREP
            [5 rows x 3 columns]

            >>> rrc = dag.read_reactionary_reason_code(update=True, hard_update=True,
            ...                                        verbose=True)
            Updating "reactionary-reason-code.pickle" at "\\data\\..." ... Done.
            >>> print(rrc.head())
              Reactionary_Reason_Code  ... Reactionary_Reason_Name
            0                       0  ...                 UNKNOWN
            1                       A  ...                 UNKNOWN
            2                      AA  ...              WTG ACCEPT
            3                      AB  ...               DOCUMENTS
            4                      AC  ...              TRAIN PREP
            [5 rows x 3 columns]
        """

        pickle_filename = "reactionary-reason-code.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            reactionary_reason_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                reactionary_reason_code = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Reactionary Reason Code")
                reactionary_reason_code.columns = [
                    x.replace(' ', '_') for x in reactionary_reason_code.columns]
                save_pickle(reactionary_reason_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Reactionary Reason Code\". {}.".format(e))
                reactionary_reason_code = None

        return reactionary_reason_code

    def read_performance_event_code(self, update=False, hard_update=False, verbose=False):
        """
        Get performance event code.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of performance event code
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> pec = dag.read_performance_event_code()
            >>> print(pec.head())
                                   Performance_Event_Group Performance_Event_Name
            Performance_Event_Code
            A                                 DELAY REPORT              Automatic
            ALL DELAYS                        DELAY REPORT             ALL DELAYS
            C                                 CANCELLATION    Cancelled At Origin
            D                                 CANCELLATION               Diverted
            F                                 DELAY REPORT         Failed to Stop

            >>> pec = dag.read_performance_event_code(update=True, hard_update=True,
            ...                                       verbose=True)
            Updating "performance-event-code.pickle" at "\\data\\..." ... Done.
            >>> print(pec.head())
                                   Performance_Event_Group Performance_Event_Name
            Performance_Event_Code
            A                                 DELAY REPORT              Automatic
            ALL DELAYS                        DELAY REPORT             ALL DELAYS
            C                                 CANCELLATION    Cancelled At Origin
            D                                 CANCELLATION               Diverted
            F                                 DELAY REPORT         Failed to Stop
        """

        pickle_filename = "performance-event-code.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            performance_event_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                performance_event_code = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Performance Event Code")
                # Rename columns
                performance_event_code.columns = [
                    x.replace(' ', '_') for x in performance_event_code.columns]
                # Set an index
                performance_event_code.set_index('Performance_Event_Code', inplace=True)
                # Save the data as .pickle
                save_pickle(performance_event_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Performance Event Code\". {}.".format(e))
                performance_event_code = None

        return performance_event_code

    def read_train_service_code(self, update=False, hard_update=False, verbose=False):
        """
        Get train service code.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of train service code
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> tsc = dag.read_train_service_code()
            >>> print(tsc.head())
              Service_Group_Code  ...                                    TSC_Description
            0               0001  ...                                            Unknown
            1               0003  ...                   National Adjustment Freight TOCs
            2               0300  ...                        DBC UK: Royal Mail Services
            3               0300  ...                          Post Office Parcels Train
            4               0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            [5 rows x 4 columns]

            >>> tsc = dag.read_train_service_code(update=True, hard_update=True,
            ...                                   verbose=True)
            Updating "train-service-code.pickle" at "\\data\\..." ... Done.
            >>> print(tsc.head())
              Service_Group_Code  ...                                    TSC_Description
            0               0001  ...                                            Unknown
            1               0003  ...                   National Adjustment Freight TOCs
            2               0300  ...                        DBC UK: Royal Mail Services
            3               0300  ...                          Post Office Parcels Train
            4               0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            [5 rows x 4 columns]
        """

        pickle_filename = "train-service-code.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            train_service_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                train_service_code = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Train Service Code")
                train_service_code.columns = [
                    x.replace(' ', '_') for x in train_service_code.columns]
                save_pickle(train_service_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Train Service Code\". {}.".format(e))
                train_service_code = None

        return train_service_code

    def read_operator_name(self, update=False, hard_update=False, verbose=False):
        """
        Get operator name.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of operator name
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> op_name = dag.read_operator_name()
            >>> print(op_name.head())
              Operator    Operator_Name
            0       A1         National
            1       A2  England & Wales
            2       A3              LSE
            3       A4               LD
            4       A5         Regional

            >>> op_name = dag.read_operator_name(update=True, hard_update=True,
            ...                                  verbose=True)
            Updating "operator-name.pickle" at "\\data\\..." ... Done.
            >>> print(op_name.head())
              Operator    Operator_Name
            0       A1         National
            1       A2  England & Wales
            2       A3              LSE
            3       A4               LD
            4       A5         Regional
        """

        pickle_filename = "operator-name.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            operator_name = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                operator_name = pd.read_excel(self.path_to_original_file(),
                                              sheet_name="Operator Name")
                operator_name.columns = [x.replace(' ', '_') for x in operator_name.columns]
                save_pickle(operator_name, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Operator Name\". {}.".format(e))
                operator_name = None

        return operator_name

    def read_service_group_code(self, update=False, hard_update=False, verbose=False):
        """
        Get service group code.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of service group code
        :rtype: pandas.DataFrame or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> sgc = dag.read_service_group_code()
            >>> print(sgc.head())
              Service_Group_Code         Service_Group_Description
            0               0001                          Dummy SG
            1               0003  National Adjustment Freight TOCs
            2               0300       DBC UK: Royal Mail Services
            3               0301                 EWS - Parcelforce
            4               0302     DBC UK: Passenger Stock Moves

            >>> sgc = dag.read_service_group_code(update=True, hard_update=True,
            ...                                   verbose=True)
            Updating "service-group-code.pickle" at "\\data\\..." ... Done.
            >>> print(sgc.head())
              Service_Group_Code         Service_Group_Description
            0               0001                          Dummy SG
            1               0003  National Adjustment Freight TOCs
            2               0300       DBC UK: Royal Mail Services
            3               0301                 EWS - Parcelforce
            4               0302     DBC UK: Passenger Stock Moves
        """

        pickle_filename = "service-group-code.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            service_group_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                service_group_code = pd.read_excel(
                    self.path_to_original_file(), sheet_name="Service Group Code")
                service_group_code.columns = [
                    x.replace(' ', '_') for x in service_group_code.columns]
                save_pickle(service_group_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Service Group Code\". {}.".format(e))
                service_group_code = None

        return service_group_code

    def get_delay_attr_glossary(self, update=False, hard_update=False, verbose=False):
        """
        Get historic delay attribution glossary.

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: data of historic delay attribution glossary
        :rtype: dict or None

        **Example**::

            >>> from misc.dag import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> dag_data = dag.get_delay_attr_glossary()
            >>> print(list(dag_data.keys()))
            ['Stanox_Codes',
             'Period_Dates',
             'Incident_Reason',
             'Responsible_Manager',
             'Reactionary_Reason_Code',
             'Performance_Event_Code',
             'Train_Service_Code',
             'Operator_Name',
             'Service_Group_Code']

            >>> dag_data = dag.get_delay_attr_glossary(update=True, hard_update=True,
            ...                                        verbose=True)
            Updating "incident-reason-metadata.pickle" at "\\data\\..." ... Done.
            Updating "operator-name.pickle" at "\\data\\..." ... Done.
            Updating "performance-event-code.pickle" at "\\data\\..." ... Done.
            Updating "period-dates.pickle" at "\\data\\..." ... Done.
            Updating "reactionary-reason-code.pickle" at "\\data\\..." ... Done.
            Updating "responsible-manager.pickle" at "\\data\\..." ... Done.
            Updating "service-group-code.pickle" at "\\data\\..." ... Done.
            Updating "stanox-codes.pickle" at "\\data\\..." ... Done.
            Updating "train-service-code.pickle" at "\\data\\..." ... Done.
            Updating "delay-attribution-glossary.pickle" at "\\data\\..." ... Done.
            >>> print(list(dag_data.keys()))
            ['Stanox_Codes',
             'Period_Dates',
             'Incident_Reason',
             'Responsible_Manager',
             'Reactionary_Reason_Code',
             'Performance_Event_Code',
             'Train_Service_Code',
             'Operator_Name',
             'Service_Group_Code']
        """

        pickle_filename = "delay-attribution-glossary.pickle"
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            delay_attr_glossary = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                glossary = []
                for func in dir(self):
                    if func.startswith('read_'):
                        glossary.append(getattr(self, func)(
                            update=update, hard_update=False, verbose=verbose))

                workbook = pd.ExcelFile(self.path_to_original_file())
                delay_attr_glossary = dict(
                    zip([x.replace(' ', '_') for x in workbook.sheet_names], glossary))
                workbook.close()

                save_pickle(delay_attr_glossary, path_to_pickle, verbose=True)

            except Exception as e:
                print("Failed to get \"historic delay attribution glossary\". ", end="")
                print(e)
                delay_attr_glossary = None

        return delay_attr_glossary
