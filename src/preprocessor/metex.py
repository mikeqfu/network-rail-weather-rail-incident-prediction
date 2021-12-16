"""
Read and cleanse data of NR_METEX_* database.

- Schedule 4 compensates train operators for the impact of planned service disruption, and
- Schedule 8 compensates train operators for the impact of unplanned service disruption.
"""

import gc
import string
import urllib.request
import zipfile

import networkx as nx
import shapely.geometry
import shapely.wkt
from pyhelpers.geom import osgb36_to_wgs84, wgs84_to_osgb36
from pyhelpers.ops import update_dict_keys
from pyhelpers.store import load_pickle, save_fig
from pyrcs import LocationIdentifiers, Stations
from pyrcs.utils import *

from utils import *


class DelayAttributionGlossary:
    """
    Historic delay attribution glossary.

    Information provided by Network Rail – https://www.networkrail.co.uk/transparency/datasets/
    """

    #: Name of the data
    NAME = 'Historic delay attribution glossary'

    def __init__(self):
        self.DataDir = cdd_metex("incidents\\delay attribution\\glossary")
        self.Filename = "Historic-Delay-Attribution-Glossary.xlsx"

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\incidents\\delay attribution\\glossary" and subdirectories / a file.

        :param sub_dir: subdirectory name(s) or filename(s)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: path to "data\\incidents\\delay attribution\\glossary" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary
            >>> import os

            >>> dag = DelayAttributionGlossary()

            >>> path_to_dag = dag.cdd()

            >>> os.path.relpath(path_to_dag)
            data\\metex\\incidents\\delay attribution\\glossary
        """

        path = cdd_metex(self.DataDir, *sub_dir, mkdir=mkdir)

        return path

    def path_to_original_file(self):
        """
        Path to the original data file of "Historic-Delay-Attribution-Glossary.xlsx".

        :return: absolute path to the original data file
        :rtype: str

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary
            >>> import os

            >>> dag = DelayAttributionGlossary()

            >>> os.path.isfile(dag.path_to_original_file())
            True
        """

        path_to_file = self.cdd(self.Filename)

        return path_to_file

    def download_dag(self, confirmation_required=True, verbose=False):
        """
        Download delay attribution glossary.

        :param confirmation_required: defaults to ``True``
        :param verbose: defaults to ``False``

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> dag.download_dag(confirmation_required=True, verbose=True)
            Replace the current version? [No]|Yes: yes
            Downloading "Historic-Delay-Attribution-Glossary.xlsx" ... Done.
        """

        path_to_file, filename = self.path_to_original_file(), self.Filename

        def _download(url_):
            filename_ = os.path.basename(path_to_file)
            print("Downloading \"{}\"".format(filename), end=" ... ") if verbose else ""
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = list(fake_requests_headers().items())
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url_, path_to_file)
                print("Done.") if verbose else ""
            except Exception as e:
                print("Failed to download \"{}\". {}".format(filename_, e))

        current_year = datetime.datetime.now().year
        years = [str(x) for x in range(current_year - 1, current_year + 1)]
        months = ['%.2d' % x for x in range(1, 13)]

        for y, m in list(itertools.product(years, months)):
            url = 'https://www.networkrail.co.uk/wp-content/uploads/{}/{}/{}'.format(y, m, filename)
            response = requests.get(url, headers=fake_requests_headers())
            if response.ok:
                if os.path.isfile(self.path_to_original_file()):
                    cfm_msg = "Replace the current version?"
                    if confirmed(prompt=cfm_msg, confirmation_required=confirmation_required):
                        _download(url)
                        break
                else:
                    _download(url)
                    break

    def get_stanox_codes(self, update=False, hard_update=False, verbose=False):
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

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> stanox_codes_sheet = dag.get_stanox_codes()
            >>> stanox_codes_sheet
                  STANOX_NO                   FULL_NAME CRS_CODE    Route_Description
            0         87904  3BDGS DN TL SDG ENTRY/EXIT                        Sussex
            1         87915  3BDGS DOWN THAMESLINK SDGS                        Sussex
            2         87907  3 BRIDGES DN DEP SIG TD127                        Sussex
            3         87950  3 BRIDGES UP DEP SIG TD165                        Sussex
            4         00005                      AACHEN                    High Speed
                     ...                         ...      ...                  ...
            11201     42241                      YORTON      YRT                Wales
            11202     78362               YSTRAD MYNACH      YSM  TfW Cardiff Valleys
            11203     78364    YSTRAD MYNACH SIG CF7420           TfW Cardiff Valleys
            11204     78363         YSTRAD MYNACH SOUTH           TfW Cardiff Valleys
            11205     78008              YSTRAD RHONDDA      YSR  TfW Cardiff Valleys

            [11206 rows x 4 columns]

            >>> stanox_codes_sheet = dag.get_stanox_codes(update=True, verbose=True)
            Updating "Stanox Codes.pickle" at "data\\...\\glossary" ... Done.
            >>> stanox_codes_sheet
                  STANOX_NO                   FULL_NAME CRS_CODE    Route_Description
            0         87904  3BDGS DN TL SDG ENTRY/EXIT                        Sussex
            1         87915  3BDGS DOWN THAMESLINK SDGS                        Sussex
            2         87907  3 BRIDGES DN DEP SIG TD127                        Sussex
            3         87950  3 BRIDGES UP DEP SIG TD165                        Sussex
            4         00005                      AACHEN                    High Speed
                     ...                         ...      ...                  ...
            11201     42241                      YORTON      YRT                Wales
            11202     78362               YSTRAD MYNACH      YSM  TfW Cardiff Valleys
            11203     78364    YSTRAD MYNACH SIG CF7420           TfW Cardiff Valleys
            11204     78363         YSTRAD MYNACH SOUTH           TfW Cardiff Valleys
            11205     78008              YSTRAD RHONDDA      YSR  TfW Cardiff Valleys

            [11206 rows x 4 columns]
        """

        sheet_name = "Stanox Codes"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            stanox_codes = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(
                    io=self.path_to_original_file(), sheet_name=sheet_name, dtype={'STANOX NO.': str},
                    skipfooter=2)

                stanox_codes = raw.copy()
                stanox_codes.columns = [x.strip('.').replace(' ', '_') for x in stanox_codes.columns]

                stanox_codes.STANOX_NO = fix_stanox(stanox_codes.STANOX_NO)

                stanox_codes.FULL_NAME = stanox_codes.FULL_NAME.str.strip('`')

                # stanox_codes = stanox_codes.where(cond=stanox_codes.notnull(), other=None)
                stanox_codes.fillna(value='', inplace=True)

                save_pickle(stanox_codes, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                stanox_codes = None

        return stanox_codes

    def get_period_dates(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> period_dates_sheet = dag.get_period_dates()
            >>> period_dates_sheet
                Financial_Year     Period Start_Date  ...   End_Date   End_Day No_of_Days
            0        2016-2017   Period 1 2016-04-01  ... 2016-04-30  Saturday         30
            1        2016-2017   Period 2 2016-05-01  ... 2016-05-28  Saturday         28
            2        2016-2017   Period 3 2016-05-29  ... 2016-06-25  Saturday         28
            3        2016-2017   Period 4 2016-06-26  ... 2016-07-23  Saturday         28
            4        2016-2017   Period 5 2016-07-24  ... 2016-08-20  Saturday         28
            ..             ...        ...        ...  ...        ...       ...        ...
            242      2034-2035   Period 9 2034-11-12  ... 2034-12-09  Saturday         28
            243      2034-2035  Period 10 2034-12-10  ... 2035-01-06  Saturday         28
            244      2034-2035  Period 11 2035-01-07  ... 2035-02-03  Saturday         28
            245      2034-2035  Period 12 2035-02-04  ... 2035-03-03  Saturday         28
            246      2034-2035  Period 13 2035-03-04  ... 2035-03-31  Saturday         28

            [247 rows x 7 columns]

            >>> period_dates_sheet = dag.get_period_dates(update=True, verbose=True)
            Updating "Period Dates.pickle" at "data\\...\\glossary" ... Done.
            >>> period_dates_sheet
                Financial_Year     Period Start_Date  ...   End_Date   End_Day No_of_Days
            0        2016-2017   Period 1 2016-04-01  ... 2016-04-30  Saturday         30
            1        2016-2017   Period 2 2016-05-01  ... 2016-05-28  Saturday         28
            2        2016-2017   Period 3 2016-05-29  ... 2016-06-25  Saturday         28
            3        2016-2017   Period 4 2016-06-26  ... 2016-07-23  Saturday         28
            4        2016-2017   Period 5 2016-07-24  ... 2016-08-20  Saturday         28
            ..             ...        ...        ...  ...        ...       ...        ...
            242      2034-2035   Period 9 2034-11-12  ... 2034-12-09  Saturday         28
            243      2034-2035  Period 10 2034-12-10  ... 2035-01-06  Saturday         28
            244      2034-2035  Period 11 2035-01-07  ... 2035-02-03  Saturday         28
            245      2034-2035  Period 12 2035-02-04  ... 2035-03-03  Saturday         28
            246      2034-2035  Period 13 2035-03-04  ... 2035-03-31  Saturday         28

            [247 rows x 7 columns]
        """

        sheet_name = "Period Dates"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            period_dates = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(
                    io=self.path_to_original_file(), sheet_name=sheet_name, skiprows=3)
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
                data = pd.concat(np.split(raw.dropna(axis='index'), no, axis=1), ignore_index=True)

                data.columns = [x.replace(' ', '_').strip('.') for x in data.columns]
                data.rename(columns={'Day_Name': 'End_Day', 'Date': 'End_Date'}, inplace=True)
                data['Start_Date'] = data.End_Date - data.No_of_Days.map(
                    lambda x: pd.Timedelta(days=x - 1))
                data['Start_Day'] = data.Start_Date.dt.day_name()
                data['Period'] = periods.Period.to_list() * no
                data['Financial_Year'] = np.repeat(financial_years, len(periods))

                sorted_column_names = [
                    'Financial_Year',
                    'Period',
                    'Start_Date',
                    'Start_Day',
                    'End_Date',
                    'End_Day',
                    'No_of_Days',
                ]
                period_dates = data[sorted_column_names]

                save_pickle(period_dates, path_to_pickle=path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                period_dates = None

        return period_dates

    def get_incident_reason(self, update=False, hard_update=False, verbose=False):
        """
        Get incident reasons (metadata).

        :param update: defaults to ``False``
        :type update: bool
        :param hard_update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool
        :return: metadata of incident reasons
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> ir_metadata = dag.get_incident_reason()
            >>> ir_metadata
                            Incident_Category  ... Incident_Category_Super_Group_Code
            Incident_Reason                    ...
            IB                            101  ...                               NTAG
            IP                            101  ...                               NTAG
            JT                            101  ...                               NTAG
            IQ                            102  ...                                MCG
            ID                            103  ...                               NTAG
                                       ...  ...                                ...
            YO                            902  ...                                  R
            YQ                            902  ...                                  R
            YR                            902  ...                                  R
            YU                            902  ...                                  R
            YX                            902  ...                                  R

            [364 rows x 6 columns]

            >>> ir_metadata = dag.get_incident_reason(update=True, verbose=True)
            Updating "Incident Reason metadata.pickle" at "data\\...\\glossary" ... Done.
            >>> ir_metadata
                            Incident_Category  ... Incident_Category_Super_Group_Code
            Incident_Reason                    ...
            IB                            101  ...                               NTAG
            IP                            101  ...                               NTAG
            JT                            101  ...                               NTAG
            IQ                            102  ...                                MCG
            ID                            103  ...                               NTAG
                                       ...  ...                                ...
            YO                            902  ...                                  R
            YQ                            902  ...                                  R
            YR                            902  ...                                  R
            YU                            902  ...                                  R
            YX                            902  ...                                  R

            [364 rows x 6 columns]
        """

        sheet_name = "Incident Reason"
        path_to_pickle = self.cdd(sheet_name + " metadata.pickle")

        if os.path.isfile(path_to_pickle) and not update:
            incident_reason = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                incident_reason = raw.copy()
                incident_reason.columns = [x.replace(' ', '_') for x in incident_reason.columns]

                incident_reason.set_index(keys='Incident_Reason', inplace=True)

                save_pickle(incident_reason, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                incident_reason = None

        return incident_reason

    def get_responsible_manager(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> rm = dag.get_responsible_manager()
            >>> rm
                 Responsible_Manager  ... Responsible_Org_NR-TOC/FOC_Others
            0                   ACCQ  ...                           TOC/FOC
            1                   ACDA  ...                           TOC/FOC
            2                   ACQA  ...                           TOC/FOC
            3                   ADAA  ...                           TOC/FOC
            4                   ADAC  ...                           TOC/FOC
                              ...  ...                               ...
            6867                ZQWL  ...                      Network Rail
            6868                ZRA4  ...                           TOC/FOC
            6869                ZRCM  ...                           TOC/FOC
            6870                ZRDS  ...                           TOC/FOC
            6871                ZXBA  ...                           TOC/FOC

            [6872 rows x 6 columns]

            >>> rm = dag.get_responsible_manager(update=True, verbose=True)
            Updating "Responsible Manager.pickle" at "data\\...\\glossary" ... Done.
            >>> rm
                 Responsible_Manager  ... Responsible_Org_NR-TOC/FOC_Others
            0                   ACCQ  ...                           TOC/FOC
            1                   ACDA  ...                           TOC/FOC
            2                   ACQA  ...                           TOC/FOC
            3                   ADAA  ...                           TOC/FOC
            4                   ADAC  ...                           TOC/FOC
                              ...  ...                               ...
            6867                ZQWL  ...                      Network Rail
            6868                ZRA4  ...                           TOC/FOC
            6869                ZRCM  ...                           TOC/FOC
            6870                ZRDS  ...                           TOC/FOC
            6871                ZXBA  ...                           TOC/FOC

            [6872 rows x 6 columns]
        """

        sheet_name = "Responsible Manager"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            responsible_manager = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                responsible_manager = raw.copy()
                responsible_manager.columns = [
                    x.replace(' ', '_').replace('Responsible_Managers.', '')
                    for x in responsible_manager.columns]
                responsible_manager.Responsible_Manager_Name = \
                    responsible_manager.Responsible_Manager_Name.str.strip()

                save_pickle(responsible_manager, path_to_pickle=path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                responsible_manager = None

        return responsible_manager

    def get_reactionary_reason_code(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> rrc = dag.get_reactionary_reason_code()
            >>> rrc
                Reactionary_Reason_Code  ... Reactionary_Reason_Name
            0                        AA  ...              WTG ACCEPT
            1                        AB  ...               DOCUMENTS
            2                        AC  ...              TRAIN PREP
            3                        AD  ...               WTG STAFF
            4                        AE  ...              CONGESTION
            ..                      ...  ...                     ...
            360                      ZU  ...              NOCAUSE ID
            361                      ZW  ...                Sys CANC
            362                      ZX  ...              SYS L-STRT
            363                      ZY  ...               SYS OTIME
            364                      ZZ  ...                 SYS LIR

            [365 rows x 3 columns]

            >>> rrc = dag.get_reactionary_reason_code(update=True, verbose=True)
            Updating "Reactionary Reason Code.pickle" at "data\\...\\glossary" ... Done.
            >>> rrc
                Reactionary_Reason_Code  ... Reactionary_Reason_Name
            0                        AA  ...              WTG ACCEPT
            1                        AB  ...               DOCUMENTS
            2                        AC  ...              TRAIN PREP
            3                        AD  ...               WTG STAFF
            4                        AE  ...              CONGESTION
            ..                      ...  ...                     ...
            360                      ZU  ...              NOCAUSE ID
            361                      ZW  ...                Sys CANC
            362                      ZX  ...              SYS L-STRT
            363                      ZY  ...               SYS OTIME
            364                      ZZ  ...                 SYS LIR

            [365 rows x 3 columns]
        """

        sheet_name = "Reactionary Reason Code"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            reactionary_reason_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                reactionary_reason_code = raw.copy()
                reactionary_reason_code.columns = [
                    x.replace(' ', '_').replace('Reactionary_Reasons.', '')
                    for x in reactionary_reason_code.columns]

                save_pickle(reactionary_reason_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                reactionary_reason_code = None

        return reactionary_reason_code

    def get_performance_event_code(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> pec = dag.get_performance_event_code()
            >>> pec
                                   Performance_Event_Group  Performance_Event_Name
            Performance_Event_Code
            A                                 DELAY REPORT               Automatic
            C                                 CANCELLATION     Cancelled At Origin
            D                                 CANCELLATION                Diverted
            F                                 DELAY REPORT          Failed to Stop
            M                                 DELAY REPORT                  Manual
            O                                CHANGE ORIGIN          Changed Origin
            P                                 CANCELLATION      Cancelled En Route
            S                                 CANCELLATION  Scheduled Cancellation

            >>> pec = dag.get_performance_event_code(update=True, verbose=True)
            Updating "Performance Event Code.pickle" at "data\\...\\glossary" ... Done.
            >>> pec
                                   Performance_Event_Group  Performance_Event_Name
            Performance_Event_Code
            A                                 DELAY REPORT               Automatic
            C                                 CANCELLATION     Cancelled At Origin
            D                                 CANCELLATION                Diverted
            F                                 DELAY REPORT          Failed to Stop
            M                                 DELAY REPORT                  Manual
            O                                CHANGE ORIGIN          Changed Origin
            P                                 CANCELLATION      Cancelled En Route
            S                                 CANCELLATION  Scheduled Cancellation
        """

        sheet_name = "Performance Event Code"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            performance_event_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                performance_event_code = raw.copy()
                performance_event_code.columns = [
                    x.replace(' ', '_').replace('Performance_Event_Types.', '')
                    for x in performance_event_code.columns]

                performance_event_code.set_index('Performance_Event_Code', inplace=True)

                save_pickle(performance_event_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                performance_event_code = None

        return performance_event_code

    def get_train_service_code(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> tsc = dag.get_train_service_code()
            >>> tsc
                 Service_Group_Code_Affected  ...                           TSC_Description_Affected
            0                           0003  ...                   National Adjustment Freight TOCs
            1                           0300  ...                        DBC UK: Royal Mail Services
            2                           0300  ...                          Post Office Parcels Train
            3                           0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            4                           0302  ...                      DBC UK: Passenger Stock Moves
                                      ...  ...                                                ...
            2805                        XXXX  ...       R.E.S Works Trains - Scottish Electric Power
            2806                        XXXX  ...               Network Rail Use Only For Bus & Ship
            2807                        XXXX  ...            Rs Deicing/Sandite MPV (Not To Be Used)
            2808                        XXXX  ...         Wrong TSC European Reserved Paths TPS Only
            2809                        XXXX  ...                                        Invalid TSC

            [2810 rows x 4 columns]

            >>> tsc = dag.get_train_service_code(update=True, verbose=True)
            Updating "Train Service Code.pickle" at "data\\...\\glossary" ... Done.
            >>> tsc
                 Service_Group_Code_Affected  ...                           TSC_Description_Affected
            0                           0003  ...                   National Adjustment Freight TOCs
            1                           0300  ...                        DBC UK: Royal Mail Services
            2                           0300  ...                          Post Office Parcels Train
            3                           0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            4                           0302  ...                      DBC UK: Passenger Stock Moves
                                      ...  ...                                                ...
            2805                        XXXX  ...       R.E.S Works Trains - Scottish Electric Power
            2806                        XXXX  ...               Network Rail Use Only For Bus & Ship
            2807                        XXXX  ...            Rs Deicing/Sandite MPV (Not To Be Used)
            2808                        XXXX  ...         Wrong TSC European Reserved Paths TPS Only
            2809                        XXXX  ...                                        Invalid TSC

            [2810 rows x 4 columns]
        """

        sheet_name = "Train Service Code"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            train_service_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                train_service_code = raw.copy()
                train_service_code.columns = [
                    x.replace(' - ', '_').replace(' ', '_') for x in train_service_code.columns]

                save_pickle(train_service_code, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                train_service_code = None

        return train_service_code

    def get_operator_name(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> op_name = dag.get_operator_name()
            >>> op_name
                Operator_Affected   Operator_Name_Affected
            0                  D1         Freightliner Inf
            1                  D2          Freightliner HH
            2                  DB  Freightliner Intermodal
            3                  E1             GWR Charters
            4                  EA                      TPE
            ..                ...                      ...
            100                XE            LUL DL R'mond
            101                XH                      DRS
            102                XX                Wrong Tsc
            103                YG            Hanson & Hall
            104                ZZ             DBCargo RITS

            [105 rows x 2 columns]

            >>> op_name = dag.get_operator_name(update=True, verbose=True)
            Updating "Operator Name.pickle" at "data\\...\\glossary" ... Done.
            >>> op_name
                Operator_Affected   Operator_Name_Affected
            0                  D1         Freightliner Inf
            1                  D2          Freightliner HH
            2                  DB  Freightliner Intermodal
            3                  E1             GWR Charters
            4                  EA                      TPE
            ..                ...                      ...
            100                XE            LUL DL R'mond
            101                XH                      DRS
            102                XX                Wrong Tsc
            103                YG            Hanson & Hall
            104                ZZ             DBCargo RITS

            [105 rows x 2 columns]
        """

        sheet_name = "Operator Name"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            operator_name = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                operator_name = raw.copy()
                operator_name.columns = [re.sub(r' - | ', '_', x) for x in operator_name.columns]

                save_pickle(operator_name, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                operator_name = None

        return operator_name

    def get_service_group_code(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> sgc = dag.get_service_group_code()
            >>> sgc
                 Service_Group_Code_Affected  ...                           TSC_Description_Affected
            0                           0003  ...                   National Adjustment Freight TOCs
            1                           0300  ...                        DBC UK: Royal Mail Services
            2                           0300  ...                          Post Office Parcels Train
            3                           0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            4                           0302  ...                      DBC UK: Passenger Stock Moves
                                      ...  ...                                                ...
            2805                        XXXX  ...       R.E.S Works Trains - Scottish Electric Power
            2806                        XXXX  ...               Network Rail Use Only For Bus & Ship
            2807                        XXXX  ...            Rs Deicing/Sandite MPV (Not To Be Used)
            2808                        XXXX  ...         Wrong TSC European Reserved Paths TPS Only
            2809                        XXXX  ...                                        Invalid TSC

            [2810 rows x 4 columns]

            >>> sgc = dag.get_service_group_code(update=True, verbose=True)
            Updating "Service Group Code.pickle" at "data\\...\\glossary" ... Done.
            >>> sgc
                 Service_Group_Code_Affected  ...                           TSC_Description_Affected
            0                           0003  ...                   National Adjustment Freight TOCs
            1                           0300  ...                        DBC UK: Royal Mail Services
            2                           0300  ...                          Post Office Parcels Train
            3                           0301  ...  Do Not Use Post Office Parcels Trains Scottish...
            4                           0302  ...                      DBC UK: Passenger Stock Moves
                                      ...  ...                                                ...
            2805                        XXXX  ...       R.E.S Works Trains - Scottish Electric Power
            2806                        XXXX  ...               Network Rail Use Only For Bus & Ship
            2807                        XXXX  ...            Rs Deicing/Sandite MPV (Not To Be Used)
            2808                        XXXX  ...         Wrong TSC European Reserved Paths TPS Only
            2809                        XXXX  ...                                        Invalid TSC

            [2810 rows x 4 columns]
        """

        sheet_name = "Service Group Code"
        path_to_pickle = self.cdd(sheet_name + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            service_group_code = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

                service_group_code = raw.copy()
                service_group_code.columns = [
                    re.sub(r' - | ', '_', x) for x in service_group_code.columns]

                save_pickle(service_group_code, path_to_pickle=path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(sheet_name, e))
                service_group_code = None

        return service_group_code

    def read_delay_attr_glossary(self, update=False, hard_update=False, verbose=False):
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

        **Test**::

            >>> from preprocessor.metex import DelayAttributionGlossary

            >>> dag = DelayAttributionGlossary()

            >>> dag_data = dag.read_delay_attr_glossary()
            >>> list(dag_data.keys())
            ['Stanox Codes',
             'Period Dates',
             'Incident Reason',
             'Responsible Manager',
             'Reactionary Reason Code',
             'Performance Event Code',
             'Service Group Code',
             'Operator Name',
             'Train Service Code']

            >>> dag_data = dag.read_delay_attr_glossary(update=True, hard_update=True, verbose=True)
            Updating "Incident Reason metadata.pickle" at "data\\...\\glossary" ... Done.
            Updating "Operator Name.pickle" at "data\\...\\glossary" ... Done.
            Updating "Performance Event Code.pickle" at "data\\...\\glossary" ... Done.
            Updating "Period Dates.pickle" at "data\\...\\glossary" ... Done.
            Updating "Reactionary Reason Code.pickle" at "data\\...\\glossary" ... Done.
            Updating "Responsible Manager.pickle" at "data\\...\\glossary" ... Done.
            Updating "Service Group Code.pickle" at "data\\...\\glossary" ... Done.
            Updating "Stanox Codes.pickle" at "data\\...\\glossary" ... Done.
            Updating "Train Service Code.pickle" at "data\\...\\glossary" ... Done.
            Updating "Historic-Delay-Attribution-Glossary.pickle" at "data\\...\\glossary" ... Done.
            >>> list(dag_data.keys())
            ['Stanox Codes',
             'Period Dates',
             'Incident Reason',
             'Responsible Manager',
             'Reactionary Reason Code',
             'Performance Event Code',
             'Service Group Code',
             'Operator Name',
             'Train Service Code']
        """

        filename = self.Filename.replace(".xlsx", "")
        path_to_pickle = self.cdd(filename + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            delay_attr_glossary = load_pickle(path_to_pickle)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            try:
                glossary = []
                for func in dir(self):
                    if func.startswith('get_'):
                        glossary.append(getattr(self, func)(
                            update=update, hard_update=False, verbose=verbose))

                workbook = pd.ExcelFile(self.path_to_original_file())
                delay_attr_glossary = dict(zip([x for x in workbook.sheet_names], glossary))
                workbook.close()

                save_pickle(delay_attr_glossary, path_to_pickle, verbose=True)

            except Exception as e:
                print("Failed to get \"{}\".".format(filename.lower(), e))
                delay_attr_glossary = None

        return delay_attr_glossary


class METExLite:
    """
    METEX database.
    """

    #: Name of the data
    NAME = 'METExLite'
    #: Brief description of the data
    DESCRIPTION = \
        'METExLite is a geographic information system (GIS) based decision support tool, ' \
        'used to assess asset and system vulnerability to weather.'

    def __init__(self, database_name='NR_METEx_20190203'):
        """
        :param database_name: name of the database, defaults to ``'NR_METEx_20190203'``
        :type database_name: str

        :ivar str DatabaseName: name of the database that stores the data
        :ivar sqlalchemy.engine.Connection DatabaseConn: connection to the database
        :ivar preprocessor.DelayAttributionGlossary DAG: instance of the DAG class
        :ivar pyrcs.LocationIdentifiers LocationID: instance of the LocationIdentifiers class
        :ivar pyrcs.Stations StationCode: instance of the Stations class

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> metex.NAME
            'METExLite'
        """

        self.DatabaseName = copy.copy(database_name)
        self.DatabaseConn = establish_mssql_connection(database_name=self.DatabaseName)

        self.DAG = DelayAttributionGlossary()
        self.LocationID = LocationIdentifiers()
        self.StationCode = Stations()

    # == Change directories ===========================================================================

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database_lite" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\metex\\database_lite" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor import METExLite
            >>> import os

            >>> metex = METExLite()

            >>> os.path.relpath(metex.cdd())
            'data\\metex\\database_lite'
        """

        path = cdd_metex("database_lite", *sub_dir, mkdir=mkdir)

        return path

    def cdd_tables(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database_lite\\tables" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\metex\\database_lite\\tables" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor import METExLite
            >>> import os

            >>> metex = METExLite()

            >>> os.path.relpath(metex.cdd_tables())
            'data\\metex\\database_lite\\tables'
        """

        path = self.cdd("tables", *sub_dir, mkdir=mkdir)

        return path

    def cdd_views(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database_lite\\views" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\metex\\database_lite\\views" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor import METExLite
            >>> import os

            >>> metex = METExLite()

            >>> os.path.relpath(metex.cdd_views())
            'data\\metex\\database_lite\\views'
        """

        path = self.cdd("views", *sub_dir, mkdir=mkdir)

        return path

    def cdd_figures(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\database_lite\\figures" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\metex\\database_lite\\figures" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor import METExLite
            >>> import os

            >>> metex = METExLite()

            >>> os.path.relpath(metex.cdd_figures())
            'data\\metex\\database_lite\\figures'
        """

        path = self.cdd("figures", *sub_dir, mkdir=mkdir)

        return path

    # == Methods to read table data from the database =================================================

    def read_table(self, table_name, schema_name='dbo', index_col=None,
                   route_name=None, weather_category=None, save_as=None, update=False, **kwargs):
        """
        Read tables stored in NR_METEX_* database.

        :param table_name: name of a table
        :type table_name: str
        :param schema_name: name of schema, defaults to ``'dbo'``
        :type schema_name: str
        :param index_col: index column(s) of the returned data frame, defaults to ``None``
        :type index_col: str or list or None
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param weather_category: weather category; if ``None`` (default), all weather categories
        :type weather_category: str or None
        :param save_as: file extension, defaults to ``None``
        :type save_as: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param kwargs: optional parameters of `pandas.read_sql`_
        :return: data of the queried table stored in NR_METEX_* database
        :rtype: pandas.DataFrame

        .. _`pandas.read_sql`:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> imdm_tbl = metex.read_table(table_name='IMDM')
            >>> imdm_tbl.head()
        """

        # Connect to the queried database
        conn_metex = establish_mssql_connection(database_name=self.DatabaseName)

        sql_query = f'SELECT * FROM %s' % f'[{schema_name}].[{table_name}]'
        # Specify possible scenarios:
        if route_name and not weather_category:
            sql_query += f" WHERE [Route]='{route_name}'"  # given Route
        elif route_name is None and weather_category is not None:
            # given Weather
            sql_query += f" WHERE [WeatherCategory]='{weather_category}'"
        elif route_name and weather_category:
            # Get data given Route and weather category e.g. wind-attributed incidents on Anglia Route
            sql_query += f" WHERE [Route]='{route_name}' AND [WeatherCategory]='{weather_category}'"

        # Create a pd.DataFrame of the queried table
        table_data = pd.read_sql(sql_query, conn_metex, index_col=index_col, **kwargs)

        # Disconnect the database
        conn_metex.close()

        if save_as:
            path_to_file = self.cdd_tables(table_name + save_as)
            if not os.path.isfile(path_to_file) or update:
                save(table_data, path_to_file, index=True if index_col else False)

        return table_data

    def get_primary_key(self, table_name):
        """
        Get primary keys of a table stored in database 'NR_METEX_*'.

        :param table_name: name of a table stored in the database 'NR_METEX_*'
        :type table_name: str
        :return: a (list of) primary key(s)
        :rtype: list

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> primary_key = metex.get_primary_key(table_name='IMDM')
            >>> primary_key
        """

        pri_key = get_table_primary_keys(self.DatabaseName, table_name=table_name)

        return pri_key

    # == Methods to get table data ====================================================================

    def get_imdm(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'IMDM'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'IMDM'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> imdm_tbl = metex.get_imdm(update=True, verbose=True)
            Updating "IMDM.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> imdm_tbl.head()
        """

        METExLite.IMDM = 'IMDM'
        path_to_file = self.cdd_tables("".join([METExLite.IMDM, ".json" if as_dict else ".pickle"]))

        if os.path.isfile(path_to_file) and not update:
            imdm = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                imdm = self.read_table(
                    table_name=METExLite.IMDM, index_col=self.get_primary_key(METExLite.IMDM),
                    save_as=save_original_as, update=update)
                imdm.index.rename(name='IMDM', inplace=True)  # Rename index

                # Update route names
                update_route_names(imdm)

                # Add regions
                regions_and_routes = load_json(cdd_network("regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
                imdm = imdm.join(regions, on='Route')

                imdm = imdm.where((pd.notnull(imdm)), None)

                if as_dict:
                    imdm_dict = imdm.to_dict()
                    imdm = imdm_dict['Route']
                    imdm.pop('None')
                save(imdm, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METExLite.IMDM, " as a dictionary" if as_dict else "", e))
                imdm = None

        return imdm

    def get_imdm_alias(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'ImdmAlias'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'ImdmAlias'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> imdm_alias_tbl = metex.get_imdm_alias(update=True, verbose=True)
            Updating "ImdmAlias.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> imdm_alias_tbl.head()

            >>> imdm_alias_tbl = metex.get_imdm_alias(as_dict=True, update=True, verbose=True)
            Updating "ImdmAlias.json" at "data\\metex\\database_lite\\tables" ... Done.
            >>> list(imdm_alias_tbl['IMDM'].keys())[:5]
        """

        METExLite.ImdmAlias = 'ImdmAlias'
        path_to_file = self.cdd_tables(METExLite.ImdmAlias + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_file) and not update:
            imdm_alias = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                imdm_alias = self.read_table(
                    table_name=METExLite.ImdmAlias, index_col=self.get_primary_key(METExLite.ImdmAlias),
                    save_as=save_original_as, update=update)
                imdm_alias.index.rename(name='ImdmAlias', inplace=True)  # Rename index
                imdm_alias.rename(columns={'Imdm': 'IMDM'}, inplace=True)  # Rename a column

                if as_dict:
                    imdm_alias = imdm_alias.to_dict()  # imdm_alias = imdm_alias['IMDM']

                save(imdm_alias, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METExLite.ImdmAlias, " as a dictionary" if as_dict else "", e))
                imdm_alias = None

        return imdm_alias

    def get_imdm_weather_cell_map(self, route_info=True, grouped=False, update=False,
                                  save_original_as=None, verbose=False):
        """
        Get data of the table 'IMDMWeatherCellMap'.

        :param route_info: get the data that contains route information, defaults to ``True``
        :type route_info: bool
        :param grouped: whether to group the data by either ``'Route'`` or ``'WeatherCellId'``,
            defaults to ``False``
        :type grouped: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'IMDMWeatherCellMap'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> wcm_tbl = metex.get_imdm_weather_cell_map(update=True, verbose=True)
            Updating "IMDMWeatherCellMap_pc.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> wcm_tbl.head()

            >>> wcm_tbl = metex.get_imdm_weather_cell_map(grouped=True, update=True, verbose=True)
            Updating "IMDMWeatherCellMap_pc-grouped.pickle" at "data\\...\tables" ... Done.
            >>> wcm_tbl.head()

            >>> wcm_tbl = metex.get_imdm_weather_cell_map(
            ...     route_info=False, grouped=True, update=True, verbose=True)
            Updating "IMDMWeatherCellMap-grouped.pickle" at "data\\...\\tables" ... Done.
            >>> wcm_tbl.head()

            >>> wcm_tbl = metex.get_imdm_weather_cell_map(
            ...     route_info=False, update=True, verbose=True)
            Updating "IMDMWeatherCellMap.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> wcm_tbl.head()
        """

        METExLite.IMDMWeatherCellMap = 'IMDMWeatherCellMap_pc' if route_info else 'IMDMWeatherCellMap'
        path_to_pickle = self.cdd_tables(
            METExLite.IMDMWeatherCellMap + ("-grouped.pickle" if grouped else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            weather_cell_map = load_pickle(path_to_pickle)

        else:
            try:
                # Read IMDMWeatherCellMap table
                weather_cell_map = self.read_table(
                    table_name=METExLite.IMDMWeatherCellMap,
                    index_col=self.get_primary_key(METExLite.IMDMWeatherCellMap),
                    coerce_float=False, save_as=save_original_as, update=update)

                if route_info:
                    update_route_names(weather_cell_map)
                    weather_cell_map[['Id', 'WeatherCell']] = weather_cell_map[
                        ['Id', 'WeatherCell']].applymap(int)
                    weather_cell_map.set_index('Id', inplace=True)

                weather_cell_map.index.rename('IMDMWeatherCellMapId', inplace=True)  # Rename index
                weather_cell_map.rename(columns={'WeatherCell': 'WeatherCellId'},
                                        inplace=True)  # Rename a column

                if grouped:  # To find out how many IMDMs each 'WeatherCellId' is associated with
                    weather_cell_map = weather_cell_map.groupby(
                        'Route' if route_info else 'WeatherCellId').aggregate(
                        lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))

                save_pickle(weather_cell_map, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METExLite.IMDMWeatherCellMap, " (being grouped)" if grouped else "", e))
                weather_cell_map = None

        return weather_cell_map

    def get_incident_reason_info(self, plus=True, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'IncidentReasonInfo'.

        :param plus: defaults to ``True``
        :type plus: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'IncidentReasonInfo'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> iri_tbl = metex.get_incident_reason_info(update=True, verbose=True)
            Updating "IncidentReasonInfo-plus.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> iri_tbl.head()

            >>> iri_tbl = metex.get_incident_reason_info(plus=False, update=True, verbose=True)
            Updating "IncidentReasonInfo.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> iri_tbl.head()
        """

        METExLite.IncidentReasonInfo = 'IncidentReasonInfo'
        suffix = "-plus" if plus else ""
        path_to_pickle = self.cdd_tables(METExLite.IncidentReasonInfo + suffix + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            incident_reason_info = load_pickle(path_to_pickle)

        else:
            try:
                # Get data from the database
                incident_reason_info = self.read_table(
                    table_name=METExLite.IncidentReasonInfo,
                    index_col=self.get_primary_key(METExLite.IncidentReasonInfo),
                    save_as=save_original_as, update=update)

                idx_name = 'IncidentReasonCode'
                incident_reason_info.index.rename(idx_name, inplace=True)
                incident_reason_info.rename(
                    columns={'Description': 'IncidentReasonDescription',
                             'Category': 'IncidentCategory',
                             'CategoryDescription': 'IncidentCategoryDescription'},
                    inplace=True)

                if plus:  # To include data of more detailed description about incident reasons
                    ir_metadata = self.DAG.get_incident_reason()
                    ir_metadata.index.name = idx_name
                    ir_metadata.columns = [x.replace('_', '') for x in ir_metadata.columns]
                    incident_reason_info = incident_reason_info.join(ir_metadata, rsuffix='_plus')
                    # incident_reason_info.dropna(axis=1, inplace=True)

                save_pickle(incident_reason_info, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METExLite.IncidentReasonInfo, " with extra information" if plus else "", e))
                incident_reason_info = None

        return incident_reason_info

    def get_weather_codes(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'WeatherCategoryLookup'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WeatherCategoryLookup'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> wc_tbl = metex.get_weather_codes(update=True, verbose=True)
            Updating "WeatherCodes.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> wc_tbl.head()

            >>> wc_tbl = metex.get_weather_codes(as_dict=True, update=True, verbose=True)
            Updating "WeatherCodes.json" at "data\\metex\\database_lite\\tables" ... Done.
            >>> list(wc_tbl['WeatherCategory'].keys())[:5]
        """

        METExLite.WeatherCodes = 'WeatherCodes'  # WeatherCodes
        path_to_file = self.cdd_tables(METExLite.WeatherCodes + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_file) and not update:
            weather_codes = load_json(path_to_file) if as_dict else load_pickle(path_to_file)

        else:
            try:
                weather_codes = self.read_table(
                    table_name=METExLite.WeatherCodes,
                    index_col=self.get_primary_key(METExLite.WeatherCodes),
                    save_as=save_original_as, update=update)

                weather_codes.rename(
                    columns={'Code': 'WeatherCategoryCode', 'Weather Category': 'WeatherCategory'},
                    inplace=True)

                if as_dict:
                    weather_codes.set_index('WeatherCategoryCode', inplace=True)
                    weather_codes = weather_codes.to_dict()

                save(weather_codes, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.WeatherCodes, e))
                weather_codes = None

        return weather_codes

    def get_incident_record(self, update=False, save_original_as=None, use_amendment_csv=True,
                            verbose=False):
        """
        Get data of the table 'IncidentRecord'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'IncidentRecord'
        :rtype: pandas.DataFrame or None

        .. note::

            None values are filled with ``NaN``.

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> ir_tbl = metex.get_incident_record(
            ...     update=True, use_amendment_csv=False, verbose=True)
            Updating "IncidentRecord.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> ir_tbl.tail()

            >>> ir_tbl = metex.get_incident_record(update=True, verbose=True)
            Updating "IncidentRecord-amended.pickle" at "data\\metex\\...\\tables" ... Done.
            >>> ir_tbl.tail()
        """

        METExLite.IncidentRecord = 'IncidentRecord'
        suffix = "-amended" if use_amendment_csv else ""
        path_to_pickle = self.cdd_tables(METExLite.IncidentRecord + suffix + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            incident_record = load_pickle(path_to_pickle)

        else:
            try:
                incident_record = self.read_table(
                    table_name=METExLite.IncidentRecord,
                    index_col=self.get_primary_key(METExLite.IncidentRecord),
                    save_as=save_original_as, update=update)

                if use_amendment_csv:
                    amendment_csv = pd.read_csv(
                        self.cdd("updates", METExLite.IncidentRecord + ".zip"), index_col='Id',
                        parse_dates=['CreateDate'], infer_datetime_format=True, dayfirst=True)
                    amendment_csv.columns = incident_record.columns
                    idx = amendment_csv[amendment_csv.WeatherCategory.isna()].index
                    amendment_csv.loc[idx, 'WeatherCategory'] = None
                    incident_record.drop(incident_record[incident_record.CreateDate >= pd.to_datetime(
                        '2018-01-01')].index, inplace=True)
                    incident_record = incident_record.append(amendment_csv)

                incident_record.index.rename(METExLite.IncidentRecord + 'Id', inplace=True)
                incident_record.rename(
                    columns={'CreateDate': METExLite.IncidentRecord + 'CreateDate',
                             'Reason': 'IncidentReasonCode'},
                    inplace=True)  # Rename column names

                # Get a weather category lookup dictionary
                weather_codes = self.get_weather_codes(as_dict=True)
                # Replace each weather category code with its full name
                incident_record.replace(weather_codes, inplace=True)
                incident_record.WeatherCategory.fillna(value='', inplace=True)

                save_pickle(incident_record, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.IncidentRecord, e))
                incident_record = None

        return incident_record

    def get_location(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Location'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Location'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> location_tbl = metex.get_location(update=True, verbose=True)
            Updating "Location.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> location_tbl.head()
        """

        METExLite.Location = 'Location'
        path_to_pickle = self.cdd_tables(METExLite.Location + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            location = load_pickle(path_to_pickle)

        else:
            try:
                location = self.read_table(
                    table_name=METExLite.Location, index_col=self.get_primary_key(METExLite.Location),
                    coerce_float=False, save_as=save_original_as, update=update)

                location.index.rename('LocationId', inplace=True)
                location.rename(columns={'Imdm': 'IMDM'}, inplace=True)
                location[['WeatherCell', 'SMDCell']] = location[['WeatherCell', 'SMDCell']].applymap(
                    lambda x: 0 if np.isnan(x) else int(x))
                # location.loc[610096, 0:4] = [-0.0751, 51.5461, -0.0751, 51.5461]

                save_pickle(location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.Location, e))
                location = None

        return location

    def get_pfpi(self, plus=True, update=False, save_original_as=None, use_amendment_csv=True,
                 verbose=False):
        """
        Get data of the table 'PfPI' (Process for Performance Improvement).

        :param plus: defaults to ``True``
        :type plus: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'PfPI'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> pfpi_tbl = metex.get_pfpi(update=True, verbose=True)
            Updating "PfPI-plus-amended.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> pfpi_tbl.tail()
        """

        METExLite.PfPI = 'PfPI'
        suffix0, suffix1 = ("-plus" if plus else "", "-amended" if use_amendment_csv else "")
        path_to_pickle = self.cdd_tables(METExLite.PfPI + suffix0 + suffix1 + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            pfpi = load_pickle(path_to_pickle)

        else:
            try:
                pfpi = self.read_table(
                    table_name=METExLite.PfPI, index_col=self.get_primary_key(METExLite.PfPI),
                    save_as=save_original_as, update=update)

                if use_amendment_csv:
                    ir_tbl = self.get_incident_record()
                    min_id = ir_tbl[
                        ir_tbl.IncidentRecordCreateDate >= pd.to_datetime('2018-01-01')].index.min()
                    pfpi.drop(pfpi[pfpi.IncidentRecordId >= min_id].index, inplace=True)
                    pfpi = pfpi.append(
                        pd.read_csv(self.cdd("updates", METExLite.PfPI + ".zip"), index_col='Id'))

                pfpi.index.rename(METExLite.PfPI + pfpi.index.name, inplace=True)

                if plus:  # To include more information for 'PerformanceEventCode'
                    performance_event_code = self.DAG.get_performance_event_code()
                    performance_event_code.index.rename('PerformanceEventCode', inplace=True)
                    performance_event_code.columns = [
                        x.replace('_', '') for x in performance_event_code.columns]
                    # Merge pfpi and performance_event_code
                    pfpi = pfpi.join(performance_event_code, on='PerformanceEventCode')

                save_pickle(pfpi, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\"{}. {}.".format(
                    METExLite.PfPI, " with performance event name" if plus else "", e))
                pfpi = None

        return pfpi

    def get_route(self, as_dict=False, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Route'.

        :param as_dict: whether to return the data as a dictionary, defaults to ``False``
        :type as_dict: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Route'
        :rtype: pandas.DataFrame or None

        .. note::

            There is only one column in the original table.

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> route_tbl = metex.get_route(as_dict=True, update=True, verbose=True)
            Updating "Route.json" at "data\\metex\\database_lite\\tables" ... Done.
            >>> list(route_tbl.keys())[:5]

            >>> route_tbl = metex.get_route(update=True, verbose=True)
            Updating "Route.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> route_tbl.head()
        """

        table_name = "Route"
        path_to_pickle = self.cdd_tables(table_name + (".json" if as_dict else ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            route = load_pickle(path_to_pickle)

        else:
            try:
                route = self.read_table(table_name=table_name, save_as=save_original_as, update=update)

                route.rename(columns={'Name': 'Route'}, inplace=True)
                update_route_names(route)

                # Add regions
                regions_and_routes = load_json(cdd_network("regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                regions = pd.DataFrame.from_dict({'Region': regions_and_routes_dict})
                route = route.join(regions, on='Route')

                route = route.where((pd.notnull(route)), None)

                if as_dict:
                    route.drop_duplicates('Route', inplace=True)
                    route = dict(zip(route.RouteAlias, route.Route))

                save(route, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(table_name, e))
                route = None

        return route

    def get_stanox_location(self, use_nr_mileage_format=True, update=False, save_original_as=None,
                            verbose=False):
        """
        Get data of the table 'StanoxLocation'.

        :param use_nr_mileage_format: defaults to ``True``
        :type use_nr_mileage_format: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'StanoxLocation'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> stanox_location_tbl = metex.get_stanox_location(
            ...     use_nr_mileage_format=False, update=True, verbose=True)
            Updating "StanoxLocation.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> stanox_location_tbl.tail()

            >>> stanox_location_tbl = metex.get_stanox_location(update=True, verbose=True)
            Saving "StanoxLocation-mileage.pickle" to "data\\metex\\database_lite\\tables" ... Done.
            >>> stanox_location_tbl.tail()
        """

        METExLite.StanoxLocation = 'StanoxLocation'
        suffix = "-mileage" if use_nr_mileage_format else ""
        path_to_pickle = self.cdd_tables(METExLite.StanoxLocation + suffix + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            stanox_location = load_pickle(path_to_pickle)

        else:
            try:
                # Read StanoxLocation table from the database
                raw = self.read_table(
                    table_name=METExLite.StanoxLocation, index_col=None, save_as=save_original_as,
                    update=update)

                # Likely errors
                raw.loc[raw.Stanox == '52053', 'ELR':'LocationId'] = ('BOK1', 6072, 534877)
                raw.loc[raw.Stanox == '52074', 'ELR':'LocationId'] = ('ELL1', 440, 610096)

                def _cleanse_stanox_location():
                    """
                    sta_loc = copy.deepcopy(stanox_location)
                    """
                    dat = copy.copy(raw)

                    # Use external data - Railway Codes
                    errata = load_json(cdd_network("railway codes", "metex-errata.json"))
                    errata_ = update_dict_keys(
                        errata, replacements=dict(zip(errata.keys(), ['Stanox', 'Description', 'Name'])))
                    # Note that {'CLAPS47': 'CLPHS47'} in err_tiploc is dubious.
                    dat.replace(errata_, inplace=True)

                    duplicated_stanox = dat[dat.Stanox.duplicated(keep=False)].sort_values('Stanox')
                    nan_idx = pd.isna(duplicated_stanox[['ELR', 'Yards', 'LocationId']]).any(axis=1)
                    dat.drop(duplicated_stanox[nan_idx].index, inplace=True)
                    # dat.drop_duplicates(subset=['Stanox'], keep='last', inplace=True)

                    location_codes = self.LocationID.fetch_codes()[self.LocationID.KEY]

                    for i, x in dat[dat.Description.isnull()].Stanox.items():
                        idx = location_codes[location_codes.STANOX == x].index
                        if len(idx) == 1:
                            idx = idx[0]
                            dat.loc[i, 'Description'] = \
                                location_codes[location_codes.STANOX == x].Location[idx]
                            dat.loc[i, 'Name'] = location_codes[location_codes.STANOX == x].STANME[idx]
                        else:
                            print("Errors occur at index \"{}\" where the corresponding STANOX is "
                                  "\"{}\"".format(i, x))
                            pass

                    for i, x in dat[dat.Name.isnull()].Stanox.items():
                        temp = location_codes[location_codes.STANOX == x]
                        if temp.shape[0] > 1:
                            desc = dat[dat.Stanox == x].Description[i]
                            if desc in temp.TIPLOC.values:
                                idx = temp[temp.TIPLOC == desc].index
                            elif desc in temp.STANME.values:
                                idx = temp[location_codes.STANME == desc].index
                            else:
                                print("Errors occur at index \"{}\" where the corresponding STANOX is "
                                      "\"{}\"".format(i, x))
                                break
                        else:
                            idx = temp.index

                        if len(idx) >= 1:
                            # Choose the first instance, and print a warning message
                            if len(idx) > 1 and verbose == 2:
                                print(f"Warning: The STANOX \"{x}\" at index \"{i}\" is not unique. "
                                      f"The first instance is chosen.")
                            idx = idx[0]
                            dat.loc[i, 'Description'] = temp.Location.loc[idx]
                            dat.loc[i, 'Name'] = temp.STANME.loc[idx]
                        else:
                            pass

                    location_stanme_dict = \
                        location_codes[['Location', 'STANME']].set_index('Location').to_dict()['STANME']
                    dat.Name.replace(location_stanme_dict, inplace=True)

                    # Use manually-created dictionary of regular expressions
                    dat.replace(fetch_loc_names_repl_dict(k='Description'), inplace=True)
                    dat.replace(fetch_loc_names_repl_dict(k='Description', regex=True), inplace=True)

                    # Use STANOX dictionary
                    stanox_dict = self.LocationID.make_xref_dict('STANOX')
                    temp = dat.join(stanox_dict, on='Stanox')[['Description', 'Location']]
                    temp.loc[temp.Location.isnull(), 'Location'] = temp.loc[
                        temp.Location.isnull(), 'Description']
                    dat.Description = temp.apply(
                        lambda y: find_similar_str(y.Description, y.Location, processor='fuzzywuzzy')
                        if isinstance(y.Location, tuple) else y.Location,
                        axis=1)

                    dat.Name = dat.Name.str.upper()

                    location_codes_cut = location_codes[['Location', 'STANME', 'STANOX']].groupby(
                        ['STANOX', 'Location']).agg({'STANME': lambda y: list(y)[0]})
                    temp = dat.join(location_codes_cut, on=['Stanox', 'Description'])
                    dat.Name = temp.STANME

                    dat.rename(columns={'Description': 'Location', 'Name': 'LocationAlias'},
                               inplace=True)

                    # dat.dropna(how='all', subset=['ELR', 'Yards', 'LocationId'], inplace=True)
                    return dat

                # Cleanse raw stanox_location
                stanox_location = _cleanse_stanox_location()

                # For 'ELR', replace NaN with ''
                stanox_location.ELR.fillna('', inplace=True)

                # For 'LocationId'
                stanox_location.Yards = stanox_location.Yards.map(
                    lambda x: '' if np.isnan(x) else int(x))
                stanox_location.LocationId = stanox_location.LocationId.map(
                    lambda x: '' if np.isnan(x) else int(x))

                # To convert yards to miles (Note: Not the 'mileage' used by Network Rail)
                if use_nr_mileage_format:
                    stanox_location['Mileage'] = stanox_location.Yards.map(yard_to_mileage)

                # Set index
                stanox_location.set_index('Stanox', inplace=True)

                save_pickle(stanox_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.StanoxLocation, e))
                stanox_location = None

        return stanox_location

    def get_stanox_section(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'StanoxSection'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'StanoxSection'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> stanox_section_tbl = metex.get_stanox_section(update=True, verbose=True)
            Updating "StanoxSection.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> stanox_section_tbl.tail()
        """

        METExLite.StanoxSection = 'StanoxSection'
        path_to_pickle = self.cdd_tables(METExLite.StanoxSection + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            stanox_section = load_pickle(path_to_pickle)

        else:
            try:
                # Read StanoxSection table from the database
                stanox_section = self.read_table(
                    table_name=METExLite.StanoxSection,
                    index_col=self.get_primary_key(METExLite.StanoxSection),
                    save_as=save_original_as, update=update)

                stanox_section.index.name = METExLite.StanoxSection + 'Id'
                stanox_section.LocationId = stanox_section.LocationId.apply(
                    lambda x: '' if np.isnan(x) else int(x))

                stanox_dat = self.LocationID.make_xref_dict('STANOX')

                # Firstly, create a stanox-to-location dictionary, and replace STANOX with location names
                for stanox_col_name in ['StartStanox', 'EndStanox']:
                    tmp_col = stanox_col_name + '_temp'
                    # Load stanox dictionary 1
                    stanox_dict = self.get_stanox_location(use_nr_mileage_format=True).Location.to_dict()
                    stanox_section[tmp_col] = stanox_section[stanox_col_name].replace(stanox_dict)
                    tmp = stanox_section.join(stanox_dat, on=tmp_col).Location
                    tmp_idx = tmp[tmp.notnull()].index
                    stanox_section.loc[tmp_idx, tmp_col] = tmp[tmp_idx]
                    stanox_section[tmp_col] = stanox_section[tmp_col].map(
                        lambda x: x[0] if isinstance(x, list) else x)

                stanme_dict = self.LocationID.make_xref_dict('STANME', as_dict=True)
                tiploc_dict = self.LocationID.make_xref_dict('TIPLOC', as_dict=True)

                # Secondly, process 'STANME' and 'TIPLOC'
                loc_name_replacement_dict = fetch_loc_names_repl_dict()
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict(regex=True)
                # Processing 'StartStanox_tmp'
                stanox_section.StartStanox_temp = stanox_section.StartStanox_temp. \
                    replace(stanme_dict).replace(tiploc_dict). \
                    replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
                # Processing 'EndStanox_tmp'
                stanox_section.EndStanox_temp = stanox_section.EndStanox_temp. \
                    replace(stanme_dict).replace(tiploc_dict). \
                    replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

                # Create 'STANOX' sections
                temp = stanox_section[stanox_section.StartStanox_temp.map(
                    lambda x: False if isinstance(x, str) else True)]
                temp_ = temp.Description.str.split(' : ', expand=True)
                temp_.columns = ['StartStanox_', 'temp']
                temp = temp.join(temp_.StartStanox_)
                stanox_section.loc[temp.index, 'StartStanox_temp'] = temp.apply(
                    lambda x: find_similar_str(
                        x.StartStanox_, x.StartStanox_temp, processor='fuzzywuzzy'),
                    axis=1)  # Temporary!

                temp = stanox_section[
                    stanox_section.EndStanox_temp.map(lambda x: False if isinstance(x, str) else True)]
                temp_ = temp.Description.str.split(' : ', expand=True)
                temp_.columns = ['temp', 'EndStanox_']
                temp = temp.join(temp_.EndStanox_.fillna(temp.Description))
                stanox_section.loc[temp.index, 'EndStanox_temp'] = temp.apply(
                    lambda x: find_similar_str(x.EndStanox_, x.EndStanox_temp, processor='fuzzywuzzy'),
                    axis=1)  # Temporary!

                start_end = stanox_section.StartStanox_temp + ' - ' + stanox_section.EndStanox_temp
                point_idx = stanox_section.StartStanox_temp == stanox_section.EndStanox_temp
                start_end[point_idx] = stanox_section.StartStanox_temp[point_idx]
                stanox_section['StanoxSection'] = start_end

                # Finalizing the cleaning process
                stanox_section.drop('Description', axis=1, inplace=True)  # Drop original
                stanox_section.rename(
                    columns={'StartStanox_temp': 'StartLocation', 'EndStanox_temp': 'EndLocation'},
                    inplace=True)
                stanox_section = stanox_section[['LocationId', 'StanoxSection',
                                                 'StartLocation', 'StartStanox', 'EndLocation',
                                                 'EndStanox',
                                                 'ApproximateLocation']]

                save_pickle(stanox_section, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.StanoxSection, e))
                stanox_section = None

        return stanox_section

    def get_trust_incident(self, start_year=2006, end_year=None, update=False, save_original_as=None,
                           use_amendment_csv=True, verbose=False):
        """
        Get data of the table 'TrustIncident'.

        :param start_year: defaults to ``2006``
        :type start_year: int or None
        :param end_year: defaults to ``None``
        :type end_year: int or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param use_amendment_csv: whether to use a supplementary .csv file
            to amend the original table data in the database, defaults to ``True``
        :type use_amendment_csv: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'TrustIncident'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> trust_incident_tbl = metex.get_trust_incident(update=True, verbose=True)
            Updating "TrustIncident-y2006-y2018-amended.pickle" at "data\\metex\\...\\tables" ... Done.
            >>> trust_incident_tbl.tail()
        """

        METExLite.TrustIncident = 'TrustIncident'
        suffix0 = "{}".format(
            "{}".format("-y{}".format(start_year) if start_year else "-up-to") +
            "{}".format("-y{}".format(2018 if not end_year or end_year >= 2019 else end_year)))
        suffix1 = "-amended" if use_amendment_csv else ""
        path_to_pickle = self.cdd_tables(METExLite.TrustIncident + suffix0 + suffix1 + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            trust_incident = load_pickle(path_to_pickle)

        else:
            try:
                trust_incident = self.read_table(
                    table_name=METExLite.TrustIncident,
                    index_col=self.get_primary_key(METExLite.TrustIncident),
                    save_as=save_original_as, update=update)

                if use_amendment_csv:
                    zip_file = zipfile.ZipFile(self.cdd("updates", METExLite.TrustIncident + ".zip"))
                    corrected_csv = pd.concat(
                        [pd.read_csv(zip_file.open(f), index_col='Id',
                                     parse_dates=['StartDate', 'EndDate'],
                                     infer_datetime_format=True, dayfirst=True)
                         for f in zip_file.infolist()])
                    zip_file.close()
                    # Remove raw data >= '2018-01-01', pd.to_datetime('2018-01-01')
                    trust_incident.drop(
                        trust_incident[trust_incident.StartDate >= '2018-01-01'].index, inplace=True)
                    # Append corrected data
                    trust_incident = trust_incident.append(corrected_csv)

                trust_incident.index.name = 'TrustIncidentId'
                trust_incident.rename(columns={'Imdm': 'IMDM', 'Year': 'FinancialYear'}, inplace=True)
                # Extract a subset of data,
                # in which the StartDateTime is between 'start_year' and 'end_year'?
                trust_incident = trust_incident[
                    (trust_incident.FinancialYear >= (start_year if start_year else 0)) &
                    (trust_incident.FinancialYear <= (
                        end_year if end_year else datetime.datetime.now().year))]
                # Convert float to int values for 'SourceLocationId'
                trust_incident.SourceLocationId = trust_incident.SourceLocationId.map(
                    lambda x: '' if pd.isna(x) else int(x))

                save_pickle(trust_incident, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.TrustIncident, e))
                trust_incident = None

        return trust_incident

    def get_weather(self, chunk_size=50000):
        """
        Get data of the table 'Weather'.

        :param chunk_size: number of rows to include in a chunk
        :type chunk_size: int
        :return: data of the table 'Weather'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> weather_tbl = metex.get_weather()
            >>> weather_tbl.tail()
        """

        METExLite.Weather = 'Weather'

        try:
            conn_db = establish_mssql_connection(database_name=self.DatabaseName)
            sql_query = "SELECT * FROM dbo.[Weather];"

            idx_col = self.get_primary_key(table_name=METExLite.Weather)

            chunks = pd.read_sql_query(
                sql=sql_query, con=conn_db, index_col=idx_col, parse_dates=['DateTime'],
                chunksize=chunk_size)

            weather = pd.concat([pd.DataFrame(chunk) for chunk in chunks], sort=False)

            del chunks
            gc.collect()

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(METExLite.Weather, e))
            weather = None

        return weather

    def query_weather_by_id_datetime(self, weather_cell_id, start_dt=None, end_dt=None, postulate=False,
                                     pickle_it=False, dat_dir=None, update=False, verbose=False):
        """
        Get weather data by ``'WeatherCell'`` and ``'DateTime'`` (Query from the database).

        :param weather_cell_id: weather cell ID
        :type weather_cell_id: int
        :param start_dt: start date and time, defaults to ``None``
        :type start_dt: datetime.datetime, str or None
        :param end_dt: end date and time, defaults to ``None``
        :type end_dt: datetime.datetime, str or None
        :param postulate: whether to add postulated data, defaults to ``False``
        :type postulate: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: weather data by ``'weather_cell_id'``, ``'start_dt'`` and ``'end_dt'``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            >>> import datetime

            >>> metex = METExLite()

            >>> cell_id = 2367
            >>> sdt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'

            >>> edt = datetime.datetime(2018, 6, 1, 13)  # '2018-06-01 13:00:00'
            >>> w_dat = metex.query_weather_by_id_datetime(cell_id, sdt, edt, update=True, verbose=True)
            >>> w_dat

            >>> edt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> w_dat = metex.query_weather_by_id_datetime(cell_id, sdt, edt, update=True, verbose=True)
            >>> w_dat
        """

        assert isinstance(weather_cell_id, (tuple, int, np.integer))

        # Make a pickle filename
        def _make_weather_pickle_filename():
            if isinstance(weather_cell_id, tuple):
                c_id = "-".join(str(x) for x in list(weather_cell_id))
            else:
                c_id = weather_cell_id
            s_dt = start_dt.strftime('_fr%Y%m%d%H%M') if start_dt else ""
            e_dt = end_dt.strftime('_to%Y%m%d%H%M') if end_dt else ""
            return "{}{}{}.pickle".format(c_id, s_dt, e_dt)

        pickle_filename = _make_weather_pickle_filename()

        # Specify a directory/path to store the pickle file (if appropriate)
        path_to_pickle = self.cdd_views("weather" if dat_dir is None else dat_dir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            weather_dat = load_pickle(path_to_pickle)

        else:
            try:
                # Establish a connection to the MSSQL server
                conn_metex = establish_mssql_connection(database_name=self.DatabaseName)
                # Specify database sql query
                sql_query = "SELECT * FROM dbo.[Weather] WHERE {} {} {} AND {} AND {};".format(
                    "[WeatherCell]", "IN" if isinstance(weather_cell_id, tuple) else "=",
                    weather_cell_id,
                    "[DateTime] >= '{}'".format(start_dt) if start_dt else "",
                    "[DateTime] <= '{}'".format(end_dt) if end_dt else "")
                # Query the weather data
                weather_dat = pd.read_sql(sql_query, conn_metex)

                if postulate:
                    i = 0
                    snowfall = weather_dat.Snowfall.tolist()
                    precipitation = weather_dat.TotalPrecipitation.tolist()
                    while i + 3 < len(weather_dat):
                        snowfall[i + 1: i + 3] = np.linspace(snowfall[i], snowfall[i + 3], 4)[1:3]
                        precipitation[i + 1: i + 3] = np.linspace(
                            precipitation[i], precipitation[i + 3], 4)[1:3]
                        i += 3
                    if i + 2 == len(weather_dat):
                        snowfall[-1:], precipitation[-1:] = snowfall[-2], precipitation[-2]
                    elif i + 3 == len(weather_dat):
                        snowfall[-2:], precipitation[-2:] = [snowfall[-3]] * 2, [precipitation[-3]] * 2
                    weather_dat.Snowfall = snowfall
                    weather_dat.TotalPrecipitation = precipitation

                if pickle_it:
                    save_pickle(weather_dat, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))
                weather_dat = None

        return weather_dat

    def get_weather_cell(self, route_name=None, update=False, save_original_as=None, show_map=False,
                         projection='tmerc', save_map_as=None, dpi=None, verbose=False):
        """
        Get data of the table 'WeatherCell'.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param show_map: whether to show a map of the weather cells, defaults to ``False``
        :type show_map: bool
        :param projection: defaults to ``'tmerc'``
        :type projection: str
        :param save_map_as: whether to save the created map or what format the created map is saved as,
            defaults to ``None``
        :type save_map_as: str or None
        :param dpi: defaults to ``None``
        :type dpi: int or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'WeatherCell'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> weather_cell_tbl = metex.get_weather_cell(update=True, verbose=True)
            Updating "WeatherCell.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> weather_cell_tbl.head()

            >>> weather_cell_tbl = metex.get_weather_cell('Anglia', update=True, verbose=True)
            Updating "WeatherCell-Anglia.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> weather_cell_tbl.head()

            >>> _ = metex.get_weather_cell(show_map=True, save_map_as=".tif", verbose=True)
            Plotting weather cells ... Done.
            Updating "WeatherCell.tif" at "data\\metex\\database_lite\\figures" ... Done.
            >>> _ = metex.get_weather_cell(route_name='Anglia', show_map=True, save_map_as=".tif")
            Plotting weather cells ... Done.
        """

        METExLite.WeatherCell = 'WeatherCell'
        pickle_filename = make_filename(METExLite.WeatherCell, route_name)
        path_to_pickle = self.cdd_tables(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            weather_cell = load_pickle(path_to_pickle)

        else:
            try:
                weather_cell = self.read_table(
                    table_name=METExLite.WeatherCell,
                    index_col=self.get_primary_key(METExLite.WeatherCell),
                    save_as=save_original_as, update=update)

                id_name = METExLite.WeatherCell + 'Id'
                weather_cell.index.rename(id_name, inplace=True)

                # Get IMDM Weather cell map
                imdm_weather_cell_map = self.get_imdm_weather_cell_map().reset_index()

                # Merge the acquired data set
                weather_cell = imdm_weather_cell_map.join(
                    weather_cell, on='WeatherCellId').sort_values('WeatherCellId')
                weather_cell.set_index('WeatherCellId', inplace=True)

                # Subset
                weather_cell = get_subset(weather_cell, route_name)

                # Lower left corner:
                weather_cell['ll_Longitude'] = weather_cell.Longitude  # - weather_cell.width / 2
                weather_cell['ll_Latitude'] = weather_cell.Latitude  # - weather_cell.height / 2
                # Upper left corner:
                weather_cell['ul_Longitude'] = weather_cell.ll_Longitude  # - weather_cell.width / 2
                weather_cell['ul_Latitude'] = weather_cell.ll_Latitude + weather_cell.height  # / 2
                # Upper right corner:
                weather_cell['ur_Longitude'] = weather_cell.ul_Longitude + weather_cell.width  # / 2
                weather_cell['ur_Latitude'] = weather_cell.ul_Latitude  # + weather_cell.height / 2
                # Lower right corner:
                weather_cell['lr_Longitude'] = weather_cell.ur_Longitude  # + weather_cell.width  # / 2
                weather_cell['lr_Latitude'] = weather_cell.ur_Latitude - weather_cell.height  # / 2

                # Create polygons WGS84 (Longitude, Latitude)
                weather_cell['Polygon_WGS84'] = weather_cell.apply(
                    lambda x: shapely.geometry.Polygon(
                        zip([x.ll_Longitude, x.ul_Longitude, x.ur_Longitude, x.lr_Longitude],
                            [x.ll_Latitude, x.ul_Latitude, x.ur_Latitude, x.lr_Latitude])), axis=1)

                # Create polygons OSGB36 (Easting, Northing)
                weather_cell['ll_Easting'], weather_cell['ll_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ll_Longitude.values, weather_cell.ll_Latitude.values)
                weather_cell['ul_Easting'], weather_cell['ul_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ul_Longitude.values, weather_cell.ul_Latitude.values)
                weather_cell['ur_Easting'], weather_cell['ur_Northing'] = \
                    wgs84_to_osgb36(weather_cell.ur_Longitude.values, weather_cell.ur_Latitude.values)
                weather_cell['lr_Easting'], weather_cell['lr_Northing'] = \
                    wgs84_to_osgb36(weather_cell.lr_Longitude.values, weather_cell.lr_Latitude.values)

                weather_cell['Polygon_OSGB36'] = weather_cell.apply(
                    lambda x: shapely.geometry.Polygon(
                        zip([x.ll_Easting, x.ul_Easting, x.ur_Easting, x.lr_Easting],
                            [x.ll_Northing, x.ul_Northing, x.ur_Northing, x.lr_Northing])), axis=1)

                regions_and_routes = load_json(cdd_network("regions", "routes.json"))
                regions_and_routes_list = [{x: k} for k, v in regions_and_routes.items() for x in v]
                # noinspection PyTypeChecker
                regions_and_routes_dict = {k: v for d in regions_and_routes_list for k, v in d.items()}
                weather_cell['Region'] = weather_cell.Route.replace(regions_and_routes_dict)

                save_pickle(weather_cell, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.WeatherCell, e))
                weather_cell = None

        # Plot the Weather cells on the map?
        if show_map:
            weather_cell_wgs84 = shapely.geometry.MultiPolygon(list(weather_cell.Polygon_WGS84))
            minx, miny, maxx, maxy = weather_cell_wgs84.bounds

            import matplotlib.pyplot as plt
            import mpl_toolkits.basemap
            import matplotlib.patches

            from pyhelpers.settings import mpl_preferences

            mpl_preferences(font_name='Cambria')

            print("Plotting weather cells", end=" ... ")
            fig, ax = plt.subplots(figsize=(5, 8))
            m = mpl_toolkits.basemap.Basemap(projection=projection,  # Transverse Mercator Projection
                                             ellps='WGS84',
                                             epsg=27700,
                                             llcrnrlon=minx - 0.285,
                                             llcrnrlat=miny - 0.255,
                                             urcrnrlon=maxx + 1.185,
                                             urcrnrlat=maxy + 0.255,
                                             lat_ts=0,
                                             resolution='h',
                                             suppress_ticks=True)

            # m.arcgisimage(service='World_Street_Map', xpixels=1500, dpi=300, verbose=False)

            m.drawlsmask(land_color='0.9', ocean_color='#9EBCD8', resolution='f', grid=1.25)
            m.fillcontinents(color='0.9')
            m.drawcountries()
            # m.drawcoastlines()

            cell_map = weather_cell.drop_duplicates(
                subset=[s for s in weather_cell.columns if '_' in s and not s.startswith('Polygon')])

            for i in cell_map.index:
                ll_x, ll_y = m(cell_map.ll_Longitude[i], cell_map.ll_Latitude[i])
                ul_x, ul_y = m(cell_map.ul_Longitude[i], cell_map.ul_Latitude[i])
                ur_x, ur_y = m(cell_map.ur_Longitude[i], cell_map.ur_Latitude[i])
                lr_x, lr_y = m(cell_map.lr_Longitude[i], cell_map.lr_Latitude[i])
                xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
                polygons = matplotlib.patches.Polygon(list(xy), fc='#D5EAFF', ec='#4b4747', alpha=0.5)
                ax.add_patch(polygons)

            plt.plot([], 's', label="Weather cell", ms=14, color='#D5EAFF', markeredgecolor='#4b4747')
            legend = plt.legend(numpoints=1, loc='best', fancybox=False)
            frame = legend.get_frame()
            frame.set_edgecolor('w')

            plt.tight_layout()

            print("Done.")

            if save_map_as:
                path_to_fig = self.cdd_figures(pickle_filename.replace(".pickle", save_map_as))
                save_fig(path_to_fig, dpi=dpi, verbose=verbose)

        return weather_cell

    def get_weather_cell_map_boundary(self, route_name=None, adjustment=(0.285, 0.255)):
        """
        Get the lower-left and upper-right corners for a weather cell map.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param adjustment: defaults to ``(0.285, 0.255)``
        :type adjustment: tuple
        :return: a boundary for a weather cell map
        :rtype: shapely.geometry.polygon.Polygon

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> boundary_poly = metex.get_weather_cell_map_boundary()
            >>> boundary_poly.wkt
            'POLYGON ((...))'
        """

        weather_cell = self.get_weather_cell()  # Get Weather cell

        if route_name:  # For a specific Route
            weather_cell = weather_cell[
                weather_cell.Route == find_similar_str(route_name, self.get_route().Route)]
        ll = tuple(weather_cell[['ll_Longitude', 'll_Latitude']].apply(min))
        lr = weather_cell.lr_Longitude.max(), weather_cell.lr_Latitude.min()
        ur = tuple(weather_cell[['ur_Longitude', 'ur_Latitude']].apply(max))
        ul = weather_cell.ul_Longitude.min(), weather_cell.ul_Latitude.max()

        if adjustment:  # Adjust the boundaries
            adj_values = np.array(adjustment)
            ll -= adj_values
            lr += (adj_values, -adj_values)
            ur += adj_values
            ul += (-adj_values, adj_values)

        boundary = shapely.geometry.Polygon((ll, lr, ur, ul))

        return boundary

    def get_track(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Track'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Track'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> track_tbl = metex.get_track(update=True, verbose=True)
            Updating "Track.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> track_tbl.tail()
        """

        METExLite.Track = 'Track'
        path_to_pickle = self.cdd_tables(METExLite.Track + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            track = load_pickle(path_to_pickle)

        else:
            try:
                track = read_table_by_query(database_name=self.DatabaseName, table_name=METExLite.Track,
                                            save_as=save_original_as).drop_duplicates()
                track.rename(columns={'S_MILEAGE': 'StartMileage', 'F_MILEAGE': 'EndMileage',
                                      'S_YARDAGE': 'StartYard', 'F_YARDAGE': 'EndYard',
                                      'MAINTAINER': 'Maintainer', 'ROUTE': 'Route',
                                      'DELIVERY_U': 'IMDM',
                                      'StartEasti': 'StartEasting', 'StartNorth': 'StartNorthing',
                                      'EndNorthin': 'EndNorthing'},
                             inplace=True)

                # Mileage and Yardage
                mileage_cols, yardage_cols = ['StartMileage', 'EndMileage'], ['StartYard', 'EndYard']
                track[mileage_cols] = track[mileage_cols].applymap(mileage_num_to_str)
                track[yardage_cols] = track[yardage_cols].applymap(int)

                # Route
                update_route_names(track, route_col_name='Route')

                # Delivery Unit and IMDM
                track.IMDM = track.IMDM.map(lambda x: 'IMDM ' + x)

                # Start and end longitude and latitude coordinates
                track['StartLongitude'], track['StartLatitude'] = osgb36_to_wgs84(
                    track.StartEasting.values, track.StartNorthing.values)
                track['EndLongitude'], track['EndLatitude'] = osgb36_to_wgs84(
                    track.EndEasting.values, track.EndNorthing.values)

                track[['StartMileage_num', 'EndMileage_num']] = track[
                    ['StartMileage', 'EndMileage']].applymap(mileage_str_to_num)

                track.sort_values(['ELR', 'StartYard', 'EndYard'], inplace=True)
                track.index = range(len(track))

                save_pickle(track, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.Track, e))
                track = None

        return track

    @staticmethod
    def create_track_geometric_graph(geom_objects, rotate_labels=None):
        """
        Create a graph to illustrate track geometry.

        :param geom_objects: geometry objects
        :type geom_objects: iterable of [WKT str, shapely.geometry.LineString,
            or shapely.geometry.MultiLineString]
        :param rotate_labels: defaults to ``None``
        :type rotate_labels: numbers.Number, None
        :return: a graph demonstrating the tracks

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> track_tbl = metex.get_track()
            >>> geom_objs = track_tbl.geom[list(range(len(track_tbl[track_tbl.ELR == 'AAV'])))]

            >>> metex.create_track_geometric_graph(geom_objs)
        """

        g = nx.Graph()

        import matplotlib.pyplot as plt
        from pyhelpers.settings import mpl_preferences

        mpl_preferences()

        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot()

        max_node_id = 0
        for geom_obj in geom_objects:

            if isinstance(geom_obj, str):
                geom_obj = shapely.wkt.loads(geom_obj)

            geom_type, geom_pos = shapely.geometry.mapping(geom_obj).values()

            if geom_type == 'MultiLineString':

                # Sort line strings of the multi-line string
                geom_pos_idx = list(range(len(geom_pos)))
                pos_idx_sorted = []
                while geom_pos_idx:
                    y = geom_pos_idx[0]
                    p1 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                    if p1 and p1[0] not in pos_idx_sorted:
                        pos_idx_sorted = p1 + [y]
                    else:
                        pos_idx_sorted = [y]

                    p2 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                    if p2:
                        pos_idx_sorted += p2

                    geom_pos_idx = [a for a in geom_pos_idx if a not in pos_idx_sorted]

                    if len(geom_pos_idx) == 1:
                        y = geom_pos_idx[0]
                        p3 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                        if p3 and p3[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p3[0]) + 1, y)
                            break

                        p4 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                        if p4 and p4[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p4[0]), y)
                            break

                geom_pos = [geom_pos[i] for i in pos_idx_sorted]
                geom_pos = [x[:-1] for x in geom_pos[:-1]] + [geom_pos[-1]]
                geom_pos = [x for g_pos in geom_pos for x in g_pos]

            # j = 0
            # n = g.number_of_nodes()
            # while j < len(geom_pos):
            #     # Nodes
            #     g_pos = geom_pos[j]
            #     for i in range(len(g_pos)):
            #         g.add_node(i + n + 1, pos=g_pos[i])
            #     # Edges
            #     current_max_node_id = g.number_of_nodes()
            #     edges = [(x, x + 1) for x in range(n + 1, n + current_max_node_id)
            #              if x + 1 <= current_max_node_id]
            #     g.add_edges_from(edges)
            #     n = current_max_node_id
            #     j += 1

            # Nodes

            for i in range(len(geom_pos)):
                g.add_node(i + 1 + max_node_id, pos=geom_pos[i])

            # Edges
            number_of_nodes = g.number_of_nodes()
            edges = [(i, i + 1) for i in range(1 + max_node_id, number_of_nodes + 1) if
                     i + 1 <= number_of_nodes]
            # edges = [(i, i + 1) for i in range(1, number_of_nodes + 1) if i + 1 <= number_of_nodes]
            g.add_edges_from(edges)

            # Plot
            nx.draw_networkx(g, pos=nx.get_node_attributes(g, name='pos'), ax=ax, node_size=0,
                             with_labels=False)

            max_node_id = number_of_nodes

        ax.tick_params(
            left=True, bottom=True, labelleft=True, labelbottom=True, gridOn=True, grid_linestyle='--')
        ax.ticklabel_format(useOffset=False)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting', fontsize=14)
        ax.set_ylabel('Northing', fontsize=14)
        if rotate_labels:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate_labels)
        plt.tight_layout()

    @staticmethod
    def _cleanse_track_summary(trk_smry_raw):
        """
        Preprocess data of the table 'TrackSummary'.

        :param trk_smry_raw: data of the table 'TrackSummary'
        :type trk_smry_raw: pandas.DataFrame
        :return: preprocessed data frame of ``dat``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> track_summary_tbl = metex.read_table(table_name='Track Summary')
            >>> track_summary_tbl.tail()

            >>> track_summary_dat = metex._cleanse_track_summary(track_summary_tbl)
            >>> track_summary_dat.tail()
        """

        dat = trk_smry_raw.copy()

        # Change column names
        rename_cols = {'GEOGIS Switch ID': 'GeoGISSwitchID',
                       'TID': 'TrackID',
                       'StartYards': 'StartYard',
                       'EndYards': 'EndYard',
                       'Sub-route': 'SubRoute',
                       'CP6 criticality': 'CP6Criticality',
                       'CP5 Start Route': 'CP5StartRoute',
                       'Adjacent S&C': 'AdjacentS&C',
                       'Rail cumulative EMGT': 'RailCumulativeEMGT',
                       'Sleeper cumulative EMGT': 'SleeperCumulativeEMGT',
                       'Ballast cumulative EMGT': 'BallastCumulativeEMGT'}
        dat.rename(columns=rename_cols, inplace=True)
        renamed_cols = list(rename_cols.values())
        upper_columns = ['ID', 'SRS', 'ELR', 'IMDM', 'TME', 'TSM', 'MGTPA', 'EMGTPA', 'LTSF', 'IRJs']
        dat.columns = [
            string.capwords(x).replace(' ', '') if x not in upper_columns + renamed_cols else x
            for x in dat.columns]

        # IMDM
        dat.IMDM = dat.IMDM.map(lambda x: 'IMDM ' + x)

        # Route
        route_names_changes = load_json(cdd_network("routes", "name-changes.json"))
        # noinspection PyTypeChecker
        temp1 = pd.DataFrame.from_dict(route_names_changes, orient='index', columns=['Route'])
        route_names_in_table = list(dat.SubRoute.unique())
        route_alt = [
            find_similar_str(x, temp1.index.tolist(), processor='fuzzywuzzy')
            for x in route_names_in_table]

        temp2 = pd.DataFrame.from_dict(dict(zip(route_names_in_table, route_alt)), 'index',
                                       columns=['RouteAlias'])
        temp = temp2.join(temp1, on='RouteAlias').dropna()
        route_names_changes_alt = dict(zip(temp.index, temp.Route))
        dat['Route'] = dat.SubRoute.replace(route_names_changes_alt)

        # Mileages
        mileage_colnames, yard_colnames = ['StartMileage', 'EndMileage'], ['StartYards', 'EndYards']
        dat[mileage_colnames] = dat[yard_colnames].applymap(yard_to_mileage)

        return dat

    def get_track_summary(self, update=False, save_original_as=None, verbose=False):
        """
        Get data of the table 'Track Summary'.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param save_original_as: file extension for saving the original data, defaults to ``None``
        :type save_original_as: str or None
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of the table 'Track Summary'
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> track_summary_tbl = metex.get_track_summary(update=True, verbose=True)
            Updating "TrackSummary.pickle" at "data\\metex\\database_lite\\tables" ... Done.
            >>> track_summary_tbl.tail()
        """

        METExLite.TrackSummary = 'Track Summary'
        path_to_pickle = self.cdd_tables(METExLite.TrackSummary.replace(' ', '') + ".pickle")

        if os.path.isfile(path_to_pickle) and not update:
            track_summary = load_pickle(path_to_pickle)

        else:
            try:
                track_summary_raw = self.read_table(
                    table_name=METExLite.TrackSummary, save_as=save_original_as, update=update)

                track_summary = self._cleanse_track_summary(track_summary_raw)

                save_pickle(track_summary, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(METExLite.TrackSummary, e))
                track_summary = None

        return track_summary

    def query_track_summary(self, elr, track_id, start_yard=None, end_yard=None, pickle_it=False,
                            dat_dir=None, update=False, verbose=False):
        """
        Get track summary data by ``'Track ID'`` and ``'Yard'`` (Query from the database).

        :param elr: ELR
        :type elr: str
        :param track_id: TrackID
        :type track_id: tuple or int or numpy.integer
        :param start_yard: start yard, defaults to ``None``
        :type start_yard: int or None
        :param end_yard: end yard, defaults to ``None``
        :type end_yard: int or None
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param dat_dir: directory where the queried data is saved, defaults to ``None``
        :type dat_dir: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of track summary queried by
            ``'elr'``, ``'track_id'``, ``'start_yard'`` and ``'end_yard'``
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> ts = metex.query_track_summary('AAV', 1100, start_yard=51150, end_yard=66220)
            >>> ts.tail()
        """

        elrs = "-".join(str(x) for x in list(elr)) if isinstance(elr, tuple) else elr
        tid = "-".join(str(x) for x in list(track_id)) if isinstance(track_id, tuple) else track_id

        pickle_filename = "{}_{}_{}_{}.pickle".format(elrs, tid, start_yard, end_yard)

        dat_dir = dat_dir if isinstance(dat_dir, str) and os.path.isabs(dat_dir) else self.cdd_views()
        path_to_pickle = cd(dat_dir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                conn_metex = establish_mssql_connection(database_name=self.DatabaseName)

                con1 = "[ELR] {} '{}'".format("=" if isinstance(elr, str) else "IN", elr)
                con2 = "[TID] {} {}".format(
                    "=" if isinstance(track_id, (int, np.integer)) else "IN", track_id)
                con3 = "[Start Yards] >= {}".format(start_yard) if start_yard else ""
                con4 = "[End Yards] <= {}".format(end_yard) if end_yard else ""

                sql_query = "SELECT * FROM dbo.[Track Summary] WHERE {} AND {} AND {} AND {};".format(
                    con1, con2, con3, con4)

                track_summary_raw = pd.read_sql(sql_query, conn_metex)

                track_summary = self._cleanse_track_summary(track_summary_raw)

                if pickle_it:
                    save_pickle(track_summary, path_to_pickle, verbose=verbose)

                return track_summary

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(
                    os.path.splitext(os.path.basename(path_to_pickle))[0], e))

    def _update_metex_table_pickles(self, update=True, verbose=True):
        """
        Update the local pickle files for all tables.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> metex._update_metex_table_pickles()
        """

        if confirmed("To update the local pickles of the Table data of the NR_METEX database?"):

            _ = self.get_imdm(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_imdm(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_imdm_alias(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_imdm_alias(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_imdm_weather_cell_map(route_info=True, grouped=False, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=True, grouped=True, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=False, grouped=False, update=update,
                                               save_original_as=None, verbose=verbose)
            _ = self.get_imdm_weather_cell_map(route_info=False, grouped=True, update=update,
                                               save_original_as=None, verbose=verbose)

            _ = self.get_incident_reason_info(plus=True, update=update, save_original_as=None,
                                              verbose=verbose)
            _ = self.get_incident_reason_info(plus=False, update=update, save_original_as=None,
                                              verbose=verbose)

            _ = self.get_weather_codes(as_dict=False, update=update, save_original_as=None,
                                       verbose=verbose)
            _ = self.get_weather_codes(as_dict=True, update=update, save_original_as=None,
                                       verbose=verbose)

            _ = self.get_incident_record(update=update, save_original_as=None, use_amendment_csv=True,
                                         verbose=verbose)

            _ = self.get_location(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_pfpi(plus=True, update=update, save_original_as=None, use_amendment_csv=True,
                              verbose=verbose)
            _ = self.get_pfpi(plus=False, update=update, save_original_as=None, use_amendment_csv=True,
                              verbose=verbose)

            _ = self.get_route(as_dict=False, update=update, save_original_as=None, verbose=verbose)
            _ = self.get_route(as_dict=True, update=update, save_original_as=None, verbose=verbose)

            _ = self.get_stanox_location(use_nr_mileage_format=True, update=update,
                                         save_original_as=None, verbose=verbose)
            _ = self.get_stanox_location(use_nr_mileage_format=False, update=update,
                                         save_original_as=None, verbose=verbose)

            _ = self.get_stanox_section(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_trust_incident(start_year=2006, end_year=None, update=update,
                                        save_original_as=None, use_amendment_csv=True, verbose=verbose)

            # _ = get_weather()

            _ = self.get_weather_cell(route_name=None, update=update, save_original_as=None,
                                      show_map=True, projection='tmerc', save_map_as=".tif", dpi=None,
                                      verbose=verbose)
            _ = self.get_weather_cell('Anglia', update=update, save_original_as=None, show_map=True,
                                      projection='tmerc', save_map_as=".tif", dpi=None, verbose=verbose)
            # _ = self.get_weather_cell_map_boundary(route=None, adjustment=(0.285, 0.255))

            _ = self.get_track(update=update, save_original_as=None, verbose=verbose)

            _ = self.get_track_summary(update=update, save_original_as=None, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")

    # == Tools to make information integration easier =================================================

    @staticmethod
    def calculate_pfpi_stats(data_set, selected_features, sort_by=None):
        """
        Calculate the 'DelayMinutes' and 'DelayCosts' for grouped data.

        :param data_set: a given data frame
        :type data_set: pandas.DataFrame
        :param selected_features: a list of selected features (column names)
        :type selected_features: list
        :param sort_by: a column or a list of columns by which the selected data is sorted,
            defaults to ``None``
        :type sort_by: str or list or None
        :return: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> # data_set = data = selected_data.copy()
            >>> data = metex.view_schedule8_data('Anglia', 'Wind', rearrange_index=True)
            >>> data.tail()

            >>> selected_feats = [
            ...     'PfPIId', 'WeatherCategory', 'Route', 'StanoxSection',
            ...     'PfPIMinutes', 'PfPICosts']
            >>> res = metex.calculate_pfpi_stats(data[selected_feats], selected_feats)
            >>> res.tail()
        """

        pfpi_stats = data_set.groupby(selected_features[1:-2]).aggregate({
            # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
            'PfPIId': np.count_nonzero,
            'PfPIMinutes': np.sum,
            'PfPICosts': np.sum})

        pfpi_stats.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']
        pfpi_stats.reset_index(inplace=True)  # Reset the grouped indexes to columns

        if sort_by:
            pfpi_stats.sort_values(sort_by, inplace=True)

        return pfpi_stats

    # == Methods to create views ======================================================================

    def view_schedule8_data(self, route_name=None, weather_category=None, rearrange_index=False,
                            weather_attributed_only=False, update=False, pickle_it=False,
                            verbose=False):
        """
        View Schedule 8 details (TRUST data).

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param rearrange_index: whether to rearrange the index of the queried data,
            defaults to ``False``
        :type rearrange_index: bool
        :param weather_attributed_only: defaults to ``False``
        :type weather_attributed_only: bool
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of Schedule 8 details
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> s8data_view = metex.view_schedule8_data(
            ...     rearrange_index=True, update=True, pickle_it=True, verbose=True)
            Updating "s8data.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8data_view.tail()

            >>> s8data_view = metex.view_schedule8_data(
            ...     route_name='Anglia', rearrange_index=True, update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8data-Anglia.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8data_view.tail()

            >>> s8data_view = metex.view_schedule8_data(
            ...     route_name='Anglia', rearrange_index=True, weather_attributed_only=True,
            ...     update=True, pickle_it=True, verbose=True)
            Updating "s8data-weather-attributed-Anglia.pickle" at "data\\metex...\\views" ... Done.
            >>> s8data_view.tail()
        """

        filename = "s8data" + ("-weather-attributed" if weather_attributed_only else "")
        pickle_filename = make_filename(filename, route_name, weather_category, save_as=".pickle")

        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            schedule8_data = load_pickle(path_to_pickle)
            if rearrange_index and schedule8_data.index.name == 'PfPIId':
                schedule8_data.reset_index(inplace=True)

        else:
            path_to_merged = self.cdd_views("{}.pickle".format(filename))

            try:

                if os.path.isfile(path_to_merged) and not update:
                    schedule8_data = load_pickle(path_to_merged)

                else:
                    pfpi = self.get_pfpi(verbose=verbose)  # Get PfPI # (260645, 6)  # (5049003, 6)
                    incident_record = self.get_incident_record(
                        verbose=verbose)  # (233452, 4)  # (4704448, 5)
                    trust_incident = self.get_trust_incident(
                        verbose=verbose)  # (192054, 11)  # (4049984, 11)
                    location = self.get_location(verbose=verbose)  # (228851, 6)  # (653882, 7)
                    imdm = self.get_imdm(verbose=verbose)  # (42, 1)  # (42, 3)
                    incident_reason_info = self.get_incident_reason_info(
                        verbose=verbose)  # (393, 7)  # (174, 9)
                    stanox_location = self.get_stanox_location(verbose=verbose)  # (7560, 5)  # (7534, 6)
                    stanox_section = self.get_stanox_section(verbose=verbose)  # (9440, 7)  # (10601, 7)

                    if weather_attributed_only:
                        incident_record = incident_record[
                            incident_record.WeatherCategory != '']  # (320942, 5) ≈ 6.8%

                    # Merge the acquired data sets - starting with (5049003, 6)
                    schedule8_data = pfpi. \
                        join(incident_record,  # (260645, 10)  # (5049003, 11)
                             on='IncidentRecordId', how='inner'). \
                        join(trust_incident,  # (260483, 21)  # (5048710, 22)
                             on='TrustIncidentId', how='inner'). \
                        join(stanox_section,  # (260483, 28)  # (5048593, 29)
                             on='StanoxSectionId', how='inner'). \
                        join(location,  # (260470, 34)  # (5045204, 36)
                             on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
                        join(stanox_location,  # (260190, 39)  # (5029204, 42)
                             on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
                        join(stanox_location,  # (260140, 44)  # (5024725, 48)
                             on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
                        join(incident_reason_info,  # (260140, 51)  # (5024703, 57)
                             on='IncidentReasonCode', how='inner'). \
                        join(imdm, on='IMDM_Location', how='inner')  # (5024674, 60)

                    del pfpi, incident_record, trust_incident, stanox_section, location, \
                        stanox_location, incident_reason_info, imdm
                    gc.collect()

                    # Note: There may be errors in
                    # e.g. IMDM data/column, location id, of the TrustIncident table.

                    idx = schedule8_data[
                        ~schedule8_data.StartLocation.eq(schedule8_data.Location_Start)].index
                    for i in idx:
                        schedule8_data.loc[i, 'StartLocation'] = schedule8_data.loc[i, 'Location_Start']
                        schedule8_data.loc[i, 'EndLocation'] = schedule8_data.loc[i, 'Location_End']
                        if schedule8_data.loc[i, 'StartLocation'] == \
                                schedule8_data.loc[i, 'EndLocation']:
                            schedule8_data.loc[i, 'StanoxSection'] = schedule8_data.loc[
                                i, 'StartLocation']
                        else:
                            schedule8_data.loc[i, 'StanoxSection'] = \
                                schedule8_data.loc[i, 'StartLocation'] + ' - ' + schedule8_data.loc[
                                    i, 'EndLocation']

                    # (5024674, 57)
                    schedule8_data.drop(
                        ['IMDM', 'Location_Start', 'Location_End'], axis=1, inplace=True)

                    # (260140, 50)  # (5155014, 57)
                    schedule8_data.rename(
                        columns={'LocationAlias_Start': 'StartLocationAlias',
                                 'LocationAlias_End': 'EndLocationAlias',
                                 'ELR_Start': 'StartELR', 'Yards_Start': 'StartYards',
                                 'ELR_End': 'EndELR', 'Yards_End': 'EndYards',
                                 'Mileage_Start': 'StartMileage',
                                 'Mileage_End': 'EndMileage',
                                 'LocationId_Start': 'StartLocationId',
                                 'LocationId_End': 'EndLocationId',
                                 'LocationId_Section': 'SectionLocationId',
                                 'IMDM_Location': 'IMDM',
                                 'StartDate': 'StartDateTime',
                                 'EndDate': 'EndDateTime'}, inplace=True)

                    # Use 'Station' data from Railway Codes website
                    station_locations = self.StationCode.fetch_station_data()[self.StationCode.StnKey]

                    station_locations = station_locations[
                        ['Station', 'Degrees Longitude', 'Degrees Latitude']]
                    station_locations = \
                        station_locations.dropna().drop_duplicates('Station', keep='first')
                    station_locations.set_index('Station', inplace=True)
                    temp = schedule8_data[['StartLocation']].join(
                        station_locations, on='StartLocation', how='left')
                    i = temp[temp['Degrees Longitude'].notna()].index
                    schedule8_data.loc[i, 'StartLongitude':'StartLatitude'] = \
                        temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()
                    temp = schedule8_data[['EndLocation']].join(
                        station_locations, on='EndLocation', how='left')
                    i = temp[temp['Degrees Longitude'].notna()].index
                    schedule8_data.loc[i, 'EndLongitude':'EndLatitude'] = \
                        temp.loc[i, 'Degrees Longitude':'Degrees Latitude'].values.tolist()

                    # data.EndELR.replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)
                    i = schedule8_data.StartLocation == 'Highbury & Islington (North London Lines)'
                    schedule8_data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.1045, 51.5460]
                    i = schedule8_data.EndLocation == 'Highbury & Islington (North London Lines)'
                    schedule8_data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.1045, 51.5460]
                    i = schedule8_data.StartLocation == 'Dalston Junction (East London Line)'
                    schedule8_data.loc[i, ['StartLongitude', 'StartLatitude']] = [-0.0751, 51.5461]
                    i = schedule8_data.EndLocation == 'Dalston Junction (East London Line)'
                    schedule8_data.loc[i, ['EndLongitude', 'EndLatitude']] = [-0.0751, 51.5461]

                schedule8_data.reset_index(inplace=True)  # (5024674, 58)

                schedule8_data = get_subset(
                    schedule8_data, route_name=route_name, weather_category=weather_category,
                    rearrange_index=rearrange_index)

                if pickle_it:
                    if not os.path.isfile(path_to_merged) or path_to_pickle != path_to_merged:
                        save_pickle(schedule8_data, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to retrieve the data about Schedule 8 incidents. {}.".format(e))
                schedule8_data = None

        return schedule8_data

    def view_schedule8_data_pfpi(self, route_name=None, weather_category=None, update=False,
                                 pickle_it=False, verbose=False):
        """
        Get a view of essential details about Schedule 8 incidents.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: essential details about Schedule 8 incidents
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> s8data_pfpi_view = metex.view_schedule8_data_pfpi(
            ...     update=True, pickle_it=True, verbose=True)
            Updating "s8data-pfpi.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8data_pfpi_view.tail()

            >>> s8data_pfpi_view = metex.view_schedule8_data_pfpi(
            ...     route_name='Anglia', update=True, pickle_it=True, verbose=True)
            Updating "s8data-pfpi-Anglia.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8data_pfpi_view.tail()

            >>> s8data_pfpi_view = metex.view_schedule8_data_pfpi(
            ...     route_name='Anglia', weather_category='Wind', update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8data-pfpi-Anglia-Wind.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8data_pfpi_view.tail()
        """

        filename = "s8data-pfpi"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)

                    data = get_subset(temp_data, route_name, weather_category)

                else:
                    # Get the merged data
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)

                    # Define the feature list
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
                        'IncidentReasonCode', 'IncidentReasonDescription',
                        'IncidentCategory', 'IncidentCategoryDescription',
                        # 'IncidentCategoryGroupDescription',
                        'IncidentFMS', 'IncidentEquipment',
                        'WeatherCell',
                        'Route', 'IMDM', 'Region',
                        'StanoxSection', 'StartLocation', 'EndLocation',
                        'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                        'ApproximateLocation']

                    data = schedule8_data[selected_features]

                if pickle_it:
                    save_pickle(data, path_to_pickle, verbose=verbose)

                return data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_location(self, route_name=None, weather_category=None, update=False,
                                         pickle_it=True, verbose=False):
        """
        Get Schedule 8 data by incident location and Weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 data by incident location and Weather category
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> s8costs_loc = metex.view_schedule8_costs_by_location(update=True, pickle_it=True,
            ...                                                      verbose=True)
            Updating "s8costs-by-location.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> s8costs_loc.tail()

            >>> s8costs_loc = metex.view_schedule8_costs_by_location(route_name='Anglia',
            ...                                                      update=True, pickle_it=True,
            ...                                                      verbose=True)
            Updating "s8costs-by-location-Anglia.pickle" at "data\\metex\\datab...\\views" ... Done.
            >>> s8costs_loc.tail()
                 WeatherCategory   Route  ... DelayMinutes DelayCost
            2492            Wind  Anglia  ...      1740.00  33284.44
            2493            Wind  Anglia  ...        32.00    590.70
            2494            Wind  Anglia  ...       555.78   5149.76
            2495            Wind  Anglia  ...      1180.00  19712.10
            2496            Wind  Anglia  ...       555.24   8131.48

            [5 rows x 20 columns]

            >>> s8costs_loc = metex.view_schedule8_costs_by_location(route_name='Anglia',
            ...                                                      weather_category='Wind',
            ...                                                      update=True, pickle_it=True,
            ...                                                      verbose=True)
            Updating "s8costs-by-location-Anglia-Wind.pickle" at "data\\metex\\...\\views" ... Done.
            >>> s8costs_loc.tail()

            >>> s8costs_loc = metex.view_schedule8_costs_by_location(route_name='Anglia',
            ...                                                      weather_category='Heat',
            ...                                                      update=True, pickle_it=True,
            ...                                                      verbose=True)
            Updating "s8costs-by-location-Anglia-Heat.pickle" at "data\\metex\\...\\views" ... Done.
            >>> s8costs_loc.tail()
        """

        filename = "s8costs-by-location"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))
                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(
                        route_name, weather_category, rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'WeatherCategory',
                        'Route', 'IMDM', 'Region', 'StanoxSection',
                        'StartLocation', 'EndLocation', 'StartELR', 'StartMileage', 'EndELR',
                        'EndMileage',
                        'StartStanox', 'EndStanox', 'StartLongitude', 'StartLatitude', 'EndLongitude',
                        'EndLatitude',
                        'PfPIMinutes', 'PfPICosts']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime_location(self, route_name=None, weather_category=None,
                                                  update=False, pickle_it=True, verbose=False):
        """
        Get Schedule 8 data by datetime and location.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 data by datetime and location
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> s8costs_dt_loc = metex.view_schedule8_costs_by_datetime_location(
            ...     update=True, pickle_it=True, verbose=True)
            Updating "s8costs-by-datetime-location.pickle" at "data\\metex\\...\\views" ... Done.
            >>> s8costs_dt_loc.tail()

            >>> s8costs_dt_loc = metex.view_schedule8_costs_by_datetime_location(
            ...     route_name='Anglia', update=True, pickle_it=True, verbose=True)
            Updating "s8costs-by-datetime-location-Anglia.pickle" at "data\\metex\\...\\views" ... Done.
            >>> s8costs_dt_loc.tail()

            >>> s8costs_dt_loc = metex.view_schedule8_costs_by_datetime_location(
            ...     route_name='Anglia', weather_category='Wind', update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8costs-by-datetime-location-Anglia-Wind.pickle" at "data\\...views" ... Done.
            >>> s8costs_dt_loc.tail()

            >>> s8costs_dt_loc = metex.view_schedule8_costs_by_datetime_location(
            ...     route_name='Anglia', weather_category='Heat', update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8costs-by-datetime-location-Anglia-Heat.pickle" at "data\\...views" ... Done.
            >>> s8costs_dt_loc.tail()
        """

        filename = "s8costs-by-datetime-location"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'FinancialYear',
                        'StartDateTime', 'EndDateTime',
                        'WeatherCategory',
                        'StanoxSection',
                        'Route', 'IMDM', 'Region',
                        'StartLocation', 'EndLocation',
                        'StartStanox', 'EndStanox',
                        'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                        'WeatherCell',
                        'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime_location_reason(self, route_name=None, weather_category=None,
                                                         update=False, pickle_it=True, verbose=False):
        """
        Get Schedule 8 costs by datetime, location and incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or list or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or list or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 costs by datetime, location and incident reason
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> s8costs_dt_loc_r = metex.view_schedule8_costs_by_datetime_location_reason(
            ...     update=True, pickle_it=True, verbose=True)
            Updating "s8costs-by-datetime-location-reason.pickle" at "data\\metex\\...\\views" ... Done.
            >>> s8costs_dt_loc_r.tail()

            >>> s8costs_dt_loc_r = metex.view_schedule8_costs_by_datetime_location_reason(
            ...     route_name='Anglia', update=True, pickle_it=True, verbose=True)
            Updating "s8costs-by-datetime-location-reason-Anglia.pickle" at "data\\...\\views" ... Done.
            >>> s8costs_dt_loc_r.tail()

            >>> s8costs_dt_loc_r = metex.view_schedule8_costs_by_datetime_location_reason(
            ...     route_name='Anglia', weather_category='Wind', update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8costs-by-datetime-location-reason-Anglia-Wind.pickle" at "...\\views" ... Done.
            >>> s8costs_dt_loc_r.tail()

            >>> s8costs_dt_loc_r = metex.view_schedule8_costs_by_datetime_location_reason(
            ...     route_name='Anglia', weather_category='Heat', update=True, pickle_it=True,
            ...     verbose=True)
            Updating "s8costs-by-datetime-location-reason-Anglia-Heat.pickle" at "...views" ... Done.
            >>> s8costs_dt_loc_r.tail()
        """

        filename = "s8costs-by-datetime-location-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle, verbose=verbose)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(
                        route_name, weather_category, rearrange_index=True)

                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'StartDateTime', 'EndDateTime',
                                         'WeatherCategory',
                                         'WeatherCell',
                                         'Route', 'IMDM', 'Region',
                                         'StanoxSection',
                                         'StartLocation', 'EndLocation',
                                         'StartStanox', 'EndStanox',
                                         'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                                         'StartLongitude', 'StartLatitude', 'EndLongitude',
                                         'EndLatitude',
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

                    selected_data = schedule8_data[selected_features]

                    extracted_data = self.calculate_pfpi_stats(
                        selected_data, selected_features, sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_datetime(self, route_name=None, weather_category=None, update=False,
                                         pickle_it=False, verbose=False):
        """
        Get Schedule 8 data by datetime and Weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 data by datetime and Weather category
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-datetime"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))
                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = [
                        'PfPIId',
                        # 'TrustIncidentId', 'IncidentRecordCreateDate',
                        'FinancialYear',
                        'StartDateTime', 'EndDateTime',
                        'WeatherCategory',
                        'Route', 'IMDM', 'Region',
                        'WeatherCell',
                        'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=['StartDateTime', 'EndDateTime'])

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_reason(self, route_name=None, weather_category=None, update=False,
                                       pickle_it=False, verbose=False):
        """
        Get Schedule 8 costs by incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 costs by incident reason
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'Route', 'IMDM', 'Region',
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
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features,
                                                               sort_by=None)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_location_reason(self, route_name=None, weather_category=None,
                                                update=False, pickle_it=False, verbose=False):
        """
        Get Schedule 8 costs by location and incident reason.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 costs by location and incident reason
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-location-reason"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(
                        route_name=route_name, weather_category=weather_category, rearrange_index=True)
                    selected_features = ['PfPIId',
                                         'FinancialYear',
                                         'WeatherCategory',
                                         'Route', 'IMDM', 'Region',
                                         'StanoxSection',
                                         'StartStanox', 'EndStanox',
                                         'StartLocation', 'EndLocation',
                                         'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                                         'StartLongitude', 'StartLatitude', 'EndLongitude',
                                         'EndLatitude',
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
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(
                        selected_data, selected_features, sort_by=None)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print("Failed to retrieve \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_schedule8_costs_by_weather_category(self, route_name=None, weather_category=None,
                                                 update=False, pickle_it=False, verbose=False):
        """
        Get Schedule 8 costs by weather category.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``False``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 costs by weather category
        :rtype: pandas.DataFrame
        """

        filename = "s8costs-by-weather_category"
        pickle_filename = make_filename(filename, route_name, weather_category)
        path_to_pickle = self.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_pickle(path_to_pickle)

        else:
            try:
                path_to_pickle_temp = self.cdd_views(make_filename(filename))

                if os.path.isfile(path_to_pickle_temp) and not update:
                    temp_data = load_pickle(path_to_pickle_temp)
                    extracted_data = get_subset(temp_data, route_name, weather_category)

                else:
                    schedule8_data = self.view_schedule8_data(route_name, weather_category,
                                                              rearrange_index=True)
                    selected_features = ['PfPIId', 'FinancialYear', 'Route', 'IMDM', 'Region',
                                         'WeatherCategory', 'PfPICosts', 'PfPIMinutes']
                    selected_data = schedule8_data[selected_features]
                    extracted_data = self.calculate_pfpi_stats(selected_data, selected_features)

                if pickle_it:
                    save_pickle(extracted_data, path_to_pickle, verbose=verbose)

                return extracted_data

            except Exception as e:
                print(
                    "Failed to retrieve \"{}.\" \n{}.".format(os.path.splitext(pickle_filename)[0], e))

    def view_metex_schedule8_incident_locations(self, route_name=None, weather_category=None,
                                                start_and_end_elr=None, update=False, verbose=False):
        """
        Get Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param start_and_end_elr: indicating if start ELR and end ELR are the same or not,
            defaults to ``False``
        :type start_and_end_elr: str, bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section
        :rtype: pandas.DataFrame

        **Test**::

            >>> from preprocessor import METExLite
            
            >>> metex = METExLite()

            >>> incid_loc = metex.view_metex_schedule8_incident_locations(update=True, verbose=True)
            Updating "s8incident-locations.pickle" at "data\\metex\\database_lite\\views" ... Done.
            >>> incid_loc.tail()
                            Route                IMDM  ...     EndEasting    EndNorthing
            23584  North and East      IMDM Newcastle  ...  439806.355306  556943.953192
            23647  North and East      IMDM Newcastle  ...  486883.443012  508107.556972
            24346      South East  IMDM London Bridge  ...  561719.798669  174386.219188
            24501           Wales        IMDM Cardiff  ...  286580.499048  204449.612520
            24755          Wessex      IMDM Eastleigh  ...  401141.953041   89970.873072

            [5 rows x 22 columns]

            >>> incid_loc = metex.view_metex_schedule8_incident_locations(
            ...     start_and_end_elr='same', update=True, verbose=True)
            Updating "s8incident-locations-sameELR.pickle" at "data\\metex\\...\\views" ... Done.
            >>> incid_loc.tail()
                            Route                IMDM  ...     EndEasting    EndNorthing
            23584  North and East      IMDM Newcastle  ...  439806.355306  556943.953192
            23647  North and East      IMDM Newcastle  ...  486883.443012  508107.556972
            24346      South East  IMDM London Bridge  ...  561719.798669  174386.219188
            24501           Wales        IMDM Cardiff  ...  286580.499048  204449.612520
            24755          Wessex      IMDM Eastleigh  ...  401141.953041   89970.873072

            [5 rows x 22 columns]

            >>> incid_loc = metex.view_metex_schedule8_incident_locations(
            ...     start_and_end_elr='diff', update=True, verbose=True)
            Updating "s8incident-locations-diffELR.pickle" at "data\\metex\\...\\views" ... Done.
            >>> incid_loc.tail()
                                  Route  ...    EndNorthing
            14978  London North Western  ...  420439.873529
            15648              Scotland  ...  805829.397554
            20581        North and East  ...  329827.328279
            21078            South East  ...  145221.750575
            21798               Western  ...  191610.009471

            [5 rows x 22 columns]

            >>> incid_loc = metex.view_metex_schedule8_incident_locations(
            ...     route_name='Anglia', update=True, verbose=True)
            Updating "s8incident-locations-Anglia.pickle" at "data\\metex\\...\\views" ... Done.
            >>> incid_loc.tail()
                   Route            IMDM  ...     EndEasting    EndNorthing
            1156  Anglia    IMDM Ipswich  ...  623193.328787  232121.661769
            1198  Anglia    IMDM Romford  ...  554085.492596  187812.105692
            1240  Anglia    IMDM Romford  ...  561234.911886  193972.676880
            1290  Anglia  IMDM Tottenham  ...  549317.204323  220825.113895
            1462  Anglia  IMDM Tottenham  ...  532432.407281  184945.970447

            [5 rows x 22 columns]
        """

        assert start_and_end_elr in (None, 'same', 'diff')

        filename = "s8incident-locations"

        start_and_end_elr_ = start_and_end_elr + 'ELR' if start_and_end_elr else start_and_end_elr
        pickle_filename = make_filename(filename, route_name, weather_category, start_and_end_elr_)
        path_to_pickle = self.cdd_views(pickle_filename)

        try:
            if os.path.isfile(path_to_pickle) and not update:
                incident_locations = load_pickle(path_to_pickle)

            else:
                # All incident locations
                s8costs_by_location = self.view_schedule8_costs_by_location(
                    route_name=route_name, weather_category=weather_category, update=update)

                incident_locations = s8costs_by_location.loc[:, 'Route':'EndLatitude'].drop_duplicates()

                # # Remove records where information of either 'StartELR' or 'EndELR' was missing
                # incident_locations = incident_locations[
                #     ~(incident_locations.StartELR.str.contains('^$')) & ~(
                #         incident_locations.EndELR.str.contains('^$'))]

                # 'FJH'
                idx = (incident_locations.StartLocation == 'Halton Junction') & (
                        incident_locations.StartELR == 'FJH')
                incident_locations.loc[idx, 'StartMileage'] = '0.0000'
                idx = (incident_locations.EndLocation == 'Halton Junction') & (
                        incident_locations.EndELR == 'FJH')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'BNE'
                idx = (incident_locations.StartLocation == 'Benton North Junction') & (
                        incident_locations.StartELR == 'BNE')
                incident_locations.loc[idx, 'StartMileage'] = '0.0000'
                idx = (incident_locations.EndLocation == 'Benton North Junction') & (
                        incident_locations.EndELR == 'BNE')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'WCI'
                idx = (incident_locations.StartLocation == 'Grangetown (Cleveland)') & \
                      (incident_locations.StartELR == 'WCI')
                incident_locations.loc[
                    idx, ('StartELR', 'StartMileage')] = 'DSN2', mile_chain_to_mileage('1.38')
                idx = (incident_locations.EndLocation == 'Grangetown (Cleveland)') & (
                        incident_locations.EndELR == 'WCI')
                incident_locations.loc[
                    idx, ('EndELR', 'EndMileage')] = 'DSN2', mile_chain_to_mileage('1.38')
                # 'SJD'
                idx = (incident_locations.EndLocation == 'Skelton Junction [Manchester]') & \
                      (incident_locations.EndELR == 'SJD')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'
                # 'HLK'
                idx = (incident_locations.EndLocation == 'High Level Bridge Junction') & \
                      (incident_locations.EndELR == 'HLK')
                incident_locations.loc[idx, 'EndMileage'] = '0.0000'

                # Create two additional columns about data of mileages (convert str to num)
                incident_locations[['StartMileage_num', 'EndMileage_num']] = \
                    incident_locations[['StartMileage', 'EndMileage']].applymap(mileage_str_to_num)

                incident_locations['StartEasting'], incident_locations['StartNorthing'] = \
                    wgs84_to_osgb36(incident_locations.StartLongitude.values,
                                    incident_locations.StartLatitude.values)
                incident_locations['EndEasting'], incident_locations['EndNorthing'] = \
                    wgs84_to_osgb36(incident_locations.EndLongitude.values,
                                    incident_locations.EndLatitude.values)

                save_pickle(incident_locations, path_to_pickle, verbose=verbose)

            if start_and_end_elr is not None:
                if start_and_end_elr == 'same':
                    # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
                    incident_locations = incident_locations[
                        incident_locations.StartELR == incident_locations.EndELR]
                elif start_and_end_elr == 'diff':
                    # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
                    incident_locations = incident_locations[
                        incident_locations.StartELR != incident_locations.EndELR]

            return incident_locations

        except Exception as e:
            print("Failed to fetch \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))

    def _update_view_pickles(self, update=True, pickle_it=True, verbose=True):
        """
        Update the local pickle files for all essential views.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``True``
        :type update: bool
        :param pickle_it: whether to save the queried data as a pickle file, defaults to ``True``
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from preprocessor import METExLite

            >>> metex = METExLite()

            >>> metex._update_view_pickles(update=True, pickle_it=True, verbose=True)
        """

        if confirmed("To update the View pickles of the NR_METEX data?"):

            _ = self.view_schedule8_costs_by_location(
                update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_location(
                route_name='Anglia', update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_location(
                route_name='Anglia', weather_category='Wind', update=update, pickle_it=pickle_it,
                verbose=verbose)
            _ = self.view_schedule8_costs_by_location(
                route_name='Anglia', weather_category='Heat', update=update, pickle_it=pickle_it,
                verbose=verbose)

            _ = self.view_schedule8_costs_by_datetime_location(
                update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location(
                route_name='Anglia', update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location(
                route_name='Anglia', weather_category='Wind', update=update, pickle_it=pickle_it,
                verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location(
                route_name='Anglia', weather_category='Heat', update=update, pickle_it=pickle_it,
                verbose=verbose)

            _ = self.view_schedule8_costs_by_datetime_location_reason(
                update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason(
                route_name='Anglia', update=update, pickle_it=pickle_it, verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason(
                route_name='Anglia', weather_category='Wind', update=update, pickle_it=pickle_it,
                verbose=verbose)
            _ = self.view_schedule8_costs_by_datetime_location_reason(
                route_name='Anglia', weather_category='Heat', update=update, pickle_it=pickle_it,
                verbose=verbose)

            _ = self.view_metex_schedule8_incident_locations(update=update, verbose=verbose)

            if verbose:
                print("\nUpdate finished.")

    # (TBC)
    def view_schedule8_incident_location_tracks(self, shift_yards=220):
        incident_locations = self.view_metex_schedule8_incident_locations()

        track = self.get_track()

        # track_summary = self.get_track_summary()

        # Testing e.g.
        elr = incident_locations.StartELR.loc[24133]
        # start_location = shapely.geometry.Point(
        #     incident_locations[['StartEasting', 'StartNorthing']].iloc[0].values)
        # end_location = shapely.geometry.Point(
        #     incident_locations[['EndEasting', 'EndNorthing']].iloc[0].values)
        start_mileage_num = incident_locations.StartMileage_num.loc[24133]
        # start_yard = nr_mileage_to_yards(start_mileage_num)
        end_mileage_num = incident_locations.EndMileage_num.loc[24133]
        # end_yard = nr_mileage_to_yards(end_mileage_num)

        #
        track_elr_mileages = track[track.ELR == elr]

        # Call OSM railway data
        from pydriosm import GeofabrikReader

        geofabrik_reader = GeofabrikReader()

        # Download/Read GB OSM data
        geofabrik_reader.read_shp_zip('Great Britain', layer_names='railways', feature_names='rail',
                                      data_dir=cdd_network("osm"), pickle_it=True, rm_extracts=True,
                                      rm_shp_zip=True)

        # Import it into PostgreSQL

        # Write a query to get track coordinates available in OSM data

        if start_mileage_num <= end_mileage_num:

            if start_mileage_num == end_mileage_num:
                start_mileage_num = shift_mileage_by_yard(start_mileage_num, -shift_yards)
                end_mileage_num = shift_mileage_by_yard(end_mileage_num, shift_yards)

            # Get adjusted mileages of start and end locations
            incident_track = track_elr_mileages[
                (start_mileage_num >= track_elr_mileages.StartMileage_num) &
                (end_mileage_num <= track_elr_mileages.EndMileage_num)]

        else:
            incident_track = track_elr_mileages[
                (start_mileage_num <= track_elr_mileages.EndMileage_num) &
                (end_mileage_num >= track_elr_mileages.StartMileage_num)]

        return incident_track


class Schedule8IncidentReports:
    """
    Reports of Schedule 8 incidents.
    """

    #: Name of the data
    NAME = 'Schedule 8 Incidents'

    FILENAME_1 = "Schedule8WeatherIncidents"
    FILENAME_2 = "Schedule8WeatherIncidents-02062006-31032014"

    def __init__(self):

        self.DataDir = os.path.relpath(cdd_metex("reports"))

        self.METExLite = METExLite()
        self.DAG = DelayAttributionGlossary()
        self.LocationID = LocationIdentifiers()
        self.StationCode = Stations()

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "data\\metex\\reports" and subdirectories / a file.

        :param sub_dir: subdirectory name(s) or filename(s)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: path to "data\\metex\\reports" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports
            >>> import os

            >>> sir = Schedule8IncidentReports()

            >>> dat_dir = sir.cdd()

            >>> os.path.relpath(dat_dir)
            'data\\metex\\reports'
        """

        path = cd(self.DataDir, *sub_dir, mkdir=mkdir)

        return path

    # == Location metadata =======================================================================

    @staticmethod
    def _pre_cleanse_location_metadata(metadata):
        """
        Pre-cleanse location metadata spreadsheet.

        :param metadata: raw data frame of location metadata
        :type metadata: pandas.DataFrame
        :return: cleansed data frame of location metadata
        :rtype: pandas.DataFrame
        """

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

    def _cleanse_location_metadata_tiploc_sheet(self, metadata, update_dict=False):
        """
        Cleanse 'TIPLOC_LocationsLyr' sheet.

        :param metadata: data frame of the 'TIPLOC_LocationsLyr' sheet in location metadata
        :type metadata: pandas.DataFrame
        :param update_dict: whether to update the location codes dictionary
        :type update_dict: bool
        :return: cleansed data frame of the 'TIPLOC_LocationsLyr' sheet in location metadata
        :rtype: pandas.DataFrame
        """

        meta_dat = self._pre_cleanse_location_metadata(metadata)

        meta_dat.TIPLOC = meta_dat.TIPLOC.fillna('').str.upper()

        ref_cols = ['STANOX', 'STANME', 'TIPLOC']
        dat = meta_dat[ref_cols + ['LOOKUP_NAME', 'DEG_LONG', 'DEG_LAT', 'LOOKUP_NAME_Raw']]

        # Rectify errors in STANOX; in errata_tiploc, {'CLAPS47': 'CLPHS47'} may be problematic.
        errata = load_json(cdd_railway_codes("metex-errata.json"))
        # noinspection PyTypeChecker
        errata_stanox, errata_tiploc, errata_stanme = [{k: v} for k, v in errata.items()]
        dat = dat.replace(errata_stanox)
        dat = dat.replace(errata_stanme)
        dat = dat.replace(errata_tiploc)

        # Rectify known issues for the location names in the data set
        dat = dat.replace(fetch_loc_names_repl_dict('LOOKUP_NAME'))
        dat = dat.replace(fetch_loc_names_repl_dict('LOOKUP_NAME', regex=True))

        # Fill in missing location names
        na_name = dat[dat.LOOKUP_NAME.isnull()]
        ref_dict = self.LocationID.make_xref_dict(ref_cols, update=update_dict)
        temp = na_name.join(ref_dict, on=ref_cols)
        temp = temp[['TIPLOC', 'Location']]
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.apply(
            lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
            if isinstance(x[1], list) else x[1],
            axis=1)

        # Rectify 'LOOKUP_NAME' according to 'TIPLOC'
        na_name = dat[dat.LOOKUP_NAME.isnull()]
        ref_dict = self.LocationID.make_xref_dict('TIPLOC', update=update_dict)
        temp = na_name.join(ref_dict, on='TIPLOC')
        dat.loc[na_name.index, 'LOOKUP_NAME'] = temp.Location.values

        not_na_name = dat[dat.LOOKUP_NAME.notnull()]
        temp = not_na_name.join(ref_dict, on='TIPLOC')

        def extract_one(lookup_name, ref_loc):
            if isinstance(ref_loc, list):
                n = find_similar_str(lookup_name.replace(' ', ''), ref_loc, processor='fuzzywuzzy')
            elif pd.isnull(ref_loc):
                n = lookup_name
            else:
                n = ref_loc
            return n

        dat.loc[not_na_name.index, 'LOOKUP_NAME'] = temp.apply(
            lambda x: extract_one(x[3], x[7]), axis=1)

        # Rectify 'STANOX'+'STANME'
        location_codes = self.LocationID.fetch_codes()[self.LocationID.KEY]
        location_codes = location_codes.drop_duplicates(['TIPLOC', 'Location']).set_index(
            ['TIPLOC', 'Location'])
        temp = dat.join(location_codes, on=['TIPLOC', 'LOOKUP_NAME'], rsuffix='_Ref').fillna('')
        dat.loc[temp.index, 'STANOX':'STANME'] = temp[['STANOX_Ref', 'STANME_Ref']].values

        # Update coordinates with reference data from RailwayCodes
        station_data = self.StationCode.fetch_station_data()[self.StationCode.StnKey]
        station_data = station_data[['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
        station_data = station_data.drop_duplicates(subset=['Station']).set_index('Station')
        temp = dat.join(station_data, on='LOOKUP_NAME')
        na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
        dat.loc[na_i, 'DEG_LONG':'DEG_LAT'] = temp.loc[
            na_i, ['Degrees Longitude', 'Degrees Latitude']].values

        # Finalising...
        meta_dat.update(dat)
        meta_dat.dropna(subset=['LOOKUP_NAME'] + ref_cols, inplace=True)
        meta_dat.fillna('', inplace=True)
        meta_dat.sort_values(['LOOKUP_NAME', 'SHAPE_LENG'], ascending=[True, False], inplace=True)
        meta_dat.index = range(len(meta_dat))

        return meta_dat

    def get_location_metadata(self, update=False, verbose=False):
        """
        Get data of location STANOX.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of location STANOX
        :rtype: dict

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> # loc_metadata = sir.get_location_metadata(update=True, verbose=True)
            >>> # Updating "stanox-locations-data.pickle" at "data\\network\\railway codes" ... Done.
            >>> loc_metadata = sir.get_location_metadata()

            >>> type(loc_metadata)
            dict
            >>> list(loc_metadata.keys())
            ['TIPLOC_LocationsLyr', 'STANOX']

            >>> loc_metadata['TIPLOC_LocationsLyr'].tail()

            >>> loc_metadata['STANOX'].tail()
        """

        pickle_filename = "stanox-locations-data.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            location_metadata = load_pickle(path_to_pickle)

        else:
            try:
                path_to_xlsx = path_to_pickle.replace(".pickle", ".xlsx")

                workbook = pd.ExcelFile(path_to_xlsx)

                # 'TIPLOC_LocationsLyr' -----------------------------------------
                raw = workbook.parse(
                    sheet_name='TIPLOC_LocationsLyr', parse_dates=['LASTEDITED', 'LAST_UPD_1'],
                    converters={'STANOX': str})

                cols_with_na = ['STATUS', 'LINE_DESCRIPTION', 'QC_STATUS', 'DESCRIPTIO', 'BusinessRef']
                raw.fillna({x: '' for x in cols_with_na}, inplace=True)

                tiploc_loc_lyr = self._cleanse_location_metadata_tiploc_sheet(raw, update_dict=update)

                # 'STANOX' -----------------------------------------------------------------------------
                stanox = workbook.parse(sheet_name='STANOX', converters={'STANOX': str, 'gridref': str})

                stanox = self._pre_cleanse_location_metadata(stanox)
                stanox.fillna({'LOOKUP_NAME_Raw': ''}, inplace=True)

                ref_cols = ['SHAPE_LENG', 'EASTING', 'NORTHING', 'GRIDREF']
                ref_data = tiploc_loc_lyr.set_index(ref_cols)
                stanox.drop_duplicates(ref_cols, inplace=True)
                temp = stanox.join(ref_data, on=ref_cols, rsuffix='_Ref').drop_duplicates(ref_cols)

                ref_cols_ok = [c for c in temp.columns if '_Ref' in c]
                stanox.loc[:, [c.replace('_Ref', '') for c in ref_cols_ok]] = temp[ref_cols_ok].values

                # Collect the above two dataframes and store them in a dictionary ---------------------
                location_metadata = dict(zip(workbook.sheet_names, [tiploc_loc_lyr, stanox]))

                workbook.close()

                save_pickle(location_metadata, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to fetch \"STANOX locations data\" from {}. {}".format(
                    os.path.dirname(path_to_pickle), e))
                location_metadata = None

        return location_metadata

    # == Location metadata from NR_METEX database ================================================

    def get_metex_location(self, update=False, verbose=False):
        """
        Get location data with LocationId and WeatherCell from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: location data
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> metex_loc_dat = sir.get_metex_location(update=True, verbose=True)
            Updating "metex-location.pickle" at "data\\network\\railway codes" ... Done.
            >>> metex_loc_dat = sir.get_metex_location()
            >>> metex_loc_dat.tail()
        """

        pickle_filename = "metex-location.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_location = load_pickle(path_to_pickle)

        else:
            try:
                metex_location = self.METExLite.read_table('Location')

                metex_location.rename(columns={'Id': 'LocationId'}, inplace=True)
                metex_location.set_index('LocationId', inplace=True)
                metex_location.WeatherCell = metex_location.WeatherCell.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))
                # # Correct a known error
                # metex_location.loc['610096', 'StartLongitude':'EndLatitude'] = \
                #     [-0.0751, 51.5461, -0.0751, 51.5461]

                save_pickle(metex_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX location data. {}".format(e))
                metex_location = None

        return metex_location

    def get_metex_stanox_location(self, update=False, verbose=False):
        """
        Get data of STANOX location from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of STANOX location
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> metex_stanox_loc_dat = sir.get_metex_stanox_location(update=True, verbose=True)
            Updating "metex-stanox-location.pickle" at "data\\network\\railway codes" ... Done.
            >>> metex_stanox_loc_dat = sir.get_metex_stanox_location()
            >>> metex_stanox_loc_dat.tail()
        """

        pickle_filename = "metex-stanox-location.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_stanox_location = load_pickle(path_to_pickle)

        else:
            try:
                metex_stanox_location = self.METExLite.read_table('StanoxLocation')

                # Cleanse "stanox_location"
                errata = load_json(cdd_railway_codes("metex-errata.json"))
                # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                errata_stanox, errata_tiploc, errata_stanme = errata.values()
                # Note that in errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                metex_stanox_location = metex_stanox_location.replace({'Stanox': errata_stanox})
                metex_stanox_location = metex_stanox_location.replace({'Description': errata_tiploc})
                metex_stanox_location = metex_stanox_location.replace({'Name': errata_stanme})

                # Get reference data from the Railway Codes website
                rc_codes = self.LocationID.fetch_codes()[self.LocationID.KEY]
                rc_codes = rc_codes[['Location', 'TIPLOC', 'STANME', 'STANOX']].drop_duplicates()

                # Fill in NA 'Description's (i.e. Location names)
                na_desc = metex_stanox_location[metex_stanox_location.Description.isnull()]
                temp = na_desc.join(rc_codes.set_index('STANOX'), on='Stanox')
                metex_stanox_location.loc[na_desc.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values

                # Fill in NA 'Name's (i.e. STANME)
                na_name = metex_stanox_location[metex_stanox_location.Name.isnull()]
                # Some records of the 'Description' are recorded by 'TIPLOC' instead
                rc_tiploc_dict = self.LocationID.make_xref_dict('TIPLOC')
                temp = na_name.join(rc_tiploc_dict, on='Description')
                temp = temp.join(rc_codes.set_index(['STANOX', 'TIPLOC', 'Location']),
                                 on=['Stanox', 'Description', 'Location'])
                temp = temp[temp.Location.notnull() & temp.STANME.notnull()]
                metex_stanox_location.loc[temp.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values
                # Still, there are some NA 'Name's remaining
                # due to incorrect spelling of 'TIPLOC' ('Description')
                na_name = metex_stanox_location[metex_stanox_location.Name.isnull()]
                temp = na_name.join(rc_codes.set_index('STANOX'), on='Stanox')
                metex_stanox_location.loc[temp.index, 'Description':'Name'] = temp[
                    ['Location', 'STANME']].values

                # Apply manually-created dictionaries
                loc_name_replacement_dict = fetch_loc_names_repl_dict('Description')
                metex_stanox_location = metex_stanox_location.replace(loc_name_replacement_dict)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('Description', regex=True)
                metex_stanox_location = metex_stanox_location.replace(loc_name_regexp_replacement_dict)

                # Check if 'Description' has STANOX code
                # instead of location name using STANOX-dictionary
                rc_stanox_dict = self.LocationID.make_xref_dict('STANOX')
                temp = metex_stanox_location.join(rc_stanox_dict, on='Description')
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2], processor='fuzzywuzzy')
                        if isinstance(x[2], list) else x[2],
                        axis=1)

                # Check if 'Description' has TIPLOC code
                # instead of location name using STANOX-TIPLOC-dictionary
                rc_stanox_tiploc_dict = self.LocationID.make_xref_dict(['STANOX', 'TIPLOC'])
                temp = metex_stanox_location.join(rc_stanox_tiploc_dict, on=['Stanox', 'Description'])
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2], processor='fuzzywuzzy')
                        if isinstance(x[2], list) else x[2],
                        axis=1)

                # Check if 'Description' has STANME code instead of location name
                # using STANOX-STANME-dictionary

                rc_stanox_stanme_dict = self.LocationID.make_xref_dict(['STANOX', 'STANME'])
                temp = metex_stanox_location.join(rc_stanox_stanme_dict, on=['Stanox', 'Description'])
                valid_loc = temp[temp.Location.notnull()][['Description', 'Name', 'Location']]
                if not valid_loc.empty:
                    metex_stanox_location.loc[valid_loc.index, 'Description'] = valid_loc.apply(
                        lambda x: find_similar_str(x[1], x[2], processor='fuzzywuzzy')
                        if isinstance(x[2], list) else x[2],
                        axis=1)

                # Finalize cleansing 'Description' (i.e. location names)
                temp = metex_stanox_location.join(rc_stanox_dict, on='Stanox')
                temp = temp[['Description', 'Name', 'Location']]
                metex_stanox_location.Description = temp.apply(
                    lambda x: find_similar_str(x[0], x[2], processor='fuzzywuzzy')
                    if isinstance(x[2], list) else x[2], axis=1)

                # Cleanse 'Name' (i.e. STANME)
                metex_stanox_location.Name = metex_stanox_location.Name.str.upper()

                loc_stanme_dict = rc_codes.groupby(['STANOX', 'Location']).agg({'STANME': list})
                loc_stanme_dict.STANME = loc_stanme_dict.STANME.map(
                    lambda x: x[0] if len(x) == 1 else x)
                temp = metex_stanox_location.join(loc_stanme_dict, on=['Stanox', 'Description'])
                # noinspection PyTypeChecker
                metex_stanox_location.Name = temp.apply(
                    lambda x: find_similar_str(x[2], x[6], processor='fuzzywuzzy')
                    if isinstance(x[6], list) else x[6], axis=1)

                # Below is not available in any reference data
                metex_stanox_location.loc[298, 'Stanox':'Name'] = [
                    '03330', 'Inverkeithing PPM Point', '']

                # Cleanse remaining NA values in 'ELR', 'Yards' and 'LocationId'
                metex_stanox_location.ELR.fillna('', inplace=True)
                metex_stanox_location.Yards = metex_stanox_location.Yards.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                # Clear duplicates by 'STANOX'
                temp = metex_stanox_location[
                    metex_stanox_location.duplicated(subset=['Stanox'], keep=False)]
                # temp.sort_values(['Description', 'LocationId'], ascending=[True, False], inplace=True)
                metex_stanox_location.drop(index=temp[temp.LocationId.isna()].index, inplace=True)

                metex_stanox_location.LocationId = metex_stanox_location.LocationId.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                # Finish
                metex_stanox_location.index = range(len(metex_stanox_location))

                save_pickle(metex_stanox_location, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX STANOX location data. {}.".format(e))
                metex_stanox_location = None

        return metex_stanox_location

    def get_metex_stanox_section(self, update=False, verbose=False):
        """
        Get data of STANOX section from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of STANOX location
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> metex_stanox_sec = sir.get_metex_stanox_section(update=True, verbose=True)
            Updating "metex-stanox-section.pickle" at "data\\network\\railway codes" ... Done.
            >>> metex_stanox_sec = sir.get_metex_stanox_section()
            >>> metex_stanox_sec.tail()
        """

        pickle_filename = "metex-stanox-section.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            metex_stanox_section = load_pickle(path_to_pickle)

        else:
            try:
                metex_stanox_section = self.METExLite.read_table('StanoxSection')

                # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
                errata = load_json(cdd_railway_codes("metex-errata.json"))
                errata_stanox = errata['STANOX']

                metex_stanox_section.rename(
                    columns={'Id': 'StanoxSectionId', 'Description': 'StanoxSection'}, inplace=True)
                metex_stanox_section = metex_stanox_section.replace({'StartStanox': errata_stanox})
                metex_stanox_section = metex_stanox_section.replace({'EndStanox': errata_stanox})
                metex_stanox_section.LocationId = metex_stanox_section.LocationId.map(
                    lambda x: '' if pd.isna(x) else str(int(x)))

                stanox_sec = metex_stanox_section.copy(deep=True)

                stanox_sec[['Start_Raw', 'End_Raw']] = stanox_sec.StanoxSection.str.split(' : ').apply(
                    pd.Series)
                stanox_sec[['Start', 'End']] = stanox_sec[['Start_Raw', 'End_Raw']]
                # Solve duplicated STANOX
                unknown_stanox_loc = load_json(cdd_railway_codes("problematic-stanox-locations.json"))
                stanox_sec = stanox_sec.replace({'Start': unknown_stanox_loc})
                stanox_sec = stanox_sec.replace({'End': unknown_stanox_loc})

                #
                stanox_location = self.get_metex_stanox_location(update)
                stanox_loc = stanox_location.set_index(['Stanox', 'LocationId'])
                temp = stanox_sec.join(stanox_loc, on=['StartStanox', 'LocationId'])
                temp = temp[temp.Description.notnull()]
                stanox_sec.loc[temp.index, 'Start'] = temp.Description
                temp = stanox_sec.join(stanox_loc, on=['EndStanox', 'LocationId'])
                temp = temp[temp.Description.notnull()]
                stanox_sec.loc[temp.index, 'End'] = temp.Description

                # Check if 'Start' and 'End' have STANOX codes
                # instead of location names using STANOX-dictionary
                rc_stanox_dict = self.LocationID.make_xref_dict('STANOX')
                temp = stanox_sec.join(rc_stanox_dict, on='Start')
                valid_loc = temp[temp.Location.notnull()]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.Location
                temp = stanox_sec.join(rc_stanox_dict, on='End')
                valid_loc = temp[temp.Location.notnull()]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.Location

                # Check if 'Start' and 'End' have STANOX/TIPLOC codes using STANOX-TIPLOC-dictionary
                rc_stanox_tiploc_dict = self.LocationID.make_xref_dict(['STANOX', 'TIPLOC'])
                temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['StartStanox', 'Start'])
                valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                        if isinstance(x[1], (list, tuple)) else
                        x[1], axis=1)
                temp = stanox_sec.join(rc_stanox_tiploc_dict, on=['EndStanox', 'End'])
                valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                        if isinstance(x[1], (list, tuple)) else x[1],
                        axis=1)

                # Check if 'Start' and 'End' have STANOX/STANME codes using STANOX-STANME-dictionary
                rc_stanox_stanme_dict = self.LocationID.make_xref_dict(['STANOX', 'STANME'])
                temp = stanox_sec.join(rc_stanox_stanme_dict, on=['StartStanox', 'Start'])
                valid_loc = temp[temp.Location.notnull()][['Start', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'Start'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                        if isinstance(x[1], (list, tuple)) else x[1],
                        axis=1)
                temp = stanox_sec.join(rc_stanox_stanme_dict, on=['EndStanox', 'End'])
                valid_loc = temp[temp.Location.notnull()][['End', 'Location']]
                if not valid_loc.empty:
                    stanox_sec.loc[valid_loc.index, 'End'] = valid_loc.apply(
                        lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                        if isinstance(x[1], (list, tuple)) else x[1],
                        axis=1)

                # Apply manually-created dictionaries
                loc_name_replacement_dict = fetch_loc_names_repl_dict('Start')
                stanox_sec = stanox_sec.replace(loc_name_replacement_dict)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('Start', regex=True)
                stanox_sec = stanox_sec.replace(loc_name_regexp_replacement_dict)
                loc_name_replacement_dict = fetch_loc_names_repl_dict('End')
                stanox_sec = stanox_sec.replace(loc_name_replacement_dict)
                loc_name_regexp_replacement_dict = fetch_loc_names_repl_dict('End', regex=True)
                stanox_sec = stanox_sec.replace(loc_name_regexp_replacement_dict)

                # Finalize cleansing
                stanox_sec.End.fillna(stanox_sec.Start, inplace=True)
                temp = stanox_sec.join(rc_stanox_dict, on='StartStanox')
                temp = temp[temp.Location.notnull()][['Start', 'Location']]
                stanox_sec.loc[temp.index, 'Start'] = temp.apply(
                    lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                    if isinstance(x[1], (list, tuple)) else x[1],
                    axis=1)
                temp = stanox_sec.join(rc_stanox_dict, on='EndStanox')
                temp = temp[temp.Location.notnull()][['End', 'Location']]
                stanox_sec.loc[temp.index, 'End'] = temp.apply(
                    lambda x: find_similar_str(x[0], x[1], processor='fuzzywuzzy')
                    if isinstance(x[1], (list, tuple)) else x[1],
                    axis=1)

                #
                section = stanox_sec.Start + ' : ' + stanox_sec.End
                non_sec = stanox_sec.Start.eq(stanox_sec.End)
                section[non_sec] = stanox_sec[non_sec].Start
                metex_stanox_section.StanoxSection = section
                metex_stanox_section.insert(2, 'StartLocation', stanox_sec.Start.values)
                metex_stanox_section.insert(3, 'EndLocation', stanox_sec.End.values)
                metex_stanox_section[['StartLocation', 'EndLocation']] = stanox_sec[['Start', 'End']]

                # Add raw data to the original dataframe
                raw_col_names = ['StanoxSection_Raw', 'StartLocation_Raw', 'EndLocation_Raw']
                metex_stanox_section[raw_col_names] = stanox_sec[
                    ['StanoxSection', 'Start_Raw', 'End_Raw']]

                save_pickle(metex_stanox_section, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get NR_METEX STANOX section data. {}.".format(e))
                metex_stanox_section = None

        return metex_stanox_section

    def get_location_metadata_plus(self, update=False, verbose=False):
        """
        Get data of location codes by assembling resources from NR_METEX_* database.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of location codes
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> loc_meta_dat_plus = sir.get_location_metadata_plus(update=True, verbose=True)
            Updating "metex-location.pickle" at "data\\network\\railway codes\\" ... Done.
            Updating "metex-stanox-location.pickle" at "data\\network\\railway codes\\" ... Done.
            Updating "metex-stanox-section.pickle" at "data\\network\railway codes\\" ... Done.
            Updating "stanox-locations-data.pickle" at "data\\network\railway codes\\" ... Done.
            Updating "metex-location-metadata.pickle" at "data\\network\\railway codes\\" ... Done.
            >>> loc_meta_dat_plus = sir.get_location_metadata_plus()
            >>> loc_meta_plus.tail()
        """

        pickle_filename = "metex-location-metadata.pickle"
        path_to_pickle = cdd_railway_codes(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            loc_meta_plus = load_pickle(path_to_pickle)
        else:
            try:
                # Location
                location = self.get_metex_location(update=update, verbose=verbose)

                # # STANOX location
                stanox_location = self.get_metex_stanox_location(update=update, verbose=verbose)
                stanox_location = stanox_location[['Stanox', 'Description', 'Name', 'ELR', 'Yards']]
                stanox_location.drop([2964, 6049, 2258, 1068, 2361, 4390], inplace=True)
                stanox_location.set_index(['Stanox', 'Description'], inplace=True)

                # STANOX section
                stanox_section = self.get_metex_stanox_section(update=update, verbose=verbose)

                # Get metadata by 'LocationId'
                location.index = location.index.astype(str)
                loc_meta_plus = stanox_section.join(location, on='LocationId')

                # --------------------------------------------------------------------------
                ref = self.get_location_metadata(update=update, verbose=verbose)['STANOX']
                ref.drop_duplicates(subset=['STANOX', 'LOOKUP_NAME'], inplace=True)
                ref.set_index(['STANOX', 'LOOKUP_NAME'], inplace=True)

                # Replace metadata coordinates data with ref coordinates if available
                temp = loc_meta_plus.join(ref, on=['StartStanox', 'StartLocation'])
                temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
                loc_meta_plus.loc[temp.index, 'StartLongitude':'StartLatitude'] = temp[
                    ['DEG_LONG', 'DEG_LAT']].values
                loc_meta_plus.loc[temp.index, 'ApproximateStartLocation'] = False
                loc_meta_plus.loc[
                    loc_meta_plus.index.difference(temp.index), 'ApproximateStartLocation'] = True

                temp = loc_meta_plus.join(ref, on=['EndStanox', 'EndLocation'])
                temp = temp[temp.DEG_LAT.notnull() & temp.DEG_LONG.notnull()]
                loc_meta_plus.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[
                    ['DEG_LONG', 'DEG_LAT']].values
                loc_meta_plus.loc[temp.index, 'ApproximateEndLocation'] = False
                loc_meta_plus.loc[
                    loc_meta_plus.index.difference(temp.index), 'ApproximateEndLocation'] = True

                # Get reference metadata from RailwayCodes
                station_data = self.StationCode.fetch_station_data()[self.StationCode.StnKey]
                station_data = station_data[
                    ['Station', 'Degrees Longitude', 'Degrees Latitude']].dropna()
                station_data.drop_duplicates(subset=['Station'], inplace=True)
                station_data.set_index('Station', inplace=True)

                temp = loc_meta_plus.join(station_data, on='StartLocation')
                na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
                loc_meta_plus.loc[na_i, 'StartLongitude':'StartLatitude'] = \
                    temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
                loc_meta_plus.loc[na_i, 'ApproximateStartLocation'] = False
                loc_meta_plus.loc[~na_i, 'ApproximateStartLocation'] = True

                temp = loc_meta_plus.join(station_data, on='EndLocation')
                na_i = temp['Degrees Longitude'].notnull() & temp['Degrees Latitude'].notnull()
                loc_meta_plus.loc[na_i, 'EndLongitude':'EndLatitude'] = \
                    temp.loc[na_i, ['Degrees Longitude', 'Degrees Latitude']].values
                loc_meta_plus.loc[na_i, 'ApproximateEndLocation'] = False
                loc_meta_plus.loc[~na_i, 'ApproximateEndLocation'] = True

                loc_meta_plus.ApproximateLocation = \
                    loc_meta_plus.ApproximateStartLocation | loc_meta_plus.ApproximateEndLocation

                # Finalise
                location_cols = stanox_location.columns
                start_cols = ['Start' + x for x in location_cols]
                loc_meta_plus = loc_meta_plus.join(stanox_location.drop_duplicates('Name'),
                                                   on=['StartStanox', 'StartLocation'])
                loc_meta_plus.rename(columns=dict(zip(location_cols, start_cols)), inplace=True)
                end_cols = ['End' + x for x in location_cols]
                loc_meta_plus = loc_meta_plus.join(stanox_location.drop_duplicates('Name'),
                                                   on=['EndStanox', 'EndLocation'])
                loc_meta_plus.rename(columns=dict(zip(location_cols, end_cols)), inplace=True)
                loc_meta_plus[start_cols + end_cols] = loc_meta_plus[start_cols + end_cols].fillna('')

                save_pickle(loc_meta_plus, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get metadata of location codes. {}.".format(e))
                loc_meta_plus = None

        return loc_meta_plus

    # == Incidents data ==========================================================================

    def _cleanse_stanox_section_column(self, data, col_name='StanoxSection', sep=' : ',
                                       update_dict=False):
        """
        Cleanse the column that contains location information in the data of incident records.

        :param data: data of incident records
        :type data: pandas.DataFrame
        :param col_name: name of the column to be cleansed, defaults to ``'StanoxSection'``
        :type col_name: str
        :param sep: separator for the column, defaults to ``' : '``
        :type sep: str
        :param update_dict: whether to update the location codes dictionary
        :type update_dict: bool
        :return: cleansed data frame
        :rtype: pandas.DataFrame
        """

        dat = data.copy()
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

        # In errata_tiploc, {'CLAPS47': 'CLPHS47'} might be problematic.
        errata = load_json(cdd_railway_codes("metex-errata.json"))
        errata_stanox, errata_tiploc, errata_stanme = errata.values()
        start_end = start_end_raw.replace(errata_stanox)
        start_end = start_end.replace(errata_tiploc)
        start_end = start_end.replace(errata_stanme)

        # Get rid of duplicated records
        def _tidy_alt_codes(code_dict_df, raw_loc):
            raw_loc.columns = [x.replace('_Raw', '') for x in raw_loc.columns]
            tmp = raw_loc.join(code_dict_df, on='StartLocation')
            tmp = tmp[tmp.Location.notnull()]
            raw_loc.loc[tmp.index, 'StartLocation'] = tmp.Location.values
            tmp = raw_loc.join(code_dict_df, on='EndLocation')
            tmp = tmp[tmp.Location.notnull()]
            raw_loc.loc[tmp.index, 'EndLocation'] = tmp.Location.values
            return raw_loc

        #
        stanox_dict = self.LocationID.make_xref_dict('STANOX', update=update_dict)
        start_end = _tidy_alt_codes(stanox_dict, start_end)
        #
        stanme_dict = self.LocationID.make_xref_dict('STANME', update=update_dict)
        start_end = _tidy_alt_codes(stanme_dict, start_end)
        #
        tiploc_dict = self.LocationID.make_xref_dict('TIPLOC', update=update_dict)
        start_end = _tidy_alt_codes(tiploc_dict, start_end)

        #
        start_end = start_end.replace(fetch_loc_names_repl_dict('Description', regex=True))
        start_end = start_end.replace(fetch_loc_names_repl_dict(old_column_name, regex=True))
        start_end = start_end.replace(fetch_loc_names_repl_dict(regex=True), regex=True)
        start_end = start_end.replace(fetch_loc_names_repl_dict())

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
        single_loc_idx = start_end.StartLocation[start_end.StartLocation == start_end.EndLocation].index
        dat.loc[single_loc_idx, col_name] = start_end.loc[single_loc_idx, 'StartLocation']

        # Resort column order
        col_names = [
            list(v)
            for k, v in itertools.groupby(list(dat.columns), lambda x: x == old_column_name) if not k]
        add_names = [old_column_name] + col_names[1][-3:] + ['StartLocation', 'EndLocation']
        col_names = col_names[0] + add_names + col_names[1][:-3]

        cleansed_data = dat.join(start_end)[col_names]

        return cleansed_data

    def _cleanse_geographical_coordinates(self, data, update_metadata=False):
        """
        Look up geographical coordinates for each incident location.

        :param data: data of incident records
        :type data: pandas.DataFrame
        :param update_metadata: whether to update the location metadata, defaults to ``False``
        :type update_metadata: bool
        :return: cleansed data frame
        :rtype: pandas.DataFrame
        """

        dat = data.copy()

        # Find geographical coordinates for each incident location
        ref_loc_dat_1, _ = self.get_location_metadata(update=update_metadata).values()
        coordinates_cols = ['EASTING', 'NORTHING', 'DEG_LONG', 'DEG_LAT']
        coordinates_cols_alt = ['Easting', 'Northing', 'Longitude', 'Latitude']
        ref_loc_dat_1.rename(columns=dict(zip(coordinates_cols, coordinates_cols_alt)), inplace=True)
        ref_loc_dat_1 = \
            ref_loc_dat_1[['LOOKUP_NAME'] + coordinates_cols_alt].drop_duplicates('LOOKUP_NAME')

        tmp, idx = [], []
        for i in ref_loc_dat_1.index:
            x_ = ref_loc_dat_1.LOOKUP_NAME.loc[i]
            if isinstance(x_, tuple):
                for y in x_:
                    tmp.append([y] + ref_loc_dat_1[coordinates_cols_alt].loc[i].to_list())
        ref_colname_1 = ['LOOKUP_NAME'] + coordinates_cols_alt
        ref_loc_dat_1 = ref_loc_dat_1[ref_colname_1].append(pd.DataFrame(tmp, columns=ref_colname_1))
        ref_loc_dat_1.drop_duplicates(subset=coordinates_cols_alt, keep='last', inplace=True)
        ref_loc_dat_1.set_index('LOOKUP_NAME', inplace=True)

        dat = dat.join(ref_loc_dat_1[coordinates_cols_alt], on='StartLocation')
        dat.rename(
            columns=dict(zip(coordinates_cols_alt, ['Start' + c for c in coordinates_cols_alt])),
            inplace=True)
        dat = dat.join(ref_loc_dat_1[coordinates_cols_alt], on='EndLocation')
        dat.rename(
            columns=dict(zip(coordinates_cols_alt, ['End' + c for c in coordinates_cols_alt])),
            inplace=True)

        # Get location metadata for reference --------------------------------
        location_metadata = self.get_location_metadata_plus(update=update_metadata)
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
        dat.loc[temp.index, 'StartEasting'], dat.loc[temp.index, 'StartNorthing'] = wgs84_to_osgb36(
            temp.Longitude.values, temp.Latitude.values)

        temp = dat[dat.EndEasting.isnull() | dat.EndLongitude.isnull()]
        temp = temp.join(loc_metadata, on='EndLocation')
        dat.loc[temp.index, 'EndLongitude':'EndLatitude'] = temp[['Longitude', 'Latitude']].values

        # Dalston Junction (East London Line)     --> Dalston Junction [-0.0751, 51.5461]
        # Ashford West Junction (CTRL)            --> Ashford West Junction [0.86601557, 51.146927]
        # Southfleet Junction                     --> ? [0.34262910, 51.419354]
        # Channel Tunnel Eurotunnel Boundary CTRL --> ? [1.1310482, 51.094808]
        na_loc = ['Dalston Junction (East London Line)', 'Ashford West Junction (CTRL)',
                  'Southfleet Junction', 'Channel Tunnel Eurotunnel Boundary CTRL']
        na_loc_longlat = [[-0.0751, 51.5461], [0.86601557, 51.146927], [0.34262910, 51.419354],
                          [1.1310482, 51.094808]]
        for x_, longlat in zip(na_loc, na_loc_longlat):
            if x_ in list(temp.EndLocation):
                idx = temp[temp.EndLocation == x_].index
                temp.loc[idx, 'EndLongitude':'Latitude'] = longlat * 2
                dat.loc[idx, 'EndLongitude':'EndLatitude'] = longlat

        dat.loc[temp.index, 'EndEasting'], dat.loc[temp.index, 'EndNorthing'] = wgs84_to_osgb36(
            temp.Longitude.values, temp.Latitude.values)

        # ref 2 ----------------------------------------------
        ref_metadata_2 = self.StationCode.fetch_station_data()[self.StationCode.StnKey]
        ref_metadata_2 = ref_metadata_2[['Station', 'Degrees Longitude', 'Degrees Latitude']]
        ref_metadata_2 = ref_metadata_2.dropna().drop_duplicates('Station')
        ref_metadata_2.columns = [x.replace('Degrees ', '') for x in ref_metadata_2.columns]
        ref_metadata_2.set_index('Station', inplace=True)

        temp = dat.join(ref_metadata_2, on='StartLocation')
        temp_start = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        dat.loc[temp_start.index, 'StartLongitude':'StartLatitude'] = temp_start[
            ['Longitude', 'Latitude']].values

        temp = dat.join(ref_metadata_2, on='EndLocation')
        temp_end = temp[temp.Longitude.notnull() & temp.Latitude.notnull()]
        dat.loc[temp_end.index, 'EndLongitude':'EndLatitude'] = temp_end[
            ['Longitude', 'Latitude']].values

        # Let (Longitude, Latitude) be almost equivalent to (Easting, Northing)
        dat.loc[:, 'StartLongitude':'StartLatitude'] = np.array(
            osgb36_to_wgs84(dat.StartEasting.values, dat.StartNorthing.values)).T
        dat.loc[:, 'EndLongitude':'EndLatitude'] = np.array(
            osgb36_to_wgs84(dat.EndEasting.values, dat.EndNorthing.values)).T

        # Convert coordinates to shapely.geometry.Point
        dat.loc[:, 'StartXY'] = [
            shapely.geometry.Point(x, y).wkt for x, y in dat[['StartEasting', 'StartNorthing']].values]
        dat.loc[:, 'EndXY'] = [
            shapely.geometry.Point(x, y).wkt for x, y in dat[['EndEasting', 'EndNorthing']].values]
        dat.loc[:, 'StartLongLat'] = [
            shapely.geometry.Point(lon, lat).wkt
            for lon, lat in dat[['StartLongitude', 'StartLatitude']].values]
        dat.loc[:, 'EndLongLat'] = [
            shapely.geometry.Point(lon, lat).wkt
            for lon, lat in dat[['EndLongitude', 'EndLatitude']].values]

        return dat

    def get_schedule8_weather_incidents(self, route_name=None, weather_category=None, update=False,
                                        verbose=False):
        """
        Get data of Schedule 8 weather incidents from spreadsheet file.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of Schedule 8 weather incidents
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> s8w_incid = sir.get_schedule8_weather_incidents(update=True, verbose=True)
            Updating "Schedule8WeatherIncidents.pickle" at "data\\metex\\reports\\" ... Done.
            >>> s8w_incid = sir.get_schedule8_weather_incidents()
            >>> s8w_incid.tail()
        """

        pickle_filename = make_filename(self.FILENAME_1, route_name, weather_category)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            s8weather_incidents = load_pickle(path_to_pickle)

        else:
            try:
                # Load data from the raw file
                raw_data = pd.read_excel(
                    path_to_pickle.replace(".pickle", ".xlsx"), parse_dates=['StartDate', 'EndDate'],
                    converters={'stanoxSection': str})

                s8weather_incidents = raw_data.rename(
                    columns={'StartDate': 'StartDateTime',
                             'EndDate': 'EndDateTime',
                             'stanoxSection': 'StanoxSection',
                             'imdm': 'IMDM',
                             'WeatherCategory': 'WeatherCategoryCode',
                             'WeatherCategory.1': 'WeatherCategory',
                             'Reason': 'IncidentReason',
                             # 'Minutes': 'DelayMinutes',
                             'Description': 'IncidentReasonDescription',
                             'Category': 'IncidentCategory',
                             'CategoryDescription': 'IncidentCategoryDescription'})

                # Add information about incident reason
                incident_reason_metadata = self.DAG.get_incident_reason()
                incident_reason_metadata.columns = [
                    c.replace('_', '') for c in incident_reason_metadata.columns]
                s8weather_incidents = s8weather_incidents.join(
                    incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
                s8weather_incidents.drop(
                    [x for x in s8weather_incidents.columns if '_meta' in x], axis=1, inplace=True)

                # Cleanse the location data
                s8weather_incidents = self._cleanse_stanox_section_column(
                    s8weather_incidents, col_name='StanoxSection', sep=' : ', update_dict=update)

                # Look up geographical coordinates for each incident location
                s8weather_incidents = self._cleanse_geographical_coordinates(s8weather_incidents)

                # Retain data for specific Route and Weather category
                s8weather_incidents = get_subset(
                    s8weather_incidents, route_name, weather_category, rearrange_index=False)

                save_pickle(s8weather_incidents, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"Schedule 8 Weather Incidents\". {}".format(e))
                s8weather_incidents = None

        return s8weather_incidents

    def get_schedule8_weather_incidents_02062006_31032014(self, route_name=None, weather_category=None,
                                                          update=False, verbose=False):
        """
        Get data of Schedule 8 weather incident from Schedule8WeatherIncidents-02062006-31032014.xlsm.

        :param route_name: name of a Route, defaults to ``None``
        :type route_name: str, None
        :param weather_category: weather category, defaults to ``None``
        :type weather_category: str, None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of Schedule 8 weather incidents
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor.metex import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()

            >>> s8wi_workbook = sir.get_schedule8_weather_incidents_02062006_31032014(
            ...     update=True, verbose=True)
            Updating "Schedule8WeatherIncidents-02062006-31032014.pickle" at ... ... Done.

            >>> type(s8wi_workbook)
            dict
            >>> list(s8wi_workbook.keys())
            ['Thresholds', 'Data', 'CategoryLookup']

            >>> s8wi_workbook['Thresholds'].tail()

            >>> s8wi_workbook['Data'].tail()

            >>> s8wi_workbook['CategoryLookup'].tail()
        """

        # Path to the file
        pickle_filename = make_filename(self.FILENAME_2, route_name, weather_category)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            s8weather_incidents = load_pickle(path_to_pickle)

        else:
            try:
                # Open the original file
                path_to_xlsm = path_to_pickle.replace(".pickle", ".xlsm")
                workbook = pd.ExcelFile(path_to_xlsm)

                # 'WeatherThresholds' -------------------------------------------------------------
                thresholds = workbook.parse(sheet_name='Thresholds', usecols='A:F').dropna()
                thresholds.columns = [col.replace(' ', '') for col in thresholds.columns]
                thresholds.WeatherHazard = thresholds.WeatherHazard.str.strip().str.upper()
                thresholds.index = range(len(thresholds))

                # 'Data' -----------------------------------------------------------------------------
                raw_data = workbook.parse(
                    sheet_name='Data', parse_dates=['StartDate', 'EndDate'],
                    converters={'stanoxSection': str})

                incident_records = raw_data.rename(
                    columns={'StartDate': 'StartDateTime',
                             'EndDate': 'EndDateTime',
                             'Year': 'FinancialYear',
                             'stanoxSection': 'StanoxSection',
                             'imdm': 'IMDM',
                             'Reason': 'IncidentReason',
                             # 'Minutes': 'DelayMinutes',
                             # 'Cost': 'DelayCost',
                             'CategoryDescription': 'IncidentCategoryDescription'})

                hazard_cols = [
                    x for x in enumerate(incident_records.columns) if 'Weather Hazard' in x[1]]
                obs_cols = [
                    (i - 1, re.search(r'(?<= \()\w+', x).group().upper()) for i, x in hazard_cols]
                hazard_cols = [(i + 1, x + '_WeatherHazard') for i, x in obs_cols]
                for i, x in obs_cols + hazard_cols:
                    incident_records.rename(columns={incident_records.columns[i]: x}, inplace=True)

                # data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')
                incident_reason_metadata = self.DAG.get_incident_reason()
                incident_reason_metadata.columns = [
                    c.replace('_', '') for c in incident_reason_metadata.columns]
                incident_records = incident_records.join(
                    incident_reason_metadata, on='IncidentReason', rsuffix='_meta')
                incident_records.drop(
                    [x for x in incident_records.columns if '_meta' in x], axis=1, inplace=True)

                # Cleanse the location data
                incident_records = self._cleanse_stanox_section_column(
                    incident_records, col_name='StanoxSection', sep=' : ', update_dict=update)

                # Look up geographical coordinates for each incident location
                incident_records = self._cleanse_geographical_coordinates(incident_records)

                # Retain data for specific Route and Weather category
                incident_records = get_subset(
                    incident_records, route_name, weather_category, rearrange_index=False)

                # Weather category lookup -------------------------------------------
                weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
                weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

                # Make a dictionary
                s8weather_incidents = dict(
                    zip(workbook.sheet_names, [thresholds, incident_records, weather_category_lookup]))

                workbook.close()

                # Save the workbook data
                save_pickle(s8weather_incidents, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get/parse \"{}.xlsm\". {}".format(self.FILENAME_2, e))
                s8weather_incidents = None

        return s8weather_incidents


class WeatherThresholds:
    """
    Weather-Thresholds_9306121.html

    Description:

      - "The following table defines Weather thresholds used to determine the classification
        of Weather as Normal, Alert, Adverse or Extreme. Note that the 'Alert' interval is
        inside the 'Normal' range."

      - "These are national thresholds. Route-specific thresholds may also be defined at
        some point."

    **Test**::

        >>> from preprocessor import WeatherThresholds

        >>> thr = WeatherThresholds()

        >>> print(thr.NAME)
        WeatherThresholds
    """

    #: Name of the data
    NAME = 'WeatherThresholds'

    FILENAME = "Weather-Thresholds_9306121.html"

    def __init__(self):

        self.DataDir = os.path.relpath(cdd_metex("thresholds"))

        self.ReportsDataDir = os.path.relpath(cdd_metex("reports"))
        self.S8WeatherIncidentsFilename = "Schedule8WeatherIncidents-02062006-31032014"

    def get_metex_weather_thresholds(self, update=False, verbose=False):
        """
        Get threshold data available in ``html_filename``.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import WeatherThresholds

            >>> thr = WeatherThresholds()

            >>> thresholds = thr.get_metex_weather_thresholds(update=True, verbose=True)
            Updating "Weather-Thresholds_9306121.pickle" at "data\\metex\\thresholds\\" ... Done.
            >>> thresholds = thr.get_metex_weather_thresholds()
            >>> thresholds.tail()
        """

        path_to_pickle = cd(self.DataDir, self.FILENAME.replace(".html", ".pickle"))

        if os.path.isfile(path_to_pickle) and not update:
            metex_weather_thresholds = load_pickle(path_to_pickle)

        else:
            path_to_html = cd(self.DataDir, self.FILENAME)

            try:
                metex_weather_thresholds = pd.read_html(path_to_html)[0]

                # Specify column names
                metex_weather_thresholds.columns = metex_weather_thresholds.loc[0].tolist()
                # Drop the first row, which has been used as the column names
                metex_weather_thresholds.drop(0, axis=0, inplace=True)

                # cls: classification
                cls = metex_weather_thresholds.Classification[
                    metex_weather_thresholds.eq(
                        metex_weather_thresholds.iloc[:, 0], axis=0).all(1)].tolist()

                cls_idx = []
                for i in range(len(cls)):
                    x = metex_weather_thresholds.index[
                        metex_weather_thresholds.Classification == cls[i]][0]
                    metex_weather_thresholds.drop(x, inplace=True)
                    if i + 1 < len(cls):
                        y = metex_weather_thresholds.index[
                            metex_weather_thresholds.Classification == cls[i + 1]][0]
                        to_rpt = y - x - 1
                    else:
                        to_rpt = metex_weather_thresholds.index[-1] - x
                    cls_idx += [cls[i]] * to_rpt
                metex_weather_thresholds.index = cls_idx

                # Add 'VariableName' and 'Unit'
                variables = ['T', 'x', 'r', 'w']
                units = ['degrees Celsius', 'cm', 'mm', 'mph']
                var_list, units_list = [], []
                for i in range(len(cls)):
                    var_temp = [variables[i]] * list(metex_weather_thresholds.index).count(cls[i])
                    units_temp = [units[i]] * list(metex_weather_thresholds.index).count(cls[i])
                    var_list += var_temp
                    units_list += units_temp
                metex_weather_thresholds.insert(1, 'VariableName', var_list)
                metex_weather_thresholds.insert(2, 'Unit', units_list)

                # Retain main description
                metex_weather_thresholds.loc[:, 'Classification'] = \
                    metex_weather_thresholds.Classification.str.replace(
                        r'( \( oC \))|(,[(\xa0) ][xrw] \(((cm)|(mm)|(mph))\))', '', regex=True)
                metex_weather_thresholds.loc[:, 'Classification'] = \
                    metex_weather_thresholds.Classification.str.replace(
                        r' (mph)|(\xa0)', ' ', regex=True)

                # Upper and lower boundaries
                def _boundary(df, col, sep1=None, sep2=None):
                    if sep1:
                        lst_lb = [metex_weather_thresholds[col].iloc[0].split(sep1)[0]]
                        lst_lb += [v.split(sep2)[0] for v in metex_weather_thresholds[col].iloc[1:]]
                        df.insert(df.columns.get_loc(col) + 1, col + 'LowerBound', lst_lb)
                    if sep2:
                        lst_ub = [metex_weather_thresholds[col].iloc[0].split(sep2)[1]]
                        lst_ub += [v.split(sep1)[-1] for v in metex_weather_thresholds[col].iloc[1:]]
                        if sep1:
                            df.insert(df.columns.get_loc(col) + 2, col + 'UpperBound', lst_ub)
                        else:
                            df.insert(df.columns.get_loc(col) + 1, col + 'Threshold', lst_ub)

                # Normal
                _boundary(metex_weather_thresholds, 'Normal', sep1=None, sep2='up to ')
                # Alert
                _boundary(metex_weather_thresholds, 'Alert', sep1=' \u003C ', sep2=' \u2264 ')
                # Adverse
                _boundary(metex_weather_thresholds, 'Adverse', sep1=' \u003C ', sep2=' \u2264 ')
                # Extreme
                extreme = [metex_weather_thresholds['Extreme'].iloc[0].split(' \u2264 ')[1]]
                extreme += [v.split(' \u2265 ')[1] for v in metex_weather_thresholds['Extreme'].iloc[1:]]
                metex_weather_thresholds['ExtremeThreshold'] = extreme

                save_pickle(metex_weather_thresholds, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"weather thresholds\" from the HTML file. {}.".format(e))
                metex_weather_thresholds = None

        return metex_weather_thresholds

    def get_schedule8_weather_thresholds(self, update=False, verbose=False):
        """
        Get threshold data available in ``workbook_filename``.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import WeatherThresholds

            >>> thr = WeatherThresholds()

            >>> thresholds = thr.get_schedule8_weather_thresholds(update=True, verbose=True)
            pdating "Schedule8WeatherIncidents-02062006-31032014-thresholds.pickle" ... ... Done.
            >>> thresholds = thr.get_schedule8_weather_thresholds()
            >>> thresholds.tail()
        """

        pickle_filename = self.S8WeatherIncidentsFilename + "-thresholds.pickle"
        path_to_pickle = cd(self.DataDir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            schedule8_weather_thresholds = load_pickle(path_to_pickle)

        else:
            path_to_spreadsheet = cd(self.ReportsDataDir, self.S8WeatherIncidentsFilename + ".xlsm")

            try:
                schedule8_weather_thresholds = pd.read_excel(
                    path_to_spreadsheet, sheet_name="Thresholds", usecols="A:F")

                schedule8_weather_thresholds.dropna(inplace=True)
                schedule8_weather_thresholds.columns = [
                    col.replace(' ', '') for col in schedule8_weather_thresholds.columns]
                schedule8_weather_thresholds.WeatherHazard = \
                    schedule8_weather_thresholds.WeatherHazard.str.strip().str.upper()
                schedule8_weather_thresholds.index = range(len(schedule8_weather_thresholds))

                save_pickle(schedule8_weather_thresholds, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"weather thresholds\" from the .xlsm file. {}.".format(e))
                schedule8_weather_thresholds = None

        return schedule8_weather_thresholds

    def get_weather_thresholds(self, update=False, verbose=False):
        """
        Get data of weather thresholds.

        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of weather thresholds
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from preprocessor import WeatherThresholds

            >>> thr = WeatherThresholds()

            >>> thresholds = thr.get_weather_thresholds(update=True, verbose=True)
            Updating "WeatherThresholds.pickle" at "data\\metex\\thresholds\\" ... Done.
            >>> thresholds = thr.get_weather_thresholds()
            >>> type(thresholds)
            dict
            >>> list(thresholds.keys())
            ['METExLite', 'Schedule8WeatherIncidents']
        """

        pickle_filename = self.NAME + ".pickle"
        path_to_pickle = cd(self.DataDir, pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            weather_thresholds = load_pickle(path_to_pickle)

        else:

            try:
                metex_weather_thresholds = self.get_metex_weather_thresholds(update)

                schedule8_weather_thresholds = self.get_schedule8_weather_thresholds(update)

                weather_thresholds = {'METExLite': metex_weather_thresholds,
                                      'Schedule8WeatherIncidents': schedule8_weather_thresholds}

                save_pickle(weather_thresholds, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"weather thresholds\". {}.".format(e))
                weather_thresholds = None

        return weather_thresholds
