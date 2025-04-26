"""
Preprocess the data of historic delay attribution glossary.
"""

import os
import re

import numpy as np
import pandas as pd
from pyhelpers._cache import _print_failure_message
from pyhelpers.dirs import cd, cdd
from pyhelpers.ops import confirmed, download_file_from_url
from pyhelpers.store import _check_saving_path, load_data, save_data
from pyrcs.converter import fix_stanox
from pyrcs.line_data import LocationIdentifiers

from src.utils import WxRailIncidentsPred


class DelayAttributionGlossary:
    """
    A class for handling data of historic delay attribution glossary.

    Data source: https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/
    transparency/open-data-feeds/
    """

    #: Name of the data.
    DATA_NAME: str = 'Historic delay attribution glossary'
    #: Pathname of the local data directory.
    DATA_DIR: str = os.path.relpath(cdd("metex/incidents/delay_attribution/glossary"))
    #: Filename of the data (online).
    DEFAULT_FILENAME: str = "Transparency page Attribution Glossary.xlsx"
    #: Filename of the data (saved locally).
    FILENAME: str = f"{DATA_NAME.lower().replace(' ', '_')}.xlsx"
    #: Name of the schema for storing the data in the project database.
    SCHEMA_NAME: str = 'NR_DelayAttributionGlossary'
    #: Download link.
    FILE_URL: str = (f'https://sacuksprodnrdigital0001.blob.core.windows.net/'
                     f'historic-delay-attribution/Reference%20Files/'
                     f'{DEFAULT_FILENAME.replace(" ", "%20")}')

    def __init__(self, db_instance=None):
        """
        :param db_instance: An instance of the class :class:`~src.utils.WxRailIncidentsPred`,
            connecting to the project database in a PostgreSQL server; defaults to ``None``.
        :type db_instance: WxRailIncidentsPred | None

        :ivar pyrcs.line_data.LocationIdentifiers lid: An instance of the class
            `pyrcs.LocationIdentifiers`_.
        :ivar WxRailIncidentsPred | None db_instance: An instance of the class
            :class:`~src.utils.WxRailIncidentsPred`, connecting to the project database
            in a PostgreSQL server.

        .. `pyrcs.LocationIdentifiers`:
            https://pyrcs.readthedocs.io/en/latest/_generated/
            pyrcs.line_data.LocationIdentifiers.html

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> dag = DelayAttributionGlossary()
            >>> dag.DATA_NAME
            'Historic delay attribution glossary'
        """

        self.lid = LocationIdentifiers()

        self.db_instance = db_instance

    def _cdd(self, *sub_dir, mkdir=False):
        """
        Change to the data directory, any of its subdirectories or a file in it.

        :param sub_dir: Subdirectory name(s) or filename(s).
        :type sub_dir: str
        :param mkdir: Whether to create a directory; defaults to ``False``.
        :type mkdir: bool
        :return: Path to the data directory, any of its subdirectories or
            a file in it, which is associated with this class
            :class:`~src.preprocessor.glossary.DelayAttributionGlossary`.
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> import os
            >>> dag = DelayAttributionGlossary()
            >>> dag.DATA_DIR == os.path.relpath(dag._cdd())
            True
        """

        path = cd(self.DATA_DIR, *sub_dir, mkdir=mkdir)

        return path

    def path_to_original_file(self):
        """
        Get the pathname of the original data file.

        :return: Pathname of the original data file.
        :rtype: str

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> import os
            >>> dag = DelayAttributionGlossary()
            >>> os.path.basename(dag.path_to_original_file())
            'historic_delay_attribution_glossary.xlsx'
            >>> os.path.isfile(dag.path_to_original_file())
            True
        """

        path_to_file = self._cdd(self.FILENAME)

        return path_to_file

    def _download_dag(self, url, verbose):
        verbose_ = True if verbose == 2 else False

        _check_saving_path(
            self.path_to_original_file(), verbose=True if verbose in {1, True} else False,
            state_verb="Downloading")

        try:
            download_file_from_url(url, path_to_file=self.path_to_original_file(), verbose=verbose_)

            if verbose in {1, True}:
                print("Done.")

        except Exception as e:
            _print_failure_message(e)

    def download_dag(self, confirmation_required=True, verbose=False):
        """
        Download delay attribution glossary.

        :param confirmation_required: Whether to ask for confirmation to proceed;
            defaults to ``True``.
        :type confirmation_required: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> dag = DelayAttributionGlossary()
            >>> dag.download_dag(confirmation_required=True, verbose=True)
            Replace the current version
            ? [No]|Yes: yes
            Downloading "Historic delay attribution glossary.xlsx" ... Done.
        """

        if os.path.isfile(self.path_to_original_file()):
            if confirmed("Replace the current version\n?", confirmation_required):
                self._download_dag(self.FILE_URL, verbose)

        else:
            self._download_dag(self.FILE_URL, verbose)

    @staticmethod
    def _make_table_name(sheet_name):
        table_name = sheet_name.lower().replace(' ', '_')
        return table_name

    def _stanox_codes(self):
        """Get STANOX codes."""
        sheet_name = "Stanox Codes"
        raw = pd.read_excel(
            io=self.path_to_original_file(), sheet_name=sheet_name, dtype={'STANOX NO.': str},
            skipfooter=2)

        stanox_codes = raw.copy()
        stanox_codes.columns = [x.strip('.').replace(' ', '_') for x in stanox_codes.columns]

        stanox_codes.STANOX_NO = stanox_codes.STANOX_NO.map(fix_stanox)
        stanox_codes.FULL_NAME = stanox_codes.FULL_NAME.str.strip('`')

        stanox_codes.fillna(value='', inplace=True)

        return stanox_codes

    @staticmethod
    def _split_raw_period_dates(raw, n=3):
        raw.columns = raw.iloc[0]
        raw.drop(0, axis=0, inplace=True)

        raw_ = raw.dropna(axis='index')
        dat_list = []
        for i in range(0, len(raw_.T), n):
            dat = raw_.T[i:i + n].T
            dat.columns.name = None
            dat_list.append(dat)

        data = pd.concat(dat_list, axis=0, ignore_index=True)

        return data

    def _period_dates(self):
        """Get period dates."""
        sheet_name = "Period Dates"
        raw = pd.read_excel(self.path_to_original_file(), sheet_name=sheet_name, skiprows=3)
        raw.dropna(axis=1, how='all', inplace=True)

        periods = raw[['Unnamed: 0']].dropna()
        raw.drop('Unnamed: 0', axis='columns', inplace=True)
        periods.columns = ['Period']

        financial_years = [
            x.replace('YEAR ', '').replace('/', '-20') for x in raw.columns
            if 'Unnamed' not in x]

        data = self._split_raw_period_dates(raw, n=3)

        data.columns = [x.replace(' ', '_').strip('.') for x in data.columns]
        data.rename(columns={'Day_Name': 'End_Day', 'Date': 'End_Date'}, inplace=True)
        data['End_Date'] = pd.to_datetime(data['End_Date'])
        data['Start_Date'] = data['End_Date'] - data['No_of_Days'].map(
            lambda x: pd.Timedelta(days=x - 1))
        data['Start_Day'] = data.Start_Date.dt.day_name()
        data['Period'] = periods['Period'].to_list() * int(raw.shape[1] / 3)
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

        return period_dates

    def _incident_reason(self):
        """Get incident reasons (metadata)."""
        sheet_name = "Incident Reason"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        incident_reason = raw.copy()
        incident_reason.columns = [x.replace(' ', '_') for x in incident_reason.columns]

        # incident_reason.set_index(keys='Incident_Reason', inplace=True)

        return incident_reason

    def _responsible_manager(self):
        """Get responsible manager."""
        sheet_name = "Responsible Manager"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        responsible_manager = raw.copy()
        responsible_manager.columns = [
            x.replace(' ', '_').replace('Responsible_Managers.', '')
            for x in responsible_manager.columns]

        responsible_manager.Responsible_Manager_Name = \
            responsible_manager.Responsible_Manager_Name.str.strip()

        return responsible_manager

    def _reactionary_reason_code(self):
        """Get reactionary reason code."""
        sheet_name = "Reactionary Reason Code"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        reactionary_reason_code = raw.copy()
        reactionary_reason_code.columns = [
            x.replace(' ', '_').replace('Reactionary_Reasons.', '')
            for x in reactionary_reason_code.columns]

        return reactionary_reason_code

    def _performance_event_code(self):
        """Get performance event code."""
        sheet_name = "Performance Event Code"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        performance_event_code = raw.copy()
        performance_event_code.columns = [
            x.replace(' ', '_').replace('Performance_Event_Types.', '')
            for x in performance_event_code.columns]

        # performance_event_code.set_index('Performance_Event_Code', inplace=True)

        return performance_event_code

    def _train_service_code(self):
        """Get train service code."""
        sheet_name = "Train Service Code"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        train_service_code = raw.copy()
        train_service_code.columns = [
            x.replace(' - ', '_').replace(' ', '_') for x in train_service_code.columns]

        return train_service_code

    def _operator_name(self):
        """Get operator name."""
        sheet_name = "Operator Name"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        operator_name = raw.copy()
        operator_name.columns = [re.sub(r' - | ', '_', x) for x in operator_name.columns]

        return operator_name

    def _service_group_code(self):
        """Get service group code."""
        sheet_name = "Service Group Code"
        raw = pd.read_excel(io=self.path_to_original_file(), sheet_name=sheet_name, skipfooter=2)

        service_group_code = raw.copy()
        service_group_code.columns = [
            re.sub(r' - | ', '_', x) for x in service_group_code.columns]

        return service_group_code

    def _read_dag_data_from_db(self, sheet_name, update=False, hard_update=False, verbose=False,
                               **kwargs):
        """
        Get data of a worksheet.

        :param sheet_name: Name of a specific sheet in the original data file
            (i.e. an Excel workbook).
        :type sheet_name: str
        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param hard_update: Whether to redownload the original data file; defaults to ``False``.
        :type hard_update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of a worksheet of the given ``sheet_name``.
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> dag = DelayAttributionGlossary()
            >>> stanox_codes = dag._read_dag_data_from_db(sheet_name='Stanox Codes')
            >>> stanox_codes.shape
            (11231, 4)
            >>> period_dates = dag._read_dag_data_from_db(sheet_name="Period Dates")
            >>> period_dates.shape
            (247, 7)
        """

        table_name = self._make_table_name(sheet_name)

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=verbose)

        if self.db_instance.table_exists(table_name, schema_name=self.SCHEMA_NAME) and not update:
            return self.db_instance.read_table(table_name, schema_name=self.SCHEMA_NAME, **kwargs)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            if verbose:
                tbl = f'"{self.SCHEMA_NAME}"."{table_name}"'
                if self.db_instance.table_exists(table_name, self.SCHEMA_NAME):
                    msg = "Updating "
                else:
                    msg = "Importing "
                print(msg + f"{tbl}", end=" ... ")

            try:
                data = getattr(self, f'_{table_name}')()

                self.db_instance.import_data(
                    data=data, table_name=table_name, schema_name=self.SCHEMA_NAME,
                    if_exists='replace',
                    index=True if data.index.name is not None else False,
                    method=self.db_instance.psql_insert_copy,
                    confirmation_required=False, verbose=False, **kwargs)

                if verbose:
                    print("Done.")

                return data

            except Exception as e:
                _print_failure_message(e, prefix=f'Failed at "{sheet_name}".')

    def _read_dag_data(self, sheet_name, update=False, hard_update=False, verbose=False):
        """

        :param sheet_name:
        :param update:
        :param hard_update:
        :param verbose:
        :return:

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> dag = DelayAttributionGlossary()
            >>> stanox_codes = dag._read_dag_data(sheet_name='Stanox Codes')

        """

        pkl_filename = f"{self._make_table_name(sheet_name)}.pkl"
        path_to_pkl = cd(self.DATA_DIR, pkl_filename)

        if os.path.isfile(path_to_pkl) and not update:
            return load_data(path_to_pkl, verbose=verbose)

        else:
            if not os.path.isfile(self.path_to_original_file()) or hard_update:
                self.download_dag(confirmation_required=False, verbose=False)

            if verbose:
                print(f"Reading {sheet_name}", end=" ... ")

            try:
                data = getattr(self, f'_{self._make_table_name(sheet_name)}')()

                if verbose:
                    print("Done.")

                save_data(data, path_to_pkl, verbose=verbose)

                return data

            except Exception as e:
                _print_failure_message(e)

    def read_data(self, update=False, hard_update=False, verbose=False, **kwargs):
        # noinspection PyShadowingNames
        """
        Get historic delay attribution glossary.

        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param hard_update: Whether to redownload the original data file; defaults to ``False``.
        :type hard_update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of historic delay attribution glossary.
        :rtype: dict | None

        **Examples**::

            >>> from src.preprocessor.glossary import DelayAttributionGlossary
            >>> dag = DelayAttributionGlossary()
            >>> delay_attr_glossary = dag.read_data()
            >>> list(delay_attr_glossary.keys())
            ['Stanox Codes',
             'Period Dates',
             'Incident Reason',
             'Responsible Manager',
             'Reactionary Reason Code',
             'Performance Event Code',
             'Service Group Code',
             'Operator Name',
             'Train Service Code']
            >>> delay_attr_glossary['Train Service Code'].shape
            (2872, 4)
        """

        if not os.path.isfile(self.path_to_original_file()) or hard_update:
            self.download_dag(confirmation_required=False, verbose=verbose)

        try:
            with pd.ExcelFile(self.path_to_original_file()) as workbook:
                glossary = []
                for sheet_name in workbook.sheet_names:
                    read_worksheet_data_params = {
                        'sheet_name': sheet_name,
                        'update': update,
                        'hard_update': False,
                        'verbose': verbose,
                    }
                    kwargs.update(read_worksheet_data_params)

                    try:
                        sheet_data = self._read_dag_data(**kwargs)
                    except Exception as e:
                        _print_failure_message(e, verbose=verbose)
                        sheet_data = self._read_dag_data_from_db(**kwargs)

                    glossary.append(sheet_data)

                delay_attr_glossary = dict(zip(workbook.sheet_names, glossary))

            if verbose:
                print("Process finished.")

            return delay_attr_glossary

        except Exception as e:
            _print_failure_message(e)


def main():
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Get historic delay attribution glossary.")

    # Add arguments
    parser.add_argument(
        '--update', required=False, default=False,
        help="Whether update the data stored in the project database; defaults to ``False``.")
    parser.add_argument(
        '--hard_update', required=False, default=False,
        help="Whether to redownload the original data file; defaults to ``False``.")
    parser.add_argument(
        '--verbose', required=False, default=False,
        help="Whether to print relevant information to the console; defaults to ``False``.")

    # Parse arguments
    args = parser.parse_args()

    dag = DelayAttributionGlossary()

    delay_attr_glossary = dag.read_data(
        update=args.update,
        hard_update=args.hard_update,
        verbose=args.verbose
    )
    print(delay_attr_glossary)


if __name__ == '__main__':
    main()

    # To Run in CMD:
    #
    # python src/preprocessor/glossary.py --update False --hard_update False --verbose False
