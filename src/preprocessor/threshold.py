import os
import re

import numpy as np
import pandas as pd
from pyhelpers.dirs import cd, cdd
from pyhelpers.store import xlsx_to_csv

from src.utils import Handler, WxRailIncidentsPred


# noinspection PyShadowingNames
class WeatherThresholds(Handler):
    """
    A class for handling data of Weather thresholds.

      - The data of Weather thresholds may be used to determine the classification
        of Weather as 'Normal', 'Alert', 'Adverse' or 'Extreme'. Note that the 'Alert' interval is
        inside the 'Normal' range."
      - The available data are national thresholds, rather than Route-specific thresholds.
    """

    #: Name of the data
    DATA_NAME: str = 'Weather thresholds'
    #: Filename of the Weather threshold data stored in HTML format
    HTML_FILENAME: str = "weather_thresholds_9306121.html"
    #: Filename of the Weather threshold data stored in Excel spreadsheet
    REPORT_FILENAME: str = "schedule8_weather_incidents_02062006_31032014.xlsm"
    #: Schema name
    SCHEMA_NAME: str = "NR_WeatherThresholds"

    def __init__(self, db_instance=None):
        """
        :param db_instance: A PostgreSQL database instance; defaults to ``None``.
        :type db_instance: src.utils.WxRailIncidentsPred | None

        :ivar str | pathlib.Path DATA_DIR: the main data directory for this class
        :ivar str | pathlib.Path reports_data_dir: directory of the data of Incidents Reports

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> thr.DATA_NAME
            'Weather thresholds'
        """

        super().__init__(db_instance=db_instance)

        self.data_dir = cdd("metex/weather")

        self.reports_data_dir = cdd("metex/incidents/reports")

        self.weather_thresholds1 = None
        self.weather_thresholds2 = None

    @staticmethod
    def specify_table_name(filename):
        table_name = re.sub(r'[ -]', '_', filename.split('.')[0]).lower()
        return table_name

    def _weather_thresholds1(self, verbose=False, **kwargs):
        """
        Read Weather threshold data stored in HTML format.

        :return: Weather threshold data stored in HTML format
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> thresholds = thr._weather_thresholds1()
            >>> thresholds.shape
            (11, 13)
        """

        path_to_html = cd(self.data_dir, self.HTML_FILENAME)

        thresholds = pd.read_html(path_to_html)[0]

        # Specify column names
        thresholds.columns = thresholds.loc[0].tolist()
        # Drop the first row, which has been used as the column names
        thresholds.drop(0, axis=0, inplace=True)

        # cls: classification
        cls = thresholds.Classification[thresholds.eq(thresholds.iloc[:, 0], axis=0).all(1)]

        cls_idx = []
        for i in range(len(cls)):
            x = thresholds.index[thresholds.Classification == cls.iloc[i]][0]
            thresholds.drop(x, inplace=True)
            if i + 1 < len(cls):
                y = thresholds.index[thresholds.Classification == cls.iloc[i + 1]][0]
                to_rpt = y - x - 1
            else:
                to_rpt = thresholds.index[-1] - x
            cls_idx += [cls.iloc[i]] * to_rpt
        thresholds.index = cls_idx

        # Add 'Variable' and 'Unit'
        variables = ['T', 'x', 'r', 'w']
        units = ['degrees Celsius', 'cm', 'mm', 'mph']
        var_list, units_list = [], []
        for i in range(len(cls)):
            var_temp = [variables[i]] * list(thresholds.index).count(cls.iloc[i])
            units_temp = [units[i]] * list(thresholds.index).count(cls.iloc[i])
            var_list += var_temp
            units_list += units_temp
        thresholds.insert(1, column='Variable', value=np.array(var_list))
        thresholds.insert(2, column='Unit', value=np.array(units_list))

        # Retain main description
        temp = thresholds.Classification.str.replace(
            r'( \( oC \))|(,[(\xa0) ][xrw] \(((cm)|(mm)|(mph))\))', '', regex=True)
        thresholds.loc[:, 'Classification'] = temp.str.replace(
            r' (mph)|(\xa0)', ' ', regex=True)

        # Upper and lower boundaries
        def _boundary(df, col, sep1=None, sep2=None):
            if sep1:
                lst_lb = [thresholds[col].iloc[0].split(sep1)[0]]
                lst_lb += [v.split(sep2)[0] for v in thresholds[col].iloc[1:]]
                df.insert(df.columns.get_loc(col) + 1, col + 'LowerBound', lst_lb)
            if sep2:
                lst_ub = [thresholds[col].iloc[0].split(sep2)[1]]
                lst_ub += [v.split(sep1)[-1] for v in thresholds[col].iloc[1:]]
                if sep1:
                    df.insert(df.columns.get_loc(col) + 2, col + 'UpperBound', lst_ub)
                else:
                    df.insert(df.columns.get_loc(col) + 1, col + 'Threshold', lst_ub)

        # Normal
        _boundary(thresholds, 'Normal', sep1=None, sep2='up to ')
        # Alert
        _boundary(thresholds, 'Alert', sep1=' \u003C ', sep2=' \u2264 ')
        # Adverse
        _boundary(thresholds, 'Adverse', sep1=' \u003C ', sep2=' \u2264 ')
        # Extreme
        extreme = [thresholds['Extreme'].iloc[0].split(' \u2264 ')[1]]
        extreme += [v.split(' \u2265 ')[1] for v in thresholds['Extreme'].iloc[1:]]
        thresholds['ExtremeThreshold'] = extreme

        num_cols = [x for x in thresholds.columns if x.endswith('Threshold') or x.endswith('Bound')]
        thresholds.loc[:, num_cols] = thresholds[num_cols].astype(int)

        thresholds.index.name = 'VariableName'

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        table_name = self.specify_table_name(self.HTML_FILENAME)
        self.dump_preprocessed_data(
            data=thresholds, table_name=table_name, verbose=verbose, pkey=[], **kwargs)

        return thresholds

    def read_weather_thresholds1(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> thr.read_weather_thresholds1()
            >>> thr.weather_thresholds1.shape
            (11, 13)
        """

        table_name = self.specify_table_name(self.HTML_FILENAME)

        self.read_data(
            table_name, ivar_name='weather_thresholds1', index_col='VariableName', **kwargs)

    def _weather_thresholds2(self, verbose=False, **kwargs):
        """
        Read Weather threshold data from the Excel spreadsheet of incident report.

        :return: Weather threshold data stored in Excel spreadsheet
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> thresholds = thr._weather_thresholds2()
            >>> thresholds.shape
            (29, 4)
        """

        path_to_spreadsheet = cd(self.reports_data_dir, self.REPORT_FILENAME)
        temp_csv_pathname = xlsx_to_csv(path_to_spreadsheet, sheet_name='1')
        raw = pd.read_csv(temp_csv_pathname, usecols=list(range(6)))
        os.remove(temp_csv_pathname)

        thresholds = raw.dropna()
        thresholds.columns = [col.replace(' ', '') for col in thresholds.columns]

        hazard_col_name = 'WeatherHazard'
        temp = thresholds[hazard_col_name].str.strip()
        del thresholds[hazard_col_name]
        thresholds.insert(1, hazard_col_name, temp)

        thresholds.set_index(['WeatherType', 'WeatherHazard'], inplace=True)

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        table_name = self.specify_table_name(self.REPORT_FILENAME)
        self.dump_preprocessed_data(
            data=thresholds, table_name=table_name, verbose=verbose, pkey=[], **kwargs)

        return thresholds

    def read_weather_thresholds2(self, **kwargs):
        """
        Read Weather threshold data stored in the Excel spreadsheet of incident report.

        :return: Weather threshold data stored in the Excel spreadsheet of incident report
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> thr.read_weather_thresholds2()
            >>> thr.weather_thresholds2.shape
            (29, 4)
        """

        table_name = self.specify_table_name(self.REPORT_FILENAME)

        self.read_data(
            table_name, ivar_name='weather_thresholds2', index_col=['WeatherType', 'WeatherHazard'],
            **kwargs)

    def read_weather_thresholds(self, update=False, verbose=False):
        """
        Read all available Weather threshold data.

        :param update: Whether to read or reprocess the original data file; defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information; defaults to ``False``.
        :type verbose: bool | int
        :return: all available Weather threshold data
        :rtype: pandas.DataFrame | None

        **Examples**::

            >>> from src.preprocessor import WeatherThresholds
            >>> thr = WeatherThresholds()
            >>> weather_thresholds = thr.read_weather_thresholds()
            >>> list(weather_thresholds.keys())
            ['METEX', 'Schedule8WeatherIncidents']
            >>> weather_thresholds['METEX'].shape
            (11, 13)
            >>> weather_thresholds['Schedule8WeatherIncidents'].shape
            (29, 4)
        """

        if self.weather_thresholds1 is None or update:
            self.read_weather_thresholds1(update=update, verbose=verbose)

        if self.weather_thresholds2 is None or update:
            self.read_weather_thresholds2(update=update, verbose=verbose)

        weather_thresholds = {
            'METEX': self.weather_thresholds1,
            'Schedule8WeatherIncidents': self.weather_thresholds2,
        }

        return weather_thresholds
