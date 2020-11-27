"""
Weather-Thresholds_9306121.html

Description:

  - "The following table defines Weather thresholds used to determine the classification
    of Weather as Normal, Alert, Adverse or Extreme. Note that the 'Alert' interval is
    inside the 'Normal' range."

  - "These are national thresholds. Route-specific thresholds may also be defined at
    some point."
"""

import os

import pandas as pd
from pyhelpers.store import load_pickle, save_pickle

from utils import cdd_incidents, cdd_metex


def get_schedule8_weather_thresholds(update=False, verbose=False):
    """
    Get threshold data available in ``workbook_filename``.

    :param update: whether to check on update and proceed to update the package data,
        defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs,
        defaults to ``False``
    :type verbose: bool, int
    :return: data of weather thresholds
    :rtype: pandas.DataFrame or None

    **Examples**::

        >>> from misc.thresholds import get_schedule8_weather_thresholds

        >>> thresholds = get_schedule8_weather_thresholds()
        >>> print(thresholds.head())
                 Description WeatherHazard  WeatherType     Period Condition  Threshold
        0   Wind Gust Normal        NORMAL    WIND_GUST  1 DAY MAX         <       60.0
        1    Wind Gust Alert         ALERT    WIND_GUST  1 DAY MAX        >=       50.0
        2  Wind Gust Adverse       ADVERSE    WIND_GUST  1 DAY MAX        >=       60.0
        3  Wind Gust Extreme       EXTREME    WIND_GUST  1 DAY MAX        >=       80.0
        4        Cold Normal        NORMAL  TEMPERATURE  1 DAY MIN         >       -4.0

        >>> thresholds = get_schedule8_weather_thresholds(update=True, verbose=True)
        Updating "schedule8-weather-thresholds.pickle" at "\\data\\..." ... Done.
        >>> print(thresholds.head())
                 Description WeatherHazard  WeatherType     Period Condition  Threshold
        0   Wind Gust Normal        NORMAL    WIND_GUST  1 DAY MAX         <       60.0
        1    Wind Gust Alert         ALERT    WIND_GUST  1 DAY MAX        >=       50.0
        2  Wind Gust Adverse       ADVERSE    WIND_GUST  1 DAY MAX        >=       60.0
        3  Wind Gust Extreme       EXTREME    WIND_GUST  1 DAY MAX        >=       80.0
        4        Cold Normal        NORMAL  TEMPERATURE  1 DAY MIN         >       -4.0
    """

    data_dir = "spreadsheets"
    pickle_filename = "schedule8-weather-thresholds.pickle"

    path_to_pickle = cdd_incidents(data_dir, pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        schedule8_weather_thresholds = load_pickle(path_to_pickle)

    else:
        path_to_spreadsheet = cdd_incidents(
            data_dir, "Schedule8WeatherIncidents-02062006-31032014.xlsm")

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
            print(
                "Failed to get \"weather thresholds\" from the .xlsm file. {}.".format(e))
            schedule8_weather_thresholds = None

    return schedule8_weather_thresholds


def get_metex_weather_thresholds(update=False, verbose=False):
    """
    Get threshold data available in ``html_filename``.

    :param update: whether to check on update and proceed to update the package data,
        defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs,
        defaults to ``False``
    :type verbose: bool, int
    :return: data of weather thresholds
    :rtype: pandas.DataFrame or None

    **Examples**::

        >>> from misc.thresholds import get_metex_weather_thresholds

        >>> thresholds = get_metex_weather_thresholds()
        >>> print(thresholds.head())
                                     Classification  ... ExtremeThreshold
        Temperature                            Cold  ...               -7
        Temperature                            Heat  ...               30
        Snow             3-Hourly (Wet or Dry Snow)  ...               15
        Rain         Hourly (normal and wet ground)  ...               40
        Rain                               3-Hourly  ...               60
        [5 rows x 13 columns]

        >>> thresholds = get_metex_weather_thresholds(update=True, verbose=True)
        Updating "metex-weather-thresholds.pickle" at "\\data\\..." ... Done.
        >>> print(thresholds.head())
                                     Classification  ... ExtremeThreshold
        Temperature                            Cold  ...               -7
        Temperature                            Heat  ...               30
        Snow             3-Hourly (Wet or Dry Snow)  ...               15
        Rain         Hourly (normal and wet ground)  ...               40
        Rain                               3-Hourly  ...               60
        [5 rows x 13 columns]
    """

    data_dir = "weather\\thresholds"
    pickle_filename = "metex-weather-thresholds.pickle"

    path_to_pickle = cdd_metex(data_dir, pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        metex_weather_thresholds = load_pickle(path_to_pickle)

    else:
        path_to_html = cdd_metex(data_dir, "Weather-Thresholds_9306121.html")

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
                var_temp = \
                    [variables[i]] * list(metex_weather_thresholds.index).count(cls[i])
                units_temp = \
                    [units[i]] * list(metex_weather_thresholds.index).count(cls[i])
                var_list += var_temp
                units_list += units_temp
            metex_weather_thresholds.insert(1, 'VariableName', var_list)
            metex_weather_thresholds.insert(2, 'Unit', units_list)

            # Retain main description
            metex_weather_thresholds.Classification = \
                metex_weather_thresholds.Classification.str.replace(
                    r'( \( oC \))|(,[(\xa0) ][xrw] \(((cm)|(mm)|(mph))\))', '')
            metex_weather_thresholds.Classification = \
                metex_weather_thresholds.Classification.str.replace(
                    ' (mph)', ' ', regex=False)

            # Upper and lower boundaries
            def boundary(df, col, sep1=None, sep2=None):
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
            boundary(metex_weather_thresholds, 'Normal', sep1=None, sep2='up to ')
            # Alert
            boundary(metex_weather_thresholds, 'Alert', sep1=' \u003C ', sep2=' \u2264 ')
            # Adverse
            boundary(metex_weather_thresholds, 'Adverse', sep1=' \u003C ', sep2=' \u2264 ')
            # Extreme
            extreme = [metex_weather_thresholds['Extreme'].iloc[0].split(' \u2264 ')[1]]
            extreme += [v.split(' \u2265 ')[1]
                        for v in metex_weather_thresholds['Extreme'].iloc[1:]]
            metex_weather_thresholds['ExtremeThreshold'] = extreme

            save_pickle(metex_weather_thresholds, path_to_pickle, verbose=verbose)

        except Exception as e:
            print(
                "Failed to get \"weather thresholds\" from the HTML file. {}.".format(e))
            metex_weather_thresholds = None

    return metex_weather_thresholds


def get_weather_thresholds(update=False, verbose=False):
    """
    Get data of weather thresholds.

    :param update: whether to check on update and proceed to update the package data,
        defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs,
        defaults to ``False``
    :type verbose: bool, int
    :return: data of weather thresholds
    :rtype: pandas.DataFrame or None

    **Examples**::

        >>> from misc.thresholds import get_weather_thresholds

        >>> thresholds = get_weather_thresholds()
        >>> type(thresholds)
        dict

        >>> print(list(thresholds.keys()))
        ['METEX', 'Schedule8WeatherIncidents']
    """

    pickle_filename = "weather-thresholds.pickle"
    path_to_pickle = cdd_metex("weather\\thresholds", pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        weather_thresholds = load_pickle(path_to_pickle)
    else:

        try:
            metex_weather_thresholds = get_metex_weather_thresholds()
            schedule8_weather_thresholds = get_schedule8_weather_thresholds(update)
            weather_thresholds = {'METEX': metex_weather_thresholds,
                                  'Schedule8WeatherIncidents': schedule8_weather_thresholds}
            save_pickle(weather_thresholds, path_to_pickle, verbose=verbose)
        except Exception as e:
            print("Failed to get \"weather thresholds\". {}.".format(e))
            weather_thresholds = None

    return weather_thresholds
