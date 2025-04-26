"""
Data categorisations.
"""

import datetime

import numpy as np
import pandas as pd
from pyhelpers.text import find_similar_str


def get_data_by_meteorological_seasons(data, seasons, datetime_col):
    """
    Get data for a meteorological season or seasons.

    The meteorological start of a season is based on the annual temperature cycle and
    the 12-month calendar. According to this definition, each season begins on the first of a
    particular month and lasts for three months:

        - Spring begins: 1 March
        - Summer begins: 1 June
        - Autumn begins: 1 September
        - Winter begins: 1 December

    (Source: https://www.almanac.com/content/first-day-seasons)

    :param data:
    :type data: pandas.DataFrame
    :param seasons: 'spring', 'summer', 'autumn', 'winter'; if ``None``, returns data of all seasons
    :type seasons: str or list or None
    :param datetime_col:
    :return: seasonal data (given meteorological seasons)
    :rtype: pandas.DataFrame
    """

    if seasons:
        default_season_names = ['spring', 'summer', 'autumn', 'winter']
        seasons_ = [seasons] if isinstance(seasons, str) else seasons
        seasons_ = [find_similar_str(s, default_season_names) for s in seasons_]

        spring_dat, summer_dat, autumn_dat, winter_dat = [pd.DataFrame()] * 4

        for year, dat in data.groupby('FinancialYear'):
            dt = pd.to_datetime(dat[datetime_col])

            # Get data for spring -----------------------------------
            spring_start1 = datetime.datetime(year=year, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = datetime.datetime(year=year + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring_dates1 = pd.date_range(spring_start1, spring_start1 + pd.DateOffset(months=2))
            spring_dates2 = pd.date_range(spring_start2, spring_start2 + pd.DateOffset(months=2))
            spring_dat = pd.concat([spring_dat, dat[dt.isin(spring_dates1) | dt.isin(spring_dates2)]])

            spring_time = \
                ((dt >= spring_start1) & (dt < spring_end1)) | \
                ((dt >= spring_start2) & (dt < spring_end2))
            spring_dat = pd.concat([spring_dat, dat.loc[spring_time]])

            # -- Get data for summer ---------------------------------------------------------------
            summer_start = datetime.datetime(year=year, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer_dat = pd.concat([summer_dat, dat.loc[(dt >= summer_start) & (dt < summer_end)]])

            # -- Get data for autumn ---------------------------------------------------------------
            autumn_start = datetime.datetime(year=year, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn_dat = pd.concat([autumn_dat, dat.loc[(dt >= autumn_start) & (dt < autumn_end)]])

            # -- Get data for winter ---------------------------------------------------------------
            winter_start = datetime.datetime(year=year, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter_dat = pd.concat([winter_dat, dat.loc[(dt >= winter_start) & (dt < winter_end)]])

        season_data = eval(
            "pd.concat([%s], ignore_index=True)" % ', '.join(['{}_dat'.format(s) for s in seasons_]))

        return season_data

    else:
        return data


def get_data_by_astronomical_seasons(data, seasons, datetime_col):
    """
    Get data for an astronomical season or seasons.

    The astronomical start of a season is based on the position of the Earth in relation to the Sun.
    More specifically, the start of each season is marked by either a solstice (for winter and summer)
    or an equinox (for spring and autumn).
    A solstice is when the Sun reaches the most southerly or northerly point in the sky,
    while an equinox is when the Sun passes over Earth’s equator.
    Because of leap years, the dates of the equinoxes and solstices can shift by a day or two over time,
    causing the start dates of the seasons to shift, too.

        - Spring begins: 19 or 20 March
        - Summer begins: 20 or 21 June
        - Autumn begins: 22 or 23 September
        - Winter begins: 21 or 22 December

    (Source: https://www.almanac.com/content/first-day-seasons)

    :param data:
    :type data: pandas.DataFrame
    :param seasons: 'spring', 'summer', 'autumn', 'winter'; if ``None``, all available data
    :type seasons: str or list or None
    :param datetime_col:
    :type datetime_col: str
    :return:
    :rtype: pandas.DataFrame
    """

    if seasons is None:
        season_data = data.copy()

    else:
        default_season_names = ['spring', 'summer', 'autumn', 'winter']

        input_season_names = [seasons] if isinstance(seasons, str) else seasons
        selected_seasons = [find_similar_str(s, default_season_names) for s in input_season_names]

        def identify_season(incident_dt):
            """
            # (incident_datetime.dt.month % 12 + 3) // 3
            """

            y = incident_dt.year
            seasons_dt = [
                ('winter', (datetime.datetime(y, 1, 1),
                            datetime.datetime(y, 3, 20) + pd.Timedelta(1, unit='day'))),
                ('spring', (datetime.datetime(y, 3, 21),
                            datetime.datetime(y, 6, 20) + pd.Timedelta(1, unit='day'))),
                ('summer', (datetime.datetime(y, 6, 21),
                            datetime.datetime(y, 9, 22) + pd.Timedelta(1, unit='day'))),
                ('autumn', (datetime.datetime(y, 9, 23),
                            datetime.datetime(y, 12, 20) + pd.Timedelta(1, unit='day'))),
                ('winter', (datetime.datetime(y, 12, 21),
                            datetime.datetime(y, 12, 31) + pd.Timedelta(1, unit='day')))]

            return next(season for season, (start, end) in seasons_dt if start <= incident_dt < end)

        mod_data_seasons = data[datetime_col].map(identify_season)

        season_data = data[mod_data_seasons.isin(selected_seasons)]

    return season_data


def label_wind_direction(degree):
    """
    Label wind direction as one of the four quadrants.

    :param degree: degree of wind direction
    :type degree: int or float
    :return: numeric label for wind direction
    :rtype: int

    **Example**::

        >>> from src.shaft.feature import label_wind_direction

        >>> label_wind_direction(45)

    """

    if (degree >= 0) & (degree < 90):
        return 1
    elif (degree >= 90) & (degree < 180):
        return 2
    elif (degree >= 180) & (degree < 270):
        return 3
    else:  # (degree >= 270) & (degree < 360):
        return 4


def define_track_orientation(lon1, lat1, lon2, lat2):
    """
    Label the orientation of a track.

    :param lon1: longitude (of start location)
    :type lon1: int or float
    :param lat1: latitude (of start location)
    :type lat1: int or float
    :param lon2: longitude (of end location)
    :type lon2: int or float
    :param lat2: latitude (of end location)
    :type lat2: int or float
    :return: a textual label for track orientation
    :rtype: str
    """

    radians = np.arctan2(lat2 - lat1, lon2 - lon1)  # Angles in radians, [-pi, pi]

    if np.logical_or(
            np.logical_and(radians >= -np.pi * 2 / 3, radians < -np.pi / 3),
            np.logical_and(radians >= np.pi / 3, radians < np.pi * 2 / 3)):
        # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
        track_orientation = 'N_S'

    elif np.logical_or(
            np.logical_and(radians >= np.pi / 6, radians < np.pi / 3),
            np.logical_and(radians >= -np.pi * 5 / 6, radians < -np.pi * 2 / 3)):
        # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
        track_orientation = 'NE_SW'

    elif np.logical_or(
            np.logical_and(radians >= np.pi * 2 / 3, radians < np.pi * 5 / 6),
            np.logical_and(radians >= -np.pi / 3, radians < -np.pi / 6)):
        track_orientation = 'NW_SE'

    else:
        # np.logical_or(np.logical_or(
        #     np.logical_and(radians >= -np.pi, radians < -np.pi * 5 / 6),
        #     np.logical_and(radians >= -np.pi/6, radians < np.pi/6)),
        #     np.logical_and(radians >= np.pi*5/6, radians < np.pi))
        track_orientation = 'E_W'

    return track_orientation


def categorise_track_orientations(data, geom_column_names=None, column_name='Track_Orientation'):
    """
    Categorise track orientations.

    :param data:
    :type data: pandas.DataFrame
    :param geom_column_names: column names of start and end geographical coordinates,
        defaults to ``None``
    :type geom_column_names: list or None
    :param column_name: defaults to ``'Track_Orientation'``
    :type column_name: str
    :return:
    :rtype:

    **Examples**::

        data = incident_location_weather.copy()

    """

    if geom_column_names is None:
        geom_col_names = ['StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
    else:
        geom_col_names = geom_column_names

    # origin = (-0.565409, 51.23622)
    start_lon, start_lat, end_lon, end_lat = map(lambda x: data[x], geom_col_names)
    # [-pi, pi]
    track_orientations = pd.DataFrame(None, index=range(len(data)), columns=[column_name])
    track_orientations[column_name + '_radians'] = np.arctan2(end_lat - start_lat, end_lon - start_lon)

    categories = ['N_S', 'NE_SW', 'NW_SE', 'E_W']

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 2 / 3))
    track_orientations.loc[n_s, column_name] = categories[0]

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 6,
                       track_orientations.Track_Orientation_radians < np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 5 / 6,
                       track_orientations.Track_Orientation_radians < -np.pi * 2 / 3))
    track_orientations.loc[ne_sw, column_name] = categories[1]

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 5 / 6),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 6))
    track_orientations.loc[nw_se, column_name] = categories[2]

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations[column_name].fillna(categories[3], inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi,
    #                    df.Track_Orientation_radians < -np.pi * 5 / 6),
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi/6,
    #                    df.Track_Orientation_radians < np.pi/6)),
    #     np.logical_and(df.Track_Orientation_radians >= np.pi*5/6,
    #                    df.Track_Orientation_radians < np.pi))
    # data[col_name][e_w] = 'E_W'

    prefix, sep = column_name, '_'
    categorical_var = pd.get_dummies(track_orientations[column_name], prefix=prefix, prefix_sep=sep)
    categorical_var = categorical_var.T.reindex(
        [prefix + sep + x for x in categories]).T.fillna(0).astype(np.int64)

    track_orientations = pd.concat([track_orientations[[column_name]], categorical_var], axis=1)

    return track_orientations


def categorise_temperatures(data, column_name='Temperature_max'):
    """
    Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30.

    :param data:
    :type data:
    :param column_name: defaults to ``'Temperature_max'``
    :type column_name: str
    :return:
    :rtype:
    """

    temp_category = pd.cut(
        data[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf],
        right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    data_ = pd.concat([temperature_category, categorical_var], axis=1)

    return data_
