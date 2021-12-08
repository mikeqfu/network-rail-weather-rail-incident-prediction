""" Data categorisations """

import datetime

import numpy as np
import pandas as pd
from pyhelpers.text import find_similar_str


def get_data_by_meteorological_seasons(incident_records, in_seasons, datetime_col):
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

    :param incident_records:
    :type incident_records: pandas.DataFrame
    :param in_seasons: 'spring', 'summer', 'autumn', 'winter'; if ``None``, returns data of all seasons
    :type in_seasons: str or list or None
    :param datetime_col:
    :return:
    :rtype: pandas.DataFrame
    """

    if in_seasons:
        default_season_names = ['spring', 'summer', 'autumn', 'winter']
        input_season_names = [in_seasons] if isinstance(in_seasons, str) else in_seasons
        valid_names = [find_similar_str(s, default_season_names) for s in input_season_names]

        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4

        for y in pd.unique(incident_records.FinancialYear):
            y_dat = incident_records[incident_records.FinancialYear == y]

            # Get data for spring -----------------------------------
            spring_start1 = datetime.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = datetime.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring_dates1 = pd.date_range(spring_start1, spring_start1 + pd.DateOffset(months=2))
            spring_dates2 = pd.date_range(spring_start2, spring_start2 + pd.DateOffset(months=2))
            spring_data = pd.concat([spring_data, y_dat[
                y_dat[datetime_col].isin(spring_dates1) | y_dat[datetime_col].isin(spring_dates2)]])

            spring_time = \
                ((y_dat[datetime_col] >= spring_start1) & (y_dat[datetime_col] < spring_end1)) | \
                ((y_dat[datetime_col] >= spring_start2) & (y_dat[datetime_col] < spring_end2))
            spring_data = pd.concat([spring_data, y_dat.loc[spring_time]])

            # Get data for summer ----------------------------------
            summer_start = datetime.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (y_dat[datetime_col] >= summer_start) & (y_dat[datetime_col] < summer_end)
            summer_data = pd.concat([summer_data, y_dat.loc[summer]])

            # Get data for autumn ----------------------------------
            autumn_start = datetime.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (y_dat[datetime_col] >= autumn_start) & (y_dat[datetime_col] < autumn_end)
            autumn_data = pd.concat([autumn_data, y_dat.loc[autumn]])

            # Get data for winter -----------------------------------
            winter_start = datetime.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (y_dat[datetime_col] >= winter_start) & (y_dat[datetime_col] < winter_end)
            winter_data = pd.concat([winter_data, y_dat.loc[winter]])

        season_data = eval(
            "pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in valid_names]))

        return season_data

    else:
        return incident_records


def get_data_by_astronomical_seasons(mod_data, in_seasons, datetime_col):
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

    :param mod_data:
    :type mod_data: pandas.DataFrame
    :param in_seasons: 'spring', 'summer', 'autumn', 'winter'; if ``None``, all available data
    :type in_seasons: str or list or None
    :param datetime_col:
    :type datetime_col: str
    :return:
    :rtype: pandas.DataFrame
    """

    if in_seasons is None:
        season_data = mod_data.copy()

    else:
        default_season_names = ['spring', 'summer', 'autumn', 'winter']

        input_season_names = [in_seasons] if isinstance(in_seasons, str) else in_seasons
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

        mod_data_seasons = mod_data[datetime_col].map(identify_season)

        season_data = mod_data[mod_data_seasons.isin(selected_seasons)]

    return season_data


def define_wind_direction(degree):
    """
    Categorise wind directions into four quadrants.

    :param degree:
    :type degree:
    :return:
    :rtype:
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
    Categorise the orientation of a track.

    :param lon1: longitude (of start location)
    :type lon1:
    :param lat1: latitude (of start location)
    :type lat1:
    :param lon2: longitude (of end location)
    :type lon2:
    :param lat2: latitude (of end location)
    :type lat2:
    :return:
    :rtype:
    """

    track_orientation_radians = np.arctan2(lat2 - lat1, lon2 - lon1)  # Angles in radians, [-pi, pi]

    if np.logical_or(
            np.logical_and(track_orientation_radians >= -np.pi * 2 / 3,
                           track_orientation_radians < -np.pi / 3),
            np.logical_and(track_orientation_radians >= np.pi / 3,
                           track_orientation_radians < np.pi * 2 / 3)):
        # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
        track_orientation = 'N_S'

    elif np.logical_or(
            np.logical_and(track_orientation_radians >= np.pi / 6,
                           track_orientation_radians < np.pi / 3),
            np.logical_and(track_orientation_radians >= -np.pi * 5 / 6,
                           track_orientation_radians < -np.pi * 2 / 3)):
        # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
        track_orientation = 'NE_SW'

    elif np.logical_or(
            np.logical_and(track_orientation_radians >= np.pi * 2 / 3,
                           track_orientation_radians < np.pi * 5 / 6),
            np.logical_and(track_orientation_radians >= -np.pi / 3,
                           track_orientation_radians < -np.pi / 6)):
        track_orientation = 'NW_SE'

    else:
        # np.logical_or(np.logical_or(
        #     np.logical_and(track_orientation_radians >= -np.pi,
        #                    track_orientation_radians < -np.pi * 5 / 6),
        #     np.logical_and(track_orientation_radians >= -np.pi/6,
        #                    track_orientation_radians < np.pi/6)),
        #     np.logical_and(track_orientation_radians >= np.pi*5/6,
        #                    track_orientation_radians < np.pi))
        track_orientation = 'E_W'

    return track_orientation


def categorise_track_orientations(data, start_lon_colname='StartLongitude',
                                  start_lat_colname='StartLatitude',
                                  end_lon_colname='EndLongitude',
                                  end_lat_colname='EndLatitude', col_name='Track_Orientation'):
    """
    Categorise track orientations.

    :param data:
    :type data:
    :param start_lon_colname: column name of start longitude
    :type start_lon_colname: str
    :param start_lat_colname: column name of start latitude
    :type start_lat_colname: str
    :param end_lon_colname: column name of end longitude
    :type end_lon_colname: str
    :param end_lat_colname: column name of end latitude
    :type end_lat_colname: str
    :param col_name:
    :return:
    :rtype:

    **Test**::

        start_lon_colname = 'StartLongitude'
        start_lat_colname = 'StartLatitude'
        end_lon_colname = 'EndLongitude'
        end_lat_colname = 'EndLatitude'

        data = incident_location_weather.copy()

    """

    track_orientations = pd.DataFrame(None, index=range(len(data)), columns=[col_name])

    categories = ['N_S', 'NE_SW', 'NW_SE', 'E_W']

    # origin = (-0.565409, 51.23622)
    start_lon, start_lat = data[start_lon_colname], data[start_lat_colname]
    end_lon, end_lat = data[end_lon_colname], data[end_lat_colname]
    # [-pi, pi]
    track_orientations[col_name + '_radians'] = np.arctan2(end_lat - start_lat, end_lon - start_lon)

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 2 / 3))
    track_orientations.loc[n_s, col_name] = categories[0]

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 6,
                       track_orientations.Track_Orientation_radians < np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 5 / 6,
                       track_orientations.Track_Orientation_radians < -np.pi * 2 / 3))
    track_orientations.loc[ne_sw, col_name] = categories[1]

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 5 / 6),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 6))
    track_orientations.loc[nw_se, col_name] = categories[2]

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations[col_name].fillna(categories[3], inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi,
    #                    df.Track_Orientation_radians < -np.pi * 5 / 6),
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi/6,
    #                    df.Track_Orientation_radians < np.pi/6)),
    #     np.logical_and(df.Track_Orientation_radians >= np.pi*5/6,
    #                    df.Track_Orientation_radians < np.pi))
    # data[col_name][e_w] = 'E_W'

    prefix, sep = col_name, '_'
    categorical_var = pd.get_dummies(
        track_orientations[col_name], prefix=prefix, prefix_sep=sep)
    categorical_var = categorical_var.T.reindex(
        [prefix + sep + x for x in categories]).T.fillna(0).astype(np.int64)

    track_orientations = pd.concat(
        [track_orientations[[col_name]], categorical_var], axis=1)

    return track_orientations


def categorise_temperatures(attr_dat, column_name='Temperature_max'):
    """
    Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30.

    :param attr_dat:
    :type attr_dat:
    :param column_name: defaults to ``'Temperature_max'``
    :type column_name: str
    :return:
    :rtype:
    """

    temp_category = pd.cut(
        attr_dat[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf],
        right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    temperature_category_data = pd.concat([temperature_category, categorical_var], axis=1)

    return temperature_category_data
