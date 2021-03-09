""" Data categorisations """

import datetime

import numpy as np
import pandas as pd
from pyhelpers.text import find_similar_str


def get_data_by_season(incident_data, season='summer'):
    """
    Get data for specified season(s).

    :param incident_data:
    :type incident_data: pandas.DataFrame
    :param season: 'spring', 'summer', 'autumn', 'winter'; if ``None``, returns data of all seasons
    :type season: str, None
    :return:
    :rtype: pandas.DataFrame
    """

    assert season is None or season in ('spring', 'summer', 'autumn', 'winter')

    if season:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4

        for y in pd.unique(incident_data.FinancialYear):
            y_dat = incident_data[incident_data.FinancialYear == y]

            # Get data for spring -----------------------------------
            spring_start1 = datetime.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = datetime.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring_dates1 = pd.date_range(spring_start1, spring_start1 + pd.DateOffset(months=2))
            spring_dates2 = pd.date_range(spring_start2, spring_start2 + pd.DateOffset(months=2))
            spring_data = pd.concat([spring_data, y_dat[
                y_dat.StartDateTime.isin(spring_dates1) | y_dat.StartDateTime.isin(spring_dates2)]])

            spring_time = \
                ((y_dat.StartDateTime >= spring_start1) & (y_dat.StartDateTime < spring_end1)) | \
                ((y_dat.StartDateTime >= spring_start2) & (y_dat.StartDateTime < spring_end2))
            spring_data = pd.concat([spring_data, y_dat.loc[spring_time]])
            # Get data for summer ----------------------------------
            summer_start = datetime.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (y_dat.StartDateTime >= summer_start) & (y_dat.StartDateTime < summer_end)
            summer_data = pd.concat([summer_data, y_dat.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = datetime.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (y_dat.StartDateTime >= autumn_start) & (y_dat.StartDateTime < autumn_end)
            autumn_data = pd.concat([autumn_data, y_dat.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = datetime.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (y_dat.StartDateTime >= winter_start) & (y_dat.StartDateTime < winter_end)
            winter_data = pd.concat([winter_data, y_dat.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [find_similar_str(s, seasons) for s in season]
        seasonal_data = eval(
            "pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data

    else:
        return incident_data


def get_data_by_season_(mod_data, in_seasons, incident_datetime_col):
    """
    An alternative to ``get_data_by_season()``.

    :param mod_data:
    :type mod_data: pandas.DataFrame
    :param in_seasons: 'spring', 'summer', 'autumn', 'winter';
        if ``None``, returns data of all seasons
    :type in_seasons: str, None
    :param incident_datetime_col:
    :type incident_datetime_col: str
    :return:
    :rtype: pandas.DataFrame
    """

    if in_seasons is None:
        return mod_data
    else:
        seasons = [in_seasons] if isinstance(in_seasons, str) else in_seasons
        selected_seasons = [find_similar_str(s, ('Spring', 'Summer', 'Autumn', 'Winter'))
                            for s in seasons]

        def identify_season(incident_dt):
            """
            # (incident_datetime.dt.month % 12 + 3) // 3
            """
            y = incident_dt.year
            seasons_dt = [
                ('Winter', (datetime.datetime(y, 1, 1),
                            datetime.datetime(y, 3, 20) + pd.Timedelta(1, unit='day'))),
                ('Spring', (datetime.datetime(y, 3, 21),
                            datetime.datetime(y, 6, 20) + pd.Timedelta(1, unit='day'))),
                ('Summer', (datetime.datetime(y, 6, 21),
                            datetime.datetime(y, 9, 22) + pd.Timedelta(1, unit='day'))),
                ('Autumn', (datetime.datetime(y, 9, 23),
                            datetime.datetime(y, 12, 20) + pd.Timedelta(1, unit='day'))),
                ('Winter', (datetime.datetime(y, 12, 21),
                            datetime.datetime(y, 12, 31) + pd.Timedelta(1, unit='day')))]
            return next(season for season, (start, end) in seasons_dt if start <= incident_dt < end)

        mod_data_seasons = mod_data[incident_datetime_col].map(identify_season)
        season_data = mod_data[mod_data_seasons.isin(selected_seasons)]

        return season_data


def categorise_wind_directions(degree):
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


def categorise_track_orientations(data, start_lon_colname='StartLongitude',
                                  start_lat_colname='StartLatitude',
                                  end_lon_colname='EndLongitude',
                                  end_lat_colname='EndLatitude'):
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
    :return:
    :rtype:

    **Test**::

        start_lon_colname = 'StartLongitude'
        start_lat_colname = 'StartLatitude'
        end_lon_colname = 'EndLongitude'
        end_lat_colname = 'EndLatitude'

        data = incident_location_weather.copy()

    """

    track_orientations = pd.DataFrame(None, index=range(len(data)), columns=['Track_Orientation'])
    categories = ['N_S', 'NE_SW', 'NW_SE', 'E_W']
    # origin = (-0.565409, 51.23622)
    start_lon, start_lat = data[start_lon_colname], data[start_lat_colname]
    end_lon, end_lat = data[end_lon_colname], data[end_lat_colname]
    track_orientations['Track_Orientation_radians'] = np.arctan2(
        end_lat - start_lat, end_lon - start_lon)  # [-pi, pi]

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 2 / 3))
    track_orientations.loc[n_s, 'Track_Orientation'] = categories[0]

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 6,
                       track_orientations.Track_Orientation_radians < np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 5 / 6,
                       track_orientations.Track_Orientation_radians < -np.pi * 2 / 3))
    track_orientations.loc[ne_sw, 'Track_Orientation'] = categories[1]

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 5 / 6),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 6))
    track_orientations.loc[nw_se, 'Track_Orientation'] = categories[2]

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations.Track_Orientation.fillna(categories[3], inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi,
    #                    df.Track_Orientation_radians < -np.pi * 5 / 6),
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi/6,
    #                    df.Track_Orientation_radians < np.pi/6)),
    #     np.logical_and(df.Track_Orientation_radians >= np.pi*5/6,
    #                    df.Track_Orientation_radians < np.pi))
    # data.Track_Orientation[e_w] = 'E_W'

    prefix, sep = 'Track_Orientation', '_'
    categorical_var = pd.get_dummies(
        track_orientations.Track_Orientation, prefix=prefix, prefix_sep=sep)
    categorical_var = categorical_var.T.reindex(
        [prefix + sep + x for x in categories]).T.fillna(0).astype(np.int64)

    track_orientations = pd.concat(
        [track_orientations[['Track_Orientation']], categorical_var], axis=1)

    return track_orientations


def categorise_track_orientations_(lon1, lat1, lon2, lat2):
    """
    An alternative to ``categorise_track_orientations()``.

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
