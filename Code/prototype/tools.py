import os

import numpy as np
import pandas as pd
import shapely.geometry
from pyhelpers.store import save_fig
from pyhelpers.text import find_similar_str

import mssqlserver.metex
from utils import cd, cdd

# ====================================================================================================================
""" Change directories """


def cdd_prototype(*sub_dir):
    path = cdd("Models\\prototype")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Model" and sub-directories
def cd_prototype_dat(*sub_dir):
    path = cdd_prototype("dat")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\...\\Figures"
def cd_prototype_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\1 - Prototype\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Save output figures """


# A function for saving the plots
def save_hotpots_fig(fig, keyword, show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                     save_as, dpi):
    """
    :param fig: [matplotlib.figure.Figure]
    :param keyword: [str] a keyword for specifying the filename
    :param show_metex_weather_cells: [bool]
    :param show_osm_landuse_forest: [bool]
    :param show_nr_hazardous_trees: [bool]
    :param save_as: [str]
    :param dpi: [int] or None
    :return:
    """
    if save_as is not None:
        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                         ['cell', 'veg', 'haz'])
            filename = '_'.join([keyword] + [v for s, v in suffix if s is True])
            path_to_file = cd_prototype_fig_pub("Hotspots", filename + save_as)
            save_fig(path_to_file, dpi=dpi)


# ====================================================================================================================
""" Data manipulations """


# Get data for specified season(s)
def get_data_by_season(incident_data, season='summer'):
    """
    :param incident_data: [pandas.DataFrame]
    :param season: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :return:
    """
    assert season is None or season in ('spring', 'summer', 'autumn', 'winter')
    if season:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4
        for y in pd.unique(incident_data.FinancialYear):
            y_dat = incident_data[incident_data.FinancialYear == y]
            # Get data for spring -----------------------------------
            spring_start1 = pd.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = pd.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring_dates1 = pd.date_range(spring_start1, spring_start1 + pd.DateOffset(months=2))
            spring_dates2 = pd.date_range(spring_start2, spring_start2 + pd.DateOffset(months=2))
            spring_data = pd.concat(
                [spring_data, y_dat[y_dat.StartDateTime.isin(spring_dates1) | y_dat.StartDateTime.isin(spring_dates2)]])

            spring_time = ((y_dat.StartDateTime >= spring_start1) & (y_dat.StartDateTime < spring_end1)) | \
                          ((y_dat.StartDateTime >= spring_start2) & (y_dat.StartDateTime < spring_end2))
            spring_data = pd.concat([spring_data, y_dat.loc[spring_time]])
            # Get data for summer ----------------------------------
            summer_start = pd.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (y_dat.StartDateTime >= summer_start) & (y_dat.StartDateTime < summer_end)
            summer_data = pd.concat([summer_data, y_dat.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = pd.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (y_dat.StartDateTime >= autumn_start) & (y_dat.StartDateTime < autumn_end)
            autumn_data = pd.concat([autumn_data, y_dat.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = pd.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (y_dat.StartDateTime >= winter_start) & (y_dat.StartDateTime < winter_end)
            winter_data = pd.concat([winter_data, y_dat.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [find_similar_str(s, seasons)[0] for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data

    else:
        return incident_data


# Find Weather Cell ID
def find_weather_cell_id(longitude, latitude):
    weather_cell = mssqlserver.metex.get_weather_cell()

    ll = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ll_Longitude, weather_cell.ll_Latitude)]
    ul = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ul_lon, weather_cell.ul_lat)]
    ur = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ur_Longitude, weather_cell.ur_Latitude)]
    lr = [shapely.geometry.Point(xy) for xy in zip(weather_cell.lr_lon, weather_cell.lr_lat)]

    poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]

    cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in range(len(weather_cell))]

    pt = shapely.geometry.Point(longitude, latitude)

    id_set = set(weather_cell.iloc[[i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
    if len(id_set) == 1:
        weather_cell_id = list(id_set)[0]
    else:
        weather_cell_id = list(id_set)
    return weather_cell_id


# ====================================================================================================================
""" Calculations """


# Get all Weather variable names
def get_weather_variable_names(weather_stats_calculations: dict):
    weather_variable_names = []
    for k, v in weather_stats_calculations.items():
        if isinstance(v, tuple):
            for v_ in v:
                weather_variable_names.append('_'.join([k, v_.__name__.replace('mean', 'avg')]).replace('_nan', '_'))
        else:
            weather_variable_names.append('_'.join([k, v.__name__.replace('mean', 'avg')]).replace('_nan', '_'))
    wind_variable_names_ = weather_variable_names + ['WindSpeed_avg', 'WindDirection_avg']
    return wind_variable_names_


# Calculate average wind speed and direction
def calculate_wind_averages(wind_speeds, wind_directions):
    u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
    v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
    uav, vav = np.nanmean(u), np.nanmean(v)  # sum up all u and v values and average it
    average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed
    # Calculate average wind direction
    if uav == 0:
        average_wind_direction = 0 if vav == 0 else (360 if vav > 0 else 180)
    else:
        average_wind_direction = (270 if uav > 0 else 90) - 180 / np.pi * np.arctan(vav / uav)
    return average_wind_speed, average_wind_direction


# Compute the statistics for all the Weather variables (except wind)
def calculate_statistics_for_weather_variables(weather_obs, weather_stats_calculations: dict):
    """
    Note: to get the n-th percentile, use percentile(n)

    This function also returns the Weather dataframe indices. The corresponding Weather conditions in that Weather
    cell might cause wind-related Incidents.
    """
    if not weather_obs.empty:
        # Calculate the statistics
        weather_stats = weather_obs.groupby('WeatherCell').aggregate(weather_stats_calculations)
        weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
            calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)
        stats_info = weather_stats.values[0].tolist()  # + [weather_obs.index.tolist()]
    else:
        stats_info = [np.nan] * 10  # + [[None]]
    return stats_info


# Calculate the cover percents across two neighbouring ELRs
def calculate_overall_cover_percent_old(start_and_end_cover_percents: tuple, total_yards_adjusted: tuple):
    # (start * end) / (start + end)
    multiplier = pd.np.prod(total_yards_adjusted) / pd.np.sum(total_yards_adjusted)
    # 1/start, 1/end
    cp_start, cp_end = start_and_end_cover_percents
    s_, e_ = pd.np.divide(1, total_yards_adjusted)
    # numerator
    n = e_ * cp_start + s_ * cp_end
    # denominator
    d = pd.np.sum(start_and_end_cover_percents) if pd.np.all(start_and_end_cover_percents) else 1
    f = multiplier * pd.np.divide(n, d)
    overall_cover_percent = f * d
    return overall_cover_percent


# Define a function the categorises wind directions into four quadrants
def categorise_wind_directions(degree):
    if (degree >= 0) & (degree < 90):
        return 1
    elif (degree >= 90) & (degree < 180):
        return 2
    elif (degree >= 180) & (degree < 270):
        return 3
    else:  # (degree >= 270) & (degree < 360):
        return 4


def categorise_track_orientations_i(lon1, lat1, lon2, lat2):
    track_orientation_radians = np.arctan2(lat2 - lat1, lon2 - lon1)  # Angles in radians, [-pi, pi]
    if np.logical_or(
            np.logical_and(track_orientation_radians >= -np.pi * 2 / 3, track_orientation_radians < -np.pi / 3),
            np.logical_and(track_orientation_radians >= np.pi / 3, track_orientation_radians < np.pi * 2 / 3)):
        # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
        track_orientation = 'N_S'
    elif np.logical_or(
            np.logical_and(track_orientation_radians >= np.pi / 6, track_orientation_radians < np.pi / 3),
            np.logical_and(track_orientation_radians >= -np.pi * 5 / 6, track_orientation_radians < -np.pi * 2 / 3)):
        # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
        track_orientation = 'NE_SW'
    elif np.logical_or(
            np.logical_and(track_orientation_radians >= np.pi * 2 / 3, track_orientation_radians < np.pi * 5 / 6),
            np.logical_and(track_orientation_radians >= -np.pi / 3, track_orientation_radians < -np.pi / 6)):
        track_orientation = 'NW_SE'
    else:
        # np.logical_or(np.logical_or(
        #     np.logical_and(track_orientation_radians >= -np.pi, track_orientation_radians < -np.pi * 5 / 6),
        #     np.logical_and(track_orientation_radians >= -np.pi/6, track_orientation_radians < np.pi/6)),
        #     np.logical_and(track_orientation_radians >= np.pi*5/6, track_orientation_radians < np.pi))
        track_orientation = 'E_W'
    return track_orientation


def categorise_track_orientations(data):
    df = data.copy()
    df['Track_Orientation'] = None
    # origin = (-0.565409, 51.23622)
    lon1, lat1, lon2, lat2 = df.StartLongitude, df.StartLatitude, df.EndLongitude, df.EndLatitude
    df['Track_Orientation_radians'] = np.arctan2(lat2 - lat1, lon2 - lon1)  # Angles in radians, [-pi, pi]

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(df.Track_Orientation_radians >= -np.pi * 2 / 3, df.Track_Orientation_radians < -np.pi / 3),
        np.logical_and(df.Track_Orientation_radians >= np.pi / 3, df.Track_Orientation_radians < np.pi * 2 / 3))
    df.Track_Orientation[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(df.Track_Orientation_radians >= np.pi / 6, df.Track_Orientation_radians < np.pi / 3),
        np.logical_and(df.Track_Orientation_radians >= -np.pi * 5 / 6, df.Track_Orientation_radians < -np.pi * 2 / 3))
    df.Track_Orientation[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(df.Track_Orientation_radians >= np.pi * 2 / 3, df.Track_Orientation_radians < np.pi * 5 / 6),
        np.logical_and(df.Track_Orientation_radians >= -np.pi / 3, df.Track_Orientation_radians < -np.pi / 6))
    df.Track_Orientation[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    df.Track_Orientation.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi, df.Track_Orientation_radians < -np.pi * 5 / 6),
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi/6, df.Track_Orientation_radians < np.pi/6)),
    #     np.logical_and(df.Track_Orientation_radians >= np.pi*5/6, df.Track_Orientation_radians < np.pi))
    # data.Track_Orientation[e_w] = 'E-W'

    return df[['Track_Orientation']]
