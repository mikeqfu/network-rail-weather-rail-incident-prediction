import functools
import os

import datetime_truncate
import geopandas as gpd
import geopy.distance
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.text import find_similar_str

import mssqlserver.metex
from utils import cd, cdd

# ====================================================================================================================
""" Change directories """


# Change directory to "Data\\Models\\prototype\\..." and sub-directories
def cdd_prototype(*sub_dir):
    path = cdd("Models\\prototype")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Models\\prototype\\dat" and sub-directories
def cd_prototype_dat(*sub_dir):
    path = cdd_prototype("dat")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures" and sub-directories
def cd_prototype_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\1 - Prototype\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Models\\intermediate\\..." and sub-directories
def cdd_intermediate(*sub_dir):
    path = cdd("Models\\intermediate")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Models\\intermediate\\dat" and sub-directories
def cd_intermediate_dat(*sub_dir):
    path = cdd_intermediate("dat")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures" and sub-directories
def cd_intermediate_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\2 - Intermediate\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Data manipulations and categorisation """


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


# Create shapely.points for StartLocations and EndLocations
def create_start_end_shapely_points(incidents_data):
    print("Creating shapely.geometry.Points for each incident location ... ", end="")
    data = incidents_data.copy()
    # Make shapely.geometry.points in longitude and latitude
    data.insert(data.columns.get_loc('StartLatitude') + 1, 'StartLonLat',
                gpd.points_from_xy(data.StartLongitude, data.StartLatitude))
    data.insert(data.columns.get_loc('EndLatitude') + 1, 'EndLonLat',
                gpd.points_from_xy(data.EndLongitude, data.EndLatitude))
    data.insert(data.columns.get_loc('EndLonLat') + 1, 'MidLonLat',
                data[['StartLonLat', 'EndLonLat']].apply(
                    lambda x: shapely.geometry.LineString([x.StartLonLat, x.EndLonLat]).centroid, axis=1))
    # Add Easting and Northing points  # Start
    start_xy = [wgs84_to_osgb36(data.StartLongitude[i], data.StartLatitude[i]) for i in data.index]
    data = pd.concat([data, pd.DataFrame(start_xy, columns=['StartEasting', 'StartNorthing'])], axis=1)
    data['StartXY'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    # End
    end_xy = [wgs84_to_osgb36(data.EndLongitude[i], data.EndLatitude[i]) for i in data.index]
    data = pd.concat([data, pd.DataFrame(end_xy, columns=['EndEasting', 'EndNorthing'])], axis=1)
    data['EndXY'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)
    # data[['StartEasting', 'StartNorthing']] = data[['StartLongitude', 'StartLatitude']].apply(
    #     lambda x: pd.Series(wgs84_to_osgb36(x.StartLongitude, x.StartLatitude)), axis=1)
    # data['StartEN'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    # data[['EndEasting', 'EndNorthing']] = data[['EndLongitude', 'EndLatitude']].apply(
    #     lambda x: pd.Series(wgs84_to_osgb36(x.EndLongitude, x.EndLatitude)), axis=1)
    # data['EndEN'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)
    print("Done.")
    return data


# Create a circle buffer for an incident location
def create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end, whisker_km=0.008, as_geom=True):
    """
    Ref: https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python

    :param midpoint: [shapely.geometry.point.Point] e.g. midpoint = incidents.MidLonLat.iloc[0]
    :param incident_start: [shapely.geometry.point.Point] e.g. incident_start = incidents.StartLonLat.iloc[0]
    :param incident_end: [shapely.geometry.point.Point] e.g. incident_end = incidents.EndLonLat.iloc[0]
    :param whisker_km: [num] an extended length to the diameter (i.e. on both sides of the start and end locations)
    :param as_geom: [bool]
    :return: [shapely.geometry.Polygon; list of tuples]

    """
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lon_0={lon} +lat_0={lat} +x_0=0 +y_0=0'
    project = functools.partial(
        pyproj.transform, pyproj.Proj(aeqd_proj.format(lon=midpoint.x, lat=midpoint.y)), pyproj.Proj(init='epsg:4326'))
    if incident_start != incident_end:
        radius_km = geopy.distance.distance(incident_start.coords, incident_end.coords).km / 2 + whisker_km
    else:
        radius_km = 2
    buffer = shapely.ops.transform(project, shapely.geometry.Point(0, 0).buffer(radius_km * 1000))
    buffer_circle = buffer if as_geom else buffer.exterior.coords[:]
    return buffer_circle


# Find all intersecting weather cells
def find_intersecting_weather_cells(x, as_geom=False):
    """
    :param x: [shapely.geometry.point.Point] e.g. x = incidents.Buffer_Zone.iloc[0]
    :param as_geom: [bool] whether to return shapely.geometry.Polygon of intersecting weather cells
    :return:
    """
    weather_cell_geoms = mssqlserver.metex.get_weather_cell().Polygon_WGS84
    intxn_weather_cells = tuple(cell for cell in weather_cell_geoms if x.intersects(cell))
    if as_geom:
        return intxn_weather_cells
    else:
        intxn_weather_cell_ids = tuple(weather_cell_geoms[weather_cell_geoms == cell].index[0]
                                       for cell in intxn_weather_cells)
        if len(intxn_weather_cell_ids) == 1:
            intxn_weather_cell_ids = intxn_weather_cell_ids[0]
        return intxn_weather_cell_ids


# Illustration of the buffer circle
def illustrate_buffer_circle(midpoint, incident_start, incident_end, whisker_km=0.008, legend_loc='best'):
    """
    :param midpoint: e.g. midpoint = incidents.MidLonLat.iloc[2]
    :param incident_start: e.g. incident_start = incidents.StartLonLat.iloc[2]
    :param incident_end: e.g. incident_end = incidents.EndLonLat.iloc[2]
    :param whisker_km:
    :param legend_loc:
    """
    buffer_circle = create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end, whisker_km)
    i_weather_cells = find_intersecting_weather_cells(buffer_circle, as_geom=True)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for g in i_weather_cells:
        x, y = g.exterior.xy
        ax.plot(x, y, color='#433f3f')
        polygons = matplotlib.patches.Polygon(g.exterior.coords[:], fc='#D5EAFF', ec='#4b4747', alpha=0.5)
        plt.gca().add_patch(polygons)
    ax.plot([], 's', label="Weather cell", ms=16, color='#D5EAFF', markeredgecolor='#4b4747')

    x_, y_ = buffer_circle.exterior.xy
    ax.plot(x_, y_)

    sx, sy, ex, ey = incident_start.xy + incident_end.xy
    if incident_start == incident_end:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
    else:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
        ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')
    ax.set_xlabel('Longitude')  # ax.set_xlabel('Easting')
    ax.set_ylabel('Latitude')  # ax.set_ylabel('Northing')
    font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
    legend = plt.legend(numpoints=1, loc=legend_loc, prop=font, fancybox=True, labelspacing=0.5)
    frame = legend.get_frame()
    frame.set_edgecolor('k')
    plt.tight_layout()


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


# (an alternative to the above)
def get_data_by_season_(mod_data, in_seasons, incident_datetime_col: str) -> pd.DataFrame:
    """
    :param mod_data: [pd.DataFrame]
    :param in_seasons: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :param incident_datetime_col: [str]
    :return:
    """
    if in_seasons is None:
        return mod_data
    else:
        seasons = [in_seasons] if isinstance(in_seasons, str) else in_seasons
        selected_seasons = [find_similar_str(s, ('Spring', 'Summer', 'Autumn', 'Winter'))[0] for s in seasons]

        def identify_season(incident_dt):
            """
            # (incident_datetime.dt.month % 12 + 3) // 3
            """
            y = incident_dt.year
            seasons_dt = [('Winter', (pd.datetime(y, 1, 1), pd.datetime(y, 3, 20) + pd.DateOffset(days=1))),
                          ('Spring', (pd.datetime(y, 3, 21), pd.datetime(y, 6, 20) + pd.DateOffset(days=1))),
                          ('Summer', (pd.datetime(y, 6, 21), pd.datetime(y, 9, 22) + pd.DateOffset(days=1))),
                          ('Autumn', (pd.datetime(y, 9, 23), pd.datetime(y, 12, 20) + pd.DateOffset(days=1))),
                          ('Winter', (pd.datetime(y, 12, 21), pd.datetime(y, 12, 31) + pd.DateOffset(days=1)))]
            return next(season for season, (start, end) in seasons_dt if start <= incident_dt < end)

        mod_data_seasons = mod_data[incident_datetime_col].map(identify_season)
        season_data = mod_data[mod_data_seasons.isin(selected_seasons)]

        return season_data


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


# Get Angle of Line between Two Points
def get_angle_of_line_between(p1, p2, in_degrees=False):
    x_diff = p2.x - p1.x
    y_diff = p2.y - p1.y
    angle = np.arctan2(y_diff, x_diff)  # in radians
    if in_degrees:
        angle = np.degrees(angle)
    return angle


# Track orientations
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


# (an alternative to the above)
def categorise_track_orientations_(lon1, lat1, lon2, lat2):
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


# Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30
def categorise_temperatures(attr_dat, column_name='Temperature_max') -> pd.DataFrame:
    temp_category = pd.cut(attr_dat[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf],
                           right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    temperature_category_data = pd.concat([attr_dat, temperature_category, categorical_var], axis=1)

    return temperature_category_data


# ====================================================================================================================
""" Calculations """


# Get weather variable names
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
def calculate_prototype_weather_statistics(weather_obs, weather_stats_calculations: dict):
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


# Get weather variable names for intermediate model
def get_intermediate_weather_variable_names(weather_stats_calculations: dict):
    stats_names = [x + '_max' for x in weather_stats_calculations.keys()]
    stats_names[stats_names.index('TotalPrecipitation_max')] = 'TotalPrecipitation_sum'
    stats_names[stats_names.index('Snowfall_max')] = 'Snowfall_sum'
    stats_names.insert(stats_names.index('Temperature_max') + 1, 'Temperature_min')
    stats_names.insert(stats_names.index('Temperature_min') + 1, 'Temperature_avg')
    stats_names.insert(stats_names.index('Temperature_avg') + 1, 'Temperature_dif')
    wind_speed_variables = ['WindSpeed_avg', 'WindDirection_avg']
    weather_variable_names = stats_names + wind_speed_variables + ['Hottest_Heretofore']
    return weather_variable_names


# Get the highest temperature of year by far
def get_highest_temperature_of_year_by_far(weather_cell_id, period_start_dt):
    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    yr_start_dt = datetime_truncate.truncate_year(period_start_dt)
    # Get weather observations
    weather_obs = mssqlserver.metex.fetch_weather_by_id_datetime(
        weather_cell_id, yr_start_dt, period_start_dt, pickle_it=False, dat_dir=cd_intermediate_dat("weather-slices"))
    #
    weather_obs_by_far = weather_obs[(weather_obs.DateTime < period_start_dt) & (weather_obs.DateTime > yr_start_dt)]
    #
    highest_temperature = weather_obs_by_far.Temperature.max()
    return highest_temperature


# Calculate the statistics for the weather-related variables (except radiation)
def calculate_intermediate_weather_statistics(weather_obs, weather_stats_calculations, values_only=True):
    if weather_obs.empty:
        weather_stats = [np.nan] * (sum(map(np.count_nonzero, weather_stats_calculations.values())) + 4)
        if not values_only:
            weather_stats = pd.DataFrame(weather_stats, columns=get_intermediate_weather_variable_names())
    else:
        # Create a pseudo id for groupby() & aggregate()
        weather_obs['Pseudo_ID'] = 0
        # Calculate basic statistics
        weather_stats = weather_obs.groupby('Pseudo_ID').aggregate(weather_stats_calculations)
        # Calculate average wind speeds and directions
        weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
            calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)
        # Lowest temperature between the time of the highest temperature and 00:00
        highest_temp_dt = weather_obs[
            weather_obs.Temperature == weather_stats.Temperature['max'][0]].DateTime.min()
        weather_stats.Temperature['min'] = weather_obs[
            weather_obs.DateTime < highest_temp_dt].Temperature.min()
        # Temperature change between the the highest and lowest temperatures
        weather_stats.insert(3, 'Temperature_dif',
                             weather_stats.Temperature['max'] - weather_stats.Temperature['min'])
        # Find out weather cell ids
        weather_cell_obs = weather_obs.WeatherCell.unique()
        weather_cell_id = weather_cell_obs[0] if len(weather_cell_obs) == 1 else tuple(weather_cell_obs)
        obs_start_dt = weather_obs.DateTime.min()  # Observation start datetime
        # Whether it is the hottest of the year by far
        highest_temp = get_highest_temperature_of_year_by_far(weather_cell_id, obs_start_dt)
        highest_temp_obs = weather_stats.Temperature['max'][0]
        weather_stats['Hottest_Heretofore'] = 1 if highest_temp_obs >= highest_temp else 0
        weather_stats.columns = get_intermediate_weather_variable_names()
        # Scale up variable
        scale_up_vars = ['WindSpeed_max', 'WindGust_max', 'WindSpeed_avg',
                         'RelativeHumidity_max', 'Snowfall_sum']
        weather_stats[scale_up_vars] = weather_stats[scale_up_vars] / 10.0
        weather_stats.index.name = None
        if values_only:
            weather_stats = weather_stats.values[0].tolist()
    return weather_stats
