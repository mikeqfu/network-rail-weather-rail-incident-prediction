""" Intermediate """

import functools
import os

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
from pyhelpers.dir import cd, cdd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.text import find_similar_str

import mssqlserver.metex

# ====================================================================================================================
""" Change directories """


# Change directory to "Modelling\\intermediate\\..." and sub-directories
def cdd_intermediate(*sub_dir):
    path = cdd("Models\\intermediate")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Modelling\\intermediate\\dat\\..." and sub-directories
def cd_intermediate_dat(*sub_dir):
    path = cdd_intermediate("dat")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\...\\Figures"
def cd_prototype_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\2 - Intermediate\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" """


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


# ====================================================================================================================


def get_data_by_season(mod_data, in_seasons, incident_datetime_col: str) -> pd.DataFrame:
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


# Get Angle of Line between Two Points
def get_angle_of_line_between(p1, p2, in_degrees=False):
    x_diff = p2.x - p1.x
    y_diff = p2.y - p1.y
    angle = np.arctan2(y_diff, x_diff)  # in radians
    if in_degrees:
        angle = np.degrees(angle)
    return angle


# Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
def categorise_track_orientations(attr_dat) -> pd.DataFrame:
    # Angles in radians, [-pi, pi]
    angles = attr_dat.apply(lambda x: get_angle_of_line_between(x.StartLonLat, x.EndLonLat), axis=1)
    track_orientation = pd.DataFrame({'Track_Orientation': None}, index=angles.index)

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(angles >= -np.pi * 2 / 3, angles < -np.pi / 3),
        np.logical_and(angles >= np.pi / 3, angles < np.pi * 2 / 3))
    track_orientation[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(angles >= np.pi / 6, angles < np.pi / 3),
        np.logical_and(angles >= -np.pi * 5 / 6, angles < -np.pi * 2 / 3))
    track_orientation[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(angles >= np.pi * 2 / 3, angles < np.pi * 5 / 6),
        np.logical_and(angles >= -np.pi / 3, angles < -np.pi / 6))
    track_orientation[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientation.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(angles >= -np.pi, angles < -np.pi * 5 / 6),
    #     np.logical_and(angles >= -np.pi/6, angles < np.pi/6)),
    #     np.logical_and(angles >= np.pi*5/6, angles < np.pi))
    # track_orientations[e_w] = 'E-W'

    categorical_var = pd.get_dummies(track_orientation, prefix='Track_Orientation')
    track_orientation_data = pd.concat([attr_dat, track_orientation, categorical_var], axis=1)

    return track_orientation_data


# Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30
def categorise_temperatures(attr_dat, column_name='Temperature_max') -> pd.DataFrame:
    temp_category = pd.cut(attr_dat[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf],
                           right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    temperature_category_data = pd.concat([attr_dat, temperature_category, categorical_var], axis=1)

    return temperature_category_data
