""" Tools for modelling. """

import functools

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
from pyhelpers.dir import cd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.text import find_similar_str

import mssqlserver.metex
from utils import cdd_models


# == Change directories ===============================================================================

def cdd_prototype(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_models("prototype", *sub_dir, mkdir=mkdir)

    return path


def cd_prototype_dat(*sub_dir, mkdir=True):
    """
    Change directory to "..\\data\\models\\prototype\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("dat", *sub_dir, mkdir=mkdir)

    return path


def cd_prototype_fig_pub(*sub_dir, mkdir=False):
    """
    Change directory to "docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures\\" and sub-directories /
    a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures\\" and sub-directories /
        a file
    :rtype: str
    """

    path = cd("docs\\5 - Publications\\1 - Prototype\\0 - Ingredients\\1 - Figures", *sub_dir, mkdir=mkdir)

    return path


def cdd_intermediate(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_models("intermediate", *sub_dir, mkdir=mkdir)

    return path


def cd_intermediate_dat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("dat", *sub_dir, mkdir=mkdir)

    return path


def cd_intermediate_fig_pub(*sub_dir, mkdir=False):
    """
    Change directory to "docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures\\"
    and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures\\"
        and sub-directories / a file
    :rtype: str
    """

    path = cd("docs\\5 - Publications\\2 - Intermediate\\0 - Ingredients\\1 - Figures", *sub_dir, mkdir=mkdir)

    return path


# == Data manipulations ===============================================================================

def find_weather_cell_id(longitude, latitude):
    """
    Find weather cell ID.

    :param longitude: longitude
    :type longitude: int, float
    :param latitude: latitude
    :type latitude: int, float
    :return: list, int
    """

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


def create_start_end_shapely_points(incidents_data, verbose=False):
    """
    Create shapely.points for 'StartLocation's and 'EndLocation's.

    :param incidents_data: data of incident records
    :type incidents_data: pandas.DataFrame
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: incident data with shapely.geometry.Points of start and end locations
    """

    print("Creating shapely.geometry.Points for each incident location ... ", end="") if verbose else ""
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
    print("Done.") if verbose else ""
    return data


def create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end, whisker_km=0.008, as_geom=True):
    """
    Create a circle buffer for an incident location.

    See also [`CCBUWC <https://gis.stackexchange.com/questions/289044/>`_]

    :param midpoint: midpoint or centre
    :type midpoint: shapely.geometry.Point
    :param incident_start: start location of an incident
    :type incident_start: shapely.geometry.Point
    :param incident_end: end location of an incident
    :type incident_end: shapely.geometry.Point
    :param whisker_km: extended length to diameter (i.e. on both sides of start/end locations), defaults to ``0.008``
    :type whisker_km: int, float
    :param as_geom: whether to return the buffer circle as shapely.geometry.Polygon, defaults to ``True``
    :type as_geom: bool
    :return: a buffer circle
    :rtype: shapely.geometry.Polygon; list of tuples

    **Example**::

        from models.tools import create_circle_buffer_upon_weather_cell

        midpoint = incidents.MidLonLat.iloc[0]
        incident_start = incidents.StartLonLat.iloc[0]
        incident_end = incidents.EndLonLat.iloc[0]

        whisker_km = 0.008
        as_geom = True
        buffer_circle = create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end,
                                                               whisker_km, as_geom)

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


def find_intersecting_weather_cells(x, as_geom=False):
    """
    Find all intersecting weather cells.

    :param x: e.g. x = incidents.Buffer_Zone.iloc[0]
    :type: x: shapely.geometry.Point
    :param as_geom: whether to return shapely.geometry.Polygon of intersecting weather cells
    :type as_geom: bool
    :return: intersecting weather cells
    :rtype: tuple

    **Example**::

        x = incidents.Buffer_Zone.iloc[0]

        as_geom = False
        intxn_weather_cell_ids = find_intersecting_weather_cells(x, as_geom)

        as_geom = True
        intxn_weather_cell_ids = find_intersecting_weather_cells(x, as_geom)
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


def illustrate_buffer_circle(midpoint, incident_start, incident_end, whisker_km=0.008, legend_loc='best'):
    """
    Illustration of the buffer circle.

    :param midpoint: e.g. midpoint = incidents.MidLonLat.iloc[2]
    :type midpoint:
    :param incident_start: e.g. incident_start = incidents.StartLonLat.iloc[2]
    :type incident_start:
    :param incident_end: e.g. incident_end = incidents.EndLonLat.iloc[2]
    :type incident_end:
    :param whisker_km: defaults to ``0.008``
    :type whisker_km: float
    :param legend_loc: defaults to ``'best'``
    :type legend_loc: str
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


def get_angle_of_line_between(p1, p2, in_degrees=False):
    """
    Get Angle of Line between two points.

    :param p1: a point
    :type p1:
    :param p2: another point
    :type p2:
    :param in_degrees: whether return a value in degrees, defaults to ``False``
    :type in_degrees: bool
    :return:
    :rtype:
    """

    x_diff = p2.x - p1.x
    y_diff = p2.y - p1.y
    angle = np.arctan2(y_diff, x_diff)  # in radians
    if in_degrees:
        angle = np.degrees(angle)
    return angle


# == Data categorisations =============================================================================

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
        season = [find_similar_str(s, seasons) for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data

    else:
        return incident_data


def get_data_by_season_(mod_data, in_seasons, incident_datetime_col):
    """
    An alternative to ``get_data_by_season()``.

    :param mod_data:
    :type mod_data: pandas.DataFrame
    :param in_seasons: 'spring', 'summer', 'autumn', 'winter'; if ``None``, returns data of all seasons
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
        selected_seasons = [find_similar_str(s, ('Spring', 'Summer', 'Autumn', 'Winter')) for s in seasons]

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


def categorise_track_orientations(data, start_lon_colname, start_lat_colname, end_lon_colname, end_lat_colname):
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
    """

    track_orientations = pd.DataFrame(None, index=range(len(data)), columns=['Track_Orientation'])
    # origin = (-0.565409, 51.23622)
    start_lon, start_lat = data[start_lon_colname], data[start_lat_colname]
    end_lon, end_lat = data[end_lon_colname], data[end_lat_colname]
    track_orientations['Track_Orientation_radians'] = np.arctan2(end_lat - start_lat, end_lon - start_lon)  # [-pi, pi]

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 2 / 3))
    track_orientations.Track_Orientation[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi / 6,
                       track_orientations.Track_Orientation_radians < np.pi / 3),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi * 5 / 6,
                       track_orientations.Track_Orientation_radians < -np.pi * 2 / 3))
    track_orientations.Track_Orientation[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(track_orientations.Track_Orientation_radians >= np.pi * 2 / 3,
                       track_orientations.Track_Orientation_radians < np.pi * 5 / 6),
        np.logical_and(track_orientations.Track_Orientation_radians >= -np.pi / 3,
                       track_orientations.Track_Orientation_radians < -np.pi / 6))
    track_orientations.Track_Orientation[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations.Track_Orientation.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi, df.Track_Orientation_radians < -np.pi * 5 / 6),
    #     np.logical_and(df.Track_Orientation_radians >= -np.pi/6, df.Track_Orientation_radians < np.pi/6)),
    #     np.logical_and(df.Track_Orientation_radians >= np.pi*5/6, df.Track_Orientation_radians < np.pi))
    # data.Track_Orientation[e_w] = 'E-W'

    categorical_var = pd.get_dummies(track_orientations.Track_Orientation, prefix='Track_Orientation', prefix_sep='_')

    track_orientations = pd.concat([track_orientations[['Track_Orientation']], categorical_var], axis=1)

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

    temp_category = pd.cut(attr_dat[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf], right=False,
                           include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    temperature_category_data = pd.concat([temperature_category, categorical_var], axis=1)

    return temperature_category_data


# == Calculations =====================================================================================

def get_weather_variable_names(weather_stats_calculations, temperature_dif=False, supplement=None):
    """
    Get weather variable names.

    :param weather_stats_calculations:
    :type weather_stats_calculations: dict
    :param temperature_dif: whether to include 'Temperature_dif', defaults to ``False``
    :type temperature_dif: bool
    :param supplement: e.g. 'Hottest_Heretofore'
    :type supplement: str, list, None
    :return: a list of names of weather variables
    :rtype: list
    """

    weather_variable_names = []
    for k, v in weather_stats_calculations.items():
        if isinstance(v, tuple):
            for v_ in v:
                weather_variable_names.append('_'.join(
                    [k, v_.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))
        else:
            weather_variable_names.append('_'.join(
                [k, v.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))
    if temperature_dif:
        weather_variable_names.insert(weather_variable_names.index('Temperature_min') + 1, 'Temperature_dif')

    if supplement:
        if isinstance(supplement, str):
            supplement = [supplement]
        wind_variable_names_ = weather_variable_names + ['WindSpeed_avg', 'WindDirection_avg'] + supplement
    else:
        wind_variable_names_ = weather_variable_names + ['WindSpeed_avg', 'WindDirection_avg']

    return wind_variable_names_


def calculate_overall_cover_percent_old(start_and_end_cover_percents, total_yards_adjusted):
    """
    Calculate the cover percents across two neighbouring ELRs.

    :param start_and_end_cover_percents:
    :type start_and_end_cover_percents: tuple
    :param total_yards_adjusted:
    :type total_yards_adjusted: tuple
    :return:
    :rtype:
    """

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


def calculate_wind_averages(wind_speeds, wind_directions):
    """
    Calculate average wind speed and direction.

    :param wind_speeds:
    :type wind_speeds:
    :param wind_directions:
    :type wind_directions:
    :return:
    :rtype:
    """

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


def calculate_prototype_weather_statistics(weather_obs, weather_stats_calculations):
    """
    Compute the statistics for all the Weather variables (except wind).

    :param weather_obs:
    :type weather_obs:
    :param weather_stats_calculations:
    :type weather_stats_calculations: dict
    :return:
    :rtype:

    .. note::

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


def get_highest_temperature_of_year_by_far(weather_cell_id, period_start_dt):
    """
    Get the highest temperature of year by far.

    :param weather_cell_id:
    :type weather_cell_id:
    :param period_start_dt:
    :type period_start_dt:
    :return:
    :rtype:
    """

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    yr_start_dt = datetime_truncate.truncate_year(period_start_dt)
    # Get weather observations
    weather_obs = mssqlserver.metex.query_weather_by_id_datetime(
        weather_cell_id, yr_start_dt, period_start_dt, pickle_it=False, dat_dir=cd_intermediate_dat("weather-slices"))
    #
    weather_obs_by_far = weather_obs[(weather_obs.DateTime <= period_start_dt) & (weather_obs.DateTime >= yr_start_dt)]
    #
    highest_temperature = weather_obs_by_far.Temperature.max()
    return highest_temperature


def calculate_intermediate_weather_statistics(weather_obs, weather_stats_calculations, values_only=True):
    """
    Calculate the statistics for the weather-related variables (except radiation).

    :param weather_obs:
    :type weather_obs:
    :param weather_stats_calculations:
    :type weather_stats_calculations:
    :param values_only: defaults to ``True``
    :type values_only: bool
    :return:
    :rtype:

    **Example**::

        weather_obs = ip_weather
        weather_stats_calculations = weather_statistics_calculations
        values_only = True
    """

    if weather_obs.empty:
        weather_variable_names = get_weather_variable_names(weather_stats_calculations)
        weather_stats = [np.nan] * len(weather_variable_names)
        if not values_only:
            weather_stats = pd.DataFrame(np.array(weather_stats).reshape((1, len(weather_stats))),
                                         columns=weather_variable_names)
    else:
        # Create a pseudo id for groupby() & aggregate()
        weather_obs['Pseudo_ID'] = 0
        # Calculate summary statistics
        weather_stats = weather_obs.groupby('Pseudo_ID').aggregate(weather_stats_calculations)
        # Rename columns
        weather_stats.columns = ['_'.join(x).replace('nan', '').replace('mean', 'avg').replace('median', 'med')
                                 for x in weather_stats.columns]

        # Calculate average wind speeds and directions
        weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
            calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)

        # Lowest temperature between the time of the highest temperature and weather_obs start
        highest_temp_dt = weather_obs[weather_obs.Temperature == weather_stats.Temperature_max.values[0]].DateTime.min()
        weather_stats.Temperature_min = weather_obs[weather_obs.DateTime < highest_temp_dt].Temperature.min()
        # Temperature change between the the highest and lowest temperatures
        weather_stats.insert(weather_stats.columns.get_loc('Temperature_min') + 1, 'Temperature_dif',
                             weather_stats.Temperature_max - weather_stats.Temperature_min)

        # Find out weather cell ids
        obs_weather_cells = weather_obs.WeatherCell.unique()
        weather_cell_id = obs_weather_cells[0] if len(obs_weather_cells) == 1 else tuple(obs_weather_cells)
        obs_start_dt = weather_obs.DateTime.min()  # Observation start datetime

        # Whether it is the hottest of the year by far
        highest_temp = get_highest_temperature_of_year_by_far(weather_cell_id, obs_start_dt)
        weather_stats['Hottest_Heretofore'] = 1 if weather_stats.Temperature_max.values[0] >= highest_temp else 0

        # # Scale up variable
        # scale_up_vars = ['WindSpeed_max', 'WindGust_max', 'WindSpeed_avg', 'RelativeHumidity_max', 'Snowfall_sum']
        # weather_stats[scale_up_vars] = weather_stats[scale_up_vars] / 10.0
        weather_stats.index.name = None

        if values_only:
            weather_stats = weather_stats.values[0].tolist()

    return weather_stats
