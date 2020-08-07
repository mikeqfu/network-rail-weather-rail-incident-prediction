import functools
import itertools

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

from mssqlserver import metex
from weather import midas, ukcp


def find_weather_cell_id(longitude, latitude):
    """
    Find weather cell ID.

    :param longitude: longitude
    :type longitude: int, float
    :param latitude: latitude
    :type latitude: int, float
    :return: list, int
    """

    weather_cell = metex.get_weather_cell()

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


def create_circle_buffer_upon_weather_cell(midpoint, start_loc, end_loc, whisker_km=0.008, as_geom=True):
    """
    Create a circle buffer for an incident location.

    See also [`CCBUWC <https://gis.stackexchange.com/questions/289044/>`_]

    :param midpoint: midpoint or centre
    :type midpoint: shapely.geometry.Point
    :param start_loc: start location of an incident
    :type start_loc: shapely.geometry.Point
    :param end_loc: end location of an incident
    :type end_loc: shapely.geometry.Point
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

    if start_loc != end_loc:
        radius_km = geopy.distance.distance(start_loc.coords, end_loc.coords).km / 2 + whisker_km
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

    weather_cell_geoms = metex.get_weather_cell().Polygon_WGS84
    intxn_weather_cells = tuple(cell for cell in weather_cell_geoms if x.intersects(cell))
    if as_geom:
        return intxn_weather_cells
    else:
        intxn_weather_cell_ids = tuple(weather_cell_geoms[weather_cell_geoms == cell].index[0]
                                       for cell in intxn_weather_cells)
        if len(intxn_weather_cell_ids) == 1:
            intxn_weather_cell_ids = intxn_weather_cell_ids[0]
        return intxn_weather_cell_ids


def illustrate_buffer_circle_on_weather_cell(midpoint, start_loc, end_loc, whisker_km=0.008, legend_pos='best'):
    """
    Illustration of the buffer circle.

    :param midpoint: e.g. midpoint = incidents.MidLonLat.iloc[2]
    :type midpoint:
    :param start_loc: e.g. incident_start = incidents.StartLonLat.iloc[2]
    :type start_loc:
    :param end_loc: e.g. incident_end = incidents.EndLonLat.iloc[2]
    :type end_loc:
    :param whisker_km: defaults to ``0.008``
    :type whisker_km: float
    :param legend_pos: defaults to ``'best'``
    :type legend_pos: str
    """

    buffer_circle = create_circle_buffer_upon_weather_cell(midpoint, start_loc, end_loc, whisker_km)
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

    sx, sy, ex, ey = start_loc.xy + end_loc.xy
    if start_loc == end_loc:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
    else:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
        ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')
    ax.set_xlabel('Longitude')  # ax.set_xlabel('Easting')
    ax.set_ylabel('Latitude')  # ax.set_ylabel('Northing')
    font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
    legend = plt.legend(numpoints=1, loc=legend_pos, prop=font, fancybox=True, labelspacing=0.5)
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


def find_closest_weather_grid(x, obs_grids, obs_centroid_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.StartNE.iloc[0]
    :param obs_grids:
    :param obs_centroid_geom:
    :return:

    **Example**::

        import copy

        x = incidents.StartXY.iloc[0]
    """

    x_ = shapely.ops.nearest_points(x, obs_centroid_geom)[1]

    pseudo_id = [i for i, y in enumerate(obs_grids.Centroid_XY) if y.equals(x_)]

    return pseudo_id[0]


def create_circle_buffer_upon_weather_grid(start, end, midpoint, whisker=500):
    """
    Create a circle buffer for start/end location.

    :param start:
    :type start: shapely.geometry.Point
    :param end:
    :type end: shapely.geometry.Point
    :param midpoint:
    :type midpoint: shapely.geometry.Point
    :param whisker: extended length on both sides of the start and end locations, defaults to ``500``
    :type whisker: int
    :return: a buffer zone
    :rtype: shapely.geometry.Polygon

    **Example**::

        whisker = 0

        start = incidents.StartXY.iloc[0]
        end = incidents.EndXY.iloc[0]
        midpoint = incidents.MidpointXY.iloc[0]
    """

    if start == end:
        buffer_circle = start.buffer(2000 + whisker)
    else:
        radius = (start.distance(end) + whisker) / 2
        buffer_circle = midpoint.buffer(radius)
    return buffer_circle


def find_intersecting_weather_grid(x, obs_grids, obs_grids_geom, as_grid_id=True):
    """
    Find all intersecting geom objects.

    :param x:
    :param obs_grids:
    :param obs_grids_geom:
    :param as_grid_id: whether to return grid id number
    :type as_grid_id: bool
    :return:

    **Example**::

        x = incidents.Buffer_Zone.iloc[0]
        as_grid_id = True
    """

    intxn_grids = [grid for grid in obs_grids_geom if x.intersects(grid)]

    if as_grid_id:
        x_ = shapely.ops.cascaded_union(intxn_grids)
        intxn_grids = [i for i, y in enumerate(obs_grids.Grid) if y.within(x_)]

    return intxn_grids


def find_closest_met_stn(x, met_stations, met_stations_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x:
    :param met_stations:
    :param met_stations_geom:
    :return:

    **Example**::

        x = incidents.MidpointXY.iloc[0]
    """

    x_1 = shapely.ops.nearest_points(x, met_stations_geom)[1]

    # rest = shapely.geometry.MultiPoint([p for p in met_stations_geom if not p.equals(x_1)])
    # x_2 = shapely.ops.nearest_points(x, rest)[1]
    # rest = shapely.geometry.MultiPoint([p for p in rest if not p.equals(x_2)])
    # x_3 = shapely.ops.nearest_points(x, rest)[1]

    idx = [i for i, y in enumerate(met_stations.EN_GEOM) if y.equals(x_1)]
    src_id = met_stations.index[idx].to_list()

    return src_id


def specify_weather_stats_calculations():
    """
    Specify the statistics needed for Weather observations (except radiation).

    :return:
    """

    weather_stats_calculations = {'Maximum_Temperature': (max, min, np.average),
                                  'Minimum_Temperature': (max, min, np.average),
                                  'Temperature_Change': np.average,
                                  'Precipitation': (max, min, np.average)}

    return weather_stats_calculations


def specify_weather_variable_names(calc):
    """
    Get all weather variable names.

    :param calc:
    :return:
    """

    var_stats_names = [[k, [i.__name__ for i in v] if isinstance(v, tuple) else [v.__name__]]
                       for k, v in calc.items()]

    weather_variable_names = [['_'.join([x, z]) for z in y] for x, y in var_stats_names]

    weather_variable_names = list(itertools.chain.from_iterable(weather_variable_names))

    return weather_variable_names


def calculate_weather_stats(weather_data):
    """
    Calculate the statistics for the Weather variables (except radiation).

    :param weather_data:
    :return:
    """

    # Specify calculations
    weather_stats_computations = specify_weather_stats_calculations()

    if weather_data.empty:
        weather_stats_info = [np.nan] * sum(map(np.count_nonzero, weather_stats_computations.values()))

    else:
        # Create a pseudo id for groupby() & aggregate()
        weather_data['Pseudo_ID'] = 0
        weather_stats = weather_data.groupby('Pseudo_ID').aggregate(weather_stats_computations)
        # a, b = [list(x) for x in weather_stats.columns.levels]
        # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
        # if not weather_stats.empty:
        #     stats_info = weather_stats.values[0].tolist()
        # else:
        #     stats_info = [np.nan] * len(weather_stats.columns)
        weather_stats_info = weather_stats.values[0].tolist()

    return weather_stats_info


def integrate_pip_ukcp09_data(grids, period):
    """
    Gather gridded weather observations of the given period for each incident record.

    :param grids: e.g. grids = incidents.Weather_Grid.iloc[0]
    :param period: e.g. period = incidents.Critical_Period.iloc[0]
    :return:

    **Example**::

        grids = incidents.Weather_Grid.iloc[0]
        period = incidents.Critical_Period.iloc[0]

    """

    # Find Weather data for the specified period
    prior_ip_weather = ukcp.query_ukcp09_by_grid_datetime(grids, period, pickle_it=True)
    # Calculate the max/min/avg for Weather parameters during the period
    weather_stats = calculate_weather_stats(prior_ip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = ukcp.query_ukcp09_by_grid_datetime_(grids, period, pickle_it=True)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


def integrate_nip_ukcp09_data(grids, period, prior_ip_data, stanox_section):
    """
    Gather gridded Weather observations of the corresponding non-incident period for each incident record.

    :param grids:
    :param period:
    :param stanox_section:
    :param prior_ip_data:
    :return:

    **Example**::

        grids = non_ip_data.Weather_Grid.iloc[0]
        period = non_ip_data.Critical_Period.iloc[0]
        stanox_section = non_ip_data.StanoxSection.iloc[0]
    """

    # Get non-IP Weather data about where and when the incident occurred
    nip_weather = ukcp.query_ukcp09_by_grid_datetime(grids, period, pickle_it=True)

    # Get all incident period data on the same section
    ip_overlap = prior_ip_data[
        (prior_ip_data.StanoxSection == stanox_section) &
        (((prior_ip_data.Critical_StartDateTime <= period.left.to_pydatetime()[0]) &
          (prior_ip_data.Critical_EndDateTime >= period.left.to_pydatetime()[0])) |
         ((prior_ip_data.Critical_StartDateTime <= period.right.to_pydatetime()[0]) &
          (prior_ip_data.Critical_EndDateTime >= period.right.to_pydatetime()[0])))]
    # Skip data of Weather causing Incidents at around the same time; but
    if not ip_overlap.empty:
        nip_weather = nip_weather[
            (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
            (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
    # Get the max/min/avg Weather parameters for those incident periods
    weather_stats = calculate_weather_stats(nip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = ukcp.query_ukcp09_by_grid_datetime_(grids, period, pickle_it=True)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


def specify_radtob_stats_calculations():
    """
    Specify the statistics needed for radiation only.

    :return:
    """

    radtob_stats_calculations = {'GLBL_IRAD_AMT': sum}
    return radtob_stats_calculations


def calculate_radtob_variables_stats(midas_radtob):
    """
    Calculate the statistics for the radiation variables.

    :param midas_radtob:
    :return:

    **Example**::

        midas_radtob = prior_ip_radtob.copy()
    """

    # Solar irradiation amount (Kjoules/ sq metre over the observation period)
    radtob_stats_calculations = specify_radtob_stats_calculations()
    if midas_radtob.empty:
        stats_info = [np.nan] * (sum(map(np.count_nonzero, radtob_stats_calculations.values())))

    else:
        # if 24 not in midas_radtob.OB_HOUR_COUNT:
        #     midas_radtob = midas_radtob.append(midas_radtob.iloc[-1, :])
        #     midas_radtob.VERSION_NUM.iloc[-1] = 0
        #     midas_radtob.OB_HOUR_COUNT.iloc[-1] = midas_radtob.OB_HOUR_COUNT.iloc[0:-1].sum()
        #     midas_radtob.GLBL_IRAD_AMT.iloc[-1] = midas_radtob.GLBL_IRAD_AMT.iloc[0:-1].sum()

        if 24 in midas_radtob.OB_HOUR_COUNT.to_list():
            temp = midas_radtob[midas_radtob.OB_HOUR_COUNT == 24]
            midas_radtob = pd.concat([temp, midas_radtob.loc[temp.last_valid_index() + 1:]])

        radtob_stats = midas_radtob.groupby('SRC_ID').aggregate(radtob_stats_calculations)
        stats_info = radtob_stats.values.flatten().tolist()

    return stats_info


def integrate_pip_midas_radtob(met_stn_id, period, route_name, use_suppl_dat):
    """
    Gather solar radiation of the given period for each incident record.

    :param met_stn_id:
    :param period:
    :param route_name:
    :param use_suppl_dat:
    :return:

    **Example**::

        met_stn_id = incidents.Met_SRC_ID.iloc[1]
        period = incidents.Critical_Period.iloc[1]
        use_suppl_dat = False
    """

    # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
    #
    # try:
    #     prior_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
    # except KeyError:
    #     prior_ip_radtob = pd.DataFrame()

    prior_ip_radtob = midas.query_midas_radtob_by_grid_datetime(met_stn_id, period, route_name, use_suppl_dat,
                                                                pickle_it=True)

    radtob_stats = calculate_radtob_variables_stats(prior_ip_radtob)

    return radtob_stats


def integrate_nip_midas_radtob(met_stn_id, period, route_name, use_suppl_dat, prior_ip_data, stanox_section):
    """
    Gather solar radiation of the corresponding non-incident period for each incident record.

    :param met_stn_id: e.g. met_stn_id = non_ip_data.Met_SRC_ID.iloc[1]
    :param period: e.g. period = non_ip_data.Critical_Period.iloc[1]
    :param route_name:
    :param use_suppl_dat:
    :param stanox_section: e.g. location = non_ip_data.StanoxSection.iloc[0]
    :param prior_ip_data:
    :return:
    """

    # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
    #
    # try:
    #     non_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
    # except KeyError:
    #     non_ip_radtob = pd.DataFrame()

    non_ip_radtob = midas.query_midas_radtob_by_grid_datetime(met_stn_id, period, route_name, use_suppl_dat,
                                                              pickle_it=True)

    # Get all incident period data on the same section
    ip_overlap = prior_ip_data[
        (prior_ip_data.StanoxSection == stanox_section) &
        (((prior_ip_data.Critical_StartDateTime <= period.left.to_pydatetime()[0]) &
          (prior_ip_data.Critical_EndDateTime >= period.left.to_pydatetime()[0])) |
         ((prior_ip_data.Critical_StartDateTime <= period.right.to_pydatetime()[0]) &
          (prior_ip_data.Critical_EndDateTime >= period.right.to_pydatetime()[0])))]
    # Skip data of Weather causing Incidents at around the same time; but
    if not ip_overlap.empty:
        non_ip_radtob = non_ip_radtob[
            (non_ip_radtob.OB_END_DATE < min(ip_overlap.Critical_StartDateTime)) |
            (non_ip_radtob.OB_END_DATE > max(ip_overlap.Critical_EndDateTime))]

    radtob_stats = calculate_radtob_variables_stats(non_ip_radtob)

    return radtob_stats
