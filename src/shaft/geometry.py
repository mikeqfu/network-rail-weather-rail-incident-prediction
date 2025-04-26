"""
Manipulation of geometric data.
"""

import functools
import glob
import itertools
import os
import re
import shutil

import geopy.distance
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops
from pydriosm import GeofabrikDownloader, GeofabrikReader, SHP
from pydriosm.errors import InvalidSubregionNameError
from pyhelpers.dirs import cdd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.store import load_pickle, save_pickle

from src.utils import points_from_xy


# == Weather grid ==

def find_closest_weather_grid(x, obs_grids, obs_centroid_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.StartNE.iloc[0]
    :param obs_grids:
    :param obs_centroid_geom:
    :return:

    **Examples**::

        import copy

        x = Incidents.StartXY.iloc[0]
    """

    x_ = shapely.ops.nearest_points(x, obs_centroid_geom)[1]

    pseudo_id = [i for i, y in enumerate(obs_grids.Centroid_XY) if y.equals(x_)]

    return pseudo_id[0]


def create_weather_grid_buffer(start, end, midpoint, min_radius=1000, whisker=500):
    """
    Create a circle buffer for start/end location.

    :param start:
    :type start: shapely.geometry.Point
    :param end:
    :type end: shapely.geometry.Point
    :param midpoint:
    :type midpoint: shapely.geometry.Point
    :param min_radius:
    :type min_radius:
    :param whisker: extended length on both sides of the start and end locations, defaults to ``500``
    :type whisker: int
    :return: a buffer zone
    :rtype: shapely.geometry.Polygon

    **Examples**::

        whisker = 0

        start = Incidents.StartXY.iloc[0]
        end = Incidents.EndXY.iloc[0]
        midpoint = Incidents.MidpointXY.iloc[0]
    """

    if (start == end) or (start.distance(end) < min_radius):
        radius = min_radius + whisker
        buffer_circle = start.buffer(radius)

    else:
        radius = (start.distance(end) + whisker) / 2
        buffer_circle = midpoint.buffer(radius)

    return buffer_circle


def find_intersecting_weather_grid(x, obs_grids, obs_grids_geom, column_name='Grid_XY',
                                   as_grid_id=True):
    """
    Find all intersecting geom objects.

    :param x:
    :type x: shapely.geometry.Polygon
    :param obs_grids:
    :type obs_grids: pandas.DataFrame
    :param obs_grids_geom:
    :type obs_grids_geom: shapely.geometry.MultiPolygon
    :param column_name:
    :param as_grid_id: whether to return grid id number
    :type as_grid_id: bool
    :return:

    **Examples**::

        x = Incidents.Buffer_Zone.iloc[0]
        as_grid_id = True
    """

    intxn_grids = [grid for grid in obs_grids_geom.geoms if x.intersects(grid)]

    if as_grid_id:
        x_ = shapely.ops.unary_union(intxn_grids)
        intxn_grids = [i for i, y in enumerate(obs_grids[column_name]) if y.within(x_)]

    return intxn_grids


def find_closest_met_stn(x, met_stations, met_stations_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x:
    :param met_stations:
    :param met_stations_geom:
    :return:

    **Examples**::

        x = Incidents.MidpointXY.iloc[0]
    """

    x_1 = shapely.ops.nearest_points(x, met_stations_geom)[1]

    rest = shapely.geometry.MultiPoint([p for p in met_stations_geom.geoms if not p.equals(x_1)])
    x_2 = shapely.ops.nearest_points(x, rest)[1]

    rest = shapely.geometry.MultiPoint([p for p in rest.geoms if not p.equals(x_2)])
    x_3 = shapely.ops.nearest_points(x, rest)[1]

    idx = [i for i, y in enumerate(met_stations['XY']) if y in (x_1, x_2, x_3)]
    src_id = met_stations.index[idx].to_list()

    return src_id


# == Weather cell ==

def find_weather_cell_id(metex_instance, longitude, latitude):
    """
    Find Weather cell ID.

    :param metex_instance:
    :type metex_instance:
    :param longitude: longitude
    :type longitude: int, float
    :param latitude: latitude
    :type latitude: int, float
    :return: list | int
    """

    weather_cell = metex_instance.read_weather_cell()

    ll = [
        shapely.geometry.Point(xy)
        for xy in zip(weather_cell['ll_Longitude'], weather_cell['ll_Latitude'])]
    ul = [
        shapely.geometry.Point(xy)
        for xy in zip(weather_cell['ul_lon'], weather_cell['ul_lat'])]
    ur = [
        shapely.geometry.Point(xy)
        for xy in zip(weather_cell['ur_Longitude'], weather_cell['ur_Latitude'])]
    lr = [
        shapely.geometry.Point(xy)
        for xy in zip(weather_cell['lr_lon'], weather_cell['lr_lat'])]

    poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]

    cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in
             range(len(weather_cell))]

    pt = shapely.geometry.Point(longitude, latitude)

    id_set = set(
        weather_cell.iloc[[i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
    if len(id_set) == 1:
        weather_cell_id = list(id_set)[0]
    else:
        weather_cell_id = list(id_set)

    return weather_cell_id


def create_start_end_shapely_points(incidents_data, verbose=False):
    """
    Create Shapely Points for 'StartLocation's and 'EndLocation's.

    :param incidents_data: data of incident records
    :type incidents_data: pandas.DataFrame
    :param verbose: whether to print relevant information in console as the function runs,
        defaults to ``False``
    :type verbose: bool, int
    :return: incident data with shapely.geometry.Points of start and end locations
    """

    data = incidents_data.copy()

    if verbose:
        print("Creating shapely.geometry.Points for each incident location", end=" ... ")

    # Make shapely.geometry.points in longitude and latitude
    data.insert(
        data.columns.get_loc('StartLatitude') + 1, 'StartLonLat',
        points_from_xy(data[['StartLongitude', 'StartLatitude']]))
    data.insert(
        data.columns.get_loc('EndLatitude') + 1, 'EndLonLat',
        points_from_xy(data[['EndLongitude', 'EndLatitude']]))
    data.insert(
        data.columns.get_loc('EndLonLat') + 1, 'MidLonLat',
        data[['StartLonLat', 'EndLonLat']].apply(
            lambda x: shapely.geometry.LineString([x.StartLonLat, x.EndLonLat]).centroid, axis=1))

    # Add Easting and Northing points  # Start
    start_xy = [wgs84_to_osgb36(data.StartLongitude[i], data.StartLatitude[i]) for i in data.index]
    data = pd.concat([data, pd.DataFrame(start_xy, columns=['StartEasting', 'StartNorthing'])], axis=1)
    data['StartXY'] = points_from_xy(data[['StartEasting', 'StartNorthing']])

    # End
    end_xy = [wgs84_to_osgb36(data.EndLongitude[i], data.EndLatitude[i]) for i in data.index]
    data = pd.concat([data, pd.DataFrame(end_xy, columns=['EndEasting', 'EndNorthing'])], axis=1)
    data['EndXY'] = points_from_xy(data[['EndEasting', 'EndNorthing']])

    # data[['StartEasting', 'StartNorthing']] = data[['StartLongitude', 'StartLatitude']].apply(
    #     lambda x: pd.Series(wgs84_to_osgb36(x.StartLongitude, x.StartLatitude)), axis=1)
    # data['StartEN'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    # data[['EndEasting', 'EndNorthing']] = data[['EndLongitude', 'EndLatitude']].apply(
    #     lambda x: pd.Series(wgs84_to_osgb36(x.EndLongitude, x.EndLatitude)), axis=1)
    # data['EndEN'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)

    if verbose:
        print("Done.")

    return data


def create_weather_cell_buffer(midpoint, start_loc, end_loc, whisker_km=0.008, as_geom=True):
    """
    Create a circle buffer for an incident location.

    See also [`CCBUWC <https://gis.stackexchange.com/questions/289044/>`_]

    :param midpoint: midpoint or centre
    :type midpoint: shapely.geometry.Point
    :param start_loc: start location of an incident
    :type start_loc: shapely.geometry.Point
    :param end_loc: end location of an incident
    :type end_loc: shapely.geometry.Point
    :param whisker_km: extended length to diameter (i.e. on both sides of start/end locations),
        defaults to ``0.008``
    :type whisker_km: int, float
    :param as_geom: whether to return the buffer circle as shapely.geometry.Polygon, defaults to ``True``
    :type as_geom: bool
    :return: a buffer circle
    :rtype: shapely.geometry.Polygon; list of tuples

    **Examples**::

        >>> from src.shaft.geometry import create_weather_cell_buffer

        midpoint = Incidents.MidLonLat.iloc[0]
        incident_start = Incidents.StartLonLat.iloc[0]
        incident_end = Incidents.EndLonLat.iloc[0]

        whisker_km = 0.008
        as_geom = True
        buffer_circle = create_weather_cell_buffer(
            midpoint, incident_start, incident_end, whisker_km, as_geom)

    """

    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lon_0={lon} +lat_0={lat} +x_0=0 +y_0=0'
    project = functools.partial(
        pyproj.transform, pyproj.Proj(aeqd_proj.format(lon=midpoint.x, lat=midpoint.y)),
        pyproj.Proj(init='epsg:4326'))

    if start_loc != end_loc:
        radius_km = geopy.distance.distance(start_loc.coords, end_loc.coords).km / 2 + whisker_km
    else:
        radius_km = 2

    buffer = shapely.ops.transform(project, shapely.geometry.Point(0, 0).buffer(radius_km * 1000))
    buffer_circle = buffer if as_geom else buffer.exterior.coords[:]

    return buffer_circle


def find_intersecting_weather_cells(x, as_geom=False, metex_instance=None):
    """
    Find all intersecting Weather cells.

    :param x: e.g. x = Incidents.Buffer_Zone.iloc[0]
    :type: x: shapely.geometry.Point
    :param as_geom: whether to return shapely.geometry.Polygon of intersecting Weather cells
    :type as_geom: bool
    :param metex_instance:
    :type metex_instance:
    :return: intersecting Weather cells
    :rtype: tuple

    **Examples**::

        x = Incidents.Buffer_Zone.iloc[0]

        as_geom = False
        intxn_weather_cell_ids = find_intersecting_weather_cells(x, as_geom)

        as_geom = True
        intxn_weather_cell_ids = find_intersecting_weather_cells(x, as_geom)
    """

    if metex_instance is None:
        from src.preprocessor.metex import METEX
        metex_instance = METEX()

    weather_cell_geoms = metex_instance.read_weather_cell().Polygon_WGS84
    intxn_weather_cells = tuple(cell for cell in weather_cell_geoms if x.intersects(cell))
    if as_geom:
        return intxn_weather_cells

    else:
        intxn_weather_cell_ids = tuple(weather_cell_geoms[weather_cell_geoms == cell].index[0]
                                       for cell in intxn_weather_cells)
        if len(intxn_weather_cell_ids) == 1:
            intxn_weather_cell_ids = intxn_weather_cell_ids[0]
        return intxn_weather_cell_ids


def illustrate_weather_cell_buffer(midpoint, start_loc, end_loc, whisker_km=0.008,
                                   legend_pos='best', metex_instance=None):
    """
    Illustration of the buffer circle.

    :param midpoint: e.g. midpoint = Incidents.MidLonLat.iloc[2]
    :type midpoint:
    :param start_loc: e.g. incident_start = Incidents.StartLonLat.iloc[2]
    :type start_loc:
    :param end_loc: e.g. incident_end = Incidents.EndLonLat.iloc[2]
    :type end_loc:
    :param whisker_km: defaults to ``0.008``
    :type whisker_km: float
    :param legend_pos: defaults to ``'best'``
    :type legend_pos: str
    :param metex_instance:
    :type metex_instance:
    """

    if metex_instance is None:
        from src.preprocessor.metex import METEX
        metex_instance = METEX()

    buffer_circle = create_weather_cell_buffer(midpoint, start_loc, end_loc, whisker_km)
    i_weather_cells = find_intersecting_weather_cells(
        buffer_circle, as_geom=True, metex_instance=metex_instance)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for g in i_weather_cells:
        x, y = g.exterior.xy
        ax.plot(x, y, color='#433f3f')
        polygons = matplotlib.patches.Polygon(g.exterior.coords[:], fc='#D5EAFF', ec='#4b4747',
                                              alpha=0.5)
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


def prepare_shp_layer_files(osm_subregion, osm_layers=None, rm_shp_zip=True):
    """
    Get shape file ready.

    :param osm_subregion:
    :type osm_subregion: str
    :param osm_layers: defaults to ``('railways', 'landuse', 'natural')``
    :type osm_layers: tuple or list or iterable
    :param rm_shp_zip: defaults to ``True``
    :type rm_shp_zip: bool
    :return: directory path of OSM files, directory name of .shp file
    :rtype: tuple

    **Examples**::

        from models.prototype.hotspots_vis import prepare_shp_layer_files

        osm_subregion = 'England'  # case-insensitive
        relevant_osm_layers = ('railways', 'landuse', 'natural')
        rm_shp_zip = True

        osm_dir, shp_file_dir_name = prepare_shp_layer_files(osm_subregion)

        print(osm_dir)

        print(shp_file_dir_name)

        osm_subregion = 'rutland'
        relevant_osm_layers = None  # ['railways', 'landuse', 'natural']
        rm_shp_zip = True

        osm_subregion = 'great britain'
        relevant_osm_layers = None  # ['railways', 'landuse', 'natural']
        rm_shp_zip = True
    """

    if osm_layers is None:
        layer_names = ['railways', 'landuse', 'natural']
    else:
        layer_names = osm_layers

    gfd = GeofabrikDownloader()

    file_format = ".shp.zip"
    osm_dir = cdd("network/osm", mkdir=True)
    subregion_name, shp_zip_filename, _, shp_zip_pathname = gfd.get_valid_download_info(
        subregion_name=osm_subregion, osm_file_format=file_format, download_dir=osm_dir)

    if shp_zip_filename is None and shp_zip_pathname is None:
        # Possibly .shp.zip file of 'osm_subregion' is unavailable from the server
        sub_subregions = gfd.get_subregions(subregion_name, deep=False)

        gfr = GeofabrikReader()
        # Try to prepare the shp layer files of the subregions of 'osm_subregion'
        shp_pathnames = []
        for layer_name in layer_names:
            shp_pathnames += gfr.merge_shp_layers(
                subregion_names=sub_subregions, layer_name=layer_name, data_dir=osm_dir,
                rm_zip_extracts=True, rm_shp_temp=True, ret_merged_shp_path=True)

        # shp_file_dirs = list(set(os.path.dirname(x) for x in shp_file_pathnames))

        if rm_shp_zip:
            for sub_subrgn_name in sub_subregions:
                _, _, _, p = gfd.get_valid_download_info(
                    subregion_name=sub_subrgn_name, osm_file_format=file_format, download_dir=osm_dir)
                shutil.rmtree(os.path.dirname(p))

    else:  # Look for the folder that may contain the extracted data
        extract_dir_name = shp_zip_filename.replace(".shp.zip", "-shp")
        shp_dir = os.path.join(osm_dir, subregion_name.lower(), extract_dir_name)

        if not os.path.isdir(shp_dir) or not any(i in layer_names for i in os.listdir(shp_dir)):
            # If there is not any folder that contains the extracted data
            try:  # Try to download .shp.zip from the server
                path_to_shp_zip = gfd.download_data(
                    subregion_names=subregion_name, osm_file_formats=file_format,
                    download_dir=osm_dir, confirmation_required=False, ret_download_path=True)
            except InvalidSubregionNameError:
                path_to_shp_zip = []

            path_to_shp_zip = path_to_shp_zip[0]
            shp_dirs = SHP.unzip_shp_zip(
                shp_zip_pathname=path_to_shp_zip, layer_names=layer_names, separate=True,
                ret_extract_dir=True)

            if rm_shp_zip:
                os.remove(path_to_shp_zip)

        else:
            shp_dirs = [os.path.join(shp_dir, layer_name) for layer_name in layer_names]

        shp_pathnames = [
            os.path.join(x, f) for x in shp_dirs for f in os.listdir(x) if f.endswith(".shp")]

    return shp_pathnames


def crop_shp_data(path_to_shp_file, boundary_polygon):
    try:
        subarea_data = SHP.read_shp(
            shp_pathname=path_to_shp_file, emulate_gpd=True, bbox=boundary_polygon.bounds)
    except Exception as e:
        print(e)
        shp_data = SHP.read_shp(path_to_shp_file, emulate_gpd=True)
        subarea_idx = [
            i for i, x in enumerate(shp_data.geometry)
            if x.intersects(boundary_polygon) or x.within(boundary_polygon)]
        subarea_data = shp_data.iloc[subarea_idx, :]

    return subarea_data


def _create_sub_area(sub_shp_pathname, shp_pathname, boundary, update):
    """

    :param sub_shp_pathname:
    :param shp_pathname:
    :param boundary:
    :param update:
    :return:

    from shapely.geometry import Point, Polygon
    import numpy as np
    from pyhelpers.geom import wgs84_to_osgb36, osgb36_to_wgs84

    bham_centroid = Point(wgs84_to_osgb36(-1.898575, 52.489471))
    eastings, northings = bham_centroid.buffer(1000).boundary.coords.xy
    bham_boundary = Polygon(osgb36_to_wgs84(eastings=eastings, northings=northings, as_array=True))

    dat = SHPReadParse.read_shp(shp_pathname, emulate_gpd=True)

    idx = [i for i, x in dat['geometry'].items() if x.intersects(boundary) or x.within(boundary)]
    subarea_data = dat.loc[idx, :]



    """
    if not os.path.isfile(sub_shp_pathname) or update:
        dat = SHP.read_shp(shp_pathname, emulate_gpd=True)

        idx = [i for i, x in dat['geometry'].items() if x.intersects(boundary) or x.within(boundary)]
        subarea_data = dat.loc[idx, :]

        SHP.write_to_shapefile(subarea_data, write_to=sub_shp_pathname)


def get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature=None, boundary=None,
                                  sub_area_name=None, update=False):
    """
    Get the path to .shp file for basemap loading.

    :param osm_subregion: e.g. osm_subregion='England'
    :type osm_subregion: str
    :param osm_layer: e.g. osm_layer='railways'
    :type osm_subregion: str
    :param osm_feature: e.g. osm_feature='rail', defaults to ``None``
    :type osm_feature: str or None
    :param boundary: coordinates of a boundary
    :type boundary: shapely.geometry.Polygon
    :param sub_area_name: defaults to ``None``
    :type sub_area_name: str or None
    :param update:
    :type update: bool
    :return: path to OSM layer (feature)
    :rtype: str

    **Examples**::

        from src.shaft.geometry import get_shp_file_path_for_basemap
        from pyhelpers.geom import get_square_vertices
        import shapely.geometry
        import os

        osm_subregion = 'England'
        osm_layer = 'railways'
        update = False

        llcrnrlon=-0.565409
        llcrnrlat=51.23622
        urcrnrlon=1.915975
        urcrnrlat=53.15000
        ll = shapely.geometry.Point((llcrnrlon, llcrnrlat))
        ur = shapely.geometry.Point((urcrnrlon, urcrnrlat))
        diagonal = shapely.geometry.LineString([ll, ur])
        radius = diagonal.length
        centroid = diagonal.centroid
        boundary = shapely.geometry.Polygon(get_square_vertices(centroid.x, centroid.y, radius))

        osm_feature = None
        sub_area_name = None
        shp_file_path_for_basemap = get_shp_file_path_for_basemap(
            osm_subregion, osm_layer, osm_feature, boundary, sub_area_name, update)
        os.path.relpath(shp_file_path_for_basemap)
        'data\\network\\osm\\england\\england-latest-free-shp\\railways\\gis_osm_railways_free_1_b'

        osm_feature = 'rail'
        sub_area_name = 'anglia'
        shp_file_path_for_basemap = get_shp_file_path_for_basemap(
            osm_subregion, osm_layer, osm_feature, boundary, sub_area_name, update)
        os.path.basename(shp_file_path_for_basemap)
        'gis_osm_railways_free_1_rail_anglia'
    """

    osm_dir = cdd("network/osm")

    gfr = GeofabrikReader()
    try:
        path_to_shp_file_ = gfr.get_shp_pathname(
            subregion_name=osm_subregion, layer_name=osm_layer, feature_name=osm_feature,
            data_dir=osm_dir)
    except TypeError:
        subregion_names = gfr.downloader.get_subregions(osm_subregion)

        prefix = "-".join(["_".join([y[:3] for y in re.split(r'[- ]', x)]) for x in subregion_names])
        merged_dirname = f"{prefix}-{osm_layer}".lower()
        path_to_merged_dir = os.path.join(osm_dir, merged_dirname)
        path_to_merged_shp = glob.glob(os.path.join(f"{path_to_merged_dir}*", "*.shp"))

        if os.path.isfile(path_to_merged_shp[0]):
            path_to_shp_file_ = path_to_merged_shp
        else:
            path_to_shp_file_ = gfr.merge_shp_layers(
                subregion_names=subregion_names, layer_name=osm_layer, data_dir=osm_dir,
                rm_zip_extracts=False, ret_merged_shp_path=True)

    if len(path_to_shp_file_) == 0:
        shp_pathnames_ = prepare_shp_layer_files(osm_subregion, osm_layers=[osm_layer])

        if isinstance(osm_feature, str):  # osm_feature is not None
            if any(f'_{osm_feature}_' in x for x in shp_pathnames_):
                shp_pathnames = [x for x in shp_pathnames_ if f'_{osm_feature}_' in x]
            else:
                shp_data, shp_pathnames = SHP.read_layer_shps(
                    shp_pathnames=shp_pathnames_, feature_names=osm_feature, save_feat_shp=True,
                    ret_feat_shp_path=True)
        else:
            shp_pathnames = shp_pathnames_

    else:
        shp_pathnames = path_to_shp_file_

    shp_file_paths = []

    for sp in shp_pathnames:
        if boundary is not None:
            sp_, ext = os.path.splitext(sp)
            suffix = "_" + re.sub(r"[ \-]", "", sub_area_name.lower()) if sub_area_name else "_b"

            sub_shp_pathname = (sp_ if sp_.endswith(suffix) else sp_ + suffix) + ext

            if not os.path.isfile(sub_shp_pathname) or update:
                shp_data = SHP.read_shp(sp, emulate_gpd=True)
                subarea_idx = [
                    i for i, x in enumerate(shp_data.geometry)
                    if x.intersects(boundary) or x.within(boundary)]
                subarea_data = shp_data.iloc[subarea_idx, :]

                SHP.write_to_shapefile(subarea_data, write_to=sub_shp_pathname)

            sp = sub_shp_pathname

        shp_file_path_for_basemap = os.path.splitext(sp)[0]

        shp_file_paths.append(shp_file_path_for_basemap)

    return shp_file_paths


def get_shp_coordinates(osm_subregion, osm_layer, osm_feature=None, boundary=None,
                        sub_area_name=None, update=False, verbose=False):
    """
    Get coordinates of points from a .shp file, by subregion, layer and feature.

    :param osm_subregion:
    :type osm_subregion: str
    :param osm_layer:
    :type osm_layer: str
    :param osm_feature:
    :type osm_feature: str or None
    :param boundary:
    :type boundary: shapely.geometry.Polygon or None
    :param sub_area_name: 
    :type sub_area_name: str or None
    :param update: defaults to ``False``
    :type update: bool
    :param verbose:
    :return: 
    :rtype: shapely.geometry.multipoint.MultiPoint

    **Examples**::

        from src.coordinator.geometry import get_shp_file_path_for_basemap, get_shp_coordinates
        from pyhelpers.geom import get_square_vertices
        import shapely.geometry
        import os

        osm_subregion = 'England'
        osm_layer = 'railways'
        osm_feature = 'rail'
        sub_area_name = 'anglia'
        update = False

        llcrnrlon=-0.565409
        llcrnrlat=51.23622
        urcrnrlon=1.915975
        urcrnrlat=53.15000
        ll = shapely.geometry.Point((llcrnrlon, llcrnrlat))
        ur = shapely.geometry.Point((urcrnrlon, urcrnrlat))
        diagonal = shapely.geometry.LineString([ll, ur])
        radius = diagonal.length
        centroid = diagonal.centroid
        boundary = shapely.geometry.Polygon(get_square_vertices(centroid.x, centroid.y, radius))

        shp_coordinates = get_shp_coordinates(
            osm_subregion, osm_layer, osm_feature, boundary, sub_area_name, update)
    """

    shp_file_paths = get_shp_file_path_for_basemap(
        osm_subregion=osm_subregion, osm_layer=osm_layer, osm_feature=osm_feature,
        boundary=boundary, sub_area_name=sub_area_name)

    shp_coordinates_ = []

    for shp_file_path in shp_file_paths:
        path_to_shp = shp_file_path + ".shp"
        path_to_shp_coordinates_pickle = shp_file_path + "_coordinates.pkl"

        if os.path.isfile(path_to_shp_coordinates_pickle) and not update:
            shp_coords = load_pickle(path_to_shp_coordinates_pickle)

        else:
            try:
                railways_shp_data = SHP.read_shp(path_to_shp)
                shp_coords = shapely.geometry.MultiPoint(
                    list(itertools.chain(*railways_shp_data['coordinates'].to_list())))
                save_pickle(shp_coords, path_to_shp_coordinates_pickle, verbose=verbose)

            except Exception as e:
                print(e)
                shp_coords = None

        shp_coordinates_.append(shp_coords)

    if len(shp_coordinates_) == 1:
        shp_coordinates = shp_coordinates_[0]
    else:
        # shp_coordinates = shapely.ops.unary_union(shp_coordinates_)
        shp_coordinates = shapely.geometry.MultiPoint(
            list(itertools.chain(*[list(x.geoms) for x in shp_coordinates_ if x is not None])))

    return shp_coordinates
