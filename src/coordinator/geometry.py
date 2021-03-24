import functools
import itertools
import os
import re
import shutil

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
from pydriosm.downloader import GeofabrikDownloader
from pydriosm.reader import GeofabrikReader, read_shp_file, unzip_shp_zip
from pyhelpers.dir import cd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.store import load_pickle, save_pickle

from preprocessor import METExLite
from utils import cdd_network

metex = METExLite()


# == Weather grid ==

def find_closest_weather_grid(x, obs_grids, obs_centroid_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.StartNE.iloc[0]
    :param obs_grids:
    :param obs_centroid_geom:
    :return:

    **Test**::

        import copy

        x = incidents.StartXY.iloc[0]
    """

    x_ = shapely.ops.nearest_points(x, obs_centroid_geom)[1]

    pseudo_id = [i for i, y in enumerate(obs_grids.Centroid_XY) if y.equals(x_)]

    return pseudo_id[0]


def create_weather_grid_buffer(start, end, midpoint, whisker=500):
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

    **Test**::

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

    **Test**::

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

    **Test**::

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


# == Weather cell ==

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
    Create shapely.points for 'StartLocation's and 'EndLocation's.

    :param incidents_data: data of incident records
    :type incidents_data: pandas.DataFrame
    :param verbose: whether to print relevant information in console as the function runs,
        defaults to ``False``
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
                    lambda x: shapely.geometry.LineString([x.StartLonLat, x.EndLonLat]).centroid,
                    axis=1))
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

    **Example**::

        from models.tools import create_weather_cell_buffer

        midpoint = incidents.MidLonLat.iloc[0]
        incident_start = incidents.StartLonLat.iloc[0]
        incident_end = incidents.EndLonLat.iloc[0]

        whisker_km = 0.008
        as_geom = True
        buffer_circle = create_weather_cell_buffer(midpoint, incident_start, incident_end,
                                                               whisker_km, as_geom)

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


def illustrate_weather_cell_buffer(midpoint, start_loc, end_loc, whisker_km=0.008, legend_pos='best'):
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

    buffer_circle = create_weather_cell_buffer(midpoint, start_loc, end_loc, whisker_km)
    i_weather_cells = find_intersecting_weather_cells(buffer_circle, as_geom=True)
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


def get_shp_coordinates(osm_subregion, osm_layer, osm_feature=None, boundary_polygon=None,
                        sub_area_name=None, update=False, verbose=False):
    """
    Get coordinates of points from a .shp file, by subregion, layer and feature.

    :param osm_subregion: [str]
    :param osm_layer: [str]
    :param osm_feature: [str; None (default)]
    :param boundary_polygon: [tuple; None (default)]
    :param sub_area_name: [str; None (default)]
    :param update: [bool] (default: False)
    :param verbose:
    :return: [shapely.geometry.multipoint.MultiPoint]

    **Test**::

        from models.prototype.hotspots_vis import get_shp_coordinates

        osm_subregion = 'England'
        osm_layer = 'railways'
        osm_feature = None
        boundary_polygon = None
        sub_area_name = 'anglia'
        update = False

        shp_coordinates = get_shp_coordinates(osm_subregion, osm_layer, osm_feature, boundary_polygon,
                                              sub_area_name, update)
    """

    suffix = "coordinates"
    path_to_shp_filename = get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature,
                                                         boundary_polygon,
                                                         sub_area_name)
    path_to_shp = path_to_shp_filename + ".shp"
    path_to_shp_coordinates_pickle = path_to_shp_filename + "-" + suffix + ".pickle"

    if os.path.isfile(path_to_shp_coordinates_pickle) and not update:
        shp_coordinates = load_pickle(path_to_shp_coordinates_pickle)

    else:
        try:
            railways_shp_data = read_shp_file(path_to_shp, mode='geopandas')
            shp_coordinates = shapely.geometry.MultiPoint(
                list(itertools.chain(*(x.coords for x in railways_shp_data.geometry))))
            save_pickle(shp_coordinates, path_to_shp_coordinates_pickle, verbose=verbose)
        except Exception as e:
            print(e)
            shp_coordinates = None

    return shp_coordinates


def prepare_shp_layer_files(osm_subregion, relevant_osm_layers=('railways', 'landuse', 'natural'),
                            rm_shp_zip=True):
    """
    Get shape file ready.

    :param osm_subregion:
    :type osm_subregion: str
    :param relevant_osm_layers: defaults to ``('railways', 'landuse', 'natural')``
    :type relevant_osm_layers: tuple or list or iterable
    :param rm_shp_zip: defaults to ``True``
    :type rm_shp_zip: bool
    :return: directory path of OSM files, directory name of .shp file
    :rtype: tuple

    **Test**::

        from models.prototype.hotspots_vis import prepare_shp_layer_files

        osm_subregion = 'England'  # case-insensitive
        relevant_osm_layers = ('railways', 'landuse', 'natural')
        rm_shp_zip = True

        osm_dir, shp_file_dir_name = prepare_shp_layer_files(osm_subregion)

        print(osm_dir)

        print(shp_file_dir_name)
    """

    osm_dir, osm_subregion_ = cdd_network("osm"), osm_subregion.lower()
    file_format = ".shp.zip"
    osm_file_list = os.listdir(osm_dir)

    # Look for the folder that may contains the extracted data
    shp_file_dir_name = [
        x for x in osm_file_list
        if x.startswith(osm_subregion_.replace(" ", "-")) and not x.endswith(file_format)]

    # If there is not any folder that contains the extracted data
    if len(shp_file_dir_name) == 0:
        # Check if the raw .shp.zip files exist
        shp_zip_file = [x for x in osm_file_list
                        if x.startswith(osm_subregion_) and x.endswith(file_format)]

        if len(shp_zip_file) == 0:  # The raw .shp.zip files do not exist
            geofabrik_downloader = GeofabrikDownloader()

            # Try to download .shp.zip from the server
            geofabrik_downloader.download_subregion_data(osm_subregion_,
                                                         osm_file_format=file_format,
                                                         download_dir=osm_dir)
            # Find out the raw .shp.zip filename
            shp_zip_filename = [x for x in os.listdir(osm_dir)
                                if x.startswith(osm_subregion_)]
            # Possibly .shp.zip file of 'osm_subregion' is unavailable from the server
            if len(shp_zip_filename) == 0:
                # Try to prepare the shp layer files of the subregions of 'osm_subregion'
                shp_file_dir_name = osm_subregion_.replace(" ", "-") + ".shp"
                osm_dir_ = cd(osm_dir, shp_file_dir_name)

                sub_subregions = geofabrik_downloader.search_for_subregions(
                    osm_subregion_, deep=False)

                geofabrik_reader = GeofabrikReader()

                for layer in relevant_osm_layers:
                    geofabrik_reader.merge_subregion_layer_shp(
                        layer_name=layer, subregion_names=sub_subregions,
                        data_dir=osm_dir_, rm_shp_temp=True)
                # Remove the files for each subregion
                for f in os.listdir(osm_dir_):
                    if f.endswith(".shp"):
                        shutil.rmtree(cd(osm_dir_, f))
                    elif f.endswith(".shp.zip"):
                        os.remove(cd(osm_dir_, f))
        else:
            shp_zip_filename = shp_zip_file[0]
            path_to_shp_zip = cd(osm_dir, shp_zip_filename)
            unzip_shp_zip(path_to_shp_zip=path_to_shp_zip, clustered=True)
            shp_file_dir_name = os.path.splitext(shp_zip_filename)[0]

            if rm_shp_zip:
                os.remove(path_to_shp_zip)

    else:
        shp_file_dir_name = shp_file_dir_name[0]

        # Delete all other files which are not wanted
        shp_file_dir = cd(osm_dir, shp_file_dir_name)
        for f in os.listdir(shp_file_dir):
            if f not in relevant_osm_layers:
                x = cd(shp_file_dir, f)
                try:
                    shutil.rmtree(x)
                except NotADirectoryError:
                    os.remove(x)

    return osm_dir, shp_file_dir_name


def get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature=None, boundary_polygon=None,
                                  sub_area_name=None, rm_other_feat=False, update=False):
    """
    Get the path to .shp file for basemap loading.

    :param osm_subregion: e.g. osm_subregion='England'
    :type osm_subregion: str
    :param osm_layer: e.g. osm_layer='railways'
    :type osm_subregion: str
    :param osm_feature: e.g. osm_feature='rail', defaults to ``None``
    :type osm_feature: str or None
    :param rm_other_feat: whether to remove other features
    :type rm_other_feat: bool
    :param boundary_polygon: coordinates of a boundary
    :type boundary_polygon: tuple or shapely.geometry.Polygon
    :param sub_area_name: defaults to ``None``
    :type sub_area_name: str or None
    :param update:
    :type update: bool
    :return: path to osm layer (feature)
    :rtype: str

    **Test**::

        from models.prototype.hotspots_vis import get_shp_file_path_for_basemap

        osm_subregion = 'England'
        osm_layer = 'railways'
        boundary_polygon = None
        rm_other_feat = False
        update = False

        osm_feature = None
        sub_area_name = None
        shp_file_path_for_basemap = get_shp_file_path_for_basemap(
            osm_subregion, osm_layer, osm_feature, boundary_polygon, sub_area_name,
            rm_other_feat, update)
        print(shp_file_path_for_basemap)
        # ..\\data\\network\\osm\\england-latest-free.shp\\railways\\gis_osm_railways_free_1'

        osm_feature = 'rail'
        sub_area_name = 'anglia'
        shp_file_path_for_basemap = get_shp_file_path_for_basemap(
            osm_subregion, osm_layer, osm_feature, boundary_polygon, sub_area_name,
            rm_other_feat, update)
        print(shp_file_path_for_basemap)
        # ..\\data\\...\\railways\\gis_osm_railways_rail_free_1-anglia
    """

    osm_dir, shp_file_dir_name = prepare_shp_layer_files(osm_subregion)

    shp_file_dir = cd(osm_dir, shp_file_dir_name, osm_layer)

    shp_filename = [
        f for f in os.listdir(shp_file_dir)
        if re.match(r"gis_osm_{}(_a)?(_free)?(_1)?\.shp".format(osm_layer), f)][0]

    geofabrik_reader = GeofabrikReader()

    try:
        path_to_shp_file = geofabrik_reader.get_path_to_osm_shp(
            subregion_name=osm_subregion, layer_name=osm_layer, feature_name=osm_feature,
            data_dir=cdd_network("osm"))

    except IndexError:
        if osm_feature is not None:
            assert isinstance(osm_feature, str)
            shp_f_name, ext = os.path.splitext(shp_filename)
            shp_feat_name = shp_f_name + "_" + osm_feature

            path_to_shp_file = cd(shp_file_dir, shp_filename)
            if shp_feat_name not in path_to_shp_file:
                shp_data = read_shp_file(path_to_shp_file, mode='geopandas')
                shp_data_feat = shp_data[shp_data.fclass == osm_feature]
                shp_data_feat.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84',
                                     'proj': 'longlat'}
                shp_data_feat.to_file(cd(shp_file_dir, shp_feat_name + ".shp"),
                                      driver='ESRI Shapefile')
                if rm_other_feat:
                    for f in os.listdir(shp_file_dir):
                        if shp_feat_name not in f:
                            os.remove(cd(shp_file_dir, f))

        path_to_shp_file = geofabrik_reader.get_path_to_osm_shp(
            subregion_name=osm_subregion, layer_name=osm_layer, feature_name=osm_feature,
            data_dir=shp_file_dir)

    if boundary_polygon is not None:
        #
        shp_filename = os.path.basename(path_to_shp_file)
        suffix = "_" + re.sub(r"[ \-]", "", sub_area_name.lower()) if sub_area_name else "_partial"
        sub_shp_fn = shp_filename if suffix in shp_filename \
            else shp_filename.replace(".shp", "") + suffix + ".shp"
        path_to_sub_shp_file = cd(shp_file_dir, sub_shp_fn)
        #
        if not os.path.isfile(path_to_sub_shp_file) or update:
            try:
                subarea_data = read_shp_file(path_to_shp_file, mode='geopandas',
                                             bbox=boundary_polygon.bounds)
            except Exception as e:
                print(e)
                shp_data = read_shp_file(path_to_shp_file, mode='geopandas')
                subarea_idx = [i for i, x in enumerate(shp_data.geometry)
                               if x.intersects(boundary_polygon) or x.within(boundary_polygon)]
                subarea_data = shp_data.iloc[subarea_idx, :]
            subarea_data.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84',
                                'proj': 'longlat'}
            subarea_data.to_file(path_to_sub_shp_file, driver='ESRI Shapefile', encoding='UTF-8')
        path_to_shp_file = path_to_sub_shp_file

    shp_file_path_for_basemap = os.path.splitext(path_to_shp_file)[0]

    return shp_file_path_for_basemap
