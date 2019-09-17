""" Plotting hotspots of Weather-related Incidents in the context of wind-related delays """

import itertools
import os
import re
import shutil

import PIL.Image
import matplotlib
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import numpy as np
import pandas as pd
import pydriosm as dri
import pysal.viz.mapclassify.classifiers
import scipy.spatial
import shapely.geometry
import shapely.ops
from pyhelpers.dir import cd
from pyhelpers.misc import colour_bar_index, confirmed
from pyhelpers.settings import mpl_preferences
from pyhelpers.store import load_pickle, save_fig, save_pickle

import models.tools
import mssqlserver.metex
import mssqlserver.vegetation
from utils import cdd_network

matplotlib.use('TkAgg')
mpl_preferences()


# ====================================================================================================================
""" Preparation """


# Make Shapefile ready
def prepare_shp_layer_files(osm_subregion, relevant_osm_layers=('railways', 'landuse', 'natural'), rm_shp_zip=True):
    """
    :param osm_subregion: [str]
    :param relevant_osm_layers: [tuple] (default: ('railways', 'landuse', 'natural'))
    :param rm_shp_zip: [bool] (default: True)
    :return: [tuple] (directory path of OSM files, directory name of .shp file)

    Testing e.g.
        osm_subregion = 'england'  # case-insensitive

        prep_shp_layer_files(osm_subregion)
    """
    osm_dir, osm_subregion = cdd_network("OSM"), osm_subregion.lower()
    osm_file_list = os.listdir(osm_dir)
    shp_file_dir_name = [x for x in osm_file_list if x.startswith(osm_subregion) and not x.endswith(".zip")]
    if len(shp_file_dir_name) == 0:
        shp_zip_file = [x for x in osm_file_list if x.startswith(osm_subregion) and x.endswith(".zip")]
        if len(shp_zip_file) == 0:
            dri.download_subregion_osm_file(osm_subregion, osm_file_format=".shp.zip", download_dir=osm_dir)
            shp_zip_filename = [x for x in os.listdir(osm_dir) if x.startswith(osm_subregion.lower())][0]
        else:
            shp_zip_filename = shp_zip_file[0]
        path_to_shp_zip = cd(osm_dir, shp_zip_filename)
        dri.extract_shp_zip(path_to_shp_zip, clustered=True)
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


# Get the path to .shp file for basemap loading
def get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature=None, boundary_polygon=None,
                                  sub_area_name=None, rm_other_feat=False, update=False):
    """
    :param osm_subregion: [str] e.g. osm_subregion='England'
    :param osm_layer: [str] e.g. osm_layer='railways'
    :param osm_feature: [str; None (default)] e.g. osm_feature='rail'
    :param rm_other_feat: [bool]
    :param boundary_polygon: [tuple]
    :param sub_area_name: [str; None (default)]
    :param update: [bool]
    :return: [str] path to osm layer (feature)

    Testing e.g.
        osm_subregion = 'England'
        osm_layer = 'railways'
        osm_feature = None
        boundary_polygon = None
        sub_area_name = None
        rm_other_feat = False
        update = False

        osm_subregion = 'England'
        osm_layer = 'railways'
        osm_feature = 'rail'
        boundary_polygon = None
        sub_area_name = 'anglia'
        rm_other_feat = False
        update = False

        get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature, boundary_polygon, sub_area_name,
                                      rm_other_feat, update)
    """
    osm_dir, shp_file_dir_name = prepare_shp_layer_files(osm_subregion)

    shp_file_dir = cd(osm_dir, shp_file_dir_name, osm_layer)

    shp_filename = [f for f in os.listdir(shp_file_dir)
                    if re.match(r"gis_osm_{}(_a)?_free_1\.shp".format(osm_layer), f)][0]

    try:
        path_to_shp_file = dri.find_osm_shp_file(osm_subregion, osm_layer, osm_feature, shp_file_dir)[0]

    except IndexError:
        if osm_feature is not None:
            assert isinstance(osm_feature, str)
            shp_feat_name = "{}-{}".format(osm_layer, osm_feature)

            path_to_shp_file = cd(shp_file_dir, shp_filename)
            if shp_feat_name not in path_to_shp_file:
                shp_data = dri.read_shp(path_to_shp_file, mode='geopandas')
                shp_data_feat = shp_data[shp_data.fclass == osm_feature]
                shp_data_feat.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}
                shp_data_feat.to_file(cd(shp_file_dir, shp_filename.replace(osm_layer, shp_feat_name)),
                                      driver='ESRI Shapefile', encoding='UTF-8')
                if rm_other_feat:
                    for f in os.listdir(shp_file_dir):
                        if shp_feat_name not in f:
                            os.remove(cd(shp_file_dir, f))

        path_to_shp_file = dri.find_osm_shp_file(osm_subregion, osm_layer, osm_feature, shp_file_dir)[0]

    if boundary_polygon is not None:
        #
        shp_filename = os.path.basename(path_to_shp_file)
        suffix = "-" + sub_area_name.lower() if sub_area_name else "-sub-area"
        if suffix in shp_filename:
            sub_shp_filename = shp_filename
        else:
            sub_shp_filename = shp_filename.replace(".shp", "") + suffix + ".shp"
        path_to_sub_shp_file = cd(shp_file_dir, sub_shp_filename)
        #
        if not os.path.isfile(path_to_sub_shp_file) or update:
            try:
                subarea_data = dri.read_shp(path_to_shp_file, mode='geopandas', bbox=boundary_polygon.bounds)
            except Exception as e:
                print(e)
                shp_data = dri.read_shp(path_to_shp_file, mode='geopandas')
                subarea_idx = [i for i, x in enumerate(shp_data.geometry)
                               if x.intersects(boundary_polygon) or x.within(boundary_polygon)]
                subarea_data = shp_data.iloc[subarea_idx, :]
            subarea_data.crs = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}
            subarea_data.to_file(path_to_sub_shp_file, driver='ESRI Shapefile', encoding='UTF-8')
        path_to_shp_file = path_to_sub_shp_file

    shp_file_path_for_basemap = os.path.splitext(path_to_shp_file)[0]

    return shp_file_path_for_basemap


# ====================================================================================================================
""" Save output figures """


# A function for saving the plots
def save_prototype_hotpots_fig(fig, keyword, category,
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose):
    """
    :param fig: [matplotlib.figure.Figure]
    :param keyword: [str] a keyword for specifying the filename
    :param category: [str]
    :param show_metex_weather_cells: [bool]
    :param show_osm_landuse_forest: [bool]
    :param show_nr_hazardous_trees: [bool]
    :param save_as: [str; None]
    :param dpi: [int; None]
    :param verbose: [bool]
    """
    if save_as is not None:
        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                         ['weather-grid', 'vegetation', 'hazard-trees'])
            filename = '-'.join([keyword] + [v for s, v in suffix if s is True])
            path_to_file = models.tools.cd_prototype_fig_pub(category, filename + save_as)
            save_fig(path_to_file, dpi=dpi, verbose=verbose)


# ====================================================================================================================
""" Prep base maps """


# Make the basemap ready
def plot_base_map(projection='tmerc', railway_line_color='#3D3D3D', legend_loc=(1.05, 0.85)):
    """
    :param projection: [str] (default: 'tmerc')
    :param railway_line_color: [str] (default: '#3D3D3D')
    :param legend_loc: [tuple] (default: (1.05, 0.85))
    :return: [tuple] (matplotlib.figure.Figure, mpl_toolkits.basemap.Basemap)

    Testing e.g.
        projection = 'tmerc'
        railway_line_color = '#3D3D3D'
        legend_loc = (1.05, 0.85)

        plot_base_map(projection, railway_line_color, legend_loc)
    """
    print("Plotting the base map ... ", end="")
    plt.style.use('ggplot')  # Default, 'classic'; matplotlib.style.available gives the list of available styles
    fig = plt.figure(figsize=(11, 9))  # fig = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(left=0.001, bottom=0.000, right=0.6035, top=1.000)

    # Plot basemap
    base_map = mpl_toolkits.basemap.Basemap(llcrnrlon=-0.565409,  # ll[0] - 0.06 * width,
                                            llcrnrlat=51.23622,  # ll[1] - 0.06 + 0.002 * height,
                                            urcrnrlon=1.915975,  # ur[0] + extra * width,
                                            urcrnrlat=53.15000,  # ur[1] + extra + 0.01 * height,
                                            ellps='WGS84',
                                            lat_ts=0,
                                            lon_0=-2.,
                                            lat_0=49.,
                                            projection=projection,  # Transverse Mercator Projection
                                            resolution='i',
                                            suppress_ticks=True,
                                            epsg=27700)

    # base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)
    base_map.drawmapboundary(color='white', fill_color='white')
    # base_map.drawcoastlines()
    base_map.fillcontinents(color='#dcdcdc')  # color='#555555'

    # Add a layer for railway tracks
    boundary_polygon = shapely.geometry.Polygon(zip(base_map.boundarylons, base_map.boundarylats))
    path_to_shp_file = get_shp_file_path_for_basemap('England', 'railways', 'rail', boundary_polygon, 'anglia')

    base_map.readshapefile(shapefile=path_to_shp_file,
                           name='anglia',
                           linewidth=1.5,
                           color=railway_line_color,  # '#626262', '#939393', '#757575', '#909090',
                           zorder=4)

    # Show legend
    plt.plot([], '-', label="Railway track", linewidth=2.2, color=railway_line_color)
    # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
    font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
    legend = plt.legend(numpoints=1, loc='best', prop=font, frameon=False, fancybox=True, bbox_to_anchor=legend_loc)
    frame = legend.get_frame()
    frame.set_edgecolor('none')
    frame.set_facecolor('none')

    print("Done.")

    return fig, base_map


# Show Weather cells on the map
def plot_weather_cells(base_map=None, update=False, weather_cell_colour='#D5EAFF', legend_loc=(1.05, 0.85)):
    """
    :param base_map: [mpl_toolkits.basemap.Basemap; None (default)] basemap object
    :param update: [bool] (default: False)
    :param weather_cell_colour: [str] '#add6ff' (default), '#99ccff', '#fff68f
    :param legend_loc: [tuple] (default: (1.05, 0.85))

    Testing e.g.
        base_map = None
        update = False
        route_name = None
        weather_cell_colour = '#D5EAFF'
        legend_loc = (1.05, 0.85)

        plot_weather_cells(base_map, update, weather_cell_colour, legend_loc)
    """
    if base_map is None:
        _, base_map = plot_base_map()

    print("Plotting the Weather cells ... ", end="")

    # Get Weather cell data
    data = mssqlserver.metex.get_weather_cell(update=update)
    data = mssqlserver.metex.get_subset(data, route_name='Anglia')
    # Drop duplicated Weather cell data
    unhashable_cols = ('Polygon_WGS84', 'Polygon_OSGB36', 'IMDM', 'Route')
    data.drop_duplicates(subset=[x for x in list(data.columns) if x not in unhashable_cols], inplace=True)

    # Plot the Weather cells one by one
    for i in range(len(data)):
        ll_x, ll_y = base_map(data['ll_Longitude'].iloc[i], data['ll_Latitude'].iloc[i])
        ul_x, ul_y = base_map(data['ul_Longitude'].iloc[i], data['ul_Latitude'].iloc[i])
        ur_x, ur_y = base_map(data['ur_Longitude'].iloc[i], data['ur_Latitude'].iloc[i])
        lr_x, lr_y = base_map(data['lr_Longitude'].iloc[i], data['lr_Latitude'].iloc[i])
        xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
        p = matplotlib.patches.Polygon(list(xy), fc=weather_cell_colour, ec='#4b4747', zorder=2)
        plt.gca().add_patch(p)

    # Add labels
    # plt.plot([], 's', label="Weather cell", ms=30, color=weather_cell_colour, markeredgecolor='#433f3f', alpha=.5)
    plt.plot([], 's', label="Weather cell", ms=25, color='#D5EAFF', markeredgecolor='#433f3f', alpha=.5)

    # Show legend  # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
    font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
    plt.legend(numpoints=1, loc='best', prop=font, frameon=False, fancybox=True, bbox_to_anchor=legend_loc)

    print("Done.")


# Demonstrate the OSM natural forest on the map
def plot_osm_forest_and_tree(base_map=None, osm_landuse_forest_colour='#72886e', fill_forest_patches=False,
                             add_osm_natural_tree=False, legend_loc=(1.05, 0.85)):
    """
    :param base_map: [mpl_toolkits.basemap.Basemap; None (default)] basemap object
    :param osm_landuse_forest_colour: [str] '#7f987b' (default), '#8ea989', '#72946c', '#72946c'
    :param add_osm_natural_tree: [bool] (default: False)
    :param fill_forest_patches: [bool] (default: False)
    :param legend_loc: [tuple] (default: (1.05, 0.85))

    Testing e.g.
        base_map = None
        osm_landuse_forest_colour = '#72886e'
        fill_forest_patches = False
        add_osm_natural_tree = False
        legend_loc = (1.05, 0.85)

        plot_osm_forest_and_tree(base_map, osm_landuse_forest_colour, fill_forest_patches,
                                 add_osm_natural_tree, legend_loc)
    """
    if base_map is None:
        _, base_map = plot_base_map()

    print("Plotting the OSM natural/forest ... ", end="")

    # OSM - landuse - forest
    boundary_polygon = shapely.geometry.Polygon(zip(base_map.boundarylons, base_map.boundarylats))
    bounded_landuse_forest_shp = get_shp_file_path_for_basemap('England', 'landuse', 'forest',
                                                               boundary_polygon, sub_area_name='anglia')

    base_map.readshapefile(bounded_landuse_forest_shp,
                           name='osm_landuse_forest',
                           color=osm_landuse_forest_colour,
                           zorder=3)

    # Fill the patches? Note this may take a long time and dramatically increase the file of the map
    if fill_forest_patches:
        print("\n")
        print("Filling the 'osm_landuse_forest' polygons ... ", end="")

        forest_polygons = [
            matplotlib.patches.Polygon(p, fc=osm_landuse_forest_colour, ec=osm_landuse_forest_colour, zorder=4)
            for p in base_map.osm_landuse_forest]

        for i in range(len(forest_polygons)):
            plt.gca().add_patch(forest_polygons[i])

    # OSM - natural - tree
    if add_osm_natural_tree:
        bounded_natural_tree_shp = get_shp_file_path_for_basemap('England', 'natural', 'tree',
                                                                 boundary_polygon, sub_area_name='anglia')
        base_map.readshapefile(bounded_natural_tree_shp,
                               name='osm_natural_tree',
                               color=osm_landuse_forest_colour,
                               zorder=3)
        natural_tree_points = [shapely.geometry.Point(p) for p in base_map.osm_natural_tree]
        base_map.scatter([geom.x for geom in natural_tree_points], [geom.y for geom in natural_tree_points],
                         marker='o', s=2, facecolor='#008000', label="Tree", alpha=0.5, zorder=3)

    # Add label
    plt.scatter([], [], marker="o",
                # hatch=3 * "x", s=580,
                facecolor=osm_landuse_forest_colour, edgecolor='none',
                label="Vegetation (OSM 'forest')")

    # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
    font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
    plt.legend(scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True, bbox_to_anchor=legend_loc)

    print("Done.")


# Show hazardous trees on the map
def plot_hazardous_trees(base_map=None, hazardous_tree_colour='#ab790a', legend_loc=(1.05, 0.85)):
    """
    :param base_map: [mpl_toolkits.basemap.Basemap; None (default)] basemap object
    :param hazardous_tree_colour: [str] '#ab790a' (default), '#886008', '#6e376e', '#5a7b6c'
    :param legend_loc: [tuple] (default: (1.05, 0.85))

    Testing e.g.
        base_map = None
        route_name = None
        hazardous_tree_colour = '#ab790a'
        legend_loc = (1.05, 0.85)

        plot_hazardous_trees(base_map, hazardous_tree_colour, legend_loc)
    """
    if base_map is None:
        _, base_map = plot_base_map()

    print("Plotting the hazardous trees ... ", end="")

    hazardous_trees = mssqlserver.vegetation.view_hazardous_trees()

    map_points = [shapely.geometry.Point(base_map(long, lat))
                  for long, lat in zip(hazardous_trees.Longitude, hazardous_trees.Latitude)]
    hazardous_trees_points = shapely.geometry.MultiPoint(map_points)

    # Plot hazardous trees on the basemap
    base_map.scatter([geom.x for geom in hazardous_trees_points], [geom.y for geom in hazardous_trees_points],
                     marker='x',  # edgecolor='w',
                     s=20, lw=1.5, facecolor=hazardous_tree_colour,
                     label="Hazardous trees", alpha=0.6, antialiased=True, zorder=3)

    # Show legend  # setfont = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
    font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
    plt.legend(scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True, bbox_to_anchor=legend_loc)

    print("Done.")


# Plot base map and associated features
def plot_base_map_plus(show_metex_weather_cells=True, show_osm_landuse_forest=True, add_osm_natural_tree=False,
                       show_nr_hazardous_trees=True, legend_loc=(1.05, 0.85), save_as=None, dpi=None):
    """
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param add_osm_natural_tree: [bool] (default: False)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param legend_loc: [tuple] (default: (1.05, 0.85))
    :param save_as: [str; None (default)]
    :param dpi: [int; None (default)]

    Testing e.g.
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        add_osm_natural_tree = False
        show_nr_hazardous_trees = True
        legend_loc = (1.05, 0.85)
        save_as= ".tif"
        dpi = None

        plot_base_map_plus(show_metex_weather_cells, show_osm_landuse_forest, add_osm_natural_tree,
                           show_nr_hazardous_trees, legend_loc, save_as, dpi)
    """
    # Plot basemap
    fig, base_map = plot_base_map(projection='tmerc', legend_loc=legend_loc)

    # Show Weather cells
    if show_metex_weather_cells:
        plot_weather_cells(base_map, legend_loc=legend_loc)

    # Show Vegetation
    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=add_osm_natural_tree, legend_loc=legend_loc)

    # Show hazardous trees
    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=legend_loc)

    # Add an axes at position [left, bottom, width, height]
    sr = fig.add_axes([0.58, 0.01, 0.40, 0.40], frameon=True)  # quantities are in fractions of figure width and height
    sr.imshow(PIL.Image.open(cdd_network("Routes\\Map", "NR-Routes-edited-1.tif")))  # "Routes-edited-0.png"
    sr.axis('off')

    # Save the figure
    save_prototype_hotpots_fig(fig, "base", "Basemap",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# ====================================================================================================================
""" Get 'hotspot' data """


# Get coordinates of points from a .shp file, by subregion, layer and feature
def get_shp_coordinates(osm_subregion, osm_layer, osm_feature=None, boundary_polygon=None, sub_area_name=None,
                        update=False):
    """
    :param osm_subregion: [str]
    :param osm_layer: [str]
    :param osm_feature: [str; None (default)]
    :param boundary_polygon: [tuple; None (default)]
    :param sub_area_name: [str; None (default)]
    :param update: [bool] (default: False)
    :return: [shapely.geometry.multipoint.MultiPoint]

    Testing e.g.
        osm_subregion = 'England'
        osm_layer = 'railways'
        osm_feature = None
        boundary_polygon = None
        sub_area_name = 'anglia'
        update = False

        get_shp_coordinates(osm_subregion, osm_layer, osm_feature, boundary_polygon, sub_area_name, update)
    """
    suffix = "coordinates"
    path_to_shp_filename = get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature,
                                                         boundary_polygon, sub_area_name)
    path_to_shp = path_to_shp_filename + ".shp"
    path_to_shp_coordinates_pickle = path_to_shp_filename + "-" + suffix + ".pickle"

    if os.path.isfile(path_to_shp_coordinates_pickle) and not update:
        shp_coordinates = load_pickle(path_to_shp_coordinates_pickle)
    else:
        try:
            railways_shp_data = dri.read_shp(path_to_shp, mode='geopandas')
            shp_coordinates = shapely.geometry.MultiPoint(
                list(itertools.chain(*(l.coords for l in railways_shp_data.geometry))))
            save_pickle(shp_coordinates, path_to_shp_coordinates_pickle)
        except Exception as e:
            print(e)
            shp_coordinates = None
    return shp_coordinates


# Get midpoints (of incident locations) for plotting hotspots
def get_midpoints_for_plotting_hotspots(route_name=None, weather_category=None, sort_by=None, update=False):
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param sort_by: [list; None (default)]
    :param update: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        weather_category = None
        sort_by = None
        update = False

        get_midpoints_for_plotting_hotspots(route_name, weather_category, sort_by, update)
    """
    pickle_filename = mssqlserver.metex.make_filename("Schedule8-hotspots", route_name, weather_category)
    path_to_pickle = mssqlserver.metex.cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_hotspots = load_pickle(path_to_pickle)

    else:
        # Get TRUST (by incident location, i.e. by STANOX section)
        schedule8_costs_by_location = mssqlserver.metex.view_schedule8_costs_by_location()

        # Find a pseudo midpoint for each recorded incident
        def get_midpoint(x1, y1, x2, y2, as_geom=False):
            """
            :param x1: [numbers.Number; np.ndarray]
            :param y1: [numbers.Number; np.ndarray]
            :param x2: [numbers.Number; np.ndarray]
            :param y2: [numbers.Number; np.ndarray]
            :param as_geom: [bool] (default: False)
            :return: [np.ndarray; (list of) shapely.geometry.Point]

            Testing e.g.
                x1, y1, x2, y2 = 1.5429, 52.6347, 1.4909, 52.6271
                as_geom = False

                get_midpoint(x1, y1, x2, y2, as_geom)
                get_midpoint(x1, y1, x2, y2, True)
            """
            mid_pts = (x1 + x2)/2, (y1 + y2)/2
            if as_geom:
                if all(isinstance(x, np.ndarray) for x in mid_pts):
                    mid_pts_ = [shapely.geometry.Point(x_, y_) for x_, y_ in zip(list(mid_pts[0]), list(mid_pts[1]))]
                else:
                    mid_pts_ = shapely.geometry.Point(mid_pts)
            else:
                mid_pts_ = np.array(mid_pts).T
            return mid_pts_

        pseudo_midpoints = get_midpoint(schedule8_costs_by_location.StartLongitude.values,
                                        schedule8_costs_by_location.StartLatitude.values,
                                        schedule8_costs_by_location.EndLongitude.values,
                                        schedule8_costs_by_location.EndLatitude.values, as_geom=False)

        # Get reference points (coordinates), given subregion and layer (i.e. 'railways' in this case) of OSM .shp file
        if route_name:
            path_to_boundary_polygon = cdd_network("Routes\\{}\\boundary-polygon.pickle".format(route_name))
            boundary_polygon = load_pickle(path_to_boundary_polygon)
            sub_area_name = route_name
        else:
            boundary_polygon, sub_area_name = None, None

        rail_points = get_shp_coordinates('England', 'railways', 'rail', boundary_polygon, sub_area_name)

        # Get rail coordinates closest to the midpoints between starts and ends
        def find_closest_points_between(pts, ref_pts, as_geom=False):
            """
            :param pts: [np.ndarray] an array of size (n, 2)
            :param ref_pts: [np.ndarray] an array of size (n, 2)
            :param as_geom: [bool] (default: False)
            :return: [np.ndarray; list of shapely.geometry.Point]

            Testing e.g.
                pts = np.array([[1.5429, 52.6347], [1.4909, 52.6271], [1.4248, 52.63075]])
                ref_pts = np.array([[2.5429, 53.6347], [2.4909, 53.6271], [2.4248, 53.63075]])
                as_geom = False

                get_closest_points_between(pts, ref_pts, as_geom)
                get_closest_points_between(pts, ref_pts, True)

            Reference: https://gis.stackexchange.com/questions/222315
            """
            import shapely.geometry
            if isinstance(ref_pts, np.ndarray):
                ref_pts_ = ref_pts
            else:
                ref_pts_ = np.concatenate([np.array(geom.coords) for geom in ref_pts])
            ref_ckd_tree = scipy.spatial.cKDTree(ref_pts_)
            distances, indices = ref_ckd_tree.query(pts, k=1)  # returns (distance, index)
            if as_geom:
                closest_pts = [shapely.geometry.Point(ref_pts_[i]) for i in indices]
            else:
                closest_pts = np.array([ref_pts_[i] for i in indices])
            return closest_pts

        midpoints = find_closest_points_between(pseudo_midpoints, rail_points)

        midpoints_ = pd.DataFrame(midpoints, schedule8_costs_by_location.index, columns=['MidLongitude', 'MidLatitude'])
        incident_hotspots = schedule8_costs_by_location.join(midpoints_)

        save_pickle(incident_hotspots, path_to_pickle, verbose=True)

    if sort_by:
        incident_hotspots.sort_values(sort_by, ascending=False, inplace=True)

    incident_hotspots = mssqlserver.metex.get_subset(incident_hotspots, route_name=route_name,
                                                     weather_category=weather_category, rearrange_index=True)

    return incident_hotspots


# Get statistics for plotting annual delays
def get_schedule8_annual_stats(route_name='Anglia', weather_category='Wind', update=False):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update = False

        get_schedule8_annual_stats(route_name, weather_category, update)
    """
    pickle_filename = mssqlserver.metex.make_filename("Schedule8-hotspots-annual-delays", route_name, weather_category)
    path_to_pickle = mssqlserver.metex.cdd_metex_db_views(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        annual_delays = load_pickle(path_to_pickle)
        annual_delays = mssqlserver.metex.get_subset(annual_delays, route_name, weather_category)

    else:
        schedule8_data = mssqlserver.metex.view_schedule8_costs_by_datetime_location(
            route_name, weather_category, update)
        selected_features = ['FinancialYear', 'WeatherCategory', 'Route', 'StanoxSection',
                             'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
        schedule8_data_ = schedule8_data.groupby(selected_features).agg(
            {'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum}).reset_index()

        incident_location_midpoints = get_midpoints_for_plotting_hotspots(route_name, weather_category, update=update)

        annual_delays = schedule8_data_.merge(
            incident_location_midpoints[selected_features[1:] + ['MidLatitude', 'MidLongitude']], how='left',
            on=selected_features[1:])
        annual_delays.sort_values(by=['DelayMinutes', 'DelayCost', 'IncidentCount'], ascending=False, inplace=True)

        save_pickle(annual_delays, path_to_pickle)

    return annual_delays


# ====================================================================================================================
""" Plot 'hotspots' """


# Plot hotspots of delays for every financial year (2006/07-2014/15)
def plot_hotspots_for_annual_delays(route_name='Anglia', weather_category='Wind', update_data=False, cmap_name='Set1',
                                    show_metex_weather_cells=True, show_osm_landuse_forest=True,
                                    show_nr_hazardous_trees=True, save_as=None, dpi=None):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)
    :param cmap_name: [str] (default: 'Set1')
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param save_as: [str; None (default)]
    :param dpi: [numbers.Number; None (default)]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False
        cmap_name = 'Set1'
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        show_nr_hazardous_trees = True
        save_as = ".tif"
        dpi = None

        plot_hotspots_for_annual_delays(route_name, weather_category, update_data, cmap_name, show_metex_weather_cells,
                                        show_osm_landuse_forest, show_nr_hazardous_trees, save_as, dpi)
    """
    schedule8_annual_stats = get_schedule8_annual_stats(route_name, weather_category, update_data)

    annual_stats = schedule8_annual_stats.groupby('FinancialYear').agg(
        {'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum})

    # Examine only 2006/07 - 2014/15
    hotspots_annual_stats = annual_stats.loc[2006:2014]

    # Labels
    years = [str(y) for y in hotspots_annual_stats.index]
    f_years = ['/'.join([y0, str(y1)[-2:]])
               for y0, y1 in zip(years, np.array(hotspots_annual_stats.index) + np.array([1]))]

    d_label = ["%s  (%s min." % (fy, format(int(d), ",")) for fy, d in zip(f_years, hotspots_annual_stats.DelayMinutes)]
    c_label = ["  / £%.2f" % round(c * 1e-6, 2) + "M)" for c in hotspots_annual_stats.DelayCost]
    label = [l for l in reversed([d + c for d, c in zip(d_label, c_label)])]

    cmap = plt.get_cmap(cmap_name)
    colours = [c for c in reversed(cmap(np.linspace(start=0, stop=1, num=9)))]

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    top_hotspots = []
    for y, fy in zip(years, f_years):
        plot_data = schedule8_annual_stats[schedule8_annual_stats.FinancialYear == int(y)][0:20]
        top_hotspots.append(fy + ':  ' + plot_data.StanoxSection.iloc[0])
        for i in plot_data.index:
            mid_x, mid_y = base_map(plot_data.MidLongitude[i], plot_data.MidLatitude[i])
            base_map.plot(mid_x, mid_y, zorder=2, marker='o', color=colours[years.index(y)], alpha=0.9,
                          markersize=26, markeredgecolor='w')

    # Add a colour bar
    cb = colour_bar_index(no_of_colours=len(label), cmap_param=cmap, shrink=0.4, labels=label, pad=0.068)
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_font_properties(matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
    cb.ax.tick_params(labelsize=14)
    cb.set_alpha(1.0)
    cb.draw_all()

    cb.ax.text(0 + 1.5, 10.00, "Annual total delays and cost",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0, 0 - 2.25, "Locations with longest delays:",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0, 0 - 7.75, "\n".join(top_hotspots),
               ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

    if show_metex_weather_cells:
        plot_weather_cells(base_map, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=(1.05, 0.975))

    # Save figure
    save_prototype_hotpots_fig(fig, "annual-delays-200607-201415", "Hotspots",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# Plot hotspots of delay minutes
def plot_hotspots_for_delays(route_name='Anglia', weather_category='Wind', update_data=False, seed=1, cmap_name='Reds',
                             show_metex_weather_cells=True, show_osm_landuse_forest=True,
                             show_nr_hazardous_trees=True, save_as=None, dpi=None):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)
    :param seed: [int] (default: 1)
    :param cmap_name: [str] (default: 'Reds')
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param save_as: [str; None (default)]
    :param dpi: [numbers.Number; None (default)]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False
        seed = 1
        cmap_name = 'Reds'
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        show_nr_hazardous_trees = True
        save_as = ".tif"
        dpi = None

        plot_hotspots_for_delays(route_name, weather_category, update_data, seed, cmap_name, show_metex_weather_cells,
                                        show_osm_landuse_forest, show_nr_hazardous_trees, save_as, dpi)
    """
    hotspots_data_init = get_midpoints_for_plotting_hotspots(route_name, weather_category,
                                                             sort_by=['DelayMinutes', 'IncidentCount', 'DelayCost'],
                                                             update=update_data)
    notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutes.notnull()]

    # Set a seed number
    np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.NaturalBreaks(y=notnull_data.DelayMinutes.values, k=6, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
    # hotspots_data['jenks_bins'].fillna(-1, inplace=True)
    jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)  # 'OrRd', 'RdPu', 'Oranges', 'YlOrBr'
    colours = cmap(np.linspace(0., 1., len(jenks_labels)))
    marker_size = np.linspace(1., 2.2, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = list(breaks.bins)
    for b in range(len(bins)):
        idx_0, idx_1 = hotspots_data.DelayMinutes <= bins[b], hotspots_data.DelayMinutes > bins[b - 1]
        if bins[b] == min(bins):
            plotting_data = hotspots_data[idx_0]
        elif bins[b] == max(bins):
            plotting_data = hotspots_data[idx_1]
        else:
            plotting_data = hotspots_data[idx_0 & idx_1]
        for i in plotting_data.index:
            mid_x, mid_y = base_map(plotting_data.MidLongitude[i], plotting_data.MidLatitude[i])
            base_map.plot(mid_x, mid_y, zorder=2, marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b],
                          markeredgecolor='w')

    # Add a colour bar
    cb = colour_bar_index(no_of_colours=len(jenks_labels), cmap_param=cmap, shrink=0.4, labels=jenks_labels, pad=0.068)
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_font_properties(matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
    cb.ax.tick_params(labelsize=14)
    cb.set_alpha(1.0)
    cb.draw_all()

    # Add descriptions
    cb.ax.text(0., 0 + 6.75, "Total delay minutes (2006/07-2018/19)",
               ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
    # Show highest delays, in descending order
    cb.ax.text(0., 0 - 1.45, "Locations accounted for most delays:",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),  # highest
               ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

    # Show Weather cells
    if show_metex_weather_cells:
        plot_weather_cells(base_map, route_name, legend_loc=(1.05, 0.95))

    # Show Vegetation
    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    # Show hazardous trees
    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=(1.05, 0.975))

    # Save figure
    save_prototype_hotpots_fig(fig, "delays",  "Hotspots",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# Plot hotspots in terms of incident frequency
def plot_hotspots_for_incident_frequency(route_name='Anglia', weather_category='Wind', update_data=False, seed=1,
                                         cmap_name='PuRd', show_metex_weather_cells=True,
                                         show_osm_landuse_forest=True, show_nr_hazardous_trees=True,
                                         save_as=None, dpi=None):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)
    :param seed: [int] (default: 1)
    :param cmap_name: [str] (default: 'Reds')
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param save_as: [str; None (default)]
    :param dpi: [numbers.Number; None (default)]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False
        seed = 1
        cmap_name = 'PuRd'
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        show_nr_hazardous_trees = True
        save_as = ".tif"
        dpi = None

        plot_hotspots_for_incident_frequency(route_name, weather_category, update_data, seed, cmap_name,
                                             show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                                             save_as, dpi)
    """
    hotspots_data_init = get_midpoints_for_plotting_hotspots(route_name, weather_category,
                                                             sort_by=['IncidentCount', 'DelayCost', 'DelayMinutes'],
                                                             update=update_data)
    notnull_data = hotspots_data_init[hotspots_data_init.IncidentCount.notnull()]

    # Set a seed number
    np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.NaturalBreaks(y=notnull_data.IncidentCount.values, k=6, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))

    jenks_labels = ["<= %d  / %d locations" % (b, c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)  # 'Oranges', 'RdPu', 'Purples'
    colours = cmap(np.linspace(0, 1., len(jenks_labels)))
    marker_size = np.linspace(1.0, 2.2, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = list(breaks.bins)
    for b in range(len(bins)):
        ind1 = hotspots_data.IncidentCount <= bins[b]
        ind2 = hotspots_data.IncidentCount > bins[b - 1]
        if np.isnan(bins[b]):
            plotting_data = hotspots_data[hotspots_data.IncidentCount.isnull()]
        elif bins[b] == np.nanmin(bins):
            plotting_data = hotspots_data[ind1]
        elif bins[b] == np.nanmax(bins):
            plotting_data = hotspots_data[ind2]
        else:
            plotting_data = hotspots_data[ind1 & ind2]

        for i in plotting_data.index:
            mid_lat = plotting_data.MidLatitude[i]
            mid_lon = plotting_data.MidLongitude[i]
            x_mid_pt, y_mid_pt = base_map(mid_lon, mid_lat)
            base_map.plot(x_mid_pt, y_mid_pt, zorder=2,
                          marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b], markeredgecolor='w')

    # Add a colour bar
    cb = colour_bar_index(no_of_colours=len(jenks_labels), cmap_param=cmap, shrink=0.4, labels=jenks_labels, pad=0.068)
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_font_properties(matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
    cb.ax.tick_params(labelsize=14)
    cb.set_alpha(1.0)
    cb.draw_all()

    # Add descriptions
    cb.ax.text(0., 0 + 6.75, "Count of Incidents (2006/07-2018/19)",
               ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
    # Show highest frequency, in descending order
    cb.ax.text(0., 0 - 1.45, "Most incident-prone locations: ",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
               ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

    if show_metex_weather_cells:
        plot_weather_cells(base_map, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=(1.05, 0.975))

    save_prototype_hotpots_fig(fig, "frequency",  "Hotspots",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# Plot hotspots of delay costs
def plot_hotspots_for_costs(route_name='Anglia', weather_category='Wind', update_data=False,
                            seed=1, cmap_name='YlGnBu', show_metex_weather_cells=False,
                            show_osm_landuse_forest=False, show_nr_hazardous_trees=False, save_as=None, dpi=None):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)
    :param seed: [int] (default: 1)
    :param cmap_name: [str] (default: 'Reds')
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param save_as: [str; None (default)]
    :param dpi: [numbers.Number; None (default)]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False
        seed = 1
        cmap_name = 'YlGnBu'
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        show_nr_hazardous_trees = True
        save_as = ".tif"
        dpi = None

        plot_hotspots_for_costs(route_name, weather_category, update_data, seed, cmap_name, show_metex_weather_cells,
                                show_osm_landuse_forest, show_nr_hazardous_trees, save_as, dpi)
    """
    hotspots_data_init = get_midpoints_for_plotting_hotspots(route_name, weather_category,
                                                             sort_by=['DelayCost', 'IncidentCount', 'DelayMinutes'],
                                                             update=update_data)
    hotspots_data_init.replace(to_replace={'DelayCost': {0: np.nan}}, inplace=True)
    notnull_data = hotspots_data_init[hotspots_data_init.DelayCost.notnull()]

    # Set a seed number
    np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.NaturalBreaks(y=notnull_data.DelayCost.values, k=5, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))
    # df.drop('jenks_bins', axis=1, inplace=True)
    hotspots_data.jenks_bins.fillna(-1, inplace=True)
    jenks_labels = ['<= £%s  / %s locations' % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]
    jenks_labels.insert(0, 'N/A (no cost)  / %s locations' % len(hotspots_data[hotspots_data['DelayCost'].isnull()]))

    cmap = plt.get_cmap(cmap_name)  # 'RdPu'
    colour_array = np.linspace(0, 1., len(jenks_labels))
    colours = cmap(colour_array)
    marker_size = np.linspace(0.8, 2.3, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.90))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = [np.nan] + list(breaks.bins)
    for b in range(len(bins)):
        idx_0, idx_1 = hotspots_data.DelayCost <= bins[b], hotspots_data.DelayCost > bins[b - 1]
        if np.isnan(bins[b]):
            plotting_data = hotspots_data[hotspots_data.DelayCost.isnull()]
        elif bins[b] == np.nanmin(bins):
            plotting_data = hotspots_data[idx_0]
        elif bins[b] == np.nanmax(bins):
            plotting_data = hotspots_data[idx_1]
        else:
            plotting_data = hotspots_data[idx_0 & idx_1]
        for i in plotting_data.index:
            mid_lat = plotting_data.MidLatitude[i]
            mid_lon = plotting_data.MidLongitude[i]
            x_mid_pt, y_mid_pt = base_map(mid_lon, mid_lat)
            base_map.plot(x_mid_pt, y_mid_pt, zorder=2,
                          marker='o', color=colours[b], alpha=0.9,
                          markersize=marker_size[b], markeredgecolor='w', markeredgewidth=1)

    # Add a colour bar
    cb = colour_bar_index(no_of_colours=len(jenks_labels), cmap_param=cmap, shrink=0.4, labels=jenks_labels, pad=0.068)
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_font_properties(matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
    cb.ax.tick_params(labelsize=14)
    cb.set_alpha(1.0)
    cb.draw_all()

    # Add descriptions
    cb.ax.text(0., 0 + 6.75, "Compensation payments (2006/07-2018/19)",
               ha='left', va='bottom', size=13, color='#555555', weight='bold', fontname='Cambria')
    # Show highest cost, in descending order
    cb.ax.text(0., 0 - 1.45, "Locations accounted for most cost: ",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
               ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

    if show_metex_weather_cells:
        plot_weather_cells(base_map, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=(1.05, 0.975))

    save_prototype_hotpots_fig(fig, "costs",  "Hotspots",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# Plot hotspots in terms of delay minutes per incident
def plot_hotspots_for_delays_per_incident(route_name='Anglia', weather_category='Wind', update_data=False,
                                          seed=1, cmap_name='BrBG', show_metex_weather_cells=False,
                                          show_osm_landuse_forest=False, show_nr_hazardous_trees=False,
                                          save_as=None, dpi=None):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)
    :param seed: [int] (default: 1)
    :param cmap_name: [str] (default: 'Reds')
    :param show_metex_weather_cells: [bool] (default: True)
    :param show_osm_landuse_forest: [bool] (default: True)
    :param show_nr_hazardous_trees: [bool] (default: True)
    :param save_as: [str; None (default)]
    :param dpi: [numbers.Number; None (default)]

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False
        seed = 1
        cmap_name = 'BrBG'
        show_metex_weather_cells = True
        show_osm_landuse_forest = True
        show_nr_hazardous_trees = True
        save_as = ".tif"
        dpi = None

        plot_hotspots_for_delays_per_incident(route_name, weather_category, update_data, seed, cmap_name,
                                              show_metex_weather_cells, show_osm_landuse_forest,
                                              show_nr_hazardous_trees, save_as, dpi)
    """
    hotspots_data_init = get_midpoints_for_plotting_hotspots(route_name, weather_category, update=update_data)
    hotspots_data_init['DelayMinutesPerIncident'] = hotspots_data_init.DelayMinutes.div(
        hotspots_data_init.IncidentCount)
    hotspots_data_init.sort_values(by='DelayMinutesPerIncident', ascending=False, inplace=True)

    notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutesPerIncident.notnull()]

    # Set a seed number
    np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.NaturalBreaks(y=notnull_data.DelayMinutesPerIncident.values, k=6)
    hotspots_data = hotspots_data_init.join(pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
    # data['jenks_bins'].fillna(-1, inplace=True)
    jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)
    colours = cmap(np.linspace(0, 1, len(jenks_labels)))
    marker_size = np.linspace(1.0, 2.2, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = list(breaks.bins)
    for b in range(len(bins)):
        idx_0 = hotspots_data.DelayMinutesPerIncident <= bins[b]
        idx_1 = hotspots_data.DelayMinutesPerIncident > bins[b - 1]
        if bins[b] == min(bins):
            plotting_data = hotspots_data[idx_0]
        elif bins[b] == max(bins):
            plotting_data = hotspots_data[idx_1]
        else:
            plotting_data = hotspots_data[idx_0 & idx_1]
        for i in plotting_data.index:
            mid_lat = plotting_data.MidLatitude[i]
            mid_lon = plotting_data.MidLongitude[i]
            mid_x, mid_y = base_map(mid_lon, mid_lat)
            base_map.plot(mid_x, mid_y, zorder=2, marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b],
                          markeredgecolor='w')

    # Add a colour bar
    cb = colour_bar_index(no_of_colours=len(jenks_labels), cmap_param=cmap, shrink=0.4, labels=jenks_labels, pad=0.068)
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_font_properties(matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold',
                                                                     fname="C:\\Windows\\Fonts\\Times.ttf"))
    cb.ax.tick_params(labelsize=14)
    cb.set_alpha(1.0)
    cb.draw_all()

    # Add descriptions
    cb.ax.text(0., 0 + 6.75, "Delay per incident (2006/07 to 2018/19)",
               ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
    # Show highest delay min. per incident, in descending order
    cb.ax.text(0., 0 - 1.45, "Longest delays per incident:",
               ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
    cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
               ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

    if show_metex_weather_cells:
        plot_weather_cells(base_map, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    # Show hazardous trees?
    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, legend_loc=(1.05, 0.975))

    save_prototype_hotpots_fig(fig, "delays-per-incident", "Hotspots",
                               show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                               save_as, dpi, verbose=True)


# Plotting
def plot_hotspots_on_route(route_name='Anglia', weather_category='Wind', update_data=False):
    """
    :param route_name: [str; None] (default: 'Anglia')
    :param weather_category: [str; None] (default: 'Wind')
    :param update_data: [bool] (default: False)

    Testing e.g.
        route_name = 'Anglia'
        weather_category = 'Wind'
        update_data = False

        plot_hotspots_on_route(route_name, weather_category, update_data)
    """
    import settings
    settings.mpl_preferences(use_cambria=False, reset=False)
    settings.np_preferences(reset=False)
    settings.pd_preferences(reset=False)

    if confirmed():  # No need of dpi for ".pdf" or ".svg"

        fmt = ".png"

        plot_base_map_plus(False, False, False, False, (1.05, 0.85), fmt)
        plot_base_map_plus(True, False, False, False, (1.05, 0.85), fmt)
        plot_base_map_plus(True, True, False, False, (1.05, 0.85), fmt)
        plot_base_map_plus(True, True, False, True, (1.05, 0.85), fmt)  # Fig. 1.
        plot_base_map_plus(True, True, False, True, (1.05, 0.85), ".tif", dpi=600)  # Fig. 1.

        # Annual delays     # Fig. 2.
        plot_hotspots_for_annual_delays(route_name, weather_category, update_data, 'Set1', True, True, True, fmt)
        plot_hotspots_for_annual_delays(route_name, weather_category, update_data, 'Set1', True, True, True, ".tif",
                                        dpi=600)

        # Delays    # Fig. 3.
        plot_hotspots_for_delays(route_name, weather_category, update_data, 1, 'Reds', False, False, False, fmt)
        plot_hotspots_for_delays(route_name, weather_category, update_data, 1, 'Reds', True, True, True, ".tif",
                                 dpi=600)

        # Cost
        plot_hotspots_for_costs(route_name, weather_category, update_data, 1, 'YlGnBu', False, False, False, fmt)
        plot_hotspots_for_costs(route_name, weather_category, update_data, 1, 'YlGnBu', True, True, True, fmt)

        # Frequency
        plot_hotspots_for_incident_frequency(route_name, weather_category, update_data,
                                             1, 'PuRd', False, False, False, fmt)
        plot_hotspots_for_incident_frequency(route_name, weather_category, update_data,
                                             1, 'PuRd', True, True, True, fmt)

        # Delay minutes per incident
        plot_hotspots_for_delays_per_incident(route_name, weather_category, update_data,
                                              1, 'BrBG', False, False, False, fmt)
        plot_hotspots_for_delays_per_incident(route_name, weather_category, update_data,
                                              1, 'BrBG', True, True, True, fmt)
