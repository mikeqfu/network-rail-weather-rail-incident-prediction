""" Plotting hotspots of Weather-related Incidents in the context of wind-related delays """

import itertools
import os
import re
import shutil

import PIL.Image
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import pandas as pd
import pydriosm as dri
import pysal.viz.mapclassify.classifiers
import shapely.geometry
import shapely.ops
from pyhelpers.dir import cd, cdd
from pyhelpers.geom import get_geometric_midpoint
from pyhelpers.misc import colour_bar_index, confirmed
from pyhelpers.store import load_pickle, save, save_pickle, save_svg_as_emf

import models.prototype.tools as proto_utils
import mssqlserver.metex
import mssqlserver.vegetation


# Create a boundary based on specified bounds (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat) ===========================
def create_boundary_polygon(bounds):
    """
    :param bounds: [tuple] (llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat)
    :return:
    """
    llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = bounds
    boundary_polygon = shapely.geometry.Polygon([(llcrnrlon, llcrnrlat),
                                                 (llcrnrlon, urcrnrlat),
                                                 (urcrnrlon, urcrnrlat),
                                                 (urcrnrlon, llcrnrlat)])
    return boundary_polygon


#
def prep_shp_layer_files(osm_subregion, relevant_osm_layers=('railways', 'landuse', 'natural'), rm_shp_zip=True):
    osm_dir, osm_subregion = cdd("Network\\OSM"), osm_subregion.lower()
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


# Get the path to .shp file for basemap loading ======================================================================
def get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature=None,
                                  boundary_polygon=None, sub_area_name=None,
                                  rm_other_feat=False, update=False):
    """
    :param osm_subregion: [str] e.g. osm_subregion='England'
    :param osm_layer: [str] e.g. osm_layer='railways'
    :param osm_feature: [str; None (default)] e.g. osm_feature='rail'
    :param rm_other_feat: [bool]
    :param boundary_polygon: [tuple]
    :param sub_area_name: [str; None (default)]
    :param update: [bool]
    :return: [str]
    """
    osm_dir, shp_file_dir_name = prep_shp_layer_files(osm_subregion)

    shp_file_dir = cd(osm_dir, shp_file_dir_name, osm_layer)

    shp_filename = [f for f in os.listdir(shp_file_dir)
                    if re.match(r"gis_osm_{}(_a)?_free_1\.shp".format(osm_layer), f)][0]

    try:
        path_to_shp_file = dri.find_osm_shp_file(osm_subregion, osm_layer, osm_feature, shp_file_dir)[0]
    except IndexError:
        if osm_feature is not None:
            assert isinstance(osm_feature, str)
            shp_feat_name = "{}_{}".format(osm_layer, osm_feature)

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
        suffix = sub_area_name.lower() if sub_area_name else "_sub_area"
        sub_shp_filename = shp_filename.strip(".shp") + "_{}.shp".format(suffix)
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
""" Prep base maps """


# Get the basemap ready
def plot_base_map(projection='tmerc', railway_line_color='#3D3D3D', legend_loc=(1.05, 0.85)):
    """
    :param projection: [str]
    :param railway_line_color: [str]
    :param legend_loc [tuple]
    :return:
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


# Show Weather cells on the map ======================================================================================
def plot_weather_cells(base_map=None, update=False, route_name=None, weather_cell_colour='#D5EAFF',
                       legend_loc=(1.05, 0.85)):
    """
    :param base_map: [mpl_toolkits.basemap.Basemap] basemap object
    :param update: [bool]
    :param route_name: [str] Route
    :param weather_cell_colour: [str] default '#add6ff'; alternative '#99ccff', '#fff68f
    :param legend_loc [tuple]
    :return:
    """
    if base_map is None:
        _, base_map = plot_base_map()

    print("Plotting the Weather cells ... ", end="")

    # Get Weather cell data
    data = mssqlserver.metex.get_weather_cell(update=update)
    data = mssqlserver.metex.get_subset(data, route_name)
    # Drop duplicated Weather cell data
    unhashable_cols = ('Polygon_WGS84', 'Polygon_OSGB36', 'IMDM', 'Route')
    data.drop_duplicates(subset=[x for x in list(data.columns) if x not in unhashable_cols], inplace=True)

    # Plot the Weather cells one by one
    for i in data.index:
        ll_x, ll_y = base_map(data['ll_Longitude'][i], data['ll_Latitude'][i])
        ul_x, ul_y = base_map(data['ul_Longitude'][i], data['ul_Latitude'][i])
        ur_x, ur_y = base_map(data['ur_Longitude'][i], data['ur_Latitude'][i])
        lr_x, lr_y = base_map(data['lr_Longitude'][i], data['lr_Latitude'][i])
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


# Demonstrate the OSM natural forest on the map ======================================================================
def plot_osm_forest_and_tree(base_map=None, osm_landuse_forest_colour='#72886e', fill_forest_patches=False,
                             add_osm_natural_tree=False, legend_loc=(1.05, 0.85)):
    """
    :param base_map: default None
    :param osm_landuse_forest_colour: default '#7f987b'; alternatives '#8ea989', '#72946c', '#72946c'
    :param add_osm_natural_tree: default False
    :param fill_forest_patches: default False
    :param legend_loc [tuple]
    :return:
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

    # # Alternative 1:
    # plt.plot([], "o", ms=15, color=osm_landuse_forest_colour, markeredgecolor='none', alpha=.65,
    #          label="Vegetation (OSM 'forest')")

    # # Alternative 2:
    # plt.fill_between([], [], 0, label="Vegetation (OSM 'forest')", alpha=.65,
    #                  color=osm_landuse_forest_colour)

    # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
    font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
    plt.legend(scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True, bbox_to_anchor=legend_loc)

    print("Done.")


# Show hazardous trees on the map
def plot_hazardous_trees(base_map=None, route_name=None, hazardous_tree_colour='#ab790a', legend_loc=(1.05, 0.85)):
    """
    :param base_map: 
    :param route_name: 
    :param hazardous_tree_colour: alternative '#886008', '#6e376e', '#5a7b6c'
    :param legend_loc:
    :return: 
    """
    if base_map is None:
        _, base_map = plot_base_map()

    print("Plotting the hazardous trees ... ", end="")

    hazardous_trees = mssqlserver.vegetation.view_hazardous_trees(route_name)

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
def plot_base_map_plus(route_name='Anglia', show_metex_weather_cells=True, show_osm_landuse_forest=True,
                       add_osm_natural_tree=False, show_nr_hazardous_trees=True,
                       legend_loc=(1.05, 0.85), save_as=".pdf", dpi=None):
    # Plot basemap
    fig, base_map = plot_base_map(projection='tmerc', legend_loc=legend_loc)

    # Show Weather cells
    if show_metex_weather_cells:
        plot_weather_cells(base_map, route_name=route_name, legend_loc=legend_loc)

    # Show Vegetation
    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=add_osm_natural_tree, legend_loc=legend_loc)

    # Show hazardous trees
    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, route_name=route_name, legend_loc=legend_loc)

    # Add a subplot of mini map of GB, e.g. ax = plt.subplot(); ax.plot(range(10))
    fig.add_subplot()

    # Add an axes at position [left, bottom, width, height]
    sr = fig.add_axes([0.58, 0.01, 0.40, 0.40], frameon=True)  # quantities are in fractions of figure width and height
    sr.imshow(PIL.Image.open(cdd("Network\\Routes\\Map", "NR-Routes-edited-1.tif")))  # "Routes-edited-0.png"
    sr.axis('off')

    # Save the figure
    if save_as:
        print("Saving the figure ... ", end="")
        filename_suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                              ['cell', 'veg', 'haz'])
        fig_filename = '_'.join(['Basemap'] + [v for s, v in filename_suffix if s is True])
        path_to_fig = proto_utils.cd_prototype_fig_pub("Basemap", fig_filename + save_as)
        fig.savefig(path_to_fig, dpi=dpi)
        print("Done.")
        if save_as == ".svg":
            save_svg_as_emf(path_to_fig, path_to_fig.replace(save_as, ".emf"))


# ====================================================================================================================
""" Get 'hotspot' data """


# Get coordinates of points from a .shp file, by subregion, layer and feature
def get_shp_coordinates(osm_subregion, osm_layer, osm_feature=None, boundary_polygon=None, sub_area_name=None,
                        update=False):
    """
    :param osm_subregion: [str] e.g. osm_subregion='England'
    :param osm_layer: [str] e.g. osm_layer='railways'
    :param osm_feature: [str] e.g. osm_feature='rail'
    :param boundary_polygon:
    :param sub_area_name:
    :param update: [bool]
    :return:
    """
    suffix = "coordinates"
    path_to_shp_filename = get_shp_file_path_for_basemap(osm_subregion, osm_layer, osm_feature,
                                                         boundary_polygon, sub_area_name)
    path_to_shp = path_to_shp_filename + ".shp"
    path_to_shp_coordinates_pickle = path_to_shp_filename + "_" + suffix + ".pickle"

    if os.path.isfile(path_to_shp_coordinates_pickle) and not update:
        shp_coordinates = load_pickle(path_to_shp_coordinates_pickle)
    else:
        try:
            railways_shp_data = dri.read_shp(path_to_shp, mode='geopandas')
            shp_coordinates = shapely.geometry.MultiPoint(
                list(itertools.chain(*(l.coords for l in railways_shp_data.geometry))))
        except Exception as e:
            print(e)
            shp_coordinates = None
        save_pickle(shp_coordinates, path_to_shp_coordinates_pickle)
    return shp_coordinates


# Get the data for plotting
def get_schedule8_incident_hotspots(route_name=None, weather_category=None, sort_by=None, update=False):
    """
    :param route_name: [NoneType] or [str]
    :param weather_category: [NoneType] or [str]
    :param sort_by: [NoneType] or [list]
    :param update: [bool]
    :return:
    """
    path_to_file = mssqlserver.metex.cdd_metex_db_views("Schedule8_incidents_hotspots.pickle")

    if os.path.isfile(path_to_file) and not update:
        hotspots_data = load_pickle(path_to_file)
    else:
        # Get TRUST (by incident location, i.e. by STANOX section)
        schedule8_costs_by_location = mssqlserver.metex.view_schedule8_costs_by_location()
        schedule8_costs_by_location['StartPoint'] = [
            shapely.geometry.Point(long, lat) for long, lat in
            zip(schedule8_costs_by_location.StartLongitude, schedule8_costs_by_location.StartLatitude)]
        schedule8_costs_by_location['EndPoint'] = [
            shapely.geometry.Point(long, lat) for long, lat in
            zip(schedule8_costs_by_location.EndLongitude, schedule8_costs_by_location.EndLatitude)]

        # Find a pseudo midpoint for each recorded incident
        pseudo_midpoints = schedule8_costs_by_location[['StartPoint', 'EndPoint']].apply(
            lambda x: get_geometric_midpoint(x[0], x[1]), axis=1)

        # Get reference points (coordinates), given subregion and layer (i.e. 'railways' in this case) of OSM .shp file
        if route_name:
            path_to_boundary_polygon = cdd("Network\\Routes\\{}\\boundary_polygon.pickle".format(route_name))
            boundary_polygon = load_pickle(path_to_boundary_polygon)
            sub_area_name = route_name
        else:
            boundary_polygon, sub_area_name = None, None
        ref_points = get_shp_coordinates('England', 'railways', 'rail', boundary_polygon, sub_area_name)

        # Get rail coordinates closest to the midpoints between starts and ends
        schedule8_costs_by_location['MidPoint'] = pseudo_midpoints.map(
            lambda x: shapely.ops.nearest_points(shapely.geometry.Point(x), ref_points)[1])

        midpoints = pd.DataFrame(((pt.x, pt.y) for pt in schedule8_costs_by_location.MidPoint),
                                 index=schedule8_costs_by_location.index,
                                 columns=['MidLongitude', 'MidLatitude'])

        hotspots_data = schedule8_costs_by_location.join(midpoints)

        save(hotspots_data, path_to_file)

    if sort_by:
        hotspots_data.sort_values(sort_by, ascending=False, inplace=True)

    hotspots_data = mssqlserver.metex.get_subset(hotspots_data, route_name, weather_category)
    hotspots_data.index = range(len(hotspots_data))

    return hotspots_data


# ====================================================================================================================
""" Plot 'hotspots' """


# Plot hotspots of delays for every financial year (2006/07-2014/15) =================================================
def hotspots_annual_delays(route_name='Anglia', weather_category='Wind', update=False,
                           cmap_name='Set1',
                           show_metex_weather_cells=False, show_osm_landuse_forest=False, show_nr_hazardous_trees=False,
                           save_as=".tif", dpi=None):
    # Get data
    path_to_pickle = mssqlserver.metex.cdd_metex_db_views("hotspots_annual_delays.pickle")
    try:
        hotspots_data = load_pickle(path_to_pickle)
        hotspots_data = mssqlserver.metex.get_subset(hotspots_data, route_name, weather_category)
    except FileNotFoundError:
        schedule8_data = mssqlserver.metex.view_schedule8_costs_by_datetime_location(update=update)
        schedule8_data = mssqlserver.metex.get_subset(schedule8_data, route_name, weather_category)
        group_features = ['FinancialYear', 'WeatherCategory', 'Route', 'StanoxSection',
                          'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
        schedule8_data = schedule8_data.groupby(group_features). \
            agg({'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum}). \
            reset_index()
        hotspots = get_schedule8_incident_hotspots(route_name, weather_category)
        hotspots_data = schedule8_data.merge(
            hotspots[group_features[1:] + ['MidLatitude', 'MidLongitude']], how='left', on=group_features[1:])
        hotspots_data.sort_values(by=['DelayMinutes', 'DelayCost', 'IncidentCount'], ascending=False, inplace=True)
        save_pickle(hotspots_data, path_to_pickle)

    yearly_cost = hotspots_data.groupby('FinancialYear'). \
        agg({'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum})

    # Examine only 2006/07 - 2014/15
    yearly_cost = yearly_cost.loc[2006:2014]

    # Labels
    years = [str(y) for y in yearly_cost.index]
    f_years = ['/'.join([y0, str(y1)[-2:]]) for y0, y1 in zip(years, pd.np.array(yearly_cost.index) + pd.np.array([1]))]

    d_label = ["%s  (%s min." % (fy, format(int(d), ",")) for fy, d in zip(f_years, yearly_cost.DelayMinutes)]
    c_label = ["  / £%.2f" % round(c * 1e-6, 2) + "M)" for c in yearly_cost.DelayCost]
    label = [l for l in reversed([d + c for d, c in zip(d_label, c_label)])]

    cmap = plt.get_cmap(cmap_name)
    colours = cmap(pd.np.linspace(start=0, stop=1, num=9))
    colours = [c for c in reversed(colours)]

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    top_hotspots = []
    for y, fy in zip(years, f_years):
        plot_data = hotspots_data[hotspots_data.FinancialYear == int(y)][0:20]
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
        plot_weather_cells(base_map, route_name=route_name, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, route_name=route_name, legend_loc=(1.05, 0.975))

    # Save figure
    proto_utils.save_hotpots_fig(fig, "hotspots_annual_delays",
                                 show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                                 save_as, dpi)


# Plot hotspots of delay minutes =====================================================================================
def hotspots_delays(route_name='Anglia', weather_category='Wind', update=False,
                    seed=1, cmap_name='Reds',
                    show_metex_weather_cells=False, show_osm_landuse_forest=False, show_nr_hazardous_trees=False,
                    save_as=".tif", dpi=None):
    # Get hotspots data
    hotspots_data_init = get_schedule8_incident_hotspots(route_name, weather_category,
                                                         sort_by=['DelayMinutes', 'IncidentCount', 'DelayCost'],
                                                         update=update)
    notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutes.notnull()]

    # Set a seed number
    pd.np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.Natural_Breaks(y=notnull_data.DelayMinutes.values, k=6, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
    # hotspots_data['jenks_bins'].fillna(-1, inplace=True)
    jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)  # 'OrRd', 'RdPu', 'Oranges', 'YlOrBr'
    colour_array = pd.np.linspace(0, 1., len(jenks_labels))
    colours = cmap(colour_array)
    marker_size = pd.np.linspace(1, 2.2, len(jenks_labels)) * 12

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
            base_map.plot(mid_x, mid_y, zorder=2,
                          marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b], markeredgecolor='w')

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
        plot_hazardous_trees(base_map, route_name, legend_loc=(1.05, 0.975))

    # Save figure
    proto_utils.save_hotpots_fig(fig, "hotspots_delays", show_metex_weather_cells, show_osm_landuse_forest,
                                 show_nr_hazardous_trees, save_as, dpi)


# Plot hotspots in terms of incident frequency =======================================================================
def hotspots_frequency(route_name='Anglia', weather_category='Wind', update=False,
                       seed=1, cmap_name='PuRd',
                       show_metex_weather_cells=False, show_osm_landuse_forest=False, show_nr_hazardous_trees=False,
                       save_as=".tif", dpi=None):
    # Get data
    hotspots_data_init = get_schedule8_incident_hotspots(route_name, weather_category,
                                                         sort_by=['IncidentCount', 'DelayCost', 'DelayMinutes'],
                                                         update=update)
    notnull_data = hotspots_data_init[hotspots_data_init.IncidentCount.notnull()]

    # Set a seed number
    pd.np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.Natural_Breaks(y=notnull_data.IncidentCount.values, k=6, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))

    jenks_labels = ["<= %d  / %d locations" % (b, c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)  # 'Oranges', 'RdPu', 'Purples'
    colours = cmap(pd.np.linspace(0, 1., len(jenks_labels)))
    marker_size = pd.np.linspace(1.0, 2.2, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.9))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = list(breaks.bins)
    for b in range(len(bins)):
        ind1 = hotspots_data.IncidentCount <= bins[b]
        ind2 = hotspots_data.IncidentCount > bins[b - 1]
        if pd.np.isnan(bins[b]):
            plotting_data = hotspots_data[hotspots_data.IncidentCount.isnull()]
        elif bins[b] == pd.np.nanmin(bins):
            plotting_data = hotspots_data[ind1]
        elif bins[b] == pd.np.nanmax(bins):
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
        plot_weather_cells(base_map, route_name=route_name, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, route_name=route_name, legend_loc=(1.05, 0.975))

    proto_utils.save_hotpots_fig(fig, "hotspots_frequency", show_metex_weather_cells, show_osm_landuse_forest,
                                 show_nr_hazardous_trees, save_as, dpi)


# Plot hotspots of delay cost ========================================================================================
def hotspots_cost(route_name='Anglia', weather_category='Wind', update=False,
                  seed=1, cmap_name='YlGnBu',
                  show_metex_weather_cells=False, show_osm_landuse_forest=False, show_nr_hazardous_trees=False,
                  save_as=".tif", dpi=None):
    # Get data
    hotspots_data_init = get_schedule8_incident_hotspots(route_name, weather_category,
                                                         sort_by=['DelayCost', 'IncidentCount', 'DelayMinutes'],
                                                         update=update)
    hotspots_data_init.replace(to_replace={'DelayCost': {0: pd.np.nan}}, inplace=True)
    notnull_data = hotspots_data_init[hotspots_data_init.DelayCost.notnull()]

    # Set a seed number
    pd.np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.Natural_Breaks(y=notnull_data.DelayCost.values, k=5, initial=100)
    hotspots_data = hotspots_data_init.join(pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))
    # df.drop('jenks_bins', axis=1, inplace=True)
    hotspots_data.jenks_bins.fillna(-1, inplace=True)
    jenks_labels = ['<= £%s  / %s locations' % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]
    jenks_labels.insert(0, 'N/A (no cost)  / %s locations' % len(hotspots_data[hotspots_data['DelayCost'].isnull()]))

    cmap = plt.get_cmap(cmap_name)  # 'RdPu'
    colour_array = pd.np.linspace(0, 1., len(jenks_labels))
    colours = cmap(colour_array)
    marker_size = pd.np.linspace(0.8, 2.3, len(jenks_labels)) * 12

    # Plot basemap (with railway tracks)
    fig, base_map = plot_base_map(legend_loc=(1.05, 0.90))
    fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

    bins = [pd.np.nan] + list(breaks.bins)
    for b in range(len(bins)):
        idx_0, idx_1 = hotspots_data.DelayCost <= bins[b], hotspots_data.DelayCost > bins[b - 1]
        if pd.np.isnan(bins[b]):
            plotting_data = hotspots_data[hotspots_data.DelayCost.isnull()]
        elif bins[b] == pd.np.nanmin(bins):
            plotting_data = hotspots_data[idx_0]
        elif bins[b] == pd.np.nanmax(bins):
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
        plot_weather_cells(base_map, route_name=route_name, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, route_name=route_name, legend_loc=(1.05, 0.975))

    proto_utils.save_hotpots_fig(fig, "hotspots_cost", show_metex_weather_cells, show_osm_landuse_forest,
                                 show_nr_hazardous_trees, save_as, dpi)


# Plot hotspots in terms of delay minutes per incident ===============================================================
def hotspots_delays_per_incident(route_name='Anglia', weather_category='Wind', update=False,
                                 seed=1, cmap_name='BrBG',
                                 show_metex_weather_cells=False, show_osm_landuse_forest=False,
                                 show_nr_hazardous_trees=False,
                                 save_as=".tif", dpi=None):
    # Get data
    hotspots_data_init = get_schedule8_incident_hotspots(route_name, weather_category, None, update)
    hotspots_data_init['DelayMinutesPerIncident'] = hotspots_data_init.DelayMinutes.div(
        hotspots_data_init.IncidentCount)
    hotspots_data_init.sort_values(by='DelayMinutesPerIncident', ascending=False, inplace=True)

    notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutesPerIncident.notnull()]

    # Set a seed number
    pd.np.random.seed(seed)

    # Calculate Jenks natural breaks for delay minutes
    breaks = pysal.viz.mapclassify.Natural_Breaks(y=notnull_data.DelayMinutesPerIncident.values, k=6)
    hotspots_data = hotspots_data_init.join(pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
    # data['jenks_bins'].fillna(-1, inplace=True)
    jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in zip(breaks.bins, breaks.counts)]

    cmap = plt.get_cmap(cmap_name)
    colours = cmap(pd.np.linspace(0, 1, len(jenks_labels)))
    marker_size = pd.np.linspace(1.0, 2.2, len(jenks_labels)) * 12

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
            base_map.plot(mid_x, mid_y, zorder=2,
                          marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b], markeredgecolor='w')

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
        plot_weather_cells(base_map, route_name=route_name, legend_loc=(1.05, 0.95))

    if show_osm_landuse_forest:
        plot_osm_forest_and_tree(base_map, add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

    # Show hazardous trees?
    if show_nr_hazardous_trees:
        plot_hazardous_trees(base_map, route_name=route_name, legend_loc=(1.05, 0.975))

    proto_utils.save_hotpots_fig(fig, "hotspots_delays_per_incident",
                                 show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                                 save_as, dpi)


#
def plot_anglia_hotspots(weather_category='Wind', update=False):
    import settings
    settings.mpl_preferences(use_cambria=False, reset=False)
    settings.np_preferences(reset=False)
    settings.pd_preferences(reset=False)

    if confirmed():  # No need of dpi for ".pdf" or ".svg"
        plot_base_map_plus('Anglia', False, False, False, False, (1.05, 0.85), ".png")
        plot_base_map_plus('Anglia', True, False, False, False, (1.05, 0.85), ".png")
        plot_base_map_plus('Anglia', True, True, False, False, (1.05, 0.85), ".png")
        plot_base_map_plus('Anglia', True, True, False, True, (1.05, 0.85), ".png")  # Fig. 1.
        plot_base_map_plus('Anglia', True, True, False, True, (1.05, 0.85), ".svg")  # Fig. 1.
        plot_base_map_plus('Anglia', True, True, False, True, (1.05, 0.85), ".tif", dpi=600)  # Fig. 1.

        # Delays yearly
        hotspots_annual_delays('Anglia', weather_category, update, 'Set1', False, False, False, ".png")
        hotspots_annual_delays('Anglia', weather_category, update, 'Set1', True, True, True, ".png")
        hotspots_annual_delays('Anglia', weather_category, update, 'Set1', True, True, True, ".svg")
        hotspots_annual_delays('Anglia', weather_category, update, 'Set1', True, True, True, ".tif", dpi=600)  # Fig. 2.

        # Delays
        hotspots_delays('Anglia', weather_category, update, 1, 'Reds', False, False, False, ".png")
        hotspots_delays('Anglia', weather_category, update, 1, 'Reds', True, True, True, ".png")
        hotspots_delays('Anglia', weather_category, update, 1, 'Reds', True, True, True, ".svg")
        hotspots_delays('Anglia', weather_category, update, 1, 'Reds', True, True, True, ".tif", dpi=600)  # Fig. 3.

        # Cost
        hotspots_cost('Anglia', weather_category, update, 1, 'YlGnBu', False, False, False, ".png")
        hotspots_cost('Anglia', weather_category, update, 1, 'YlGnBu', True, True, True, ".png")

        # Frequency
        hotspots_frequency('Anglia', weather_category, update, 1, 'PuRd', False, False, False, ".png")
        hotspots_frequency('Anglia', weather_category, update, 1, 'PuRd', True, True, True, ".png")

        # Delay minutes per incident
        hotspots_delays_per_incident(weather_category, 'Wind', update, 1, 'BrBG', False, False, False, ".png")
        hotspots_delays_per_incident(weather_category, 'Wind', update, 1, 'BrBG', True, True, True, ".png")
