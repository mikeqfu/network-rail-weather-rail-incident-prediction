"""
Hotspots of weather-related incidents (in the context of wind-related delays).
"""

import os

import mapclassify
import matplotlib
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import numpy as np
import pandas as pd
import PIL.Image
import scipy.spatial
import shapely.geometry
import shapely.ops
from pyhelpers.ops import colour_bar_index, confirmed
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_pickle, save_fig, save_pickle

from coordinator.geometry import get_shp_coordinates, get_shp_file_path_for_basemap
from preprocessor import METExLite, Vegetation
from utils import cd_models, cdd_network, get_subset, make_filename


class Hotspots:

    def __init__(self, database_name='NR_METEx_20150331'):
        self.Name = 'Hotspots of weather-related incidents in the context of wind-related delays'

        self.METEx = METExLite(database_name=database_name)
        self.Vegetation = Vegetation()

        self.Route = 'Anglia'
        self.WeatherCategory = 'Wind'

        self.Projection = 'tmerc'  # Transverse Mercator Projection

        self.LegendLoc = (1.05, 0.85)

        # matplotlib.use('TkAgg')
        mpl_preferences(font_name='Cambria')
        pd_preferences()

    # == Save outputs =================================================================================

    def save_prototype_hotpots_fig(self, fig, keyword, category,
                                   show_metex_weather_cells, show_osm_landuse_forest,
                                   show_nr_hazardous_trees,
                                   save_as, dpi, verbose):
        """
        Save a figure.

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

        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                         ['weather-grid', 'vegetation', 'hazard-trees'])
            filename = '-'.join([keyword] + [v for s, v in suffix if s is True])

            path_to_file = cd_models(
                "prototype", self.WeatherCategory.lower(), category, filename + save_as)

            save_fig(path_to_file, dpi=dpi, conv_svg_to_emf=True, verbose=verbose)

    # == Prepare base maps ============================================================================

    def plot_base_map(self, railway_line_color='#3D3D3D', legend_loc=(1.05, 0.85)):
        """
        Create a base map.

        :param legend_loc:
        :param railway_line_color: [str] (default: '#3D3D3D')
        :return: [tuple] (matplotlib.figure.Figure, mpl_toolkits.basemap.Basemap)

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_base_map()
        """

        print("Plotting the base map ... ", end="")

        plt.style.use('ggplot')
        # Default style: 'classic';
        # matplotlib.style.available gives the list of available styles
        fig = plt.figure(figsize=(11, 9))  # fig = plt.subplots(figsize=(11, 9))
        plt.subplots_adjust(left=0.001, bottom=0.000, right=0.6035, top=1.000)

        # Plot basemap
        base_map = mpl_toolkits.basemap.Basemap(
            llcrnrlon=-0.565409,  # ll[0] - 0.06 * width,
            llcrnrlat=51.23622,  # ll[1] - 0.06 + 0.002 * height,
            urcrnrlon=1.915975,  # ur[0] + extra * width,
            urcrnrlat=53.15000,  # ur[1] + extra + 0.01 * height,
            ellps='WGS84',
            lat_ts=0,
            lon_0=-2.,
            lat_0=49.,
            projection=self.Projection,
            resolution='i',
            suppress_ticks=True,
            epsg=27700)

        # base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)
        base_map.drawmapboundary(color='white', fill_color='white')
        # base_map.drawcoastlines()
        base_map.fillcontinents(color='#dcdcdc')  # color='#555555'

        # Add a layer for railway tracks
        boundary_polygon = shapely.geometry.Polygon(zip(base_map.boundarylons, base_map.boundarylats))

        path_to_shp_file = get_shp_file_path_for_basemap(
            osm_subregion='England', osm_layer='railways', osm_feature='rail',
            boundary_polygon=boundary_polygon, sub_area_name=self.Route.lower())

        base_map.readshapefile(
            shapefile=path_to_shp_file, name=self.Route.lower(), linewidth=1.5,
            color=railway_line_color,  # '#626262', '#939393', '#757575'
            zorder=4)

        # Show legend
        plt.plot([], '-', label="Railway track", linewidth=2.2, color=railway_line_color)

        # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        legend = plt.legend(
            numpoints=1, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc)
        frame = legend.get_frame()
        frame.set_edgecolor('none')
        frame.set_facecolor('none')

        print("Done.")

        self.__setattr__('BaseMap', base_map)

        return fig, base_map

    def plot_weather_cells(self, update=False, legend_loc=(1.05, 0.85)):
        """
        Show weather cells on the base map.

        :param legend_loc:
        :param update: [bool] (default: False)

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_weather_cells()
        """

        try:
            base_map = self.__getattribute__('BaseMap')
        except AttributeError:
            _, base_map = self.plot_base_map()

        print("Plotting the Weather cells ... ", end="")

        # Get Weather cell data

        data = self.METEx.get_weather_cell(update=update)
        data = get_subset(data, route_name='Anglia')
        # Drop duplicated Weather cell data
        unhashable_cols = ('Polygon_WGS84', 'Polygon_OSGB36', 'IMDM', 'Route')
        data.drop_duplicates(
            subset=[x for x in list(data.columns) if x not in unhashable_cols], inplace=True)

        weather_cell_colour = '#D5EAFF'  # '#add6ff', '#99ccff', '#fff68f

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
        plt.plot(
            [], 's', label="Weather cell", ms=25, color=weather_cell_colour, markeredgecolor='#433f3f',
            alpha=.5)

        # Show legend  # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        plt.legend(
            numpoints=1, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc)

        print("Done.")

    def plot_osm_forest_and_tree(self, fill_forest_patches=False, add_osm_natural_tree=False,
                                 legend_loc=(1.05, 0.85)):
        """
        Show the OSM natural forest on the base map.

        :param legend_loc:
        :param add_osm_natural_tree: [bool] (default: False)
        :param fill_forest_patches: [bool] (default: False)

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_osm_forest_and_tree()
        """

        try:
            base_map = self.__getattribute__('BaseMap')
        except AttributeError:
            _, base_map = self.plot_base_map()

        print("Plotting the OSM natural/forest ... ", end="")

        # OSM - landuse - forest
        boundary_polygon = shapely.geometry.Polygon(zip(base_map.boundarylons, base_map.boundarylats))
        bounded_landuse_forest_shp = get_shp_file_path_for_basemap(
            osm_subregion='England', osm_layer='landuse', osm_feature='forest',
            boundary_polygon=boundary_polygon, sub_area_name='anglia')

        osm_landuse_forest_colour = '#72886E'  # '#7f987b', '#8ea989', '#72946c', '#72946c'

        base_map.readshapefile(
            bounded_landuse_forest_shp, name='osm_landuse_forest', color=osm_landuse_forest_colour,
            zorder=3)

        # Fill the patches? Note this may take a long time and dramatically increase the file of the map
        if fill_forest_patches:
            print("\n")
            print("Filling the 'osm_landuse_forest' polygons ... ", end="")

            forest_polygons = [
                matplotlib.patches.Polygon(
                    p, fc=osm_landuse_forest_colour, ec=osm_landuse_forest_colour, zorder=4)
                for p in base_map.__getattribute__('osm_landuse_forest')]

            for i in range(len(forest_polygons)):
                plt.gca().add_patch(forest_polygons[i])

        # OSM - natural - tree
        if add_osm_natural_tree:
            bounded_natural_tree_shp = get_shp_file_path_for_basemap(
                'England', 'natural', 'tree', boundary_polygon, sub_area_name='anglia')
            base_map.readshapefile(
                bounded_natural_tree_shp, name='osm_natural_tree', color=osm_landuse_forest_colour,
                zorder=3)
            natural_tree_points = [
                shapely.geometry.Point(p) for p in base_map.__getattribute__('osm_natural_tree')]
            base_map.scatter(
                [geom.x for geom in natural_tree_points], [geom.y for geom in natural_tree_points],
                marker='o', s=2, facecolor='#008000', label="Tree", alpha=0.5, zorder=3)

        # Add label
        plt.scatter(
            [], [], marker="o",  # hatch=3 * "x", s=580,
            facecolor=osm_landuse_forest_colour, edgecolor='none', label="Vegetation (OSM 'forest')")

        # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        plt.legend(
            scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc)

        print("Done.")

    def plot_hazardous_trees(self, legend_loc=(1.05, 0.85)):
        """
        Show hazardous trees on the base map.

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_hazardous_trees()
        """

        try:
            base_map = self.__getattribute__('BaseMap')
        except AttributeError:
            _, base_map = self.plot_base_map()

        print("Plotting the hazardous trees ... ", end="")

        hazardous_trees = self.Vegetation.view_hazardous_trees()

        map_points = [
            shapely.geometry.Point(base_map(long, lat))
            for long, lat in zip(hazardous_trees.Longitude, hazardous_trees.Latitude)]
        hazardous_trees_points = shapely.geometry.MultiPoint(map_points)

        # Plot hazardous trees on the basemap
        hazardous_tree_colour = '#ab790a'  # '#886008', '#6e376e', '#5a7b6c'

        base_map.scatter(
            [geom.x for geom in hazardous_trees_points], [geom.y for geom in hazardous_trees_points],
            marker='x',  # edgecolor='w',
            s=20, lw=1.5, facecolor=hazardous_tree_colour, label="Hazardous trees", alpha=0.6,
            antialiased=True, zorder=3)

        # Show legend  # setfont = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        plt.legend(
            scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc)

        print("Done.")

    def plot_base_map_plus(self, show_metex_weather_cells=True, show_osm_landuse_forest=True,
                           add_osm_natural_tree=False, show_nr_hazardous_trees=True,
                           legend_loc=(1.05, 0.85), save_as=".tif", dpi=600, verbose=True):
        """
        Illustrate weather cell and associated natural features (incl. forest and hazardous trees)
        with the base map.

        :param legend_loc:
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param add_osm_natural_tree: [bool] (default: False)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [int; None (default)]
        :param verbose:

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_base_map_plus(save_as=None)
        """

        # Plot basemap
        fig, base_map = self.plot_base_map(legend_loc=legend_loc)

        # Show Weather cells
        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=legend_loc)

        # Show Vegetation
        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(
                add_osm_natural_tree=add_osm_natural_tree, legend_loc=legend_loc)

        # Show hazardous trees
        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=legend_loc)

        # Add an axes at position [left, bottom, width, height]
        sr = fig.add_axes([0.58, 0.01, 0.40, 0.40], frameon=True)
        # The quantities are in fractions of figure width and height

        sr.imshow(PIL.Image.open(cdd_network("routes\\map", "NR-Routes-edited-1.tif")))
        # Alternative: "Routes-edited-0.png"
        sr.axis('off')

        # Save the figure
        if save_as:
            self.save_prototype_hotpots_fig(
                fig, keyword="base", category="basemap",
                show_metex_weather_cells=show_metex_weather_cells,
                show_osm_landuse_forest=show_osm_landuse_forest,
                show_nr_hazardous_trees=show_nr_hazardous_trees, save_as=save_as, dpi=dpi,
                verbose=verbose)

    # == Data of HOTSPOTS =============================================================================

    def get_midpoints_for_plotting_hotspots(self, sort_by=None, update=False, verbose=False):
        """
        Get midpoints (of incident locations) for plotting hotspots.

        :param sort_by: [list; None (default)]
        :param update: [bool] (default: False)
        :param verbose:
        :return: [pd.DataFrame]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> incid_hotspots = hotspots.get_midpoints_for_plotting_hotspots()
        """

        pickle_filename = make_filename("s8hotspots", self.Route, self.WeatherCategory)
        path_to_pickle = self.METEx.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_hotspots = load_pickle(path_to_pickle)

        else:
            # Get TRUST (by incident location, i.e. by STANOX section)
            schedule8_costs_by_location = self.METEx.view_schedule8_costs_by_location()

            # Find a pseudo midpoint for each recorded incident
            def get_midpoint(x1, y1, x2, y2, as_geom=False):
                """
                :param x1: [numbers.Number; np.ndarray]
                :param y1: [numbers.Number; np.ndarray]
                :param x2: [numbers.Number; np.ndarray]
                :param y2: [numbers.Number; np.ndarray]
                :param as_geom: [bool] (default: False)
                :return: [np.ndarray; (list of) shapely.geometry.Point]

                **Test**::

                    x1, y1, x2, y2 = 1.5429, 52.6347, 1.4909, 52.6271
                    as_geom = False

                    get_midpoint(x1, y1, x2, y2, as_geom)
                    get_midpoint(x1, y1, x2, y2, True)
                """
                mid_pts = (x1 + x2) / 2, (y1 + y2) / 2
                if as_geom:
                    if all(isinstance(x, np.ndarray) for x in mid_pts):
                        mid_pts_ = [shapely.geometry.Point(x_, y_) for x_, y_ in
                                    zip(list(mid_pts[0]), list(mid_pts[1]))]
                    else:
                        mid_pts_ = shapely.geometry.Point(mid_pts)
                else:
                    mid_pts_ = np.array(mid_pts).T
                return mid_pts_

            pseudo_midpoints = get_midpoint(schedule8_costs_by_location.StartLongitude.values,
                                            schedule8_costs_by_location.StartLatitude.values,
                                            schedule8_costs_by_location.EndLongitude.values,
                                            schedule8_costs_by_location.EndLatitude.values,
                                            as_geom=False)

            # Get reference points (coordinates),
            # given subregion and layer (i.e. 'railways' in this case) of OSM .shp file
            if self.Route:
                path_to_boundary_polygon = cdd_network(
                    "routes\\{}".format(self.Route), "boundary-polygon.pickle")
                boundary_polygon = load_pickle(path_to_boundary_polygon)
                sub_area_name = self.Route
            else:
                boundary_polygon, sub_area_name = None, None

            railway_coordinates = get_shp_coordinates(
                'England', 'railways', 'rail', boundary_polygon, sub_area_name)

            # Get rail coordinates closest to the midpoints between starts and ends
            # noinspection PyUnresolvedReferences
            def find_closest_points_between(pts, ref_pts, as_geom=False):
                """
                :param pts: [np.ndarray] an array of size (n, 2)
                :param ref_pts: [np.ndarray] an array of size (n, 2)
                :param as_geom: [bool] (default: False)
                :return: [np.ndarray; list of shapely.geometry.Point]

                **Test**::

                    pts = np.array([[1.5429, 52.6347], [1.4909, 52.6271], [1.4248, 52.63075]])
                    ref_pts = np.array([[2.5429, 53.6347], [2.4909, 53.6271], [2.4248, 53.63075]])
                    as_geom = False

                    get_closest_points_between(pts, ref_pts, as_geom)
                    get_closest_points_between(pts, ref_pts, True)

                Reference: https://gis.stackexchange.com/questions/222315
                """

                if isinstance(ref_pts, np.ndarray):
                    ref_pts_ = ref_pts
                else:
                    ref_pts_ = np.concatenate([np.array(geom.coords) for geom in ref_pts])

                # noinspection PyArgumentList
                ref_ckd_tree = scipy.spatial.cKDTree(ref_pts_)
                distances, indices = ref_ckd_tree.query(pts, k=1)  # returns (distance, index)

                if as_geom:
                    closest_pts = [shapely.geometry.Point(ref_pts_[i]) for i in indices]
                else:
                    closest_pts = np.array([ref_pts_[i] for i in indices])

                return closest_pts

            midpoints = find_closest_points_between(pseudo_midpoints, railway_coordinates)

            midpoints_ = pd.DataFrame(
                midpoints, schedule8_costs_by_location.index, columns=['MidLongitude', 'MidLatitude'])
            incident_hotspots = schedule8_costs_by_location.join(midpoints_)

            save_pickle(incident_hotspots, path_to_pickle, verbose=verbose)

        if sort_by:
            incident_hotspots.sort_values(sort_by, ascending=False, inplace=True)

        incident_hotspots = get_subset(
            incident_hotspots, route_name=self.Route, weather_category=self.WeatherCategory,
            rearrange_index=True)

        return incident_hotspots

    def get_schedule8_annual_stats(self, update=False, verbose=False):
        """
        Get statistics for plotting annual delays.

        :param update: [bool] (default: False)
        :param verbose:
        :return: [pd.DataFrame]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> annual_statistics = hotspots.get_schedule8_annual_stats()
        """

        pickle_filename = make_filename("s8hotspots-annual-delays", self.Route, self.WeatherCategory)
        path_to_pickle = self.METEx.cdd_views(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            annual_stats = load_pickle(path_to_pickle)
            annual_stats = get_subset(annual_stats, self.Route, self.WeatherCategory)

        else:
            schedule8_data = self.METEx.view_schedule8_costs_by_datetime_location(
                self.Route, self.WeatherCategory, update=update)
            selected_features = [
                'FinancialYear', 'WeatherCategory', 'Route', 'StanoxSection',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
            schedule8_data_ = schedule8_data.groupby(selected_features).agg(
                {'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum}).reset_index()

            incident_location_midpoints = self.get_midpoints_for_plotting_hotspots(update=update)

            annual_stats = schedule8_data_.merge(
                incident_location_midpoints[selected_features[1:] + ['MidLatitude', 'MidLongitude']],
                how='left', on=selected_features[1:])
            annual_stats.sort_values(
                by=['DelayMinutes', 'DelayCost', 'IncidentCount'], ascending=False, inplace=True)

            save_pickle(annual_stats, path_to_pickle, verbose=verbose)

        return annual_stats

    # == Visualise the HOTSPOTS =======================================================================

    def show_annual_stats(self, cmap_name='Set1',
                          show_metex_weather_cells=True,
                          show_osm_landuse_forest=True,
                          show_nr_hazardous_trees=True,
                          save_as=".tif", dpi=600, update=False):
        """
        Plot hotspots of delays for every financial year (2006/07-2014/15).

        :param update: [bool] (default: False)
        :param cmap_name: [str] (default: 'Set1')
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [numbers.Number; None (default)]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.show_annual_stats(save_as=None)
        """

        schedule8_annual_stats = self.get_schedule8_annual_stats(update)

        annual_stats = schedule8_annual_stats.groupby('FinancialYear').agg(
            {'DelayMinutes': sum, 'DelayCost': sum, 'IncidentCount': sum})

        # Examine only 2006/07 - 2014/15
        hotspots_annual_stats = annual_stats.loc[2006:2014]

        # Labels
        years = [str(y) for y in hotspots_annual_stats.index]
        # noinspection PyTypeChecker
        f_years = [
            '/'.join([y0, str(y1)[-2:]])
            for y0, y1 in zip(years, np.array(hotspots_annual_stats.index) + np.array([1]))]

        d_label = [
            "%s  (%s min." % (fy, format(int(d), ","))
            for fy, d in zip(f_years, hotspots_annual_stats.DelayMinutes)]
        c_label = [
            "  / £%.2f" % round(c * 1e-6, 2) + "M)" for c in hotspots_annual_stats.DelayCost]
        label = [x for x in reversed([d + c for d, c in zip(d_label, c_label)])]

        cmap = plt.get_cmap(cmap_name)
        colours = [c for c in reversed(cmap(np.linspace(start=0, stop=1, num=9)))]

        # Plot basemap (with railway tracks)
        fig, base_map = self.plot_base_map(legend_loc=(1.05, 0.9))
        fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

        top_hotspots = []
        for y, fy in zip(years, f_years):
            plot_data = schedule8_annual_stats[schedule8_annual_stats.FinancialYear == int(y)][0:20]
            top_hotspots.append(fy + ':  ' + plot_data.StanoxSection.iloc[0])
            for i in plot_data.index:
                mid_x, mid_y = base_map(plot_data.MidLongitude[i], plot_data.MidLatitude[i])
                base_map.plot(
                    mid_x, mid_y, zorder=2, marker='o', color=colours[years.index(y)], alpha=0.9,
                    markersize=26, markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(cmap=cmap, n_colours=len(label), labels=label, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
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
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        # Save figure
        if save_as:
            self.save_prototype_hotpots_fig(
                fig, "annual-delays-200607-201415", "Hotspots",
                show_metex_weather_cells, show_osm_landuse_forest,
                show_nr_hazardous_trees,
                save_as, dpi, verbose=True)

    def show_delays(self, random_seed=1, cmap_name='Reds',
                    show_metex_weather_cells=True,
                    show_osm_landuse_forest=True,
                    show_nr_hazardous_trees=True,
                    save_as=".tif", dpi=600, update=False):
        """
        Plot hotspots in terms of delay minutes.

        :param update: [bool] (default: False)
        :param random_seed: [int] (default: 1)
        :param cmap_name: [str] (default: 'Reds')
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [numbers.Number; None (default)]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.show_delays(save_as=None)
        """

        hotspots_data_init = self.get_midpoints_for_plotting_hotspots(
            sort_by=['DelayMinutes', 'IncidentCount', 'DelayCost'], update=update)
        notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutes.notnull()]

        # Set a random_seed number
        np.random.seed(random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.DelayMinutes.values, k=6, initial=100)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
        # hotspots_data['jenks_bins'].fillna(-1, inplace=True)
        jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in
                        zip(breaks.bins, breaks.counts)]

        cmap = plt.get_cmap(cmap_name)  # 'OrRd', 'RdPu', 'Oranges', 'YlOrBr'
        colours = cmap(np.linspace(0., 1., len(jenks_labels)))
        marker_size = np.linspace(1., 2.2, len(jenks_labels)) * 12

        # Plot basemap (with railway tracks)
        fig, base_map = self.plot_base_map()
        fig.subplots_adjust(left=0.001, bottom=0.000, right=0.7715, top=1.000)

        bins = list(breaks.bins)
        for b in range(len(bins)):
            idx_0, idx_1 = hotspots_data.DelayMinutes <= bins[b], hotspots_data.DelayMinutes > bins[
                b - 1]
            if bins[b] == min(bins):
                plotting_data = hotspots_data[idx_0]
            elif bins[b] == max(bins):
                plotting_data = hotspots_data[idx_1]
            else:
                plotting_data = hotspots_data[idx_0 & idx_1]
            for i in plotting_data.index:
                mid_x, mid_y = base_map(plotting_data.MidLongitude[i], plotting_data.MidLatitude[i])
                base_map.plot(mid_x, mid_y, zorder=2, marker='o', color=colours[b], alpha=0.9,
                              markersize=marker_size[b],
                              markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(
            cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Total delay minutes (2006/07-2018/19)",
                   ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
        # Show the highest delays, in descending order
        cb.ax.text(0., 0 - 1.45, "Locations accounted for most delays:",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
        cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),  # highest
                   ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

        # Show Weather cells
        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        # Show Vegetation
        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        # Show hazardous trees
        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        # Save figure
        if save_as:
            self.save_prototype_hotpots_fig(fig, "delays", "Hotspots",
                                            show_metex_weather_cells, show_osm_landuse_forest,
                                            show_nr_hazardous_trees,
                                            save_as, dpi, verbose=True)

    def show_incident_frequency(self, random_seed=1, cmap_name='PuRd',
                                show_metex_weather_cells=True,
                                show_osm_landuse_forest=True,
                                show_nr_hazardous_trees=True,
                                save_as=".tif", dpi=600, update=False):
        """
        Plot hotspots in terms of incident frequency.

        :param update: [bool] (default: False)
        :param random_seed: [int] (default: 1)
        :param cmap_name: [str] (default: 'Reds')
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [numbers.Number; None (default)]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.show_incident_frequency(save_as=None)
        """

        hotspots_data_init = self.get_midpoints_for_plotting_hotspots(
            sort_by=['IncidentCount', 'DelayCost', 'DelayMinutes'], update=update)
        notnull_data = hotspots_data_init[hotspots_data_init.IncidentCount.notnull()]

        # Set a random_seed number
        np.random.seed(random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.IncidentCount.values, k=6, initial=100)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))

        jenks_labels = ["<= %d  / %d locations" % (b, c) for b, c in zip(breaks.bins, breaks.counts)]

        cmap = plt.get_cmap(cmap_name)  # 'Oranges', 'RdPu', 'Purples'
        colours = cmap(np.linspace(0, 1., len(jenks_labels)))
        marker_size = np.linspace(1.0, 2.2, len(jenks_labels)) * 12

        # Plot basemap (with railway tracks)
        fig, base_map = self.plot_base_map(legend_loc=(1.05, 0.9))
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
                              marker='o', color=colours[b], alpha=0.9, markersize=marker_size[b],
                              markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4,
                              pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Count of Incidents (2006/07-2018/19)",
                   ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
        # Show the highest frequency, in descending order
        cb.ax.text(0., 0 - 1.45, "Most incident-prone locations: ",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
        cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
                   ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        if save_as:
            self.save_prototype_hotpots_fig(fig, "frequency", "Hotspots",
                                            show_metex_weather_cells, show_osm_landuse_forest,
                                            show_nr_hazardous_trees,
                                            save_as, dpi, verbose=True)

    def show_costs(self, random_seed=1, cmap_name='YlGnBu',
                   show_metex_weather_cells=True,
                   show_osm_landuse_forest=True,
                   show_nr_hazardous_trees=True,
                   save_as=".tif", dpi=600, update=False):
        """
        Plot hotspots in terms of delay cost.

        :param update: [bool] (default: False)
        :param random_seed: [int] (default: 1)
        :param cmap_name: [str] (default: 'Reds')
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [numbers.Number; None (default)]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.show_costs(save_as=None)
        """

        hotspots_data_init = self.get_midpoints_for_plotting_hotspots(
            sort_by=['DelayCost', 'IncidentCount',
                     'DelayMinutes'],
            update=update)
        hotspots_data_init.replace(to_replace={'DelayCost': {0: np.nan}}, inplace=True)
        notnull_data = hotspots_data_init[hotspots_data_init.DelayCost.notnull()]

        # Set a random_seed number
        np.random.seed(random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.DelayCost.values, k=5, initial=100)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))
        # df.drop('jenks_bins', axis=1, inplace=True)
        hotspots_data.jenks_bins.fillna(-1, inplace=True)
        jenks_labels = ['<= £%s  / %s locations' % (format(int(b), ','), c) for b, c in
                        zip(breaks.bins, breaks.counts)]
        jenks_labels.insert(0, 'N/A (no cost)  / %s locations' % len(
            hotspots_data[hotspots_data['DelayCost'].isnull()]))

        cmap = plt.get_cmap(cmap_name)  # 'RdPu'
        colour_array = np.linspace(0, 1., len(jenks_labels))
        colours = cmap(colour_array)
        marker_size = np.linspace(0.8, 2.3, len(jenks_labels)) * 12

        # Plot basemap (with railway tracks)
        fig, base_map = self.plot_base_map(legend_loc=(1.05, 0.90))
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
        cb = colour_bar_index(cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4,
                              pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Compensation payments (2006/07-2018/19)",
                   ha='left', va='bottom', size=13, color='#555555', weight='bold', fontname='Cambria')
        # Show the highest cost, in descending order
        cb.ax.text(0., 0 - 1.45, "Locations accounted for most cost: ",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
        cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
                   ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        if save_as:
            self.save_prototype_hotpots_fig(
                fig, "costs", "Hotspots",
                show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                save_as, dpi, verbose=True)

    def show_delays_per_incident(self, random_seed=1, cmap_name='BrBG',
                                 show_metex_weather_cells=True,
                                 show_osm_landuse_forest=True,
                                 show_nr_hazardous_trees=True,
                                 save_as=".tif", dpi=600, update=False):
        """
        Plot hotspots in terms of delay minutes per incident.

        :param update: [bool] (default: False)
        :param random_seed: [int] (default: 1)
        :param cmap_name: [str] (default: 'Reds')
        :param show_metex_weather_cells: [bool] (default: True)
        :param show_osm_landuse_forest: [bool] (default: True)
        :param show_nr_hazardous_trees: [bool] (default: True)
        :param save_as: defaults to ``None``
        :type save_as: str or None
        :param dpi: [numbers.Number; None (default)]

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.show_delays_per_incident(save_as=None)
        """

        hotspots_data_init = self.get_midpoints_for_plotting_hotspots(update=update)
        hotspots_data_init['DelayMinutesPerIncident'] = hotspots_data_init.DelayMinutes.div(
            hotspots_data_init.IncidentCount)
        hotspots_data_init.sort_values(by='DelayMinutesPerIncident', ascending=False, inplace=True)

        notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutesPerIncident.notnull()]

        # Set a random_seed number
        np.random.seed(random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.DelayMinutesPerIncident.values, k=6)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
        # data['jenks_bins'].fillna(-1, inplace=True)
        jenks_labels = ["<= %s min.  / %s locations" % (format(int(b), ','), c) for b, c in
                        zip(breaks.bins, breaks.counts)]

        cmap = plt.get_cmap(cmap_name)
        colours = cmap(np.linspace(0, 1, len(jenks_labels)))
        marker_size = np.linspace(1.0, 2.2, len(jenks_labels)) * 12

        # Plot basemap (with railway tracks)
        fig, base_map = self.plot_base_map(legend_loc=(1.05, 0.9))
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
                base_map.plot(mid_x, mid_y, zorder=2, marker='o', color=colours[b], alpha=0.9,
                              markersize=marker_size[b],
                              markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4,
                              pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold',
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
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        # Show hazardous trees?
        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        if save_as:
            self.save_prototype_hotpots_fig(fig, "delays-per-incident", "Hotspots",
                                            show_metex_weather_cells, show_osm_landuse_forest,
                                            show_nr_hazardous_trees,
                                            save_as, dpi, verbose=True)

    def plot_hotspots_on_route(self, save_as=".tif", dpi=600, update=False, confirmation_required=True):
        """
        :param save_as:
        :param dpi:
        :param update: [bool] (default: False)
        :param confirmation_required:

        **Test**::

            >>> from illustrator.hotspot import Hotspots

            >>> hotspots = Hotspots()

            >>> hotspots.plot_hotspots_on_route()
        """

        if confirmed(confirmation_required=confirmation_required):
            # Fig. 1.
            self.plot_base_map_plus(save_as=save_as, dpi=dpi)

            # Fig. 2: Annual delays
            self.show_annual_stats(save_as=save_as, dpi=dpi, update=update)

            # Fig. 3: Delays
            self.show_delays(save_as=save_as, dpi=dpi, update=update)

            # Cost
            self.show_costs(save_as=".png", update=update)

            # Frequency
            self.show_incident_frequency(save_as=".png", update=update)

            # Delay minutes per incident
            self.show_delays_per_incident(save_as=".png", update=update)
