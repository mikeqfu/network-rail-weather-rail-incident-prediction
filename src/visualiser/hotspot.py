"""
Hotspots of Weather-related Incidents (in the context of wind-related delays).
"""

import os

import PIL.Image
import mapclassify
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
from pyhelpers.dirs import cd, cdd
from pyhelpers.geom import find_closest_points, get_midpoint
from pyhelpers.ops import colour_bar_index, confirmed
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_data, save_data

from src.preprocessor import METEX, Vegetation
from src.shaft.geometry import get_shp_coordinates, get_shp_file_path_for_basemap
from src.utils import WxRailIncidentsPred, get_subset, make_filename


class Hotspots:
    NAME = 'Hotspots of Weather-related Incidents in the context of wind-related delays'

    METEX = METEX(use_old_db=True, db_instance=WxRailIncidentsPred(verbose=False))

    VEGETATION = Vegetation(db_instance=WxRailIncidentsPred(verbose=False))

    def __init__(self, route_name='Anglia', weather_category='Wind', projection='tmerc',
                 font_name='Cambria'):
        """
        :param route_name: defaults to ``'Anglia'``
        :type route_name: str
        :param weather_category: defaults to ``'Wind'``
        :type weather_category: str
        :param projection: defaults to ``'tmerc'``
        :type projection:
        :param font_name: defaults to ``'Cambria'``
        :type font_name:

        **Examples**::
        
            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.route_name
            'Anglia'
            >>> hotspots.weather_category
            'Wind'
        """
        
        self.route_name = route_name
        self.weather_category = weather_category

        self.projection = projection  # Transverse Mercator Projection
        self.legend_loc = (1.05, 0.85)
        self.figure, self.base_map = None, None

        self.railway_line_colour = '#3D3D3D'

        self.random_seed = 1

        pd_preferences()
        mpl_preferences(backend='TkAgg', font_name=font_name)

    # == Save outputs ==============================================================================

    def _save_fig(self, fig, keyword, category, show_metex_weather_cells, show_osm_landuse_forest,
                  show_nr_hazardous_trees, save_as, dpi, verbose):
        """
        Save a figure.

        :param fig:
        :type fig: matplotlib.figure.Figure
        :param keyword: a keyword for specifying the filename
        :type keyword: str
        :param category:
        :type category: str
        :param show_metex_weather_cells:
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest:
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees:
        :type show_nr_hazardous_trees: bool
        :param save_as:
        :type save_as: str or None
        :param dpi:
        :type dpi: int or None
        :param verbose:
        :type verbose: bool or int
        """

        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip(
                [show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                ['weather_cells', 'Vegetation', 'hazard_trees'])
            filename = '_'.join([keyword] + [v for s, v in suffix if s])

            path_to_file = cd(
                "models", "prototype", self.weather_category.lower(), category, filename + save_as)

            save_data(None, path_to_file, dpi=dpi, conv_svg_to_emf=True, verbose=verbose)

    # == Prepare base maps =========================================================================

    def plot_base_map(self, railway_line_colour=None, legend_loc=None):
        """
        Create a base Map.

        :param legend_loc: defaults to ``None``
        :type legend_loc:
        :param railway_line_colour: defaults to ``'#3D3D3D'``
        :type railway_line_colour: str
        :return:
        :rtype: typing.Tuple[matplotlib.figure.Figure, mpl_toolkits.basemap.Basemap]

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> f, bm = hotspots.plot_base_map()
        """

        print("Plotting the base Map ... ", end="")

        plt.style.use('ggplot')
        # Default style: 'classic';
        # matplotlib.style.available gives the list of available styles
        fig = plt.figure(figsize=(11, 9))  # figsize=(9, 7)
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
            projection=self.projection,
            resolution='i',
            suppress_ticks=True,
            epsg=27700)

        # base_map.arcgisimage(service='World_Shaded_Relief', xpixels=1500, dpi=300, verbose=False)
        base_map.drawmapboundary(color='white', fill_color='white')
        # base_map.drawcoastlines()
        base_map.fillcontinents(color='#dcdcdc')  # color='#555555'

        # Add a layer for railway tracks
        boundary = shapely.geometry.Polygon(zip(base_map.boundarylons, base_map.boundarylats))

        path_to_shp_file = get_shp_file_path_for_basemap(
            osm_subregion='England', osm_layer='railways', osm_feature='rail',
            boundary=boundary, sub_area_name=self.route_name.lower())

        if railway_line_colour is None:
            railway_line_colour = self.railway_line_colour

        for sf in path_to_shp_file:
            base_map.readshapefile(
                shapefile=sf, name=self.route_name.lower(), linewidth=1.5,
                color=railway_line_colour,  # '#626262', '#939393', '#757575'
                zorder=4)

        # Show legend
        plt.plot([], '-', label="Railway track", linewidth=2.2, color=railway_line_colour)

        # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        legend = plt.legend(
            numpoints=1, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc if legend_loc else self.legend_loc)
        frame = legend.get_frame()
        frame.set_edgecolor('none')
        frame.set_facecolor('none')

        print("Done.")

        self.base_map = base_map
        self.figure = fig

        return fig, base_map

    def plot_weather_cells(self, update=False, legend_loc=(1.05, 0.85)):
        """
        Show Weather cells on the base Map.

        :param legend_loc: defaults to ``(1.05, 0.85)``
        :type legend_loc: tuple
        :param update: defaults to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.plot_weather_cells()
        """

        if self.base_map is None:
            _ = self.plot_base_map()

        print("Plotting the Weather cells ... ", end="")

        # Get Weather cell data
        self.METEX.read_weather_cell(update=update)
        data = self.METEX.weather_cell.copy()
        data = get_subset(data, route_name='Anglia')
        # Drop duplicated Weather cell data
        unhashable_cols = ('Polygon_WGS84', 'Polygon_OSGB36', 'IMDM', 'Route')
        data.drop_duplicates(
            subset=[x for x in list(data.columns) if x not in unhashable_cols], inplace=True)

        weather_cell_colour = '#D5EAFF'  # '#add6ff', '#99ccff', '#fff68f

        # Plot the Weather cells one by one
        for i in range(len(data)):
            ll_x, ll_y = self.base_map(data['ll_Longitude'].iloc[i], data['ll_Latitude'].iloc[i])
            ul_x, ul_y = self.base_map(data['ul_Longitude'].iloc[i], data['ul_Latitude'].iloc[i])
            ur_x, ur_y = self.base_map(data['ur_Longitude'].iloc[i], data['ur_Latitude'].iloc[i])
            lr_x, lr_y = self.base_map(data['lr_Longitude'].iloc[i], data['lr_Latitude'].iloc[i])
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
        Show the OSM natural forest on the base Map.

        :param add_osm_natural_tree: defaults to ``False``
        :type add_osm_natural_tree: bool
        :param fill_forest_patches: defaults to ``False``
        :type fill_forest_patches: bool
        :param legend_loc: defaults to ``(1.05, 0.85)``
        :type legend_loc: tuple

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots

            >>> hotspots = Hotspots()
            >>> hotspots.plot_osm_forest_and_tree()

            >>> hotspots = Hotspots()
            >>> hotspots.plot_osm_forest_and_tree(add_osm_natural_tree=True)
        """

        if self.base_map is None:
            _ = self.plot_base_map()

        print("Plotting the OSM natural/forest ... ", end="")

        # OSM - landuse - forest
        boundary = shapely.geometry.Polygon(
            zip(self.base_map.boundarylons, self.base_map.boundarylats))

        path_to_shp_file = get_shp_file_path_for_basemap(
            osm_subregion='England', osm_layer='landuse', osm_feature='forest',
            boundary=boundary, sub_area_name=self.route_name.lower())
        osm_landuse_forest_colour = '#72886E'  # '#7f987b', '#8ea989', '#72946c', '#72946c'
        self.base_map.readshapefile(
            path_to_shp_file[0], name='osm_landuse_forest', color=osm_landuse_forest_colour,
            zorder=3)

        # Fill the patches?
        # Note this may take a long time and dramatically increase the file of the Map
        if fill_forest_patches:
            print("\n")
            print("Filling the 'osm_landuse_forest' polygons ... ", end="")

            forest_polygons = [
                matplotlib.patches.Polygon(
                    p, fc=osm_landuse_forest_colour, ec=osm_landuse_forest_colour, zorder=4)
                for p in self.base_map.__getattribute__('osm_landuse_forest')]

            for i in range(len(forest_polygons)):
                plt.gca().add_patch(forest_polygons[i])

        # OSM - natural - tree
        if add_osm_natural_tree:
            bounded_natural_tree_shp = get_shp_file_path_for_basemap(
                osm_subregion='England', osm_layer='natural', osm_feature='tree', boundary=boundary,
                sub_area_name=self.route_name.lower())
            for bnt in bounded_natural_tree_shp:
                self.base_map.readshapefile(
                    bnt, name='osm_natural_tree', color=osm_landuse_forest_colour, zorder=3)
            natural_tree_points = [
                shapely.geometry.Point(p)
                for p in self.base_map.__getattribute__('osm_natural_tree')]
            self.base_map.scatter(
                [geom.x for geom in natural_tree_points], [geom.y for geom in natural_tree_points],
                marker='o', s=2, facecolor='#008000', label="Tree", alpha=0.5, zorder=3)

        # Add label
        plt.scatter(
            [], [], marker="o",  # hatch=3 * "x", s=580,
            facecolor=osm_landuse_forest_colour, edgecolor='none',
            label="Vegetation (OSM 'forest')")

        # font = {'family': 'Georgia', 'size': 16, 'weight': 'bold'}
        font = matplotlib.font_manager.FontProperties(family='Cambria', weight='normal', size=16)
        plt.legend(
            scatterpoints=10, loc='best', prop=font, frameon=False, fancybox=True,
            bbox_to_anchor=legend_loc)

        print("Done.")

    def plot_hazardous_trees(self, legend_loc=(1.05, 0.85)):
        """
        Show hazardous trees on the base Map.

        :param legend_loc: defaults to ``(1.05, 0.85)``
        :type legend_loc: tuple

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.plot_hazardous_trees()

        """

        if self.base_map is None:
            _ = self.plot_base_map()

        print("Plotting the hazardous trees ... ", end="")

        self.VEGETATION.view_hazardous_trees()
        hazardous_trees = self.VEGETATION.hazardous_trees.copy()

        map_points = [
            shapely.geometry.Point(self.base_map(long, lat))
            for long, lat in zip(hazardous_trees.Longitude, hazardous_trees.Latitude)]
        hazardous_trees_points = shapely.geometry.MultiPoint(map_points)

        # Plot hazardous trees on the basemap
        hazardous_tree_colour = '#ab790a'  # '#886008', '#6e376e', '#5a7b6c'

        self.base_map.scatter(
            [p.x for p in hazardous_trees_points.geoms], [p.y for p in hazardous_trees_points.geoms],
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
                           legend_loc=(1.05, 0.85), save_as=".svg", dpi=600, verbose=True):
        """
        Illustrate Weather cell and associated natural features (incl. forest and hazardous trees)
        with the base Map.

        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param add_osm_natural_tree: defaults to ``False``
        :type add_osm_natural_tree: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :param legend_loc: defaults to ``(1.05, 0.85)``
        :type legend_loc: tuple
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param verbose:
        :type verbose: bool or int

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.plot_base_map_plus(save_as=None)

        """

        # Plot basemap
        if self.base_map is None or self.figure is None:
            self.figure, self.base_map = self.plot_base_map(legend_loc=legend_loc)

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
        sr = self.figure.add_axes([0.58, 0.01, 0.40, 0.40], frameon=True)
        # The quantities are in fractions of figure width and height

        sr.imshow(PIL.Image.open(cdd("network/routes/map", "NR_Routes_edited_1.tif")))
        # Alternative: "Routes-edited-0.png"
        sr.axis('off')

        # Save the figure
        if save_as:
            self._save_fig(
                self.figure, keyword="base", category="basemap",
                show_metex_weather_cells=show_metex_weather_cells,
                show_osm_landuse_forest=show_osm_landuse_forest,
                show_nr_hazardous_trees=show_nr_hazardous_trees, save_as=save_as, dpi=dpi,
                verbose=verbose)

    # == Data of HOTSPOTS ==========================================================================

    def get_centroids_for_plotting_hotspots(self, sort_by=None, update=False, verbose=False):
        """
        Get centroids (of incident locations) for plotting Hotspots.

        :param sort_by: defaults to ``None``
        :type sort_by: list or None
        :param update: defaults to ``False``
        :type update: bool
        :param verbose:
        :type verbose: bool or int
        :return:
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> incid_hotspots = hotspots.get_centroids_for_plotting_hotspots(verbose=True)
            >>> incid_hotspots.shape
            (258, 22)
        """

        filename_ = "hotspots"
        pickle_filename = make_filename(filename_, self.route_name, self.weather_category, sep="_")
        path_to_pickle = self.METEX.cdd("views", pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_hotspots = load_data(path_to_pickle)

        else:
            # Get TRUST (by incident location, i.e. by STANOX section)
            self.METEX.view_schedule8_cost_by_location()
            schedule8_costs_by_location = self.METEX.schedule8_cost_by_location.copy()

            # Find a pseudo midpoint for each recorded incident
            pseudo_midpoints = get_midpoint(
                schedule8_costs_by_location['StartLongitude'].values,
                schedule8_costs_by_location['StartLatitude'].values,
                schedule8_costs_by_location['EndLongitude'].values,
                schedule8_costs_by_location['EndLatitude'].values,
                as_geom=False)

            # Get reference points (coordinates),
            # given subregion and layer (i.e. 'railways' in this case) of OSM .shp file
            if self.route_name:
                path_to_boundary_polygon = cdd(f"network/routes/{self.route_name}", "boundary.pkl")
                boundary_polygon = load_data(path_to_boundary_polygon)
                sub_area_name = self.route_name.lower()
            else:
                boundary_polygon, sub_area_name = None, None

            railway_coordinates = get_shp_coordinates(
                osm_subregion='England', osm_layer='railways', osm_feature='rail',
                boundary=boundary_polygon, sub_area_name=sub_area_name)

            # Get rail coordinates closest to the midpoints between starts and ends
            centroids_ = find_closest_points(pseudo_midpoints, railway_coordinates)
            centroids = pd.DataFrame(
                centroids_, index=schedule8_costs_by_location.index,
                columns=['MidLongitude', 'MidLatitude'])
            incident_hotspots = schedule8_costs_by_location.join(centroids)

            save_data(incident_hotspots, path_to_pickle, verbose=verbose)

        if sort_by:
            incident_hotspots.sort_values(sort_by, ascending=False, inplace=True)

        incident_hotspots = get_subset(
            incident_hotspots, route_name=self.route_name, weather_category=self.weather_category,
            rearrange_index=True)

        return incident_hotspots

    def get_schedule8_annual_stats(self, update=False, verbose=False):
        """
        Get statistics for plotting annual delays.

        :param update: defaults to ``False``
        :type update: bool
        :param verbose: defaults to ``False``
        :type verbose: bool or int
        :return:
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> annual_statistics = hotspots.get_schedule8_annual_stats(verbose=True)
            >>> annual_statistics.shape
            (1024, 13)
        """

        filename_ = "hotspots_annual_delays"
        pickle_filename = make_filename(filename_, self.route_name, self.weather_category, sep="_")
        path_to_pickle = self.METEX.cdd("views", pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            annual_stats = load_data(path_to_pickle)
            annual_stats = get_subset(annual_stats, self.route_name, self.weather_category)

        else:
            self.METEX.view_schedule8_cost_by_day_location(self.route_name, self.weather_category)
            schedule8_data = self.METEX.schedule8_cost_by_day_location
            selected_features = [
                'FinancialYear', 'WeatherCategory', 'Route', 'StanoxSection',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
            schedule8_data_ = schedule8_data.groupby(selected_features).agg(
                {'DelayMinutes': 'sum', 'DelayCost': 'sum', 'IncidentCount': 'sum'}).reset_index()

            incident_location_midpoints = self.get_centroids_for_plotting_hotspots(update=update)

            annual_stats = schedule8_data_.merge(
                incident_location_midpoints[selected_features[1:] + ['MidLatitude', 'MidLongitude']],
                how='left', on=selected_features[1:])
            annual_stats.sort_values(
                by=['DelayMinutes', 'DelayCost', 'IncidentCount'], ascending=False, inplace=True)

            save_data(annual_stats, path_to_pickle, verbose=verbose)

        return annual_stats

    # == Visualise the HOTSPOTS ====================================================================

    def visualise_annual_stats(self, cmap_name='Set1',
                               show_metex_weather_cells=True,
                               show_osm_landuse_forest=True,
                               show_nr_hazardous_trees=True,
                               save_as=".svg", dpi=600, update=False):
        """
        Plot Hotspots of delays for every financial year (2006/07-2014/15).

        :param cmap_name: default to ``'Set1'``
        :type cmap_name: str
        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :type show_nr_hazardous_trees: bool
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param update: default to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.visualise_annual_stats(save_as=None)

        """

        schedule8_annual_stats = self.get_schedule8_annual_stats(update=update)

        annual_stats = schedule8_annual_stats.groupby('FinancialYear').agg(
            {'DelayMinutes': 'sum', 'DelayCost': 'sum', 'IncidentCount': 'sum'})

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
            for fy, d in zip(f_years, hotspots_annual_stats['DelayMinutes'])]
        c_label = [
            "  / £%.2f" % round(c * 1e-6, 2) + "M)" for c in hotspots_annual_stats['DelayCost']]
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
                    mid_x, mid_y, zorder=4, marker='o', color=colours[years.index(y)], alpha=0.9,
                    markersize=26, markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(cmap=cmap, n_colours=len(label), labels=label, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        # cb.draw_all()

        cb.ax.text(0 + 1.5, 10.00, "Annual total delays and cost",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold',
                   fontname='Cambria')
        cb.ax.text(0, 0 - 1.95, "Locations with longest delays:",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold',
                   fontname='Cambria')
        cb.ax.text(0, 0 - 7.75, "\n".join(top_hotspots),
                   ha='left', va='bottom', size=14, color='#555555',
                   fontname='Times New Roman')

        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        # Save figure
        if save_as:
            self._save_fig(
                fig, "annual_delays_200607_201415", "hotspots",
                show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                save_as=save_as, dpi=dpi, verbose=True)

    def visualise_delays(self, cmap_name='Reds',
                         show_metex_weather_cells=True,
                         show_osm_landuse_forest=True,
                         show_nr_hazardous_trees=True,
                         save_as=".svg", dpi=600, update=False):
        """
        Plot Hotspots in terms of delay minutes.

        :param cmap_name: default to ``'Reds'``
        :type cmap_name: str
        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :type show_nr_hazardous_trees: bool
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param update: default to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.visualise_delays(save_as=None)

        """

        hotspots_data_init = self.get_centroids_for_plotting_hotspots(
            sort_by=['DelayMinutes', 'IncidentCount', 'DelayCost'], update=update)
        notnull_data = hotspots_data_init[hotspots_data_init.DelayMinutes.notnull()]

        # Set a random_seed number
        np.random.seed(self.random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.DelayMinutes.values, k=6, initial=100)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
        # hotspots_data['jenks_bins'].fillna(-1, inplace=True)
        jenks_labels = [
            "<= %s min.  / %s locations" % (format(int(b), ','), c)
            for b, c in zip(breaks.bins, breaks.counts)]

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
                base_map.plot(
                    mid_x, mid_y, zorder=4, marker='o', color=colours[b], alpha=0.9,
                    markersize=marker_size[b], markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(
            cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        # cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Total delay minutes (2006/07-2018/19)",
                   ha='left', va='bottom', size=14, color='#555555', weight='bold',
                   fontname='Cambria')
        # Show the highest delays, in descending order
        cb.ax.text(0., 0 - 1.15, "Locations accounted for most delays:",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold',
                   fontname='Cambria')
        cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),  # highest
                   ha='left', va='bottom', size=14, color='#555555',
                   fontname='Times New Roman')

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
            self._save_fig(fig, "delays", "hotspots",
                           show_metex_weather_cells, show_osm_landuse_forest,
                           show_nr_hazardous_trees,
                           save_as, dpi, verbose=True)

    def visualise_incident_frequency(self, cmap_name='PuRd',
                                     show_metex_weather_cells=True,
                                     show_osm_landuse_forest=True,
                                     show_nr_hazardous_trees=True,
                                     save_as=".svg", dpi=600, update=False):
        """
        Plot Hotspots in terms of incident frequency.

        :param cmap_name: default to ``'PuRd'``
        :type cmap_name: str
        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :type show_nr_hazardous_trees: bool
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param update: default to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.visualise_incident_frequency(save_as=None)

        """

        hotspots_data_init = self.get_centroids_for_plotting_hotspots(
            sort_by=['IncidentCount', 'DelayCost', 'DelayMinutes'], update=update)
        notnull_data = hotspots_data_init[hotspots_data_init.IncidentCount.notnull()]

        # Set a random_seed number
        np.random.seed(self.random_seed)

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
                base_map.plot(
                    x_mid_pt, y_mid_pt, zorder=4, marker='o', color=colours[b], alpha=0.9,
                    markersize=marker_size[b], markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(
            cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        # cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Count of Incidents (2006/07-2018/19)",
                   ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
        # Show the highest frequency, in descending order
        cb.ax.text(0., 0 - 1.15, "Most incident-prone locations: ",
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
            self._save_fig(
                fig, "frequency", "hotspots", show_metex_weather_cells, show_osm_landuse_forest,
                show_nr_hazardous_trees, save_as, dpi, verbose=True)

    def visualise_costs(self, cmap_name='YlGnBu',
                        show_metex_weather_cells=True,
                        show_osm_landuse_forest=True,
                        show_nr_hazardous_trees=True,
                        save_as=".svg", dpi=600, update=False):
        """
        Plot Hotspots in terms of delay cost.

        :param cmap_name: default to ``'YlGnBu'``
        :type cmap_name: str
        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :type show_nr_hazardous_trees: bool
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param update: default to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.visualise_costs(save_as=None)

        """

        hotspots_data_init = self.get_centroids_for_plotting_hotspots(
            sort_by=['DelayCost', 'IncidentCount', 'DelayMinutes'], update=update)
        hotspots_data_init.replace({'DelayCost': {0: np.nan}}, inplace=True)
        notnull_data = hotspots_data_init[hotspots_data_init.DelayCost.notnull()]

        # Set a random_seed number
        np.random.seed(self.random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data.DelayCost.values, k=5, initial=100)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame(data={'jenks_bins': breaks.yb}, index=notnull_data.index))
        # df.drop('jenks_bins', axis=1, inplace=True)
        hotspots_data['jenks_bins'] = hotspots_data['jenks_bins'].fillna(-1)
        jenks_labels = [
            '<= £%s  / %s locations' % (format(int(b), ','), c)
            for b, c in zip(breaks.bins, breaks.counts)]
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
                base_map.plot(
                    x_mid_pt, y_mid_pt, zorder=4, marker='o', color=colours[b], alpha=0.9,
                    markersize=marker_size[b], markeredgecolor='w', markeredgewidth=1)

        # Add a colour bar
        cb = colour_bar_index(
            cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(family='Times New Roman', weight='bold'))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        # cb.draw_all()

        # Add descriptions
        cb.ax.text(0., 0 + 6.75, "Compensation payments (2006/07-2018/19)",
                   ha='left', va='bottom', size=13, color='#555555', weight='bold',
                   fontname='Cambria')
        # Show the highest cost, in descending order
        cb.ax.text(0., 0 - 1.15, "Locations accounted for most cost: ",
                   ha='left', va='bottom', size=15, color='#555555', weight='bold',
                   fontname='Cambria')
        cb.ax.text(0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
                   ha='left', va='bottom', size=14, color='#555555',
                   fontname='Times New Roman')

        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        if save_as:
            self._save_fig(
                fig, "costs", "hotspots",
                show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                save_as, dpi, verbose=True)

    def visualise_delays_per_incident(self, cmap_name='BrBG',
                                      show_metex_weather_cells=True,
                                      show_osm_landuse_forest=True,
                                      show_nr_hazardous_trees=True,
                                      save_as=".svg", dpi=600, update=False):
        """
        Plot Hotspots in terms of delay minutes per incident.

        :param cmap_name: default to ``'BrBG'``
        :type cmap_name: str
        :param show_metex_weather_cells: defaults to ``True``
        :type show_metex_weather_cells: bool
        :param show_osm_landuse_forest: defaults to ``True``
        :type show_osm_landuse_forest: bool
        :param show_nr_hazardous_trees: defaults to ``True``
        :type show_nr_hazardous_trees: bool
        :param save_as: defaults to ``".svg"``
        :type save_as: str or None
        :param dpi: defaults to ``600``
        :type dpi: int or None
        :param update: default to ``False``
        :type update: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.visualise_delays_per_incident(save_as=None)

        """

        hotspots_data_init = self.get_centroids_for_plotting_hotspots(update=update)
        hotspots_data_init['DelayMinutesPerIncident'] = hotspots_data_init['DelayMinutes'].div(
            hotspots_data_init['IncidentCount'])
        hotspots_data_init.sort_values(by='DelayMinutesPerIncident', ascending=False, inplace=True)

        notnull_data = hotspots_data_init[hotspots_data_init['DelayMinutesPerIncident'].notnull()]

        # Set a random_seed number
        np.random.seed(self.random_seed)

        # Calculate Jenks natural breaks for delay minutes
        breaks = mapclassify.NaturalBreaks(y=notnull_data['DelayMinutesPerIncident'].values, k=6)
        hotspots_data = hotspots_data_init.join(
            pd.DataFrame({'jenks_bins': breaks.yb}, index=notnull_data.index))
        # data['jenks_bins'].fillna(-1, inplace=True)
        jenks_labels = [
            "<= %s min.  / %s locations" % (format(int(b), ','), c)
            for b, c in zip(breaks.bins, breaks.counts)]

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
                base_map.plot(
                    mid_x, mid_y, zorder=4, marker='o', color=colours[b], alpha=0.9,
                    markersize=marker_size[b], markeredgecolor='w')

        # Add a colour bar
        cb = colour_bar_index(
            cmap=cmap, n_colours=len(jenks_labels), labels=jenks_labels, shrink=0.4, pad=0.068)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_font_properties(
                matplotlib.font_manager.FontProperties(
                    family='Times New Roman', weight='bold', fname="C:\\Windows\\Fonts\\Times.ttf"))
        cb.ax.tick_params(labelsize=14)
        cb.set_alpha(1.0)
        # cb.draw_all()

        # Add descriptions
        cb.ax.text(
            0., 0 + 6.75, "Delay per incident (2006/07 to 2018/19)",
            ha='left', va='bottom', size=14, color='#555555', weight='bold', fontname='Cambria')
        # Show highest delay min. per incident, in descending order
        cb.ax.text(
            0., 0 - 1.15, "Longest delays per incident:",
            ha='left', va='bottom', size=15, color='#555555', weight='bold', fontname='Cambria')
        cb.ax.text(
            0., 0 - 5.65, "\n".join(hotspots_data.StanoxSection[:10]),
            ha='left', va='bottom', size=14, color='#555555', fontname='Times New Roman')

        if show_metex_weather_cells:
            self.plot_weather_cells(legend_loc=(1.05, 0.95))

        if show_osm_landuse_forest:
            self.plot_osm_forest_and_tree(add_osm_natural_tree=False, legend_loc=(1.05, 0.96))

        # Show hazardous trees?
        if show_nr_hazardous_trees:
            self.plot_hazardous_trees(legend_loc=(1.05, 0.975))

        if save_as:
            self._save_fig(
                fig, "delays_per_incident", "hotspots", show_metex_weather_cells,
                show_osm_landuse_forest, show_nr_hazardous_trees, save_as, dpi, verbose=True)

    def plot_hotspots_on_route(self, save_as=".svg", dpi=600, update=False,
                               confirmation_required=True):
        """


        :param save_as: defaults to ``".svg"``
        :type save_as: str | None
        :param dpi: defaults to ``600``
        :type dpi: int | None
        :param update: defaults to ``False``
        :type update: bool
        :param confirmation_required: defaults to ``True``
        :type confirmation_required: bool

        **Examples**::

            >>> from src.visualiser.hotspot import Hotspots
            >>> hotspots = Hotspots()
            >>> hotspots.plot_hotspots_on_route()

        """

        if confirmed(confirmation_required=confirmation_required):

            plot_args = {'save_as': save_as, 'dpi': dpi}

            # Fig. 1.
            self.plot_base_map_plus(**plot_args)

            plot_args.update({'update': update})

            # Fig. 2: Annual delays
            self.visualise_annual_stats(**plot_args)

            # Fig. 3: Delays
            self.visualise_delays(**plot_args)

            # Cost
            self.visualise_costs(**plot_args)

            # Frequency
            self.visualise_incident_frequency(**plot_args)

            # Delay minutes per incident
            self.visualise_delays_per_incident(**plot_args)
