"""
Read and cleanse data of rail Incidents and Weather extracted from METEX Database.

- Schedule 4 compensates train operators for the impact of planned service disruption, and
- Schedule 8 compensates train operators for the impact of unplanned service disruption.
"""

import datetime
import fractions
import gc
import multiprocessing
import os

import numpy as np
import shapely.geometry
import shapely.wkt
from pyhelpers._cache import _print_failure_message
from pyhelpers.dirs import cdd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.ops import confirmed
from pyhelpers.store import load_data, save_data
from pyrcs.converter import fix_stanox, mileage_str_to_num, shift_mileage_by_yard
from pyrcs.other_assets import Stations

from src.preprocessor._base import BaseMETEX
from src.utils import WxRailIncidentsPred, get_subset
from utils import make_filename


class METEX(BaseMETEX):
    """
    A class for handling data samples extracted from METEX.

    METEX is a geographic information system (GIS) based decision support tool,
    used to assess asset and system vulnerability to Weather.'
    """

    def __init__(self, db_instance=None, use_old_db=False):
        """
        :param db_instance: A PostgreSQL database instance; defaults to ``None``.
        :type db_instance: src.utils.WxRailIncidentsPred | None

        :ivar str | os.PathLike[str] DATA_DIR: the main data directory for this class

        :ivar sqlalchemy.engine.Connection mssql: connection to the database
        :ivar sqlalchemy.engine.Connection db_instance: connection to the database

        **Examples**::

            >>> from src.preprocessor.metex import METEX

            >>> mtx = METEX()

            >>> mtx.DATA_NAME
            'METExLite'
        """

        super().__init__()

        self.db_instance = db_instance

        if use_old_db:
            self.MSSQL_DATABASE_NAME = self.POSTGRES_SCHEMA_NAME = 'NR_METEx_20150331'

        self.schedule8_details = None
        self.schedule8_cost_by_location = None
        self.schedule8_cost_by_day_location = None
        self.schedule8_cost_by_day_location_reason = None
        self.schedule8_incident_locations = None

    # == Get table data ============================================================================

    def read_imdm(self, as_dict=False, **kwargs):
        """

        :param as_dict: Whether to return the data as a dictionary; defaults to ``False``.
        :type as_dict: bool
        :return:

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> # mtx.read_imdm(update=True, verbose=True)
            >>> mtx.read_imdm()
            >>> # mtx.read_imdm(as_dict=True, update=True, verbose=True, ret_data=True)
            >>> mtx.read_imdm(as_dict=True, ret_data=True)
        """

        self._read_data(table_name='IMDM', **kwargs)

        if as_dict:
            imdm_dict = self.imdm.to_dict()
            self.imdm = imdm_dict['Route']

            try:
                self.imdm.pop(np.nan)
            except KeyError:
                pass

            if kwargs.get('ret_data') is True:
                return self.imdm

    def read_imdm_alias(self, as_dict=False, **kwargs):
        """

        :param as_dict:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> # mtx.read_imdm_alias(update=True, verbose=True)
            >>> mtx.read_imdm_alias()
        """

        self._read_data(table_name='ImdmAlias', **kwargs)

        if as_dict:
            self.imdm_alias = self.imdm_alias.to_dict()

        if kwargs.get('ret_data') is True:
            return self.imdm_alias

    def read_imdm_weather_cell_map(self, grouped=False, **kwargs):

        self._read_data(table_name='IMDMWeatherCellMap', ivar_name='weather_cell_map', **kwargs)

        if grouped:  # Find out how many IMDMs each 'WeatherCellId' is associated with
            self.weather_cell_map = self.weather_cell_map.groupby('Route').aggregate(
                lambda x: list(set(x))[0] if len(list(set(x))) == 1 else list(set(x)))

        if kwargs.get('ret_data') is True:
            return self.weather_cell_map

    def read_incident_reason_info(self, **kwargs):

        self._read_data(table_name='IncidentReasonInfo', **kwargs)

        if kwargs.get('ret_data') is True:
            return self.incident_reason_info

    def read_weather_codes(self, as_dict=False, **kwargs):
        """

        :param as_dict:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_weather_codes()
            >>> mtx.weather_codes.shape
            (9, 1)
        """

        self._read_data(table_name='WeatherCodes', **kwargs)

        if as_dict:
            self.weather_codes = self.weather_codes.to_dict()

        if kwargs.get('ret_data') is True:
            return self.weather_codes

    def read_incident_record(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_incident_record()
            >>> mtx.incident_record.shape
            (4704448, 5)
        """

        self._read_data(
            table_name='IncidentRecord', parse_dates=['IncidentRecordCreateDate'], **kwargs)

        self.incident_record.fillna({'WeatherCategory': ''}, inplace=True)

        if kwargs.get('ret_data') is True:
            return self.incident_record

    def read_location(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_location()
            >>> mtx.location.shape
            (653882, 7)
        """

        self._read_data(table_name='Location', **kwargs)

        if kwargs.get('ret_data') is True:
            return self.location

    def read_pfpi(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_pfpi()
            >>> mtx.pfpi.shape
            (5764333, 6)
        """

        self._read_data(table_name='PfPI', ivar_name='pfpi', **kwargs)

        if kwargs.get('ret_data') is True:
            return self.pfpi

    def read_route(self, as_dict=False, **kwargs):
        """

        :param as_dict:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_route()
            >>> len(mtx.route)
            14
            >>> mtx.read_route(as_dict=True, ret_data=True)
        """

        self._read_data(table_name='Route', **kwargs)

        if as_dict:
            self.route = dict(self.route['Route'])

        if kwargs.get('ret_data') is True:
            return self.route

    def read_stanox_location(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_stanox_location()
            >>> mtx.stanox_location.shape
            (7548, 5)
        """

        table_name = 'StanoxLocation'
        read_args = {
            'table_name': table_name,
            'dtype': {'LocationId': int, 'Stanox': str, 'Mileage': str},
            'pkey': [],
            'keep_default_na': False,
        }
        kwargs.update(read_args)
        self._read_data(**kwargs)

        self.stanox_location['Stanme'] = self.stanox_location['Stanme'].fillna('')

        # self.stanox_location.index.name is None
        if all(x in self.stanox_location.columns for x in ['LocationId', 'Stanox']):
            self.stanox_location.set_index(['LocationId', 'Stanox'], inplace=True)

        if kwargs.get('ret_data') is True:
            return self.stanox_location

    def read_stanox_section(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_stanox_section()
            >>> mtx.stanox_section.shape
            (10601, 7)
        """

        table_name = 'StanoxSection'
        self._read_data(
            table_name=table_name, dtype={'StartStanox': str, 'EndStanox': str},
            true_values=['t', 'true'], false_values=['f', 'false'],  # keep_default_na=False,
            **kwargs)

        if kwargs.get('ret_data') is True:
            return self.stanox_section

    def read_trust_incident(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_trust_incident()
            >>> mtx.trust_incident.shape
            (4049984, 11)
        """

        table_name = 'TrustIncident'
        self._read_data(
            table_name=table_name, parse_dates=['StartDate', 'EndDate'],
            true_values=['t', 'true'], false_values=['f', 'false'], keep_default_na=False,
            low_memory=False, **kwargs)

        if kwargs.get('ret_data') is True:
            return self.trust_incident

    def query_weather(self, weather_cell_id, start_dt=None, end_dt=None, **kwargs):
        # noinspection PyShadowingNames
        """
        Get Weather data by ``'WeatherCell'`` and ``'DateTime'`` (Query from the database).

        :param weather_cell_id: Weather cell ID
        :type weather_cell_id: int
        :param start_dt: start date and time; defaults to ``None``.
        :type start_dt: datetime.datetime, str | None
        :param end_dt: end date and time; defaults to ``None``.
        :type end_dt: datetime.datetime, str | None
        :return: Weather data by ``'weather_cell_id'``, ``'start_dt'`` and ``'end_dt'``
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> from src.utils import WxRailIncidentsPred
            >>> import datetime

            >>> mtx = METEX(db_instance=WxRailIncidentsPred())

            >>> weather_cell_id = 2367

            >>> start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> end_dt = datetime.datetime(2018, 6, 1, 13)  # '2018-06-01 13:00:00'
            >>> mtx.query_weather(weather_cell_id, start_dt, end_dt)

            >>> start_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> end_dt = datetime.datetime(2018, 6, 1, 12)  # '2018-06-01 12:00:00'
            >>> mtx.query_weather(weather_cell_id, start_dt, end_dt)

            >>> start_dt = datetime.datetime(2018, 3, 1)
            >>> end_dt = datetime.datetime(2018, 3, 31)
            >>> mtx.query_weather(weather_cell_id, start_dt, end_dt)
        """

        # assert isinstance(weather_cell_id, (tuple, int, np.integer))

        table_name = 'Weather'

        # Make a pickle filename
        # def _make_weather_pickle_filename():
        #     if isinstance(weather_cell_id, tuple):
        #         c_id = "-".join(str(x) for x in list(weather_cell_id))
        #     else:
        #         c_id = weather_cell_id
        #     s_dt = start_dt.strftime('_fr%Y%m%d%H%M') if start_dt else ""
        #     e_dt = end_dt.strftime('_to%Y%m%d%H%M') if end_dt else ""
        #     return "{}{}{}.pickle".format(c_id, s_dt, e_dt)

        # Specify database sql query
        # sql_query = "SELECT * FROM dbo.[Weather] WHERE {} {} {} AND {} AND {};".format(
        #     "[WeatherCell]", "IN" if isinstance(weather_cell_id, tuple) else "=",
        #     weather_cell_id,
        #     "[DateTime] >= '{}'".format(start_dt) if start_dt else "",
        #     "[DateTime] <= '{}'".format(end_dt) if end_dt else "")

        tbl = f'"{self.POSTGRES_SCHEMA_NAME}"."{table_name}"'
        sql_query = f'SELECT * FROM {tbl}'
        if isinstance(weather_cell_id, tuple):
            sql_query += f' WHERE "WeatherCell" IN {weather_cell_id}'
        else:
            sql_query += f' WHERE "WeatherCell" = {weather_cell_id}'
        if start_dt:
            sql_query += f' AND "DateTime" >= \'{start_dt}\''
        if end_dt:
            sql_query += f' AND "DateTime" <= \'{end_dt}\''

        kwargs.update({'parse_dates': ['DateTime']})

        weather_dat = self.db_instance.read_sql_query(sql_query, **kwargs)

        return weather_dat

    def read_weather_cell(self, route_name=None, **kwargs):
        """

        :param route_name:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_weather_cell()
            >>> mtx.weather_cell.shape
            (1838, 27)
        """

        self._read_data(table_name='WeatherCell', pkey=[], **kwargs)
        # self.weather_cell.set_index('WeatherCellId', inplace=True)

        try:
            self.weather_cell['polygon_WGS84'] = \
                self.weather_cell['polygon_WGS84'].map(shapely.to_wkt)
            self.weather_cell['polygon_OSGB36'] = \
                self.weather_cell['polygon_OSGB36'].map(shapely.to_wkt)
        except TypeError:
            pass

        self.weather_cell = get_subset(data=self.weather_cell, route_name=route_name)

        if kwargs.get('ret_data') is True:
            return self.weather_cell

    def view_weather_cell_map(self, route_name=None, save_as=None, verbose=False, **kwargs):
        """

        :param route_name:
        :param save_as:
        :param verbose:
        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> # mtx.view_weather_cell_map(save_as=".png")
            >>> # mtx.view_weather_cell_map(save_as=".svg")
            >>> mtx.view_weather_cell_map()
        """

        from pyhelpers.settings import mpl_preferences

        mpl_preferences(font_name='Cambria', backend='TkAgg')

        import matplotlib.pyplot as plt
        import mpl_toolkits.basemap
        import matplotlib.patches

        self.read_weather_cell(route_name=route_name)

        if verbose:
            print("Plotting Weather cells", end=" ... ")

        try:
            weather_cell_wgs84 = shapely.geometry.MultiPolygon(
                self.weather_cell['polygon_WGS84'].map(shapely.wkt.loads).to_list())
            minx, miny, maxx, maxy = weather_cell_wgs84.bounds

            fs = fractions.Fraction((maxx - minx) / (maxy - miny)).limit_denominator(8)

            fig, ax = plt.subplots(figsize=(fs.numerator, fs.denominator))
            m = mpl_toolkits.basemap.Basemap(
                projection='tmerc',  # Transverse Mercator Projection
                ellps='WGS84',
                epsg=27700,
                llcrnrlon=minx - 0.285,
                llcrnrlat=miny - 0.255,
                urcrnrlon=maxx + 1.185,
                urcrnrlat=maxy + 0.255,
                lat_ts=0,
                resolution='f',
                suppress_ticks=True)

            # m.arcgisimage(service='World_Street_Map', xpixels=1500, dpi=300, verbose=False)

            m.drawlsmask(land_color='0.9', ocean_color='#9EBCD8', resolution='f', grid=1.25)
            m.fillcontinents(color='0.9')
            m.drawcountries()
            # m.drawcoastlines()

            subset_cols = [
                s for s in self.weather_cell.columns if '_' in s and not s.startswith('polygon')]
            cell_map = self.weather_cell.drop_duplicates(subset=subset_cols)

            for i in cell_map.index:
                ll_x, ll_y = m(cell_map.ll_Longitude[i], cell_map.ll_Latitude[i])
                ul_x, ul_y = m(cell_map.ul_Longitude[i], cell_map.ul_Latitude[i])
                ur_x, ur_y = m(cell_map.ur_Longitude[i], cell_map.ur_Latitude[i])
                lr_x, lr_y = m(cell_map.lr_Longitude[i], cell_map.lr_Latitude[i])
                xy = zip([ll_x, ul_x, ur_x, lr_x], [ll_y, ul_y, ur_y, lr_y])
                polygons = matplotlib.patches.Polygon(
                    list(xy), fc='#D5EAFF', ec='#4b4747', alpha=0.5)
                ax.add_patch(polygons)

            plt.plot(
                [], 's', label="Weather cell", ms=14, color='#D5EAFF', markeredgecolor='#4b4747')
            legend = plt.legend(numpoints=1, loc='best', fancybox=False)
            frame = legend.get_frame()
            frame.set_edgecolor('w')

            plt.tight_layout()

            plt.show()

            if verbose:
                print("Done.")

            if save_as:
                filename = "weather_cell"
                if route_name:
                    filename += f"_{route_name.lower().replace(' ', '_')}"
                path_to_fig = self.cdd("images", filename + save_as)

                save_data(fig, path_to_file=path_to_fig, verbose=verbose, **kwargs)

        except Exception as e:
            print(f"Failed. {e}")

    def get_weather_cell_map_boundary(self, route_name=None, **kwargs):
        """
        Get the lower-left and upper-right corners for a Weather cell Map.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str | None
        :return: a boundary for a Weather cell Map
        :rtype: shapely.geometry.polygon.Polygon

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> import shapely.geometry
            >>> mtx = METEX()
            >>> mtx.get_weather_cell_map_boundary()
            >>> isinstance(mtx.weather_cell_map_boundary, shapely.geometry.Polygon)
        """

        if self.weather_cell_map_boundary is None:

            if self.weather_cell is None:
                self.read_weather_cell(route_name=route_name)

            ll = tuple(self.weather_cell[['ll_Longitude', 'll_Latitude']].apply(min))
            lr = self.weather_cell.lr_Longitude.max(), self.weather_cell.lr_Latitude.min()
            ur = tuple(self.weather_cell[['ur_Longitude', 'ur_Latitude']].apply(max))
            ul = self.weather_cell.ul_Longitude.min(), self.weather_cell.ul_Latitude.max()

            # Adjust the boundaries
            adj_values = np.array((0.285, 0.255))
            ll -= adj_values
            lr += np.array((adj_values[0], -adj_values[1]))
            ur += adj_values
            ul += np.array((-adj_values[0], adj_values[1]))

            self.weather_cell_map_boundary = shapely.geometry.Polygon((ll, lr, ur, ul))

        if kwargs.get('ret_data') is True:
            return self.weather_cell_map_boundary

    def read_track(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_track()
            >>> mtx.track.shape
            (16736, 19)
        """

        self._read_data(table_name='Track', dtype={'StartMileage': str, 'EndMileage': str}, **kwargs)

        self.track.sort_index(inplace=True)

        if kwargs.get('ret_data') is True:
            return self.track

    @staticmethod
    def view_track(geom_objects, rotate_labels=None):
        """
        Create a graph to illustrate track geometry.

        :param geom_objects: geometry objects
        :type geom_objects: iterable of [WKT str, shapely.geometry.LineString,
            or shapely.geometry.MultiLineString]
        :param rotate_labels: defaults to ``None``
        :type rotate_labels: numbers.Number, None
        :return: a graph demonstrating the tracks

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_track()
            >>> mtx.view_track(geom_objects=mtx.track.xs('AAV', level='ELR').geom.to_list())
        """

        import networkx as nx
        import matplotlib.pyplot as plt
        from pyhelpers.settings import mpl_preferences

        mpl_preferences(font_name='Cambria', backend='TkAgg')

        g = nx.Graph()

        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot()

        max_node_id = 0
        for geom_obj in geom_objects:

            if isinstance(geom_obj, str):
                geom_obj = shapely.wkt.loads(geom_obj)

            geom_type, geom_pos = shapely.geometry.mapping(geom_obj).values()

            if geom_type == 'MultiLineString':

                # Sort line strings of the multi-line string
                geom_pos_idx = list(range(len(geom_pos)))
                pos_idx_sorted = []
                while geom_pos_idx:
                    y = geom_pos_idx[0]
                    p1 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                    if p1 and p1[0] not in pos_idx_sorted:
                        pos_idx_sorted = p1 + [y]
                    else:
                        pos_idx_sorted = [y]

                    p2 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                    if p2:
                        pos_idx_sorted += p2

                    geom_pos_idx = [a for a in geom_pos_idx if a not in pos_idx_sorted]

                    if len(geom_pos_idx) == 1:
                        y = geom_pos_idx[0]
                        p3 = [i for i, x in enumerate([x[-1] for x in geom_pos]) if x == geom_pos[y][0]]
                        if p3 and p3[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p3[0]) + 1, y)
                            break

                        p4 = [i for i, x in enumerate([x[0] for x in geom_pos]) if x == geom_pos[y][-1]]
                        if p4 and p4[0] in pos_idx_sorted:
                            pos_idx_sorted.insert(pos_idx_sorted.index(p4[0]), y)
                            break

                geom_pos = [geom_pos[i] for i in pos_idx_sorted]
                geom_pos = [x[:-1] for x in geom_pos[:-1]] + [geom_pos[-1]]
                geom_pos = [x for g_pos in geom_pos for x in g_pos]

            # j = 0
            # n = g.number_of_nodes()
            # while j < len(geom_pos):
            #     # Nodes
            #     g_pos = geom_pos[j]
            #     for i in range(len(g_pos)):
            #         g.add_node(i + n + 1, pos=g_pos[i])
            #     # Edges
            #     current_max_node_id = g.number_of_nodes()
            #     edges = [(x, x + 1) for x in range(n + 1, n + current_max_node_id)
            #              if x + 1 <= current_max_node_id]
            #     g.add_edges_from(edges)
            #     n = current_max_node_id
            #     j += 1

            # Nodes

            for i in range(len(geom_pos)):
                g.add_node(i + 1 + max_node_id, pos=geom_pos[i])

            # Edges
            number_of_nodes = g.number_of_nodes()
            edges = [
                (i, i + 1) for i in range(1 + max_node_id, number_of_nodes + 1)
                if i + 1 <= number_of_nodes]
            # edges = [(i, i + 1) for i in range(1, number_of_nodes + 1) if i + 1 <= number_of_nodes]
            g.add_edges_from(edges)

            # Plot
            nx.draw_networkx(g, pos=nx.get_node_attributes(g, name='pos'), ax=ax, node_size=0,
                             with_labels=False)

            max_node_id = number_of_nodes

        ax.tick_params(
            left=True, bottom=True, labelleft=True, labelbottom=True, gridOn=True,
            grid_linestyle='--')
        ax.ticklabel_format(useOffset=False)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting', fontsize=14)
        ax.set_ylabel('Northing', fontsize=14)

        if rotate_labels:
            for tick in ax.get_xticklabels():
                tick.set_rotation(rotate_labels)

        plt.tight_layout()

    def read_track_summary(self, **kwargs):
        """

        :param kwargs:
        :return:

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.read_track_summary()
            >>> mtx.track_summary.shape
            (654843, 63)
        """

        self._read_data(
            table_name='TrackSummary', dtype={'StartMileage': str, 'EndMileage': str},
            low_memory=False, **kwargs)

        if kwargs.get('ret_data') is True:
            return self.track_summary

    def query_track_summary(self, elr, track_id, start_yard=None, end_yard=None, **kwargs):
        """
        Get track summary data by ``'Track ID'`` and ``'Yard'`` (Query from the database).

        :param elr: ELR
        :type elr: str
        :param track_id: TrackID
        :type track_id: tuple | int | numpy.integer
        :param start_yard: start yard; defaults to ``None``.
        :type start_yard: int | None
        :param end_yard: end yard; defaults to ``None``.
        :type end_yard: int | None
        :return: Data of track summary queried by
            ``'elr'``, ``'track_id'``, ``'start_yard'`` and ``'end_yard'``
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> ts = mtx.query_track_summary('AAV', track_id=1100, start_yard=51150, end_yard=66220)
            >>> ts.shape
            (40, 67)
        """

        table_name = 'TrackSummary'

        tbl = f'"{self.SCHEMA_NAME}"."{table_name}"'
        sql_query = f'SELECT * FROM {tbl}'
        if isinstance(elr, str):
            sql_query += f' WHERE "ELR" = \'{elr}\''
        else:
            sql_query += f' WHERE "ELR" IN {elr}'

        if isinstance(track_id, (int, np.integer)):
            sql_query += f' AND "TrackID" = {track_id}'
        else:
            sql_query += f' AND "TrackID" IN {track_id}'

        if start_yard:
            sql_query += f' AND "StartYards" >= {start_yard}'
        if end_yard:
            sql_query += f' AND "EndYards" >= {end_yard}'

        track_summary = self.db_instance.read_sql_query(sql_query, **kwargs)

        return track_summary

    def _update_tables(self, verbose=True):
        """
        Update the local pickle files for all Tables.

        :param verbose: Whether to print relevant information in console; defaults to ``True``.
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx._update_tables()
        """

        if confirmed(f'To update the Tables of "{self.SCHEMA_NAME}"\n?'):
            update_args = {'update': True, 'verbose': verbose}

            self.read_imdm(**update_args)
            self.read_imdm_alias(**update_args)
            self.read_imdm_weather_cell_map(**update_args)
            self.read_incident_reason_info(**update_args)
            self.read_weather_codes(**update_args)
            self.read_incident_record(**update_args)
            self.read_location(**update_args)
            self.read_pfpi(**update_args)
            self.read_route(**update_args)
            self.read_stanox_location(**update_args)
            self.read_stanox_section(**update_args)
            self.read_trust_incident(**update_args)
            self.read_weather_cell(**update_args)
            self.get_weather_cell_map_boundary()
            self.view_weather_cell_map(save_as=".svg", verbose=verbose)
            self.view_weather_cell_map(save_as=".png", dpi=600, verbose=verbose)
            self.view_weather_cell_map(route_name='Anglia', save_as=".svg", verbose=verbose)
            self.view_weather_cell_map(route_name='Anglia', save_as=".png", dpi=600, verbose=verbose)

            self.read_track(**update_args)
            self.read_track_summary(**update_args)

            if verbose:
                print("Update finished.")

    # == Tools to make information integration easier ==============================================

    @staticmethod
    def calculate_pfpi_stats(data, variables, sort_by=None):
        """
        Calculate the 'DelayMinutes' and 'DelayCosts' for grouped data.

        :param data: a given data frame
        :type data: pandas.DataFrame
        :param variables: a list of selected features (column names)
        :type variables: list
        :param sort_by: a column or a list of columns by which the selected data is sorted,
            defaults to ``None``
        :type sort_by: str | list | None
        :return: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_details('Anglia', 'Wind')
            >>> mtx.schedule8_details.shape
            (3323, 40)
            >>> selected_feats = [
            ...     'PfPIId', 'WeatherCategory', 'Route', 'StanoxSection',
            ...     'PfPIMinutes', 'PfPICosts']
            >>> res = mtx.calculate_pfpi_stats(mtx.schedule8_details[selected_feats], selected_feats)
            >>> res.shape
            (258, 6)
        """

        pfpi_stats = data.groupby(variables[1:-2]).aggregate(
            {
                # 'IncidentId_and_CreateDate': {'IncidentCount': np.count_nonzero},
                'PfPIId': np.count_nonzero,
                'PfPIMinutes': 'sum',
                'PfPICosts': 'sum'
            }
        )

        pfpi_stats.columns = ['IncidentCount', 'DelayMinutes', 'DelayCost']
        pfpi_stats.reset_index(inplace=True)  # Reset the grouped indexes to columns

        if sort_by:
            pfpi_stats.sort_values(sort_by, inplace=True)

        return pfpi_stats

    # == Create views ==============================================================================

    @staticmethod
    def _check_location_names(data):
        dat = data.copy()

        diff_start = dat[~dat['StartLocation'].eq(dat['Location_Start'])]
        if not diff_start.empty:
            diff_si = diff_start.iloc[
                np.where(dat.loc[diff_start.index, 'Location_Start'].notna())].index
            dat.loc[diff_si, 'StartLocation'] = dat.loc[diff_si, 'Location_Start']

        diff_end = dat[~dat['EndLocation'].eq(dat['Location_End'])]
        if not diff_end.empty:
            diff_ei = diff_end.iloc[np.where(dat.loc[diff_end.index, 'Location_End'].notna())].index
            dat.loc[diff_ei, 'EndLocation'] = dat.loc[diff_ei, 'Location_End']

        eq_start_end = dat[dat['StartLocation'].eq(dat['EndLocation'])]
        if not eq_start_end.empty:
            dat.loc[eq_start_end.index, 'StanoxSection'] = eq_start_end['StartLocation'].values

        diff_start_end = dat[~dat['StartLocation'].eq(dat['EndLocation'])]
        if not diff_start_end.empty:
            temp = diff_start_end['StartLocation'] + ' : ' + diff_start_end['EndLocation']
            dat.loc[diff_start_end.index, 'StanoxSection'] = temp.values

        return dat

    @staticmethod
    def _check_geom_coords(data):
        dat = data.copy()

        # Use 'Station' data from Railway Codes website
        stn = Stations()
        stn_loc = stn.fetch_locations()[stn.KEY_TO_STN]

        lon_lat_columns = ['Degrees Longitude', 'Degrees Latitude']
        start_lon_lat_columns = ['StartLongitude', 'StartLatitude']
        end_lon_lat_columns = ['EndLongitude', 'EndLatitude']

        stn_loc = stn_loc[['Station'] + lon_lat_columns]
        stn_loc.dropna(subset=lon_lat_columns, inplace=True)
        stn_loc = stn_loc.drop_duplicates(subset=['Station']).set_index('Station')

        temp = dat[['StartLocation']].join(stn_loc, on='StartLocation', how='left')
        i = temp[temp['Degrees Longitude'].notna() & temp['Degrees Latitude'].notna()].index
        dat.loc[i, start_lon_lat_columns] = temp.loc[i, lon_lat_columns].values

        temp = dat[['EndLocation']].join(stn_loc, on='EndLocation', how='left')
        i = temp[temp['Degrees Longitude'].notna() & temp['Degrees Latitude'].notna()].index
        dat.loc[i, end_lon_lat_columns] = temp.loc[i, lon_lat_columns].values

        # data['EndELR'].replace({'STM': 'SDC', 'TIR': 'TLL'}, inplace=True)

        loc_name = 'Highbury & Islington (North London Lines)'
        lon_lat = [-0.1045, 51.5460]
        dat.loc[dat['StartLocation'] == loc_name, start_lon_lat_columns] = lon_lat
        dat.loc[dat['EndLocation'] == loc_name, end_lon_lat_columns] = lon_lat

        loc_name = 'Dalston Junction (East London Line)'
        lon_lat = [-0.0751, 51.5461]
        dat.loc[dat['StartLocation'] == loc_name, start_lon_lat_columns] = lon_lat
        dat.loc[dat['EndLocation'] == loc_name, end_lon_lat_columns] = lon_lat

        return dat

    def _schedule8_details(self, route_name=None, weather_category=None,
                           weather_attributed_only=False, update=False, verbose=False, **kwargs):
        """
        View Schedule 8 details (TRUST data).

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :param weather_attributed_only: defaults to ``False``
        :type weather_attributed_only: bool
        :param update: Whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: Whether to print relevant information in console; defaults to ``False``.
        :type verbose: bool | int
        :return: Data of Schedule 8 details
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> dat = mtx._schedule8_details(update=True, verbose=True)
            >>> dat.shape
            (5184316, 54)
        """

        if self.db_instance is None:
            self.db_instance = WxRailIncidentsPred(verbose=False)

        read_args = {'update': update, 'verbose': verbose}
        self.read_pfpi(**read_args)
        self.read_incident_record(**read_args)
        self.read_trust_incident(**read_args)
        self.read_stanox_section(**read_args)
        self.read_location(**read_args)
        self.read_stanox_location(**read_args)
        self.read_incident_reason_info(**read_args)
        self.read_imdm(**read_args)

        if weather_attributed_only:  # ≈ 6.8%
            incident_record = self.incident_record.query("`weather_category` != ''")
        else:
            incident_record = self.incident_record.copy()

        stanox_location = self.stanox_location.reset_index().sort_values(['Yards'])
        stanox_location = stanox_location.drop_duplicates(subset=['Stanox']).set_index('Stanox')

        # Merge the acquired data sets - starting with (5764333, 6)
        data = self.pfpi. \
            join(incident_record,  # (5213189, 11)
                 on='IncidentRecordId', how='inner'). \
            join(self.trust_incident,  # (5211144, 22)
                 on='TrustIncidentId', how='inner'). \
            join(self.stanox_section,  # (5211027, 29)
                 on='StanoxSectionId', how='inner'). \
            join(self.location,  # (5207471, 36)
                 on='LocationId', how='inner', lsuffix='', rsuffix='_Location'). \
            join(stanox_location,  # (5194206, 42)
                 on='StartStanox', how='inner', lsuffix='_Section', rsuffix=''). \
            join(stanox_location,  # (5189992, 48)
                 on='EndStanox', how='inner', lsuffix='_Start', rsuffix='_End'). \
            join(self.incident_reason_info,  # (5188756, 54)
                 on='IncidentReasonCode', how='inner'). \
            join(self.imdm, on='IMDM_Location', how='inner')  # (5188724, 57)
        # Note: There may be errors in e.g. IMDM data/column, location id, of the TrustIncident table.
        gc.collect()

        # Check / update location names
        data = self._check_location_names(data)  # (5188724, 57)
        gc.collect()

        # Drop unused columns
        data.drop(columns=['IMDM', 'Location_Start', 'Location_End'], inplace=True)  # (5188724, 54)

        # Rename columns
        rename_columns = {
            'LocationAlias_Start': 'StartLocationAlias',
            'LocationAlias_End': 'EndLocationAlias',
            'IMDM_Location': 'IMDM',
            'ELR_Start': 'StartELR',
            'Yards_Start': 'StartYards',
            'ELR_End': 'EndELR',
            'Yards_End': 'EndYards',
            'Mileage_Start': 'StartMileage',
            'Mileage_End': 'EndMileage',
            'LocationId_Start': 'StartLocationId',
            'LocationId_End': 'EndLocationId',
            'LocationId_Section': 'SectionLocationId',
            'StartDate': 'StartDateTime',
            'EndDate': 'EndDateTime',
        }
        data.rename(columns=rename_columns, inplace=True)

        # Check / update longitudes and latitudes
        data = self._check_geom_coords(data)
        gc.collect()

        # Import data into database
        view_name = 'schedule8_details'
        self._dump_prep_data(data=data, table_name=view_name, verbose=verbose, pkey=[], **kwargs)
        self.db_instance.null_text_to_empty_string(
            table_name=view_name, schema_name=self.SCHEMA_NAME)

        schedule8_details = get_subset(
            data, route_name=route_name, weather_category=weather_category)
        gc.collect()

        return schedule8_details

    def _view_schedule8_details(self, **kwargs):
        """
        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx._view_schedule8_details()
            >>> mtx.schedule8_details
            (5184316, 55)
        """

        read_args = {
            'table_name': 'schedule8_details',
            'true_values': ['t', 'true'],
            'false_values': ['f', 'false'],
            'dtype': {'StartMileage': str, 'EndMileage': str, 'IncidentCategory': str},
            'low_memory': False,
            'keep_default_na': False,
        }
        kwargs.update(read_args)
        self._read_data(**kwargs)

        # fill_na_cols = ['WeatherCategory', 'IncidentFMS', 'IncidentEquipment']
        # self.schedule8_details.fillna(
        #     dict(zip(fill_na_cols, [''] * len(fill_na_cols))), inplace=True)

        if 'PfPIId' in self.schedule8_details.columns:
            self.schedule8_details.set_index('PfPIId', inplace=True)

    @staticmethod
    def _add_query_condition(sql_query, name, cond):
        if cond:
            v1 = 'WHERE' if 'WHERE' not in sql_query else 'AND'
            v2 = 'IN' if isinstance(cond, tuple) else '='
            sql_query += f' {v1} "{name}" {v2} \'{cond}\''
        return sql_query

    def view_schedule8_details(self, route_name=None, weather_category=None, column_names=None,
                               update=False, verbose=False, **kwargs):
        """
        Get a view of essential details about Schedule 8 Incidents.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :param column_names:
        :type column_names:
        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: essential details about Schedule 8 Incidents
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_details()
            >>> mtx.schedule8_details.shape
            (5188756, 40)
            >>> mtx.view_schedule8_details(route_name='Anglia')
            >>> mtx.schedule8_details.shape
            (375523, 40)
            >>> mtx.view_schedule8_details(route_name='Anglia', weather_category='Wind')
            >>> mtx.schedule8_details.shape
            (3323, 40)
            >>> mtx.view_schedule8_details(route_name='Anglia', weather_category='Heat')
            >>> mtx.schedule8_details.shape
            (1441, 40)
        """

        if column_names is None:
            col_names_ = [
                'PfPIId',
                'IncidentRecordId',
                'TrustIncidentId',
                'IncidentNumber',
                'PerformanceEventCode', 'PerformanceEventGroup', 'PerformanceEventName',
                'PfPIMinutes', 'PfPICosts', 'FinancialYear',
                'IncidentRecordCreateDate',
                'StartDateTime', 'EndDateTime',
                'IncidentDescription', 'IncidentJPIPCategory',
                'WeatherCategory',
                'IncidentReasonCode', 'IncidentReasonDescription',
                'IncidentCategory', 'IncidentCategoryDescription',
                # 'IncidentCategoryGroupDescription',
                'IncidentFMS', 'IncidentEquipment',
                'WeatherCell',
                'Route', 'IMDM', 'Region',
                'StanoxSection', 'StartLocation', 'EndLocation',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'ApproximateLocation']
        else:
            col_names_ = column_names.copy()

        pkl_filename = make_filename(
            "schedule8_details", route_name=route_name, weather_category=weather_category, sep="_",
            save_as=".pkl.xz")
        path_to_pkl_xz = self.cdd("views", pkl_filename)

        if os.path.isfile(path_to_pkl_xz) and not update:
            schedule8_details = load_data(path_to_pkl_xz, verbose=verbose)
            self.schedule8_details = schedule8_details[col_names_]

        else:
            if self.db_instance is None:
                self.db_instance = WxRailIncidentsPred(verbose=False)

            col_names = ', '.join([f'"{x}"' for x in col_names_])

            tbl = f'"{self.SCHEMA_NAME}"."schedule8_details"'
            query = f'SELECT {col_names} FROM {tbl}'

            query = self._add_query_condition(query, 'Route', route_name)
            query = self._add_query_condition(query, 'WeatherCategory', weather_category)

            parse_dates_columns = [
                x for x in col_names_
                if x in ['IncidentRecordCreateDate', 'StartDateTime', 'EndDateTime']]
            read_csv_args = {
                'true_values': ['t', 'true'],
                'false_values': ['f', 'false'],
                'dtype': {
                    'StartMileage': str, 'EndMileage': str, 'IncidentCategory': str,
                    'StartStanox': str, 'EndStanox': str},
                'parse_dates': parse_dates_columns,
                'keep_default_na': False,
                'low_memory': False,
            }
            kwargs.update(read_csv_args)

            self.schedule8_details = self.db_instance.read_sql_query(query, **kwargs)

            stanox_columns = ['StartStanox', 'EndStanox']
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
                for col in stanox_columns:
                    if col in self.schedule8_details.columns:
                        self.schedule8_details[col] = p.map(
                            fix_stanox, self.schedule8_details[col])

            save_data(self.schedule8_details, path_to_pkl_xz, verbose=verbose)

            gc.collect()

    def view_schedule8_cost_by_location(self, route_name=None, weather_category=None, update=False,
                                        verbose=False, **kwargs):
        """
        Get Schedule 8 data by incident location and Weather category.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Schedule 8 data by incident location and Weather category
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_cost_by_location()
            >>> mtx.schedule8_cost_by_location.shape
            (25186, 20)
            >>> mtx.view_schedule8_cost_by_location(route_name='Anglia')
            >>> mtx.schedule8_cost_by_location.shape
            (2498, 20)
            >>> mtx.view_schedule8_cost_by_location(route_name='Anglia', weather_category='Wind')
            >>> mtx.schedule8_cost_by_location.shape
            (258, 20)
            >>> mtx.view_schedule8_cost_by_location(route_name='Anglia', weather_category='Heat')
            >>> mtx.schedule8_cost_by_location.shape
            (154, 20)
        """

        pkl_filename = make_filename(
            "schedule8_cost_by_location", route_name=route_name,
            weather_category=weather_category, sep="_", save_as=".pkl.xz")
        path_to_pkl_xz = self.cdd("views", pkl_filename)

        if os.path.isfile(path_to_pkl_xz) and not update:
            self.schedule8_cost_by_location = load_data(path_to_pkl_xz, verbose=verbose)

        else:
            column_names = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'WeatherCategory', 'Route', 'IMDM', 'Region', 'StanoxSection',
                'StartLocation', 'EndLocation', 'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartStanox', 'EndStanox',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'PfPIMinutes', 'PfPICosts'
            ]

            self.view_schedule8_details(
                route_name=route_name, weather_category=weather_category, column_names=column_names,
                **kwargs)

            self.schedule8_cost_by_location = self.calculate_pfpi_stats(
                self.schedule8_details, variables=column_names)

            save_data(self.schedule8_cost_by_location, path_to_pkl_xz, verbose=verbose)

    def view_schedule8_cost_by_day_location(self, route_name=None, weather_category=None,
                                            update=False, verbose=False, **kwargs):
        """
        Get Schedule 8 data by datetime and location.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Schedule 8 data by datetime and location
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_cost_by_day_location()
            >>> mtx.schedule8_cost_by_day_location.shape
            (3915368, 24)
            >>> mtx.view_schedule8_cost_by_day_location(route_name='Anglia')
            >>> mtx.schedule8_cost_by_day_location.shape
            (262852, 24)
            >>> mtx.view_schedule8_cost_by_day_location('Anglia', weather_category='Wind')
            >>> mtx.schedule8_cost_by_day_location.shape
            (1748, 24)
            >>> mtx.view_schedule8_cost_by_day_location('Anglia', weather_category='Heat')
            >>> mtx.schedule8_cost_by_day_location.shape
            (914, 24)
        """

        pkl_filename = make_filename(
            "schedule8_cost_by_day_location", route_name=route_name,
            weather_category=weather_category, sep="_", save_as=".pkl.xz")
        path_to_pkl_xz = self.cdd("views", pkl_filename)

        if os.path.isfile(path_to_pkl_xz) and not update:
            self.schedule8_cost_by_day_location = load_data(path_to_pkl_xz, verbose=verbose)

        else:
            column_names = [
                'PfPIId',
                # 'TrustIncidentId', 'IncidentRecordCreateDate',
                'FinancialYear',
                'StartDateTime', 'EndDateTime',
                'WeatherCategory',
                'StanoxSection', 'Route', 'IMDM', 'Region',
                'StartLocation', 'EndLocation', 'StartStanox', 'EndStanox',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'WeatherCell',
                'PfPICosts', 'PfPIMinutes']

            self.view_schedule8_details(
                route_name=route_name, weather_category=weather_category, column_names=column_names,
                **kwargs)

            self.schedule8_cost_by_day_location = self.calculate_pfpi_stats(
                self.schedule8_details, variables=column_names,
                sort_by=['StartDateTime', 'EndDateTime'])

            save_data(self.schedule8_cost_by_day_location, path_to_pkl_xz, verbose=verbose)

    def view_schedule8_cost_by_day_location_reason(self, route_name=None, weather_category=None,
                                                   update=False, verbose=False, **kwargs):
        """
        Get Schedule 8 costs by datetime, location and incident reason.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | list | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | list | None
        :param update: Whether update the data stored in the project database;
            defaults to ``False``.
        :type update: bool
        :param verbose: Whether to print relevant information to the console; defaults to ``False``.
        :type verbose: bool | int
        :return: Schedule 8 costs by datetime, location and incident reason
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_cost_by_day_location_reason()
            >>> mtx.schedule8_cost_by_day_location_reason.shape
            (3970759, 30)
            >>> mtx.view_schedule8_cost_by_day_location_reason(route_name='Anglia')
            >>> mtx.schedule8_cost_by_day_location_reason.shape
            (263997, 30)
            >>> mtx.view_schedule8_cost_by_day_location_reason('Anglia', weather_category='Wind')
            >>> mtx.schedule8_cost_by_day_location_reason.shape
            (1748, 30)
            >>> mtx.view_schedule8_cost_by_day_location_reason('Anglia', weather_category='Heat')
            >>> mtx.schedule8_cost_by_day_location_reason.shape
            (916, 30)
        """

        pkl_filename = make_filename(
            "schedule8_cost_by_day_location_reason", route_name=route_name,
            weather_category=weather_category, sep="_", save_as=".pkl.xz")
        path_to_pkl_xz = self.cdd("views", pkl_filename)

        if os.path.isfile(path_to_pkl_xz) and not update:
            self.schedule8_cost_by_day_location_reason = load_data(path_to_pkl_xz, verbose=verbose)

        else:
            column_names = [
                'PfPIId',
                'FinancialYear', 'StartDateTime', 'EndDateTime',
                'WeatherCategory',
                'WeatherCell', 'Route', 'IMDM', 'Region',
                'StanoxSection', 'StartLocation', 'EndLocation', 'StartStanox', 'EndStanox',
                'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
                'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
                'IncidentDescription', 'IncidentCategory', 'IncidentCategoryDescription',
                # 'IncidentCategorySuperGroupCode',
                # 'IncidentCategoryGroupDescription',
                'IncidentReasonCode', 'IncidentReasonDescription',
                # 'IncidentReasonName',
                'IncidentJPIPCategory',
                'PfPIMinutes', 'PfPICosts']

            self.view_schedule8_details(
                route_name=route_name, weather_category=weather_category, column_names=column_names,
                **kwargs)

            self.schedule8_cost_by_day_location_reason = self.calculate_pfpi_stats(
                self.schedule8_details, variables=column_names,
                sort_by=['StartDateTime', 'EndDateTime'])

            save_data(self.schedule8_cost_by_day_location_reason, path_to_pkl_xz, verbose=verbose)

    def view_schedule8_cost_by_day(self, route_name=None, weather_category=None, **kwargs):
        """
        Get Schedule 8 data by datetime and Weather category.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :return: Schedule 8 data by datetime and Weather category
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> dat = mtx.view_schedule8_cost_by_day()
            >>> dat.shape
            (3907378, 11)
            >>> dat = mtx.view_schedule8_cost_by_day(route_name='Anglia', weather_category='Wind')
            >>> dat.shape
            (1748, 11)
            >>> dat = mtx.view_schedule8_cost_by_day(route_name='Anglia', weather_category='Heat')
            >>> dat.shape
            (914, 11)
        """

        column_names = [
            'PfPIId',
            # 'TrustIncidentId', 'IncidentRecordCreateDate',
            'FinancialYear', 'StartDateTime', 'EndDateTime',
            'WeatherCategory',
            'Route', 'IMDM', 'Region', 'WeatherCell',
            'PfPICosts', 'PfPIMinutes']

        self.view_schedule8_details(
            route_name=route_name, weather_category=weather_category, column_names=column_names,
            **kwargs)

        s8c_by_day = self.calculate_pfpi_stats(
            self.schedule8_details, variables=column_names, sort_by=['StartDateTime', 'EndDateTime'])

        return s8c_by_day

    def view_schedule8_cost_by_reason(self, route_name=None, weather_category=None, **kwargs):
        """
        Get Schedule 8 costs by incident reason.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :return: Schedule 8 costs by incident reason
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> dat = mtx.view_schedule8_cost_by_reason(route_name='Anglia', weather_category='Wind')
            >>> dat.shape
            (1705, 16)
            >>> dat = mtx.view_schedule8_cost_by_reason(route_name='Anglia', weather_category='Heat')
            >>> dat.shape
            (804, 16)
        """

        column_names = [
            'PfPIId',
            'FinancialYear',
            'Route', 'IMDM', 'Region',
            'WeatherCategory',
            'IncidentDescription', 'IncidentCategory', 'IncidentCategoryDescription',
            # 'IncidentCategorySuperGroupCode',
            # 'IncidentCategoryGroupDescription',
            'IncidentReasonCode', 'IncidentReasonDescription',
            # 'IncidentReasonName',
            'IncidentJPIPCategory',
            'PfPIMinutes', 'PfPICosts']

        self.view_schedule8_details(
            route_name=route_name, weather_category=weather_category, column_names=column_names,
            **kwargs)

        s8c_by_reason = self.calculate_pfpi_stats(data=self.schedule8_details, variables=column_names)

        return s8c_by_reason

    def view_schedule8_cost_by_location_reason(self, route_name=None, weather_category=None,
                                               **kwargs):
        """
        Get Schedule 8 costs by location and incident reason.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :return: Schedule 8 costs by location and incident reason
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> dat = mtx.view_schedule8_cost_by_location_reason('Anglia', weather_category='Wind')
            >>> dat.shape
            (1715, 29)
            >>> dat = mtx.view_schedule8_cost_by_location_reason('Anglia', weather_category='Heat')
            >>> dat.shape
            (824, 29)
        """

        column_names = [
            'PfPIId',
            'FinancialYear',
            'WeatherCategory',
            'Route', 'IMDM', 'Region',
            'StanoxSection', 'StartStanox', 'EndStanox', 'StartLocation', 'EndLocation',
            'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
            'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude',
            'IncidentDescription', 'IncidentCategory', 'IncidentCategoryDescription',
            # 'IncidentCategorySuperGroupCode',
            # 'IncidentCategoryGroupDescription',
            'IncidentReasonCode', 'IncidentReasonDescription',
            # 'IncidentReasonName',
            'IncidentJPIPCategory',
            'PfPIMinutes', 'PfPICosts']

        self.view_schedule8_details(
            route_name=route_name, weather_category=weather_category, column_names=column_names,
            **kwargs)

        s8c_by_location_reason = self.calculate_pfpi_stats(
            data=self.schedule8_details, variables=column_names)

        return s8c_by_location_reason

    def view_schedule8_cost_by_weather(self, route_name=None, **kwargs):
        """
        Get Schedule 8 costs by Weather category.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :return: Schedule 8 costs by Weather category
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> dat = mtx.view_schedule8_cost_by_weather(route_name='Anglia')
            >>> dat.shape
            (367, 8)
        """

        column_names = [
            'PfPIId', 'FinancialYear', 'Route', 'IMDM', 'Region', 'WeatherCategory',
            'PfPICosts', 'PfPIMinutes']

        self.view_schedule8_details(
            route_name=route_name, weather_category=None, column_names=column_names,
            **kwargs)

        s8c_by_weather = self.calculate_pfpi_stats(self.schedule8_details, variables=column_names)

        return s8c_by_weather

    def view_schedule8_incident_locations(self, route_name=None, weather_category=None,
                                          start_end_elr=None, **kwargs):
        """
        Get Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section.

        :param route_name: name of a Route; defaults to ``None``.
        :type route_name: str | None
        :param weather_category: Weather category; defaults to ``None``.
        :type weather_category: str | None
        :param start_end_elr: indicating if start ELR and end ELR are the same or not,
            defaults to ``None``
        :type start_end_elr: None | bool | str
        :return: Schedule 8 costs (delay minutes & costs) aggregated for each STANOX section
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx.view_schedule8_incident_locations()
            >>> mtx.schedule8_incident_locations.shape
            (10042, 22)
            >>> mtx.view_schedule8_incident_locations(start_end_elr=True)
            >>> mtx.schedule8_incident_locations.shape
            (5367, 22)
            >>> mtx.view_schedule8_incident_locations(start_end_elr=False)
            >>> mtx.schedule8_incident_locations.shape
            (4675, 22)
            >>> mtx.view_schedule8_incident_locations(route_name='Anglia')
            >>> mtx.schedule8_incident_locations.shape
            (1101, 22)
            >>> mtx.view_schedule8_incident_locations(weather_category='Wind')
            >>> mtx.schedule8_incident_locations.shape
            (2595, 22)
            >>> mtx.view_schedule8_incident_locations(weather_category='Heat')
            >>> mtx.schedule8_incident_locations.shape
            (1687, 22)
            >>> mtx.view_schedule8_incident_locations(route_name='Anglia', weather_category='Wind')
            >>> mtx.schedule8_incident_locations.shape
            (258, 22)
            >>> mtx.view_schedule8_incident_locations(route_name='Anglia', weather_category='Heat')
            >>> mtx.schedule8_incident_locations.shape
            (154, 22)
        """

        # All incident locations
        self.view_schedule8_cost_by_location(
            route_name=route_name, weather_category=weather_category, **kwargs)

        selected_columns = [
            'Route', 'IMDM', 'Region',
            'StanoxSection', 'StartLocation', 'EndLocation',
            'StartELR', 'StartMileage', 'EndELR', 'EndMileage',
            'StartStanox', 'EndStanox',
            'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude'
        ]

        data = self.schedule8_cost_by_location.loc[:, selected_columns].drop_duplicates()

        if start_end_elr is True:
            # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
            data = data[data['StartELR'] == data['EndELR']]
        elif start_end_elr is False:
            # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
            data = data[data['StartELR'] != data['EndELR']]

        # # Remove records where information of either 'StartELR' or 'EndELR' was missing
        # incident_locations = incident_locations[
        #     ~(incident_locations.StartELR.str.contains('^$')) & ~(
        #         incident_locations.EndELR.str.contains('^$'))]

        # # 'FJH'
        # elr, loc_name, mileage = 'FJH', 'Halton Junction', '0.0000'
        # idx = (data['StartELR'] == elr) & (data['StartLocation'] == loc_name)
        # data.loc[idx, 'StartMileage'] = mileage
        # idx = (data['EndELR'] == elr) & (data['EndLocation'] == loc_name)
        # data.loc[idx, 'EndMileage'] = mileage
        #
        # # 'BNE'
        # elr, loc_name, mileage = 'BNE', 'Benton North Junction', '0.0000'
        # idx = (data['StartELR'] == elr) & (data['StartLocation'] == loc_name)
        # data.loc[idx, 'StartMileage'] = mileage
        # idx = (data['EndELR'] == elr) & (data['EndLocation'] == loc_name)
        # data.loc[idx, 'EndMileage'] = mileage

        # # 'WCI'
        # elr, loc_name, mileage = 'WCI', 'Grangetown (Cleveland)', mile_chain_to_mileage('1.38')
        # idx = (data['StartELR'] == elr) & (data['StartLocation'] == loc_name)
        # data.loc[idx, 'StartMileage'] = mileage
        # idx = (data['EndELR'] == elr) & (data['EndLocation'] == loc_name)
        # data.loc[idx, 'EndMileage'] = mileage

        # # 'SJD'
        # idx = (data['EndELR'] == 'SJD') & (data['EndLocation'] == 'Skelton Junction [Manchester]')
        # data.loc[idx, 'EndMileage'] = '0.0000'
        #
        # # 'HLK'
        # idx = (data['EndELR'] == 'HLK') & (data['EndLocation'] == 'High Level Bridge Junction')
        # data.loc[idx, 'EndMileage'] = '0.0000'

        # Create two additional columns about data of Mileages (convert str to num)
        data[['StartMileage_num', 'EndMileage_num']] = data[['StartMileage', 'EndMileage']].map(
            mileage_str_to_num)

        data['StartEasting'], data['StartNorthing'] = wgs84_to_osgb36(
            data['StartLongitude'], data['StartLatitude'])
        data['EndEasting'], data['EndNorthing'] = wgs84_to_osgb36(
            data['EndLongitude'], data['EndLatitude'])

        self.schedule8_incident_locations = data

    # (TBC)
    def __view_schedule8_incident_location_tracks(self, shift_yards=220):
        incident_locations = self.view_schedule8_incident_locations()

        if self.track is None:
            self.read_track()

        # track_summary = self.get_track_summary()

        # Testing e.g.
        elr = incident_locations.StartELR.loc[24133]
        # start_location = shapely.geometry.Point(
        #     incident_locations[['StartEasting', 'StartNorthing']].iloc[0].values)
        # end_location = shapely.geometry.Point(
        #     incident_locations[['EndEasting', 'EndNorthing']].iloc[0].values)
        start_mileage_num = incident_locations.StartMileage_num.loc[24133]
        # start_yard = nr_mileage_to_yards(start_mileage_num)
        end_mileage_num = incident_locations.EndMileage_num.loc[24133]
        # end_yard = nr_mileage_to_yards(end_mileage_num)

        #
        track_elr_mileages = self.track[self.track.ELR == elr]

        # Call OSM railway data
        from pydriosm import GeofabrikReader

        geofabrik_reader = GeofabrikReader()

        # Download/Read GB OSM data
        geofabrik_reader.read_shp(
            'Great Britain', layer_names='railways', feature_names='rail',
            data_dir=cdd("network/osm"), pickle_it=True, rm_extracts=True,
            rm_shp_zip=True)

        # Import it into PostgreSQL

        # Write a query to get track coordinates available in OSM data

        if start_mileage_num <= end_mileage_num:

            if start_mileage_num == end_mileage_num:
                start_mileage_num = shift_mileage_by_yard(start_mileage_num, -shift_yards)
                end_mileage_num = shift_mileage_by_yard(end_mileage_num, shift_yards)

            # Get adjusted Mileages of start and end locations
            incident_track = track_elr_mileages[
                (start_mileage_num >= track_elr_mileages.StartMileage_num) &
                (end_mileage_num <= track_elr_mileages.EndMileage_num)]

        else:
            incident_track = track_elr_mileages[
                (start_mileage_num <= track_elr_mileages.EndMileage_num) &
                (end_mileage_num >= track_elr_mileages.StartMileage_num)]

        return incident_track

    def _update_views(self, verbose=True):
        """

        :param verbose: Whether to print relevant information in console as the function runs,
            defaults to ``True``
        :type verbose: bool | int

        **Examples**::

            >>> from src.preprocessor.metex import METEX
            >>> mtx = METEX()
            >>> mtx._update_views()
        """

        if confirmed(f'To update the views of "{self.SCHEMA_NAME}"\n?'):
            if self.db_instance is None:
                self.db_instance = WxRailIncidentsPred(verbose=False)

            _ = self._schedule8_details(update=True, verbose=verbose)

            route_name = 'Anglia'

            update_args = {'update': True, 'verbose': False}

            if verbose:
                print("\nProcessing ... ")
                print("\tSchedule 8 cost (by location)", end=" ... ")

            try:
                self.view_schedule8_cost_by_location(**update_args)
                self.view_schedule8_cost_by_location(route_name, **update_args)
                self.view_schedule8_cost_by_location(route_name, 'Wind', **update_args)
                self.view_schedule8_cost_by_location(route_name, 'Heat', **update_args)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_message(e, verbose=verbose, raise_error=True)

            if verbose:
                print("\tSchedule 8 cost (by day and location)", end=" ... ")

            try:
                self.view_schedule8_cost_by_day_location(**update_args)
                self.view_schedule8_cost_by_day_location(route_name, **update_args)
                self.view_schedule8_cost_by_day_location(route_name, 'Wind', **update_args)
                self.view_schedule8_cost_by_day_location(route_name, 'Heat', **update_args)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_message(e, verbose=verbose, raise_error=True)

            if verbose:
                print("\tSchedule 8 cost (by day, location and incident reason)", end=" ... ")

            try:
                self.view_schedule8_cost_by_day_location_reason(**update_args)
                self.view_schedule8_cost_by_day_location_reason(route_name, **update_args)
                self.view_schedule8_cost_by_day_location_reason(route_name, 'Wind', **update_args)
                self.view_schedule8_cost_by_day_location_reason(route_name, 'Heat', **update_args)

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_message(e, verbose=verbose, raise_error=True)

            if verbose:
                print("\tSchedule 8 incident locations", end=" ... ")

            try:
                self.view_schedule8_incident_locations()
                self.view_schedule8_incident_locations(start_end_elr=True)
                self.view_schedule8_incident_locations(start_end_elr=False)
                self.view_schedule8_incident_locations(route_name)
                self.view_schedule8_incident_locations(route_name, 'Wind')
                self.view_schedule8_incident_locations(route_name, 'Heat')

                if verbose:
                    print("Done.")

            except Exception as e:
                _print_failure_message(e, verbose=verbose, raise_error=True)

            if verbose:
                print("Update finished.")
