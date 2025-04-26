"""Processing of furlong data."""

import functools
import itertools
import multiprocessing
import os

import numpy as np
import pandas as pd
from pyhelpers.dirs import cdd
from pyhelpers.store import load_data, save_data
from pyrcs.converter import mileage_num_to_str, mileage_str_to_num, shift_mileage_by_yard
from pyrcs.line_data import ELRMileages

from src.utils import WxRailIncidentsPred, get_subset, make_filename


# noinspection PyShadowingNames
class FurlongHandler:
    from src.preprocessor import METEX, Vegetation

    METEX = METEX()

    VEGETATION = Vegetation()

    @classmethod
    def cdd(cls, *sub_dir, mkdir=False):
        """
        Change directory to "data\\Network\\geodata" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: full path to "data\\Network\\geodata" and subdirectories / a file
        :rtype: str

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler
            >>> import os
            >>> fur = FurlongHandler()
            >>> os.path.relpath(fur.cdd())
            'data\\network\\furlongs'
        """

        path = cdd("network/furlongs", *sub_dir, mkdir=mkdir)

        return path

    # == Tools =====================================================================================

    @classmethod
    def adjust_incident_mileages(cls, critical_variables, ref_furlongs, shift_yards):
        """
        Get adjusted Mileages of the start and end locations for each recorded incident,
        based on ``'StartELR'`` (or ``'EndELR'``), ``'StartMileage_num'`` and ``'EndMileage_num'``.

        :param critical_variables:
        :type critical_variables:
        :param ref_furlongs: reference furlong data
        :type ref_furlongs: pandas.DataFrame
        :param shift_yards: yards by which the start/end mileage is shifted for adjustment
        :type shift_yards: int or float
        :return: adjusted Mileages of incident locations and critical furlong IDs
        :rtype: tuple

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler
            >>> from src.preprocessor import METEX, Vegetation

            >>> veg = Vegetation()
            >>> veg.view_vegetation_condition2(route_name='Anglia')
            >>> ref_furlongs = veg.vegetation_condition2.copy()

            >>> METEX = METEX()
            >>> METEX.view_schedule8_incident_locations('Anglia', 'Wind', start_end_elr=True)
            >>> loc_same_elr = METEX.schedule8_incident_locations.copy()

            >>> critical_var_cols = ['StartELR', 'StartMileage_num', 'EndMileage_num']
            >>> loc_same_elr['Critical_Variables'] = loc_same_elr[critical_var_cols].values.tolist()

            >>> flh = FurlongHandler()

            >>> shift_yards = 220

            >>> i = 0
            >>> critical_var = loc_same_elr['Critical_Variables'].iloc[i]
            >>> adj_mileages = flh.adjust_incident_mileages(critical_var, ref_furlongs, shift_yards)
            >>> adj_mileages
            ['5.1100', '6.0000', 5.11, 6.0, 1566.3999999999994, [60043, 30669, 35531]]

            >>> i = 18
            >>> critical_var = loc_same_elr['Critical_Variables'].iloc[i]
            >>> adj_mileages = flh.adjust_incident_mileages(critical_var, ref_furlongs, shift_yards)
            >>> adj_mileages
            ['59.0722', '69.0000', 59.0722, 69.0, 17472.927999999996, [47137, ..., 54266]]

            >>> i = 3
            >>> critical_var = loc_same_elr['Critical_Variables'].iloc[i]
            >>> adj_mileages = flh.adjust_incident_mileages(critical_var, ref_furlongs, shift_yards)
            >>> adj_mileages
            ['51.1100', '46.0880', 51.11, 46.088, 8838.719999999998, [44675, ..., 53373]]
        """

        elr, start_mileage_num, end_mileage_num = critical_variables

        column_names = ['ELR', 'StartMileage', 'EndMileage', 'StartMileage_num', 'EndMileage_num']

        try:
            elr_furlongs = ref_furlongs.query(f"ELR == '{elr}'")[column_names]

            # Merge the Mileages (num) of both start and end
            elr_mileages = pd.concat(
                [elr_furlongs['StartMileage_num'], elr_furlongs['EndMileage_num']])
            elr_mileages = elr_mileages.drop_duplicates(keep='first').sort_values()

            m_indices, s_indices, e_indices = map(
                pd.Index, [elr_mileages, elr_furlongs['StartMileage'], elr_furlongs['EndMileage']])

            if start_mileage_num <= end_mileage_num:

                if start_mileage_num == end_mileage_num:
                    start_mileage_num = shift_mileage_by_yard(start_mileage_num, -shift_yards)
                    end_mileage_num = shift_mileage_by_yard(end_mileage_num, shift_yards)

                # Get adjusted Mileages and 'FurlongID' for the start location
                try:
                    adj_start_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([start_mileage_num], method='ffill')].values[0]
                except (ValueError, KeyError):
                    adj_start_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([start_mileage_num], method='nearest')].values[0]

                try:
                    s_idx = s_indices.get_loc(mileage_num_to_str(adj_start_mileage_num))
                except (ValueError, KeyError):
                    s_idx = e_indices.get_loc(mileage_num_to_str(adj_start_mileage_num))
                    adj_start_mileage_num = mileage_str_to_num(
                        elr_furlongs['StartMileage'].iloc[s_idx])

                # Get adjusted Mileages and 'FurlongID' for the start location
                try:
                    adj_end_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([end_mileage_num], method='bfill')].values[0]
                except (ValueError, KeyError):
                    adj_end_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([end_mileage_num], method='nearest')].values[0]

                try:
                    e_idx = e_indices.get_loc(mileage_num_to_str(adj_end_mileage_num))
                except (ValueError, KeyError):
                    e_idx = s_indices.get_loc(mileage_num_to_str(adj_end_mileage_num))
                    adj_end_mileage_num = mileage_str_to_num(elr_furlongs['EndMileage'].iloc[e_idx])

            else:  # start_mileage_num > end_mileage_num
                # Get adjusted Mileages of start and end locations
                try:
                    adj_start_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([start_mileage_num], method='bfill')].values[0]
                except (ValueError, KeyError):
                    adj_start_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([start_mileage_num], method='nearest')].values[0]
                try:
                    adj_end_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([end_mileage_num], method='ffill')].values[0]
                except (ValueError, KeyError):
                    adj_end_mileage_num = elr_mileages.iloc[
                        m_indices.get_indexer([end_mileage_num], method='nearest')].values[0]

                # Get 'FurlongID's
                try:
                    s_idx = e_indices.get_loc(mileage_num_to_str(adj_start_mileage_num))
                except (ValueError, KeyError):
                    s_idx = s_indices.get_loc(mileage_num_to_str(adj_start_mileage_num))
                    adj_start_mileage_num = mileage_str_to_num(
                        elr_furlongs['EndMileage'].iloc[s_idx])
                try:
                    e_idx = s_indices.get_loc(mileage_num_to_str(adj_end_mileage_num))
                except (ValueError, KeyError):
                    e_idx = e_indices.get_loc(mileage_num_to_str(adj_end_mileage_num))
                    adj_end_mileage_num = mileage_str_to_num(
                        elr_furlongs['StartMileage'].iloc[e_idx])

            if s_idx <= e_idx:
                e_idx = e_idx + 1 if e_idx < len(elr_mileages) else e_idx
                nr_elr_furlongs_dat = elr_furlongs.iloc[s_idx:e_idx]
            else:  # s_idx > e_idx
                s_idx = s_idx + 1 if s_idx < len(elr_mileages) else s_idx
                nr_elr_furlongs_dat = elr_furlongs.iloc[e_idx:s_idx]
            critical_furlong_id = list(set(nr_elr_furlongs_dat.index))

            adj_start_mileage, adj_end_mileage = map(
                mileage_num_to_str, [adj_start_mileage_num, adj_end_mileage_num])
            distance = np.abs(adj_end_mileage_num - adj_start_mileage_num) * 1760

        except (IndexError, KeyError):
            adj_start_mileage, adj_end_mileage = '', ''
            adj_start_mileage_num, adj_end_mileage_num, distance = np.nan, np.nan, np.nan
            critical_furlong_id = []

        adjusted_incident_mileages = [
            adj_start_mileage,
            adj_end_mileage,
            adj_start_mileage_num,
            adj_end_mileage_num,
            distance,
            critical_furlong_id,
        ]

        return adjusted_incident_mileages

    @classmethod
    def get_connecting_nodes(cls, diff_start_end_elr_dat, route_name=None, update=False,
                             verbose=False):
        """
        Get data of connecting points for different ELRs.

        :param diff_start_end_elr_dat: data frame where StartELR != EndELR
        :type diff_start_end_elr_dat: pandas.DataFrame
        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of connecting points for different ELRs
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler
            >>> from src.preprocessor import METEX

            >>> flh = FurlongHandler()
            >>> mtx = METEX()

            >>> mtx.view_schedule8_incident_locations(route_name='Anglia', start_end_elr=False)
            >>> diff_start_end_elr_dat = mtx.schedule8_incident_locations.copy()  # .iloc[:10]
            >>> connecting_nodes = flh.get_connecting_nodes(
            ...     diff_start_end_elr_dat, route_name='Anglia', verbose=True)
            >>> connecting_nodes.shape
            (429, 23)

            >>> mtx.view_schedule8_incident_locations(start_end_elr=False)
            >>> diff_start_end_elr_dat = mtx.schedule8_incident_locations.copy()  # .iloc[:10]
            >>> connecting_nodes = flh.get_connecting_nodes(diff_start_end_elr_dat, verbose=True)
            >>> connecting_nodes.shape
            (4668, 23)
        """

        filename = "connections_for_diff_ELRs"
        pickle_filename = make_filename(filename, route_name=route_name)
        path_to_pickle = cls.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            return load_data(path_to_pickle, verbose=verbose)

        else:
            try:
                pickle_filename_temp = make_filename(filename)
                path_to_pickle_temp = cls.cdd(pickle_filename_temp)

                if os.path.isfile(path_to_pickle_temp) and not update:
                    connecting_nodes_all = load_data(path_to_pickle_temp)
                    connecting_nodes = get_subset(data=connecting_nodes_all, route_name=route_name)

                else:
                    diff_elr_mileages = diff_start_end_elr_dat.drop_duplicates()

                    if verbose:
                        print("Searching for connecting ELRs", end=" ... ")
                    mileage_file_dir = cdd(
                        "../../data/Network/Railway_codes/Line-data/ELR-and-mileages/Mileages")

                    em = ELRMileages()
                    conn_mileages = diff_elr_mileages.apply(
                        lambda x: em.get_conn_mileages(
                            x['StartELR'], x['EndELR'], update=update, dump_dir=mileage_file_dir),
                        axis=1)
                    # # Debugging:
                    # for i, x in diff_elr_mileages.iterrows():
                    #     print(i)
                    #     try:
                    #         c = em.get_conn_mileages(x['StartELR'], x['EndELR'], update=True)
                    #     except Exception as e:
                    #         print(e)
                    #         break

                    if verbose:
                        print("Done.")

                    conn_column_names = [
                        'StartELR_EndMileage',
                        'ConnELR',
                        'ConnELR_StartMileage',
                        'ConnELR_EndMileage',
                        'EndELR_StartMileage',
                    ]
                    conn_mileages_data = pd.DataFrame(
                        conn_mileages.to_list(), index=diff_elr_mileages.index,
                        columns=conn_column_names)

                    connecting_nodes = diff_elr_mileages.join(conn_mileages_data)
                    connecting_nodes.set_index(
                        ['StartELR', 'StartMileage', 'EndELR', 'EndMileage'], inplace=True)

                save_data(connecting_nodes, path_to_file=path_to_pickle, verbose=verbose)

                return connecting_nodes

            except Exception as e:
                print(f'Failed to get "{filename}". {e}.')

    # == Furlongs of incident locations ============================================================

    def _adjust_mileages_for_same_elr(self, incident_locations, ref_furlongs, shift_yards):
        """

        :param incident_locations:
        :param ref_furlongs:
        :param shift_yards:
        :return:

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> flh.METEX.view_schedule8_incident_locations('Anglia', 'Wind', start_end_elr=True)
            >>> incident_locations = flh.METEX.schedule8_incident_locations.copy()

            >>> flh.VEGETATION.view_vegetation_condition2('Anglia')
            >>> ref_furlongs = flh.VEGETATION.vegetation_condition2.copy()

            >>> flh._adjust_mileages_for_same_elr(incident_locations, ref_furlongs, shift_yards=220)

        """

        critical_var_cols = ['StartELR', 'StartMileage_num', 'EndMileage_num']
        incident_locations['Critical_Variables'] = incident_locations[critical_var_cols].values.tolist()

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
            adj_mileages = p.map(
                functools.partial(
                    self.adjust_incident_mileages, ref_furlongs=ref_furlongs, shift_yards=shift_yards),
                incident_locations['Critical_Variables'])

        column_names = [
            'StartMileage_Adj',
            'EndMileage_Adj',
            'StartMileage_num_Adj',
            'EndMileage_num_Adj',
            'Section_Length_Adj',  # yards
            'Critical_FurlongIDs',
        ]
        adjusted_mileages = pd.DataFrame(
            data=adj_mileages, index=incident_locations.index, columns=column_names)

        return adjusted_mileages

    def adjust_mileages_for_same_elr(self, route_name, weather_category, shift_yards, update=False,
                                     verbose=False):
        """
        Get adjusted Mileages for each incident location where StartELR == EndELR.

        :param route_name: name of a Route; if ``None``, all Routes
        :type route_name: str or None
        :param weather_category: Weather category; if ``None``, all Weather categories
        :type weather_category: str or None
        :param shift_yards: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR == EndELR
        :type shift_yards: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: adjusted Mileages for each incident location where StartELR == EndELR
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> adj_mileages = flh.adjust_mileages_for_same_elr(
            ...     route_name='Anglia', weather_category='Wind', shift_yards=220, verbose=True)
            >>> adj_mileages.shape
            (183, 6)

            >>> adj_mileages = flh.adjust_mileages_for_same_elr(
            ...     route_name=None, weather_category=None, shift_yards=220, verbose=True)
            >>> adj_mileages.shape
            (5359, 6)
        """

        filename = "adj_mileages_for_same_ELRs"
        pickle_filename = make_filename(filename, route_name, weather_category, shift_yards)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            adjusted_mileages = load_data(path_to_pickle)

        else:
            if verbose:
                print("Getting adjusted Mileages for the same start-end ELR pairs", end=" ... ")

            try:
                if self.METEX.db_instance is None:
                    self.METEX.db_instance = WxRailIncidentsPred(verbose=False)

                # Get data of incident locations where the 'StartELR' and 'EndELR' are THE SAME
                self.METEX.view_schedule8_incident_locations(
                    route_name=route_name, weather_category=weather_category, start_end_elr=True)
                incident_locations = self.METEX.schedule8_incident_locations.copy()

                if self.VEGETATION.db_instance is None:
                    self.VEGETATION.db_instance = WxRailIncidentsPred(verbose=False)

                # Get furlong information as reference
                self.VEGETATION.view_vegetation_condition2()
                ref_furlongs = self.VEGETATION.vegetation_condition2.copy()

                # Calculate adjusted furlongs for each incident location regarding Vegetation
                adjusted_mileages = self._adjust_mileages_for_same_elr(
                    incident_locations=incident_locations, ref_furlongs=ref_furlongs,
                    shift_yards=shift_yards)

                if verbose:
                    print("Done.")

                save_data(adjusted_mileages, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                adjusted_mileages = None

        return adjusted_mileages

    def get_furlongs_for_same_elr(self, route_name=None, weather_category=None, shift_yards=220,
                                  update=False, verbose=False):
        """
        Get furlongs data for incident locations each identified by the same start and end ELRs,
        i.e. StartELR == EndELR.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param weather_category: Weather category; if ``None`` (default), all Weather categories
        :type weather_category: str or None
        :param shift_yards: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR == EndELR, defaults to ``220``
        :type shift_yards: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: furlongs data of incident locations each identified by the same start and end ELRs
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> furlongs_for_same_elr = flh.get_furlongs_for_same_elr(
            ...     'Anglia', weather_category='Wind', shift_yards=220, verbose=True)
            >>> furlongs_for_same_elr.shape
            (3265, 75)

            >>> furlongs_for_same_elr = flh.get_furlongs_for_same_elr(shift_yards=220, verbose=True)
            >>> furlongs_for_same_elr.shape
            (50590, 75)
        """

        filename = "furlongs_for_same_ELRs"
        pickle_filename = make_filename(filename, route_name, weather_category, shift_yards)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            furlongs_for_same_elr = load_data(path_to_pickle)

        else:
            adjusted_mileages = self.adjust_mileages_for_same_elr(
                route_name=route_name, weather_category=weather_category, shift_yards=shift_yards,
                verbose=verbose)

            if verbose:
                print("Getting furlong data for the same start-end ELR pairs", end=" ... ")

            try:
                if self.VEGETATION.db_instance is None:
                    self.VEGETATION.db_instance = WxRailIncidentsPred(verbose=False)

                self.VEGETATION.view_vegetation_condition2()
                ref_furlongs = self.VEGETATION.vegetation_condition2.copy()

                # Form a list containing all the furlong IDs
                furlong_ids = list(set(itertools.chain(*adjusted_mileages['Critical_FurlongIDs'])))
                # Select critical (i.e. incident) furlongs
                furlongs_for_same_elr = ref_furlongs.loc[furlong_ids]

                if verbose:
                    print("Done.")

                save_data(furlongs_for_same_elr, path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                furlongs_for_same_elr = None

        return furlongs_for_same_elr

    def _get_connecting_elr_mileages(self, incident_locations, route_name):
        """
        Get connecting points for different start-end ELR pairs.

        :param incident_locations:
        :param route_name:
        :return:
        """

        connecting_nodes = self.get_connecting_nodes(
            diff_start_end_elr_dat=incident_locations, route_name=route_name)
        connecting_nodes.set_index(['StanoxSection'], append=True, inplace=True)

        # Find End Mileage and Start Mileage of StartELR and EndELR, respectively
        locations_conn = incident_locations.join(
            connecting_nodes, on=list(connecting_nodes.index.names), rsuffix='_conn')  # dropna
        locations_conn.drop(
            columns=[x for x in locations_conn.columns if '_conn' in x], inplace=True)

        # # Remove the data records where connecting nodes are unknown
        # locations_conn = locations_conn[
        #     ~((locations_conn['StartELR_EndMileage'] == '') |
        #       (locations_conn['EndELR_StartMileage'] == ''))]

        # Convert str Mileages to num
        str_conn_cols = [
            'StartELR_EndMileage',
            'EndELR_StartMileage',
            'ConnELR_StartMileage',
            'ConnELR_EndMileage',
        ]
        num_conn_cols = [x + '_num' for x in str_conn_cols]
        locations_conn[num_conn_cols] = locations_conn[str_conn_cols].applymap(mileage_str_to_num)

        return locations_conn

    def _adjust_mileages_for_diff_elr(self, locations_conn, ref_furlongs, shift_yards):
        """

        :param locations_conn:
        :param ref_furlongs:
        :param shift_yards:
        :return:

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> route_name = 'Anglia'

            >>> flh.METEX.view_schedule8_incident_locations(route_name, 'Wind', start_end_elr=False)
            >>> incident_locations = flh.METEX.schedule8_incident_locations.copy()

            >>> flh.VEGETATION.view_vegetation_condition2(route_name)
            >>> ref_furlongs = flh.VEGETATION.vegetation_condition2.copy()

            >>> locations_conn = flh._get_connecting_elr_mileages(incident_locations, route_name)

            >>> shift_yards = 220

            >>> adjusted_mileages = flh._adjust_mileages_for_diff_elr(
            ...     locations_conn, ref_furlongs, shift_yards)
            >>> adjusted_mileages.shape
            (75, 6)
        """

        # -- Connections ---------------------------------------------------------------------------

        critical_var_cols_1 = ['ConnELR', 'ConnELR_StartMileage_num', 'ConnELR_EndMileage_num']
        locations_conn['Critical_Variables'] = locations_conn[critical_var_cols_1].values.tolist()

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
            adj_conn_mileages = p.map(
                functools.partial(
                    self.adjust_incident_mileages, ref_furlongs=ref_furlongs, shift_yards=shift_yards),
                locations_conn['Critical_Variables'])

        adj_conn_columns = [
            'Conn_StartMileage_Adj',
            'ConnELR_EndMileage_Adj',
            'Conn_StartMileage_num_Adj',
            'ConnELR_EndMileage_num_Adj',
            'ConnELR_Length_Adj',  # yards
            'ConnELR_Critical_FurlongIDs',
        ]
        adjusted_conn_mileages = pd.DataFrame(
            data=adj_conn_mileages, index=locations_conn.index, columns=adj_conn_columns)

        # -- Start ---------------------------------------------------------------------------------

        critical_var_cols_2 = ['StartELR', 'StartMileage_num', 'StartELR_EndMileage_num']
        locations_conn['Critical_Variables'] = locations_conn[critical_var_cols_2].values.tolist()

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
            adj_start_mileages = p.map(
                functools.partial(
                    self.adjust_incident_mileages, ref_furlongs=ref_furlongs, shift_yards=shift_yards),
                locations_conn['Critical_Variables'])

        adj_start_columns = [
            'StartMileage_Adj',
            'StartELR_EndMileage_Adj',
            'StartMileage_num_Adj',
            'StartELR_EndMileage_num_Adj',
            'StartELR_Length_Adj',  # yards
            'StartELR_Critical_FurlongIDs',
        ]
        adjusted_start_mileages = pd.DataFrame(
            data=adj_start_mileages, index=locations_conn.index, columns=adj_start_columns)

        # -- End -----------------------------------------------------------------------------------

        critical_var_cols_3 = ['EndELR', 'EndELR_StartMileage_num', 'EndMileage_num']
        locations_conn['Critical_Variables'] = locations_conn[critical_var_cols_3].values.tolist()

        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
            adj_end_mileages = p.map(
                functools.partial(
                    self.adjust_incident_mileages, ref_furlongs=ref_furlongs, shift_yards=shift_yards),
                locations_conn['Critical_Variables'])

        adj_end_columns = [
            'EndELR_StartMileage_Adj',
            'EndMileage_Adj',
            'EndELR_StartMileage_num_Adj',
            'EndMileage_num_Adj',
            'EndELR_Length_Adj',  # yards
            'EndELR_Critical_FurlongIDs',
        ]
        adjusted_end_mileages = pd.DataFrame(
            data=adj_end_mileages, index=locations_conn.index, columns=adj_end_columns)

        # -- Finalising ----------------------------------------------------------------------------

        adjusted_mileages = pd.concat(
            [adjusted_start_mileages, adjusted_conn_mileages, adjusted_end_mileages], axis=1)

        # adjusted_mileages.dropna(
        #     subset=['StartMileage_num_Adj', 'EndMileage_num_Adj'], inplace=True)

        # Sum up section lengths
        temp = zip(
            adjusted_mileages['StartELR_Length_Adj'],
            adjusted_mileages['ConnELR_Length_Adj'],
            adjusted_mileages['EndELR_Length_Adj'],
        )
        adjusted_mileages['Section_Length_Adj'] = [
            np.nan if np.isnan(x).all() else np.nansum(x) for x in temp]

        # Gather all furlong IDs
        furlong_id_cols = [
            'StartELR_Critical_FurlongIDs',
            'ConnELR_Critical_FurlongIDs',
            'EndELR_Critical_FurlongIDs',
        ]
        temp = adjusted_mileages[furlong_id_cols].sum(axis=1)
        adjusted_mileages['Critical_FurlongIDs'] = temp.map(lambda x: list(set(x)))

        # Keep only essential columns
        selected_columns = [
            'StartMileage_Adj',
            'EndMileage_Adj',
            'StartMileage_num_Adj',
            'EndMileage_num_Adj',
            'Section_Length_Adj',
            'Critical_FurlongIDs',
        ]
        adjusted_mileages = adjusted_mileages[selected_columns]

        return adjusted_mileages

    def adjust_mileages_for_diff_elr(self, route_name, weather_category, shift_yards, update=False,
                                     verbose=False):
        """
        Get adjusted Mileages for each incident location where StartELR != EndELR.

        :param route_name: name of a Route; if ``None``, all Routes
        :type route_name: str or None
        :param weather_category: Weather category; if ``None``, all Weather categories
        :type weather_category: str or None
        :param shift_yards: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR == EndELR
        :type shift_yards: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: adjusted Mileages for each incident location where StartELR != EndELR
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> adj_mileages = flh.adjust_mileages_for_diff_elr(
            ...     route_name='Anglia', weather_category='Wind', shift_yards=220, verbose=True)
            >>> adj_mileages.shape
            (75, 6)

            >>> adj_mileages = flh.adjust_mileages_for_diff_elr(
            ...     route_name=None, weather_category=None, shift_yards=220, verbose=True)
            >>> adj_mileages.shape
            (4668, 6)
        """

        filename = "adj_mileages_for_diff_ELRs"
        pickle_filename = make_filename(filename, route_name, weather_category, shift_yards)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            adjusted_mileages = load_data(path_to_pickle)

        else:
            if verbose:
                print("Getting adjusted Mileages for different start-end ELR pairs", end=" ... ")

            try:
                if self.METEX.db_instance is None:
                    self.METEX.db_instance = WxRailIncidentsPred(verbose=False)

                # Get data for which the 'StartELR' and 'EndELR' are DIFFERENT
                self.METEX.view_schedule8_incident_locations(
                    route_name=route_name, weather_category=weather_category, start_end_elr=False)
                incident_locations = self.METEX.schedule8_incident_locations.copy()

                # Get connecting points for different (ELRs, Mileages)
                locations_conn = self._get_connecting_elr_mileages(
                    incident_locations=incident_locations, route_name=route_name)

                if self.VEGETATION.db_instance is None:
                    self.VEGETATION.db_instance = WxRailIncidentsPred(verbose=False)

                # Get furlong information
                self.VEGETATION.view_vegetation_condition2()
                ref_furlongs = self.VEGETATION.vegetation_condition2.copy()

                # Get adjusted Mileages
                adjusted_mileages = self._adjust_mileages_for_diff_elr(
                    locations_conn=locations_conn, ref_furlongs=ref_furlongs, shift_yards=shift_yards)

                if verbose:
                    print("Done.")

                save_data(adjusted_mileages, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                adjusted_mileages = None

        return adjusted_mileages

    def get_furlongs_for_diff_elr(self, route_name=None, weather_category=None, shift_yards=220,
                                  update=False, verbose=False):
        """
        Get furlongs data for incident locations each identified by the same start and end ELRs,
        i.e. StartELR != EndELR.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param weather_category: Weather category; if ``None`` (default), all Weather categories
        :type weather_category: str or None
        :param shift_yards: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR == EndELR, defaults to ``220``
        :type shift_yards: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: furlongs data of incident locations each identified by the same start and end ELRs
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> furlongs_for_diff_elr = flh.get_furlongs_for_diff_elr(
            ...     route_name='Anglia', weather_category='Wind', shift_yards=220, verbose=True)
            >>> furlongs_for_diff_elr.shape
            (4063, 75)

            >>> furlongs_for_diff_elr = flh.get_furlongs_for_diff_elr(shift_yards=220, verbose=True)
            >>> furlongs_for_diff_elr.shape
            (64303, 75)
        """

        filename = "furlongs_for_diff_ELRs"
        pickle_filename = make_filename(filename, route_name, weather_category, shift_yards)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            furlongs_for_diff_elr = load_data(path_to_pickle)

        else:
            adjusted_mileages = self.adjust_mileages_for_diff_elr(
                route_name=route_name, weather_category=weather_category, shift_yards=shift_yards,
                verbose=verbose)

            if verbose:
                print("Getting furlong data for different start-end ELR pairs", end=" ... ")

            try:
                if self.VEGETATION.db_instance is None:
                    self.VEGETATION.db_instance = WxRailIncidentsPred(verbose=False)

                # Get furlong information
                self.VEGETATION.view_vegetation_condition2()
                ref_furlongs = self.VEGETATION.vegetation_condition2.copy()

                # Form a list containing all the furlong IDs
                furlong_ids = list(set(itertools.chain(*adjusted_mileages['Critical_FurlongIDs'])))
                # Select critical (i.e. incident) furlongs
                furlongs_for_diff_elr = ref_furlongs.loc[furlong_ids]

                if verbose:
                    print("Done.")

                save_data(furlongs_for_diff_elr, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                furlongs_for_diff_elr = None

        return furlongs_for_diff_elr

    # == Integrate data ============================================================================

    def get_furlongs_data(self, route_name=None, weather_category=None, shift_yards_same_elr=220,
                          shift_yards_diff_elr=220, update=False, verbose=False):
        """
        Get furlongs data.

        :param route_name: name of a Route; if ``None`` (default), all Routes
        :type route_name: str or None
        :param weather_category: Weather category, defaults to ``None``
        :type weather_category: str or None
        :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR == EndELR, defaults to ``220``
        :type shift_yards_same_elr: int or float
        :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
            given that StartELR != EndELR, defaults to ``220``
        :type shift_yards_diff_elr: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of furlongs for incident locations
        :rtype: pandas.DataFrame

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> furlongs_data = flh.get_furlongs_data('Anglia', 'Wind', verbose=True)
            >>> furlongs_data.shape
            (5139, 75)

            >>> furlongs_data = flh.get_furlongs_data(verbose=True)
            >>> furlongs_data.shape
            (67338, 75)
        """

        filename = "furlongs"
        pickle_filename = make_filename(
            filename, route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr)
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            furlongs_data = load_data(path_to_pickle)

        else:
            if verbose:
                print("Getting furlongs data", end=" ... ")

            try:
                # Data of incident furlongs: both start and end identified by the same ELR
                furlongs_data_same_elr = self.get_furlongs_for_same_elr(
                    route_name=route_name, weather_category=weather_category,
                    shift_yards=shift_yards_same_elr, verbose=False)

                # Data of incident furlongs: start and end are identified by different ELRs
                furlongs_data_diff_elr = self.get_furlongs_for_diff_elr(
                    route_name=route_name, weather_category=weather_category,
                    shift_yards=shift_yards_diff_elr, verbose=False)

                # Merge the above two data sets
                furlongs_data_ = pd.concat([furlongs_data_same_elr, furlongs_data_diff_elr], axis=0)
                furlongs_data = furlongs_data_.drop_duplicates().sort_index()

                if verbose:
                    print("Done.")

                save_data(furlongs_data, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                furlongs_data = None

        return furlongs_data

    def get_incident_location_furlongs(self, route_name=None, weather_category=None,
                                       shift_yards_same_elr=220, shift_yards_diff_elr=220,
                                       update=False, verbose=False):
        """
        Get data of furlongs for incident locations.

        :param route_name: name of a Route; if ``None`` (default), all available Routes
        :type route_name: str or None
        :param weather_category: Weather category;
            if ``None`` (default), all available Weather categories
        :type weather_category: str or None
        :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
            given that ``StartELR == EndELR``, defaults to ``220``
        :type shift_yards_same_elr: int or float
        :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
            given that ``StartELR != EndELR``, defaults to ``220``
        :type shift_yards_diff_elr: int or float
        :param update: whether to check on update and proceed to update the package data,
            defaults to ``False``
        :type update: bool
        :param verbose: whether to print relevant information in console as the function runs,
            defaults to ``False``
        :type verbose: bool or int
        :return: data of furlongs for incident locations
        :rtype: pandas.DataFrame or None

        **Examples**::

            >>> from src.shaft.furlong import FurlongHandler

            >>> flh = FurlongHandler()

            >>> incident_location_furlongs = flh.get_incident_location_furlongs(
            ...     route_name='Anglia', weather_category='Wind', verbose=True)
            >>> incident_location_furlongs.shape
            (258, 24)

            >>> incident_location_furlongs = flh.get_incident_location_furlongs(verbose=True)
            >>> incident_location_furlongs.shape
            (10027, 24)
        """

        filename = "incident_location_furlongs"
        pickle_filename = make_filename(
            filename, route_name, weather_category,
            f"s{shift_yards_same_elr}", f"d{shift_yards_diff_elr}")
        path_to_pickle = self.cdd(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_furlongs = load_data(path_to_pickle)

        else:
            if verbose:
                print("Getting furlong data for incident locations", end=" ... ")

            try:
                key_columns = ['Section_Length_Adj', 'Critical_FurlongIDs']

                adj_args = {
                    'route_name': route_name,
                    'weather_category': weather_category,
                    'shift_yards': shift_yards_same_elr,
                    'verbose': False,
                }
                adj_mileages_for_same_elr = \
                    self.adjust_mileages_for_same_elr(**adj_args)[key_columns]

                adj_args.update({'shift_yards': shift_yards_diff_elr})
                adj_mileages_for_diff_elr = \
                    self.adjust_mileages_for_diff_elr(**adj_args)[key_columns]

                furlongs_dat = pd.concat(
                    [adj_mileages_for_same_elr, adj_mileages_for_diff_elr], axis=0)
                furlongs_dat.sort_index(inplace=True)

                if self.METEX.db_instance is None:
                    self.METEX.db_instance = WxRailIncidentsPred(verbose=False)
                self.METEX.view_schedule8_incident_locations(
                    route_name=route_name, weather_category=weather_category, verbose=False)
                incident_locations = self.METEX.schedule8_incident_locations.copy()

                incident_location_furlongs = pd.concat([incident_locations, furlongs_dat], axis=1)

                if verbose:
                    print("Done.")

                save_data(incident_location_furlongs, path_to_file=path_to_pickle, verbose=verbose)

            except Exception as e:
                print(f"Failed. {e}.")
                incident_location_furlongs = None

        return incident_location_furlongs


# if __name__ == '__main__':
#     from src.preprocessor.METEX import METEX
#
#     flh = FurlongHandler()
#     METEX = METEX()
#
#     METEX.view_schedule8_incident_locations(route_name='Anglia', start_end_elr=False)
#     diff_start_end_elr_dat = METEX.schedule8_incident_locations.copy()  # .iloc[:10]
#     connecting_nodes = flh.get_connecting_nodes(
#         diff_start_end_elr_dat, route_name='Anglia', verbose=True)
#     assert connecting_nodes.shape == (429, 23)
#
#     METEX.view_schedule8_incident_locations(start_end_elr=False)
#     diff_start_end_elr_dat = METEX.schedule8_incident_locations.copy()  # .iloc[:10]
#     connecting_nodes = flh.get_connecting_nodes(diff_start_end_elr_dat, verbose=True)
#     assert connecting_nodes.shape == (4668, 23)
#
#     adj_mileages = flh.adjust_mileages_for_same_elr(
#         route_name='Anglia', weather_category='Wind', shift_yards=220, verbose=True)
#     assert adj_mileages.shape == (183, 6)
#
#     adj_mileages = flh.adjust_mileages_for_same_elr(
#         route_name=None, weather_category=None, shift_yards=220, verbose=True)
#     assert adj_mileages.shape == (5359, 6)
