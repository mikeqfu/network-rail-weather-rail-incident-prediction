""" Processing furlong data """

import itertools
import os

import measurement.measures
import numpy as np
import pandas as pd
from pyhelpers.store import load_pickle, save_pickle
from pyrcs.line_data import ELRMileages
from pyrcs.utils import nr_mileage_num_to_str, nr_mileage_str_to_num, shift_num_nr_mileage

from models.tools import cd_prototype_dat
from mssqlserver.metex import view_metex_schedule8_incident_locations
from mssqlserver.vegetation import view_nr_vegetation_furlong_data
from utils import cdd_railway_codes, get_subset, make_filename


# == Tools ============================================================================================

def adjust_incident_mileages(ref_furlongs, elr, start_mileage_num, end_mileage_num, shift_yards):
    """
    Get adjusted Start and End mileages.

    :param ref_furlongs: reference furlong data
    :type ref_furlongs: pandas.DataFrame
    :param elr: ELR
    :type elr: str
    :param start_mileage_num: start mileage
    :type start_mileage_num: float
    :param end_mileage_num: end mileage
    :type end_mileage_num: float
    :param shift_yards: yards by which the start/end mileage is shifted for adjustment
    :type shift_yards: int, float
    :return: adjusted mileages of incident locations and critical furlong IDs
    :rtype: tuple

    **Examples**::

        from mssqlserver.metex import view_metex_schedule8_incident_locations
        from mssqlserver.vegetation import view_nr_vegetation_furlong_data
        from models.prototype.furlong import adjust_incident_mileages

        ref_furlongs                = view_nr_vegetation_furlong_data()
        incident_locations_same_elr = view_metex_schedule8_incident_locations(start_and_end_elr='same')
        shift_yards_same_elr        = 220
        shift_yards_diff_elr        = 220

        elr               = incident_locations_same_elr.StartELR.iloc[109]
        start_mileage_num = incident_locations_same_elr.StartMileage_num.iloc[109]
        end_mileage_num   = incident_locations_same_elr.EndMileage_num.iloc[109]
        shift_yards       = shift_yards_same_elr

        adjusted_incident_mileages = adjust_incident_mileages(ref_furlongs, elr, start_mileage_num,
                                                              end_mileage_num, shift_yards)
        print(adjusted_incident_mileages)
        # ('0.0000', '0.0440', 0.0, 0.044, 77.44000000000001, [10026, 54275])

        elr               = incident_locations_same_elr.StartELR.iloc[0]
        start_mileage_num = incident_locations_same_elr.StartMileage_num.iloc[0]
        end_mileage_num   = incident_locations_same_elr.EndMileage_num.iloc[0]
        shift_yards       = shift_yards_same_elr

        adjusted_incident_mileages = adjust_incident_mileages(ref_furlongs, elr, start_mileage_num,
                                                              end_mileage_num, shift_yards)
        print(adjusted_incident_mileages)
        # ('10.1100', '11.0000', 10.11, 11.0, 1566.400000000001, [25698, 16397, 28861])

        elr               = incident_locations_same_elr.StartELR.iloc[4]
        start_mileage_num = incident_locations_same_elr.StartMileage_num.iloc[4]
        end_mileage_num   = incident_locations_same_elr.EndMileage_num.iloc[4]
        shift_yards       = shift_yards_same_elr

        adjusted_incident_mileages = adjust_incident_mileages(ref_furlongs, elr, start_mileage_num,
                                                              end_mileage_num, shift_yards)
        print(adjusted_incident_mileages)
        # ('74.0440', '68.0880', 74.044, 68.088, 10482.560000000007, [43654, ..., 44668])

        elr               = incident_locations_same_elr.StartELR.iloc[-2]
        start_mileage_num = incident_locations_same_elr.StartMileage_num.iloc[-2]
        end_mileage_num   = incident_locations_same_elr.EndMileage_num.iloc[-2]
        shift_yards       = shift_yards_same_elr

        adjusted_incident_mileages = adjust_incident_mileages(ref_furlongs, elr, start_mileage_num,
                                                              end_mileage_num, shift_yards)
        print(adjusted_incident_mileages)
        # ('17.0000', '21.1540', 17.0, 21.154, 7311.04, [63503, ..., 63480])
    """

    nr_elr_furlongs = ref_furlongs[ref_furlongs.ELR == elr]

    try:
        # Merge the mileages (num) of both start and end
        elr_mileages = nr_elr_furlongs.StartMileage_num.append(nr_elr_furlongs.EndMileage_num)
        elr_mileages.drop_duplicates(keep='first', inplace=True)
        elr_mileages.sort_values(inplace=True)

        m_indices = pd.Index(elr_mileages)
        s_indices = pd.Index(nr_elr_furlongs.StartMileage)
        e_indices = pd.Index(nr_elr_furlongs.EndMileage)

        if start_mileage_num <= end_mileage_num:

            if start_mileage_num == end_mileage_num:
                start_mileage_num = shift_num_nr_mileage(start_mileage_num, -shift_yards)
                end_mileage_num = shift_num_nr_mileage(end_mileage_num, shift_yards)

            # Get adjusted mileages and 'FurlongID' for the start location
            try:
                adjusted_start_mileage_num = elr_mileages.iloc[m_indices.get_loc(start_mileage_num, 'ffill')]
            except (ValueError, KeyError):
                adjusted_start_mileage_num = elr_mileages.iloc[m_indices.get_loc(start_mileage_num, 'nearest')]

            try:
                s_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
            except (ValueError, KeyError):
                s_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
                adjusted_start_mileage_num = nr_mileage_str_to_num(nr_elr_furlongs.StartMileage.iloc[s_idx])

            # Get adjusted mileages and 'FurlongID' for the start location
            try:
                adjusted_end_mileage_num = elr_mileages.iloc[m_indices.get_loc(end_mileage_num, 'bfill')]
            except (ValueError, KeyError):
                adjusted_end_mileage_num = elr_mileages.iloc[m_indices.get_loc(end_mileage_num, 'nearest')]

            try:
                e_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
            except (ValueError, KeyError):
                e_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
                adjusted_end_mileage_num = nr_mileage_str_to_num(nr_elr_furlongs.EndMileage.iloc[e_idx])

        else:  # start_mileage_num > end_mileage_num
            # Get adjusted mileages of start and end locations
            try:
                adjusted_start_mileage_num = elr_mileages.iloc[m_indices.get_loc(start_mileage_num, 'bfill')]
            except (ValueError, KeyError):
                adjusted_start_mileage_num = elr_mileages.iloc[m_indices.get_loc(start_mileage_num, 'nearest')]
            try:
                adjusted_end_mileage_num = elr_mileages.iloc[m_indices.get_loc(end_mileage_num, 'ffill')]
            except (ValueError, KeyError):
                adjusted_end_mileage_num = elr_mileages.iloc[m_indices.get_loc(end_mileage_num, 'nearest')]

            # Get 'FurlongID's
            try:
                s_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
            except (ValueError, KeyError):
                s_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
                adjusted_start_mileage_num = nr_mileage_str_to_num(nr_elr_furlongs.EndMileage.iloc[s_idx])
            try:
                e_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
            except (ValueError, KeyError):
                e_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
                adjusted_end_mileage_num = nr_mileage_str_to_num(nr_elr_furlongs.StartMileage.iloc[e_idx])

        if s_idx <= e_idx:
            e_idx = e_idx + 1 if e_idx < len(elr_mileages) else e_idx
            nr_elr_furlongs_dat = nr_elr_furlongs.iloc[s_idx:e_idx]
        else:  # s_idx > e_idx
            s_idx = s_idx + 1 if s_idx < len(elr_mileages) else s_idx
            nr_elr_furlongs_dat = nr_elr_furlongs.iloc[e_idx:s_idx]
        critical_furlong_id = list(set(nr_elr_furlongs_dat.index))

        adjusted_start_mileage = nr_mileage_num_to_str(adjusted_start_mileage_num)
        adjusted_end_mileage = nr_mileage_num_to_str(adjusted_end_mileage_num)
        distance = measurement.measures.Distance(mile=np.abs(adjusted_end_mileage_num - adjusted_start_mileage_num)).yd

    except IndexError:
        adjusted_start_mileage, adjusted_end_mileage = '', ''
        adjusted_start_mileage_num, adjusted_end_mileage_num, distance = np.nan, np.nan, np.nan
        critical_furlong_id = []

    adjusted_incident_mileages = (adjusted_start_mileage, adjusted_end_mileage, adjusted_start_mileage_num,
                                  adjusted_end_mileage_num, distance, critical_furlong_id)

    return adjusted_incident_mileages


def get_connecting_nodes(diff_start_end_elr_dat, route_name=None, update=False, verbose=False):
    """
    Get data of connecting points for different ELRs.

    :param diff_start_end_elr_dat: data frame where StartELR != EndELR
    :type diff_start_end_elr_dat: pandas.DataFrame
    :param route_name: name of a Route; if ``None`` (default), all Routes
    :type route_name: str, None
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of connecting points for different ELRs
    :rtype: pandas.DataFrame

    **Examples**::

        from mssqlserver.metex import view_metex_schedule8_incident_locations
        from models.prototype.furlong import get_connecting_nodes

        update = False
        verbose = True

        route_name = None
        diff_start_end_elr_dat = view_metex_schedule8_incident_locations(
            route_name=route_name, start_and_end_elr='diff', verbose=verbose)
        connecting_nodes = get_connecting_nodes(diff_start_end_elr_dat, route_name, update, verbose)
        print(connecting_nodes)

        route_name = 'Anglia'
        diff_start_end_elr_dat = view_metex_schedule8_incident_locations(
            route_name=route_name, start_and_end_elr='diff', verbose=verbose)
        connecting_nodes = get_connecting_nodes(diff_start_end_elr_dat, route_name, update, verbose)
        print(connecting_nodes)
    """

    filename = "connections-between-different-ELRs"
    pickle_filename = make_filename(filename, route_name)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        try:
            pickle_filename_temp = make_filename(filename)
            path_to_pickle_temp = cd_prototype_dat(pickle_filename_temp)

            if os.path.isfile(path_to_pickle_temp) and not update:
                connecting_nodes_all = load_pickle(path_to_pickle_temp)
                connecting_nodes = get_subset(connecting_nodes_all, route_name)

            else:
                diff_elr_mileages = diff_start_end_elr_dat.drop_duplicates()

                em = ELRMileages()
                print("Searching for connecting ELRs ... ", end="") if verbose else ""
                mileage_file_dir = cdd_railway_codes("line data\\elrs-and-mileages\\mileages")
                conn_mileages = diff_elr_mileages.apply(
                    lambda x: em.get_conn_mileages(x.StartELR, x.EndELR, update, pickle_mileage_file=True,
                                                   data_dir=mileage_file_dir), axis=1)

                print("\nFinished.") if verbose else ""

                conn_mileages_data = pd.DataFrame(conn_mileages.to_list(), index=diff_elr_mileages.index,
                                                  columns=['StartELR_EndMileage', 'ConnELR', 'ConnELR_StartMileage',
                                                           'ConnELR_EndMileage', 'EndELR_StartMileage'])

                connecting_nodes = diff_elr_mileages.join(conn_mileages_data)
                connecting_nodes.set_index(['StartELR', 'StartMileage', 'EndELR', 'EndMileage'], inplace=True)

            save_pickle(connecting_nodes, path_to_pickle, verbose=verbose)

            return connecting_nodes

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# == Furlongs of incident locations ===================================================================

def get_adjusted_mileages_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr, update=False,
                                              verbose=False):
    """
    Get adjusted mileages for each incident location where StartELR == EndELR.

    :param route_name: name of a Route; if ``None``, all Routes
    :type route_name: str, None
    :param weather_category: weather category; if ``None``, all weather categories
    :type weather_category: str, None
    :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR
    :type shift_yards_same_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: adjusted mileages for each incident location where StartELR == EndELR
    :rtype: pandas.DataFrame

    **Example**::

        from models.prototype.furlong import get_adjusted_mileages_same_start_end_elrs

        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        update               = True
        verbose              = True

        adj_mileages = get_adjusted_mileages_same_start_end_elrs(route_name, weather_category,
                                                                 shift_yards_same_elr, update, verbose)
        print(adj_mileages)
    """

    filename = "adjusted-mileages-same-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        adj_mileages = load_pickle(path_to_pickle)
        return adj_mileages

    else:
        try:
            # Get data about for which the 'StartELR' and 'EndELR' are THE SAME
            incident_locations = view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr='same', verbose=verbose)

            # Get furlong information as reference
            ref_furlongs = view_nr_vegetation_furlong_data(verbose=verbose)

            # Calculate adjusted furlong locations for each incident (for extracting vegetation conditions)
            adjusted_mileages = incident_locations.apply(
                lambda x: adjust_incident_mileages(ref_furlongs, x.StartELR, x.StartMileage_num, x.EndMileage_num,
                                                   shift_yards_same_elr), axis=1)

            # Get adjusted mileage data
            adj_mileages = pd.DataFrame(list(adjusted_mileages), index=incident_locations.index,
                                        columns=['StartMileage_Adj', 'EndMileage_Adj',
                                                 'StartMileage_num_Adj', 'EndMileage_num_Adj',
                                                 'Section_Length_Adj',  # yards
                                                 'Critical_FurlongIDs'])

            save_pickle(adj_mileages, path_to_pickle, verbose=verbose)

            return adj_mileages

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


def get_furlongs_same_start_end_elrs(route_name=None, weather_category=None, shift_yards_same_elr=220,
                                     update=False, verbose=False):
    """
    Get furlongs data for incident locations each identified by the same start and end ELRs, i.e. StartELR == EndELR.

    :param route_name: name of a Route; if ``None`` (default), all Routes
    :type route_name: str, None
    :param weather_category: weather category; if ``None`` (default), all weather categories
    :type weather_category: str, None
    :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR, defaults to ``220``
    :type shift_yards_same_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: furlongs data of incident locations each identified by the same start and end ELRs
    :rtype: pandas.DataFrame

    **Examples**::

        from models.prototype.furlong import get_furlongs_same_start_end_elrs

        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        update               = True
        verbose              = True

        furlongs_same_start_end_elr = get_furlongs_same_start_end_elrs(
            route_name, weather_category, shift_yards_same_elr, update, verbose)
        print(furlongs_same_start_end_elr)
    """

    filename = "furlongs-same-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        furlongs_same_start_end_elr = load_pickle(path_to_pickle)
        return furlongs_same_start_end_elr

    else:
        adj_mileages = get_adjusted_mileages_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr,
                                                                 verbose=verbose)

        try:
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)
            # Form a list containing all the furlong IDs
            furlong_ids = list(set(itertools.chain(*adj_mileages.Critical_FurlongIDs)))
            # Select critical (i.e. incident) furlongs
            furlongs_same_start_end_elr = nr_furlong_data.loc[furlong_ids]

            # Save 'incident_furlongs_same_start_end_elr'
            save_pickle(furlongs_same_start_end_elr, path_to_pickle, verbose=verbose)

            return furlongs_same_start_end_elr

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


def get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr, update=False,
                                              verbose=False):
    """
    Get adjusted mileages for each incident location where StartELR != EndELR.

    :param route_name: name of a Route; if ``None``, all Routes
    :type route_name: str, None
    :param weather_category: weather category; if ``None``, all weather categories
    :type weather_category: str, None
    :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR
    :type shift_yards_diff_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: adjusted mileages for each incident location where StartELR != EndELR
    :rtype: pandas.DataFrame

    **Example**::

        from models.prototype.furlong import get_adjusted_mileages_diff_start_end_elrs

        route_name           = None
        weather_category     = None
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        adj_mileages = get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category,
                                                                 shift_yards_diff_elr, update, verbose)
        print(adj_mileages)
    """

    filename = "adjusted-mileages-diff-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            # Get data for which the 'StartELR' and 'EndELR' are DIFFERENT
            incident_locations_diff_start_end_elr = view_metex_schedule8_incident_locations(
                route_name, weather_category, start_and_end_elr='diff', verbose=verbose)
            # Get connecting points for different (ELRs, mileages)
            connecting_nodes = get_connecting_nodes(incident_locations_diff_start_end_elr,
                                                    route_name, update=False, verbose=False)

            # Find End Mileage and Start Mileage of StartELR and EndELR, respectively
            locations_conn = incident_locations_diff_start_end_elr.join(
                connecting_nodes.set_index(['StanoxSection'], append=True),
                on=list(connecting_nodes.index.names) + ['StanoxSection'], rsuffix='_conn').dropna()
            locations_conn.drop(columns=[x for x in locations_conn.columns if '_conn' in x], inplace=True)
            # Remove the data records where connecting nodes are unknown
            locations_conn = locations_conn[~((locations_conn.StartELR_EndMileage == '') |
                                              (locations_conn.EndELR_StartMileage == ''))]
            # Convert str mileages to num
            num_conn_colnames = ['StartELR_EndMileage_num', 'EndELR_StartMileage_num',
                                 'ConnELR_StartMileage_num', 'ConnELR_EndMileage_num']
            str_conn_colnames = ['StartELR_EndMileage', 'EndELR_StartMileage',
                                 'ConnELR_StartMileage', 'ConnELR_EndMileage']
            locations_conn[num_conn_colnames] = locations_conn[str_conn_colnames].applymap(nr_mileage_str_to_num)

            # Get furlong information
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)

            adjusted_conn_elr_mileages = locations_conn.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.ConnELR, x.ConnELR_StartMileage_num,
                                                   x.ConnELR_EndMileage_num, 0)
                if x.ConnELR != '' else tuple([''] * 2 + [np.nan] * 2 + [0.0, []]), axis=1)
            adjusted_conn_mileages = pd.DataFrame(adjusted_conn_elr_mileages.tolist(), index=locations_conn.index,
                                                  columns=['Conn_StartMileage_Adj', 'ConnELR_EndMileage_Adj',
                                                           'Conn_StartMileage_num_Adj', 'ConnELR_EndMileage_num_Adj',
                                                           'ConnELR_Length_Adj',  # yards
                                                           'ConnELR_Critical_FurlongIDs'])

            # Processing Start locations
            adjusted_start_elr_mileages = locations_conn.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.StartELR, x.StartMileage_num,
                                                   x.StartELR_EndMileage_num, shift_yards_diff_elr), axis=1)

            # Create a dataframe adjusted mileage data of the Start ELRs
            adjusted_start_mileages = pd.DataFrame(adjusted_start_elr_mileages.tolist(),
                                                   index=locations_conn.index,
                                                   columns=['StartMileage_Adj', 'StartELR_EndMileage_Adj',
                                                            'StartMileage_num_Adj', 'StartELR_EndMileage_num_Adj',
                                                            'StartELR_Length_Adj',  # yards
                                                            'StartELR_Critical_FurlongIDs'])

            # Processing End locations
            adjusted_end_elr_mileages = locations_conn.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.EndELR, x.EndELR_StartMileage_num,
                                                   x.EndMileage_num, shift_yards_diff_elr),
                axis=1)

            # Create a dataframe of adjusted mileage data of the EndELRs
            adjusted_end_mileages = pd.DataFrame(adjusted_end_elr_mileages.tolist(), index=locations_conn.index,
                                                 columns=['EndELR_StartMileage_Adj', 'EndMileage_Adj',
                                                          'EndELR_StartMileage_num_Adj', 'EndMileage_num_Adj',
                                                          'EndELR_Length_Adj',  # yards
                                                          'EndELR_Critical_FurlongIDs'])

            # Combine 'adjusted_start_mileages' and 'adjusted_end_mileages'
            adj_mileages = adjusted_start_mileages.join(adjusted_conn_mileages).join(adjusted_end_mileages)

            adj_mileages.dropna(subset=['StartMileage_num_Adj', 'EndMileage_num_Adj'], inplace=True)

            adj_mileages['Section_Length_Adj'] = list(zip(
                adj_mileages.StartELR_Length_Adj, adj_mileages.ConnELR_Length_Adj, adj_mileages.EndELR_Length_Adj))

            adj_mileages['Critical_FurlongIDs'] = \
                adj_mileages.StartELR_Critical_FurlongIDs + \
                adj_mileages.EndELR_Critical_FurlongIDs + \
                adj_mileages.ConnELR_Critical_FurlongIDs
            adj_mileages.Critical_FurlongIDs = adj_mileages.Critical_FurlongIDs.map(lambda x: list(set(x)))

            save_pickle(adj_mileages, path_to_pickle, verbose=update)

            return adj_mileages

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


def get_furlongs_diff_start_end_elrs(route_name=None, weather_category=None, shift_yards_diff_elr=220,
                                     update=False, verbose=False):
    """
    Get furlongs data for incident locations each identified by the same start and end ELRs, i.e. StartELR != EndELR.

    :param route_name: name of a Route; if ``None`` (default), all Routes
    :type route_name: str, None
    :param weather_category: weather category; if ``None`` (default), all weather categories
    :type weather_category: str, None
    :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR, defaults to ``220``
    :type shift_yards_diff_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: furlongs data of incident locations each identified by the same start and end ELRs
    :rtype: pandas.DataFrame

    **Examples**::

        from models.prototype.furlong import get_furlongs_diff_start_end_elrs

        route_name           = None
        weather_category     = None
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        furlongs_diff_start_end_elr = get_furlongs_diff_start_end_elrs(
            route_name, weather_category, shift_yards_diff_elr, update, verbose)
        print(furlongs_diff_start_end_elr)
    """

    filename = "furlongs-diff-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        furlongs_diff_start_end_elr = load_pickle(path_to_pickle)
        return furlongs_diff_start_end_elr

    else:
        adj_mileages = get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr,
                                                                 verbose=verbose)

        try:
            # Get furlong information
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)
            # Form a list containing all the furlong IDs
            furlong_ids = list(set(itertools.chain(*adj_mileages.Critical_FurlongIDs)))

            # Select critical (i.e. incident) furlongs
            furlongs_diff_start_end_elr = nr_furlong_data.loc[furlong_ids]

            # Save 'incident_furlongs_diff_start_end_elr'
            save_pickle(furlongs_diff_start_end_elr, path_to_pickle, verbose=verbose)

            return furlongs_diff_start_end_elr

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# == Integrate data ===================================================================================

def get_furlongs_data(route_name=None, weather_category=None, shift_yards_same_elr=220, shift_yards_diff_elr=220,
                      update=False, verbose=False) -> pd.DataFrame:
    """
    Get furlongs data.

    :param route_name: name of a Route; if ``None`` (default), all Routes
    :type route_name: str, None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str, None
    :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR, defaults to ``220``
    :type shift_yards_same_elr: int, float
    :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR != EndELR, defaults to ``220``
    :type shift_yards_diff_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of furlongs for incident locations
    :rtype: pandas.DataFrame

    **Example**::

        from models.prototype.furlong import get_furlongs_data

        weather_category     = None
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        route_name = None
        furlongs_data = get_furlongs_data(route_name, weather_category, shift_yards_same_elr,
                                          shift_yards_diff_elr, update, verbose)
        print(furlongs_data)

        route_name = 'Anglia'
        furlongs_data = get_furlongs_data(route_name, weather_category, shift_yards_same_elr,
                                          shift_yards_diff_elr, update, verbose)
        print(furlongs_data)
    """

    filename = "furlongs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        furlongs_data = load_pickle(path_to_pickle)
        return furlongs_data

    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            furlongs_data_same_elr = get_furlongs_same_start_end_elrs(
                route_name, weather_category, shift_yards_same_elr, verbose=verbose)

            # Data of incident furlongs: start and end are identified by different ELRs
            furlongs_data_diff_elr = get_furlongs_diff_start_end_elrs(
                route_name, weather_category, shift_yards_diff_elr, verbose=verbose)

            # Merge the above two data sets
            furlongs_data = furlongs_data_same_elr.append(furlongs_data_diff_elr)
            furlongs_data.drop_duplicates(['AssetNumber', 'StructuredPlantNumber'], inplace=True)
            furlongs_data.sort_index(inplace=True)

            save_pickle(furlongs_data, path_to_pickle, verbose=verbose)

            return furlongs_data

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


def get_incident_location_furlongs(route_name=None, weather_category=None, shift_yards_same_elr=220,
                                   shift_yards_diff_elr=220, update=False, verbose=False) -> pd.DataFrame:
    """
    Get data of furlongs for incident locations.

    :param route_name: name of a Route; if ``None`` (default), all Routes
    :type route_name: str, None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str, None
    :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR, defaults to ``220``
    :type shift_yards_same_elr: int, float
    :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR != EndELR, defaults to ``220``
    :type shift_yards_diff_elr: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return: data of furlongs for incident locations
    :rtype: pandas.DataFrame

    **Example**::

        from models.prototype.furlong import get_incident_location_furlongs

        weather_category     = None
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        route_name = None
        incident_location_furlongs = get_incident_location_furlongs(route_name, weather_category,
                                                                    shift_yards_same_elr, shift_yards_diff_elr,
                                                                    update, verbose)
        print(incident_location_furlongs)

        route_name = 'Anglia'
        incident_location_furlongs = get_incident_location_furlongs(route_name, weather_category,
                                                                    shift_yards_same_elr, shift_yards_diff_elr,
                                                                    update, verbose)
        print(incident_location_furlongs)
    """

    filename = "incident-location-furlongs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_location_furlongs = load_pickle(path_to_pickle)
        return incident_location_furlongs

    else:
        try:
            adjusted_mileages_same_start_end_elrs = get_adjusted_mileages_same_start_end_elrs(
                route_name, weather_category, shift_yards_same_elr, verbose=verbose)
            ilf_same = adjusted_mileages_same_start_end_elrs[['Section_Length_Adj', 'Critical_FurlongIDs']]

            adjusted_mileages_diff_start_end_elrs = get_adjusted_mileages_diff_start_end_elrs(
                route_name, weather_category, shift_yards_diff_elr, verbose=verbose)
            ilf_diff = adjusted_mileages_diff_start_end_elrs[['Section_Length_Adj', 'Critical_FurlongIDs']]

            incident_locations = view_metex_schedule8_incident_locations(route_name, weather_category, verbose=verbose)

            # Merge the above data sets
            incident_location_furlongs = incident_locations.join(pd.concat([ilf_same, ilf_diff]), how='right')
            incident_location_furlongs.drop(['StartMileage_num', 'EndMileage_num'], axis=1, inplace=True)
            incident_location_furlongs.index = range(len(incident_location_furlongs))

            save_pickle(incident_location_furlongs, path_to_pickle, verbose=verbose)

            return incident_location_furlongs

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))
