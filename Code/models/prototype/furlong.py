""" Processing furlong data """

import itertools
import os

import measurement.measures
import numpy as np
import pandas as pd
from pyhelpers.store import load_pickle, save_pickle
from pyrcs.line_data_cls import elrs_mileages
from pyrcs.utils import nr_mileage_num_to_str, nr_mileage_to_yards, str_to_num_mileage, yards_to_nr_mileage

from models.tools import cd_prototype_dat
from mssqlserver.metex import fetch_incident_locations_from_nr_metex, get_subset, make_filename
from mssqlserver.vegetation import view_nr_vegetation_furlong_data
from utils import cdd_network


#
def merge_start_and_end(start_array, end_array):
    end_array = np.array(end_array)
    assert len(start_array) == len(end_array)
    return pd.Series(np.append(start_array, end_array[-1]))


#
def num_mileage_shifting(mileage, y):
    yards = nr_mileage_to_yards(mileage) + y
    str_mileage = yards_to_nr_mileage(yards)
    return str_to_num_mileage(str_mileage)


# Get adjusted Start and End mileages
def adjust_incident_mileages(nr_furlong_data, elr, start_mileage_num, end_mileage_num, shift_yards) -> tuple:
    """
    :param nr_furlong_data: [pd.DataFrame]
    :param elr: [str]
    :param start_mileage_num: [float]
    :param end_mileage_num: [float]
    :param shift_yards: [numeric]
    :return: [tuple] (Adjusted start mileage [str], Adjusted end mileage [str],
                      Adjusted start mileage [numeric], adjusted end mileage [numeric],
                      Distance [numbers.Number], Furlong ID [list])

    Testing e.g.
        from mssqlserver.metex import fetch_incident_locations_from_nr_metex
        from mssqlserver.vegetation import view_nr_vegetation_furlong_data

        nr_furlong_data             = view_nr_vegetation_furlong_data()
        incident_locations_same_elr = fetch_incident_locations_from_nr_metex(start_and_end_elr='same')
        shift_yards_same_elr        = 220
        shift_yards_diff_elr        = 220

        elr               = incident_locations_same_elr.StartELR.iloc[0]
        start_mileage_num = incident_locations_same_elr.StartMileage_num.iloc[0]
        end_mileage_num   = incident_locations_same_elr.EndMileage_num.iloc[0]
        shift_yards       = shift_yards_same_elr

        adjust_incident_mileages(nr_furlong_data, elr, start_mileage_num, end_mileage_num, shift_yards)

        elr                 = locations_conn.EndELR[5083]
        start_mileage_num   = locations_conn.EndELR_StartMileage_num[5083]
        end_mileage_num     = locations_conn.EndMileage_num[5083]
        shift_yards         = shift_yards_diff_elr

        adjust_incident_mileages(nr_furlong_data, elr, start_mileage_num, end_mileage_num, shift_yards)

    """
    nr_elr_furlongs = nr_furlong_data[nr_furlong_data.ELR == elr]

    try:
        elr_mileages = merge_start_and_end(nr_elr_furlongs.StartMileage_num, nr_elr_furlongs.EndMileage_num)
    except IndexError:
        return '', '', np.nan, np.nan, np.nan, []

    m_indices = pd.Index(elr_mileages)
    s_indices = pd.Index(nr_elr_furlongs.StartMileage)
    e_indices = pd.Index(nr_elr_furlongs.EndMileage)

    if start_mileage_num <= end_mileage_num:

        if start_mileage_num == end_mileage_num:
            start_mileage_num = num_mileage_shifting(start_mileage_num, -shift_yards)
            end_mileage_num = num_mileage_shifting(end_mileage_num, shift_yards)

        # Get adjusted mileages of start and end locations
        try:
            adjusted_start_mileage_num = elr_mileages[m_indices.get_loc(start_mileage_num, 'ffill')]
        except (ValueError, KeyError):
            adjusted_start_mileage_num = elr_mileages[m_indices.get_loc(start_mileage_num, 'nearest')]

        try:
            adjusted_end_mileage_num = elr_mileages[m_indices.get_loc(end_mileage_num, 'bfill')]
        except (ValueError, KeyError):
            adjusted_end_mileage_num = elr_mileages[m_indices.get_loc(end_mileage_num, 'nearest')]

        # Get 'FurlongID's
        try:
            s_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
        except (ValueError, KeyError):
            s_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
            adjusted_start_mileage_num = str_to_num_mileage(nr_elr_furlongs.StartMileage.iloc[s_idx])

        try:
            e_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
        except (ValueError, KeyError):
            e_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
            adjusted_end_mileage_num = str_to_num_mileage(nr_elr_furlongs.EndMileage.iloc[e_idx])

    else:  # start_mileage > end_mileage:
        # Get adjusted mileages of start and end locations
        try:
            adjusted_start_mileage_num = elr_mileages[m_indices.get_loc(start_mileage_num, 'bfill')]
        except (ValueError, KeyError):
            adjusted_start_mileage_num = elr_mileages[m_indices.get_loc(start_mileage_num, 'nearest')]
        try:
            adjusted_end_mileage_num = elr_mileages[m_indices.get_loc(end_mileage_num, 'ffill')]
        except (ValueError, KeyError):
            adjusted_end_mileage_num = elr_mileages[m_indices.get_loc(end_mileage_num, 'nearest')]

        # Get 'FurlongID's
        try:
            s_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
        except (ValueError, KeyError):
            s_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_start_mileage_num))
            adjusted_start_mileage_num = str_to_num_mileage(nr_elr_furlongs.EndMileage.iloc[s_idx])
        try:
            e_idx = s_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
        except (ValueError, KeyError):
            e_idx = e_indices.get_loc(nr_mileage_num_to_str(adjusted_end_mileage_num))
            adjusted_end_mileage_num = str_to_num_mileage(nr_elr_furlongs.StartMileage.iloc[e_idx])

    if s_idx <= e_idx:
        e_idx = e_idx + 1 if e_idx < len(elr_mileages) else e_idx
        nr_elr_furlongs_dat = nr_elr_furlongs.iloc[s_idx:e_idx]
    else:  # s_idx > e_idx
        s_idx = s_idx + 1 if s_idx < len(elr_mileages) else s_idx
        nr_elr_furlongs_dat = nr_elr_furlongs.iloc[e_idx:s_idx]
    critical_furlong_id = nr_elr_furlongs_dat.index.to_list()

    adjusted_start_mileage = nr_mileage_num_to_str(adjusted_start_mileage_num)
    adjusted_end_mileage = nr_mileage_num_to_str(adjusted_end_mileage_num)
    distance = measurement.measures.Distance(mile=np.abs(adjusted_end_mileage_num - adjusted_start_mileage_num)).yd

    return \
        adjusted_start_mileage, adjusted_end_mileage, \
        adjusted_start_mileage_num, adjusted_end_mileage_num, \
        distance, critical_furlong_id


# Get information of connecting points for different ELRs
def get_connecting_nodes(route_name=None, update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name = None
        update = False
        verbose = True

        get_connecting_nodes(route_name, update, verbose)
        get_connecting_nodes('Anglia', update, verbose)
    """
    filename = "connections-between-different-ELRs"
    pickle_filename = make_filename(filename, route_name)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        try:
            path_to_pickle_temp = cd_prototype_dat(make_filename(filename))

            if os.path.isfile(path_to_pickle_temp) and not update:
                connecting_nodes_all = load_pickle(path_to_pickle_temp, verbose=verbose)
                connecting_nodes = get_subset(connecting_nodes_all, route_name)

            else:
                diff_start_end_elr = fetch_incident_locations_from_nr_metex(route_name, start_and_end_elr='diff',
                                                                            verbose=verbose)

                diff_elr_mileages = diff_start_end_elr.drop_duplicates()

                em_cls = elrs_mileages.ELRMileages()
                print("Searching for connecting ELRs ... ")
                mileage_file_dir = cdd_network("Railway Codes\\Line data\\ELRs and mileages\\Mileage files")
                conn_mileages = diff_elr_mileages.apply(
                    lambda x: em_cls.get_conn_mileages(x.StartELR, x.EndELR, update, pickle_mileage_file=True,
                                                       data_dir=mileage_file_dir),
                    axis=1)
                print("\nFinished.")

                conn_mileages_data = pd.DataFrame(conn_mileages.to_list(), index=diff_elr_mileages.index,
                                                  columns=['StartELR_EndMileage', 'ConnELR', 'ConnELR_StartMileage',
                                                           'ConnELR_EndMileage', 'EndELR_StartMileage'])

                connecting_nodes = diff_elr_mileages.join(conn_mileages_data)
                connecting_nodes.set_index(['StartELR', 'StartMileage', 'EndELR', 'EndMileage'], inplace=True)

            save_pickle(connecting_nodes, path_to_pickle, verbose=verbose)

            return connecting_nodes

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get adjusted mileages for each incident location (where StartELR == EndELR)
def get_adjusted_mileages_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr, update=False,
                                              verbose=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param shift_yards_same_elr: [numbers.Number]
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        update               = True
        verbose              = True

        get_adjusted_mileages_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr, update, verbose)
    """
    filename = "adjusted-incident-mileages-same-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        adjusted_incident_mileages = load_pickle(path_to_pickle, verbose=verbose)
        return adjusted_incident_mileages

    else:
        try:
            # Get data about for which the 'StartELR' and 'EndELR' are THE SAME
            incident_locations_same_start_end_elr = fetch_incident_locations_from_nr_metex(
                route_name, weather_category, start_and_end_elr='same', verbose=verbose)

            # Get furlong information
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)

            # Calculate adjusted furlong locations for each incident (for extracting vegetation conditions)
            adjusted_mileages = incident_locations_same_start_end_elr.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.StartELR, x.StartMileage_num, x.EndMileage_num,
                                                   shift_yards_same_elr),
                axis=1)

            # Get adjusted mileage data
            adjusted_incident_mileages = pd.DataFrame(list(adjusted_mileages),
                                                      index=incident_locations_same_start_end_elr.index,
                                                      columns=['StartMileage_Adj', 'EndMileage_Adj',
                                                               'StartMileage_num_Adj', 'EndMileage_num_Adj',
                                                               'Section_Length_Adj',  # yards
                                                               'Critical_FurlongIDs'])

            save_pickle(adjusted_incident_mileages, path_to_pickle, verbose=verbose)

            return adjusted_incident_mileages

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get furlongs data of incident locations each identified by the same ELRs (StartELR == EndELR)
def get_incident_furlongs_same_start_end_elrs(route_name=None, weather_category=None, shift_yards_same_elr=220,
                                              update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param shift_yards_same_elr: [numbers.Number] (default: 220)
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        update               = True
        verbose              = True

        get_incident_furlongs_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr, update, verbose)
    """
    filename = "incident-furlongs-same-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        adj_mileages = get_adjusted_mileages_same_start_end_elrs(route_name, weather_category, shift_yards_same_elr,
                                                                 verbose=verbose)

        try:
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)
            # Form a list containing all the furlong IDs
            veg_furlongs_idx = list(set(itertools.chain(*adj_mileages.Critical_FurlongIDs)))
            # Select critical (i.e. incident) furlongs
            incident_furlongs_same_start_end_elr = nr_furlong_data.loc[veg_furlongs_idx]

            # Save 'incident_furlongs_same_start_end_elr'
            save_pickle(incident_furlongs_same_start_end_elr, path_to_pickle, verbose=verbose)

            return incident_furlongs_same_start_end_elr

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get adjusted mileages for each incident location (where StartELR != EndELR)
def get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr, update=False,
                                              verbose=False):
    """
    :param route_name: [str; None]
    :param weather_category: [str; None]
    :param shift_yards_diff_elr: [numbers.Number]
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr, update, verbose)
    """
    filename = "adjusted-incident-mileages-diff-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        try:
            # Get data for which the 'StartELR' and 'EndELR' are DIFFERENT
            incident_locations_diff_start_end_elr = fetch_incident_locations_from_nr_metex(
                route_name, weather_category, start_and_end_elr='diff', verbose=verbose)
            # Get connecting points for different (ELRs, mileages)
            connecting_nodes = get_connecting_nodes(route_name, update=False, verbose=False)

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
            locations_conn[num_conn_colnames] = locations_conn[str_conn_colnames].applymap(str_to_num_mileage)

            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)  # Get furlong information

            adjusted_conn_elr_mileages = locations_conn.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.ConnELR, x.ConnELR_StartMileage_num,
                                                   x.ConnELR_EndMileage_num, 0)
                if x.ConnELR != '' else tuple([''] * 2 + [np.nan] * 2 + [0.0, []]), axis=1)
            adjusted_conn_mileages = pd.DataFrame(adjusted_conn_elr_mileages.tolist(),
                                                  index=locations_conn.index,
                                                  columns=['Conn_StartMileage_Adj', 'ConnELR_EndMileage_Adj',
                                                           'Conn_StartMileage_num_Adj', 'ConnELR_EndMileage_num_Adj',
                                                           'ConnELR_Length_Adj',  # yards
                                                           'ConnELR_Critical_FurlongIDs'])

            # Processing Start locations
            adjusted_start_elr_mileages = locations_conn.apply(
                lambda x: adjust_incident_mileages(nr_furlong_data, x.StartELR, x.StartMileage_num,
                                                   x.StartELR_EndMileage_num, shift_yards_diff_elr),
                axis=1)

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
            adjusted_end_mileages = pd.DataFrame(adjusted_end_elr_mileages.tolist(),
                                                 index=locations_conn.index,
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

            # Save the combined 'adjusted_incident_mileages'
            save_pickle(adj_mileages, path_to_pickle, verbose=update)

            return adj_mileages

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get furlongs data of incident locations each identified by the same ELRs (StartELR != EndELR)
def get_incident_furlongs_diff_start_end_elrs(route_name=None, weather_category=None, shift_yards_diff_elr=220,
                                              update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param shift_yards_diff_elr: [numbers.Number] (default: 220)
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        get_incident_furlongs_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr, update, verbose)
        get_incident_furlongs_diff_start_end_elrs('Anglia', weather_category, shift_yards_diff_elr, update, verbose)
    """
    filename = "incident-furlongs-diff-start-end-ELRs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        adj_mileages = get_adjusted_mileages_diff_start_end_elrs(route_name, weather_category, shift_yards_diff_elr,
                                                                 verbose=verbose)

        try:
            # Get furlong information
            nr_furlong_data = view_nr_vegetation_furlong_data(verbose=verbose)
            # Form a list containing all the furlong IDs
            veg_furlongs_idx = list(set(itertools.chain(*adj_mileages.Critical_FurlongIDs)))

            # Select critical (i.e. incident) furlongs
            incident_furlongs_diff_start_end_elr = nr_furlong_data.loc[veg_furlongs_idx]

            # Save 'incident_furlongs_diff_start_end_elr'
            save_pickle(incident_furlongs_diff_start_end_elr, path_to_pickle, verbose=verbose)

            return incident_furlongs_diff_start_end_elr

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


# Get furlongs data of incident locations (combining the data of incident furlongs of both the above)
def get_incident_furlongs(route_name=None, weather_category=None, shift_yards_same_elr=220, shift_yards_diff_elr=220,
                          update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param shift_yards_same_elr: [numbers.Number] (default: 220)
    :param shift_yards_diff_elr: [numbers.Number] (default: 220)
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        get_incident_furlongs(route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr, update, verbose)
        get_incident_furlongs('Anglia', weather_category, shift_yards_same_elr, shift_yards_diff_elr, update, verbose)
    """
    filename = "incident-furlongs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            incident_furlongs_same_elr = get_incident_furlongs_same_start_end_elrs(
                route_name, weather_category, shift_yards_same_elr, verbose=verbose)

            # Data of incident furlongs: start and end are identified by different ELRs
            incident_furlongs_diff_elr = get_incident_furlongs_diff_start_end_elrs(
                route_name, weather_category, shift_yards_diff_elr, verbose=verbose)

            # Merge the above two data sets
            incident_furlongs = incident_furlongs_same_elr.append(incident_furlongs_diff_elr)
            incident_furlongs.drop_duplicates(['AssetNumber', 'StructuredPlantNumber'], inplace=True)
            incident_furlongs.sort_index(inplace=True)

            save_pickle(incident_furlongs, path_to_pickle, verbose=verbose)

            return incident_furlongs

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))


def get_incident_location_furlongs(route_name=None, weather_category=None, shift_yards_same_elr=220,
                                   shift_yards_diff_elr=220, update=False, verbose=False) -> pd.DataFrame:
    """
    :param route_name: [str; None (default)]
    :param weather_category: [str; None (default)]
    :param shift_yards_same_elr: [numbers.Number] (default: 220)
    :param shift_yards_diff_elr: [numbers.Number] (default: 220)
    :param update: [bool] (default: False)
    :param verbose: [bool] (default: False)
    :return: [pd.DataFrame]

    Testing e.g.
        route_name           = None
        weather_category     = None
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        update               = True
        verbose              = True

        get_incident_location_furlongs(route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr,
                                       update, verbose)
        get_incident_location_furlongs('Anglia', weather_category, shift_yards_same_elr, shift_yards_diff_elr,
                                       update, verbose)
    """
    filename = "incident-location-furlongs"
    pickle_filename = make_filename(filename, route_name, weather_category, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_pickle = cd_prototype_dat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle, verbose=verbose)

    else:
        try:
            adjusted_mileages_same_start_end_elrs = get_adjusted_mileages_same_start_end_elrs(
                route_name, weather_category, shift_yards_same_elr, verbose=verbose)
            ilf_same = adjusted_mileages_same_start_end_elrs[['Section_Length_Adj', 'Critical_FurlongIDs']]

            adjusted_mileages_diff_start_end_elrs = get_adjusted_mileages_diff_start_end_elrs(
                route_name, weather_category, shift_yards_diff_elr, verbose=verbose)
            ilf_diff = adjusted_mileages_diff_start_end_elrs[['Section_Length_Adj', 'Critical_FurlongIDs']]

            incident_locations = fetch_incident_locations_from_nr_metex(route_name, weather_category, verbose=verbose)

            # Merge the above data sets
            incident_location_furlongs = incident_locations.join(pd.concat([ilf_same, ilf_diff]), how='right')
            incident_location_furlongs.drop(['StartMileage_num', 'EndMileage_num'], axis=1, inplace=True)
            incident_location_furlongs.index = range(len(incident_location_furlongs))

            save_pickle(incident_location_furlongs, path_to_pickle, verbose=verbose)

            return incident_location_furlongs

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))
