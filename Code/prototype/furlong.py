""" Processing furlong data """

import itertools
import os
import re

import measurement.measures
import numpy as np
import pandas as pd
from pyhelpers.store import load_pickle, save_pickle
from pyrcs.line_data_cls import elrs_mileages
from pyrcs.utils import nr_mileage_num_to_str, nr_mileage_to_yards, str_to_num_mileage, yards_to_nr_mileage

import mssqlserver.metex
import mssqlserver.vegetation
import prototype.utils


# Get Schedule 8 costs (minutes & cost) aggregated for each STANOX section
def get_incident_locations_from_nr_metex(route=None, weather=None, same_elr=None):
    """
    :param route:
    :param weather:
    :param same_elr:
    :return:
    """
    # Load Schedule 8 costs data aggregated by financial year and STANOX section
    s8data = mssqlserver.metex.view_schedule8_cost_by_location(route, weather).loc[:, 'Route':]

    # Aggregate the data for each STANOX section
    incident_locations = s8data.groupby(list(s8data.columns)[:-3]).agg(np.sum)
    incident_locations.reset_index(inplace=True)

    # Create two additional columns about data of mileages (convert str to num)
    incident_locations[['start_mileage', 'end_mileage']] = \
        incident_locations[['StartMileage', 'EndMileage']].applymap(str_to_num_mileage)

    # Remove records for which ELR information was missing
    incident_locations = incident_locations[
        ~(incident_locations.StartELR.str.contains('^$')) & ~(incident_locations.EndELR.str.contains('^$'))]

    # # Remove records of 'WTS', as Vegetation data is unavailable for this ELR
    # incident_locations = incident_locations[
    #     ~(incident_locations.StartELR.str.contains(re.compile('^$|WTS'))) &
    #     ~(incident_locations.EndELR.str.contains(re.compile('^$|WTS')))]

    # Get "ilocations_same_elr" / "ilocations_diff_elr", and "furlongs_veg_db"
    if same_elr is None:
        return incident_locations
    elif same_elr is True:
        # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
        same_elr_idx = incident_locations.StartELR == incident_locations.EndELR
        incident_locations_same_elr = incident_locations[same_elr_idx]
        return incident_locations_same_elr
    elif same_elr is False:
        # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
        diff_elr_idx = incident_locations.StartELR != incident_locations.EndELR
        incident_locations_diff_elr = incident_locations[diff_elr_idx]
        return incident_locations_diff_elr


# Get the ELR & mileage data of furlong locations
def get_furlongs_info_from_nr_vegetation(location_data_only=False, update=False):
    """
    :param location_data_only:
    :param update:
    :return:
    """
    filename = "furlongs_veg_db"
    path_to_file = prototype.utils.cd_prototype_dat(filename + ".pickle")

    if location_data_only:
        path_to_file = path_to_file.replace(filename, filename + "_loc_only")

    if os.path.isfile(path_to_file) and not update:
        furlongs = load_pickle(path_to_file)
    else:
        try:
            # Get the data of furlong location
            if location_data_only:  # using the original 'FurlongLocation'?
                furlong_location = mssqlserver.vegetation.get_furlong_location(useful_columns_only=True, update=update)
            else:  # using the merged data set 'furlong_vegetation_data'
                furlong_vegetation_data = mssqlserver.vegetation.get_furlong_vegetation_conditions(update=update)
                furlong_vegetation_data.set_index('FurlongID', inplace=True)
                furlong_location = furlong_vegetation_data.sort_index()

            # Column names of mileage data (as string)
            str_mileage_colnames = ['StartMileage', 'EndMileage']
            # Column names of ELR and mileage data (as string)
            elr_mileage_colnames = ['ELR'] + str_mileage_colnames

            furlongs = furlong_location.drop_duplicates(elr_mileage_colnames)

            # Create two new columns of mileage data (as float)
            num_mileage_colnames = ['start_mileage', 'end_mileage']
            furlongs[num_mileage_colnames] = furlongs[str_mileage_colnames].applymap(str_to_num_mileage)

            # Sort the furlong data by ELR and mileage
            furlongs.sort_values(['ELR'] + num_mileage_colnames, inplace=True)

            save_pickle(furlongs, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            furlongs = None

    return furlongs


# Get adjusted Start and End mileages
def adjust_start_end(furlongs, elr, start_mileage, end_mileage, shift_yards):
    """
    :param furlongs:
    :param elr:
    :param start_mileage:
    :param end_mileage:
    :param shift_yards:
    :return:
    """
    elr_furlongs = furlongs[furlongs.ELR == elr]

    def merge_start_and_end(start_array, end_array):
        end_array = np.array(end_array)
        assert len(start_array) == len(end_array)
        return pd.Series(np.append(start_array, end_array[-1]))

    try:
        elr_mileages = merge_start_and_end(elr_furlongs.start_mileage, elr_furlongs.end_mileage)
    except IndexError:
        return '', '', np.nan, np.nan, np.nan, []

    m_indices = pd.Index(elr_mileages)
    s_indices = pd.Index(elr_furlongs.StartMileage)
    e_indices = pd.Index(elr_furlongs.EndMileage)

    def num_mileage_shifting(mileage, y):
        yards = nr_mileage_to_yards(mileage) + y
        str_mileage = yards_to_nr_mileage(yards)
        return str_to_num_mileage(str_mileage)

    if start_mileage <= end_mileage:
        if start_mileage == end_mileage:
            start_mileage = num_mileage_shifting(start_mileage, -shift_yards)
            end_mileage = num_mileage_shifting(end_mileage, shift_yards)
        else:  # start_mileage < end_mileage
            pass
        # Get adjusted mileages of start and end locations ---------------
        try:
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'ffill')]
        except (ValueError, KeyError):
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'nearest')]
        try:
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'bfill')]
        except (ValueError, KeyError):
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'nearest')]
        # Get 'FurlongID's for extracting Vegetation data ----------------
        try:
            s_idx = s_indices.get_loc(nr_mileage_num_to_str(adj_start_mileage))
        except (ValueError, KeyError):
            s_idx = e_indices.get_loc(nr_mileage_num_to_str(adj_start_mileage))
            adj_start_mileage = str_to_num_mileage(elr_furlongs.StartMileage.iloc[s_idx])
        try:
            e_idx = e_indices.get_loc(nr_mileage_num_to_str(adj_end_mileage))
        except (ValueError, KeyError):
            e_idx = s_indices.get_loc(nr_mileage_num_to_str(adj_end_mileage))
            adj_end_mileage = str_to_num_mileage(elr_furlongs.EndMileage.iloc[e_idx])
    else:  # start_mileage > end_mileage:
        # Get adjusted mileages of start and end locations ---------------
        try:
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'bfill')]
        except (ValueError, KeyError):
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'nearest')]
        try:
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'ffill')]
        except (ValueError, KeyError):
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'nearest')]
        # Get 'FurlongID's for extracting Vegetation data ----------------
        try:
            s_idx = e_indices.get_loc(nr_mileage_num_to_str(adj_start_mileage))
        except (ValueError, KeyError):
            s_idx = s_indices.get_loc(nr_mileage_num_to_str(adj_start_mileage))
            adj_start_mileage = str_to_num_mileage(elr_furlongs.EndMileage.iloc[s_idx])
        try:
            e_idx = s_indices.get_loc(nr_mileage_num_to_str(adj_end_mileage))
        except (ValueError, KeyError):
            e_idx = e_indices.get_loc(nr_mileage_num_to_str(adj_end_mileage))
            adj_end_mileage = str_to_num_mileage(elr_furlongs.StartMileage.iloc[e_idx])

    if s_idx <= e_idx:
        e_idx = e_idx + 1 if e_idx < len(elr_mileages) else e_idx
        veg_furlongs = elr_furlongs.iloc[s_idx:e_idx]
    else:  # s_idx > e_idx
        s_idx = s_idx + 1 if s_idx < len(elr_mileages) else s_idx
        veg_furlongs = elr_furlongs.iloc[e_idx:s_idx]

    return \
        nr_mileage_num_to_str(adj_start_mileage), \
        nr_mileage_num_to_str(adj_end_mileage), \
        adj_start_mileage, adj_end_mileage, \
        measurement.measures.Distance(mile=np.abs(adj_end_mileage - adj_start_mileage)).yd, \
        veg_furlongs.index.tolist()


# Get furlongs data of incident locations each identified by the same ELRs
def get_incident_location_furlongs_same_elr(route=None, weather=None, shift_yards_same_elr=220, update=False):
    """
    :param route:
    :param weather:
    :param shift_yards_same_elr: yards
    :param update:
    :return:
    """
    filename = mssqlserver.metex.make_filename("incident_location_furlongs_same_ELRs", route, weather,
                                               shift_yards_same_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs_same_elr = load_pickle(path_to_file)
    else:
        try:
            # Get data about for which the 'StartELR' and 'EndELR' are THE SAME
            incident_locations_same_elr = get_incident_locations_from_nr_metex(route, weather, same_elr=True)

            # Get furlong information
            furlongs = get_furlongs_info_from_nr_vegetation(location_data_only=False, update=update)

            # Get data of each incident's furlong locations for extracting Vegetation
            adjusted_mileages = incident_locations_same_elr.apply(
                lambda record: adjust_start_end(
                    furlongs, record.StartELR, record.start_mileage, record.end_mileage, shift_yards_same_elr), axis=1)
            # Column names
            colnames = ['StartMileage_adjusted',
                        'EndMileage_adjusted',
                        'start_mileage_adjusted',
                        'end_mileage_adjusted',
                        'total_yards_adjusted',  # yards
                        'critical_FurlongIDs']
            # Get adjusted mileage data
            adjusted_mileages_data = pd.DataFrame(list(adjusted_mileages), incident_locations_same_elr.index, colnames)

            save_pickle(adjusted_mileages_data,
                        prototype.utils.cd_prototype_dat("adjusted_mileages_same_ELRs_{}.pickle".format(route)))

            incident_locations_same_elr.drop(['start_mileage', 'end_mileage'], axis=1, inplace=True)
            incident_location_furlongs_same_elr = incident_locations_same_elr.join(
                adjusted_mileages_data[['total_yards_adjusted', 'critical_FurlongIDs']], how='inner').dropna()

            save_pickle(incident_location_furlongs_same_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs_same_elr = None

    return incident_location_furlongs_same_elr


# Get furlongs data by the same ELRs
def get_incident_furlongs_same_elr(route=None, weather=None, shift_yards_same_elr=220, update=False):
    filename = mssqlserver.metex.make_filename("incident_furlongs_same_elr", route, weather, shift_yards_same_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_furlongs_same_elr = load_pickle(path_to_file)
    else:
        try:
            incident_location_furlongs_same_elr = \
                get_incident_location_furlongs_same_elr(route, weather, shift_yards_same_elr, update)
            # Form a list containing all the furlong ID's
            veg_furlongs_idx = list(itertools.chain(*incident_location_furlongs_same_elr.critical_FurlongIDs))
            # Get furlong information
            furlongs = get_furlongs_info_from_nr_vegetation(location_data_only=False, update=update)

            incident_furlongs_same_elr = furlongs.loc[veg_furlongs_idx]. \
                drop(['start_mileage', 'end_mileage'], axis=1).drop_duplicates(subset='AssetNumber')

            incident_furlongs_same_elr['IncidentReported'] = 1

            save_pickle(incident_furlongs_same_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_furlongs_same_elr = None

    return incident_furlongs_same_elr


# Get information of connecting points for different ELRs
def get_connecting_nodes(route=None, update=False):
    filename = mssqlserver.metex.make_filename("connecting_nodes_between_ELRs", route, weather_category=None)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        connecting_nodes = load_pickle(path_to_file)
    else:
        try:
            # Get data about where Incidents occurred
            incident_locations_diff_elr = get_incident_locations_from_nr_metex(route, same_elr=False)

            elr_mileage_cols = ['StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'start_mileage', 'end_mileage']
            diff_elr_mileages = incident_locations_diff_elr[elr_mileage_cols].drop_duplicates()

            em_cls = elrs_mileages.ELRMileages()

            # Trying to get the connecting nodes ...
            def get_conn_mileages(start_elr, start_mileage, end_elr, end_mileage):
                s_end_mileage, e_start_mileage = em_cls.get_conn_end_start_mileages(start_elr, end_elr)
                if s_end_mileage is None:
                    s_end_mileage = start_mileage
                if e_start_mileage is None:
                    e_start_mileage = end_mileage
                return s_end_mileage, e_start_mileage

            conn_mileages = diff_elr_mileages.apply(
                lambda x: pd.Series(get_conn_mileages(x.StartELR, x.start_mileage, x.EndELR, x.end_mileage)), axis=1)
            conn_mileages.columns = ['StartELR_EndMileage', 'EndELR_StartMileage']

            idx_columns = ['StartELR', 'StartMileage', 'EndELR', 'EndMileage']
            connecting_nodes = diff_elr_mileages[idx_columns].join(conn_mileages).set_index(idx_columns)

            save_pickle(connecting_nodes, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            connecting_nodes = None

    return connecting_nodes


# Get furlongs data of incident locations each identified by different ELRs
def get_incident_location_furlongs_diff_elr(route=None, weather=None, shift_yards_diff_elr=220, update=False):
    filename = mssqlserver.metex.make_filename("incident_location_furlongs_diff_ELRs", route, weather,
                                               shift_yards_diff_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs_diff_elr = load_pickle(path_to_file)
    else:
        try:
            # Get data for which the 'StartELR' and 'EndELR' are DIFFERENT
            incident_locations_diff_elr = get_incident_locations_from_nr_metex(route, weather, same_elr=False)

            # Get furlong information
            furlongs = get_furlongs_info_from_nr_vegetation(location_data_only=False, update=update)

            # Get connecting points for different (ELRs, mileages)
            connecting_nodes = get_connecting_nodes(route, update=update)

            incident_locations_diff_elr = incident_locations_diff_elr.join(
                connecting_nodes, on=connecting_nodes.index.names, how='inner')
            str_conn_colnames = ['StartELR_EndMileage', 'EndELR_StartMileage']
            num_conn_colnames = ['StartELR_end_mileage', 'EndELR_start_mileage']
            incident_locations_diff_elr[num_conn_colnames] = \
                incident_locations_diff_elr[str_conn_colnames].applymap(str_to_num_mileage)

            """ Get data of each incident's furlong locations for extracting Vegetation """
            # Processing Start locations
            adjusted_start_elr_mileages = incident_locations_diff_elr.apply(
                lambda x: adjust_start_end(
                    furlongs, x.StartELR, x.start_mileage, x.StartELR_end_mileage, shift_yards_diff_elr),
                axis=1)

            # Column names for adjusted_start_elr_mileages_data
            start_elr_colnames = [
                'StartMileage_adjusted',
                'StartELR_EndMileage_adjusted',
                'start_mileage_adjusted',
                'StartELR_end_mileage_adjusted',
                'StartELR_total_yards_adjusted',  # yards
                'StartELR_FurlongIDs']

            # Form a dataframe for adjusted_start_elr_mileages_data
            adjusted_start_elr_mileages_data = pd.DataFrame(list(adjusted_start_elr_mileages),
                                                            index=incident_locations_diff_elr.index,
                                                            columns=start_elr_colnames)

            # Find the index for null values in adjusted_start_elr_mileages_data
            start_elr_null_idx = \
                adjusted_start_elr_mileages_data[adjusted_start_elr_mileages_data.isnull().any(axis=1)].index

            # Processing End locations
            adjusted_end_elr_mileages = incident_locations_diff_elr.apply(
                lambda record: adjust_start_end(
                    furlongs, record.EndELR, record.EndELR_start_mileage, record.end_mileage, shift_yards_diff_elr),
                axis=1)

            # Column names for adjusted_end_elr_mileages_data
            end_elr_colnames = [
                'EndELR_StartMileage_adjusted',
                'EndMileage_adjusted',
                'EndELR_start_mileage_adjusted',
                'end_mileage_adjusted',
                'EndELR_total_yards_adjusted',  # yards
                'EndELR_FurlongIDs']

            # Form a dataframe for adjusted_end_elr_mileages_data
            adjusted_end_elr_mileages_data = pd.DataFrame(list(adjusted_end_elr_mileages),
                                                          index=incident_locations_diff_elr.index,
                                                          columns=end_elr_colnames)

            # Find the index for null values in adjusted_end_elr_mileages_data
            end_elr_null_idx = adjusted_end_elr_mileages_data[adjusted_end_elr_mileages_data.isnull().any(axis=1)].index

            # --------------------------------------------------------------------------------------------
            adjusted_mileages_data = adjusted_start_elr_mileages_data.join(adjusted_end_elr_mileages_data)
            adjusted_mileages_data['total_yards_adjusted'] = list(zip(
                adjusted_mileages_data.StartELR_total_yards_adjusted.fillna(0),
                adjusted_mileages_data.EndELR_total_yards_adjusted.fillna(0)))
            adjusted_mileages_data['critical_FurlongIDs'] = \
                adjusted_mileages_data.StartELR_FurlongIDs + adjusted_mileages_data.EndELR_FurlongIDs

            # Save the adjusted_mileages_data
            save_pickle(adjusted_mileages_data, prototype.utils.cd_prototype_dat("adjusted_mileages_diff_ELRs.pickle"))

            incident_locations_diff_elr.drop(
                str_conn_colnames + num_conn_colnames + ['start_mileage', 'end_mileage'], axis=1, inplace=True)

            colnames = incident_locations_diff_elr.columns
            start_loc_cols = [x for x in colnames if re.match('^Start(?!Location)', x)]
            end_loc_cols = [x for x in colnames if re.match('^End(?!Location)', x)]
            incident_locations_diff_elr.loc[start_elr_null_idx, start_loc_cols] = \
                incident_locations_diff_elr.loc[start_elr_null_idx, end_loc_cols].values
            incident_locations_diff_elr.loc[end_elr_null_idx, end_loc_cols] = \
                incident_locations_diff_elr.loc[end_elr_null_idx, start_loc_cols].values

            incident_location_furlongs_diff_elr = incident_locations_diff_elr.join(
                adjusted_mileages_data[['total_yards_adjusted', 'critical_FurlongIDs']], how='inner').dropna()

            save_pickle(incident_location_furlongs_diff_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs_diff_elr = None

    return incident_location_furlongs_diff_elr


# Get furlongs data by different ELRS
def get_incident_furlongs_diff_elr(route=None, weather=None, shift_yards_diff_elr=220, update=False):
    filename = mssqlserver.metex.make_filename("incident_furlongs_diff_ELRs", route, weather, shift_yards_diff_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incid_furlongs_diff_elr = load_pickle(path_to_file)
    else:
        try:
            incident_location_furlongs_diff_elr = get_incident_location_furlongs_diff_elr(route, weather,
                                                                                          shift_yards_diff_elr)

            # Form a list containing all the furlong ID's
            veg_furlongs_idx = list(itertools.chain(*incident_location_furlongs_diff_elr.critical_FurlongIDs))

            # Get furlong information
            furlongs = get_furlongs_info_from_nr_vegetation(location_data_only=False)

            # Merge the data of the starts and ends
            incid_furlongs_diff_elr = furlongs.loc[veg_furlongs_idx]. \
                drop(['start_mileage', 'end_mileage'], axis=1).drop_duplicates(subset='AssetNumber')

            incid_furlongs_diff_elr['IncidentReported'] = 1

            save_pickle(incid_furlongs_diff_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incid_furlongs_diff_elr = None

    return incid_furlongs_diff_elr


# Combine the incident furlong data of both of the above
def get_incident_location_furlongs(route=None, shift_yards_same_elr=220, shift_yards_diff_elr=220, update=False):
    filename = mssqlserver.metex.make_filename("incident_location_furlongs", route, None, shift_yards_same_elr,
                                               shift_yards_diff_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs = load_pickle(path_to_file)
    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            incident_location_furlongs_same_elr = \
                get_incident_location_furlongs_same_elr(route, None, shift_yards_same_elr)
            # Data of incident furlongs: start and end are identified by different ELRs
            incident_location_furlongs_diff_elr = \
                get_incident_location_furlongs_diff_elr(route, None, shift_yards_diff_elr)
            # Merge the above two data sets
            incident_location_furlongs = incident_location_furlongs_same_elr.append(
                incident_location_furlongs_diff_elr)
            incident_location_furlongs.sort_index(inplace=True)
            # incident_location_furlongs['IncidentReported'] = 1
            save_pickle(incident_location_furlongs, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs = None

    return incident_location_furlongs


# Combine the incident furlong data of both of the above
def get_incident_furlongs(route=None, shift_yards_same_elr=220, shift_yards_diff_elr=220, update=False):
    filename = mssqlserver.metex.make_filename("incident_furlongs", route, None, shift_yards_same_elr,
                                               shift_yards_diff_elr)
    path_to_file = prototype.utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_furlongs = load_pickle(path_to_file)
    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            incid_furlongs_same_elr = get_incident_furlongs_same_elr(route, None, shift_yards_same_elr, update=update)
            # Data of incident furlongs: start and end are identified by different ELRs
            incid_furlongs_diff_elr = get_incident_furlongs_diff_elr(route, None, shift_yards_diff_elr, update=update)
            # Merge the above two data sets
            furlong_incidents = incid_furlongs_same_elr.append(incid_furlongs_diff_elr)
            furlong_incidents.drop_duplicates(subset='AssetNumber', inplace=True)
            # Data of furlong Vegetation coverage and hazardous trees
            furlong_vegetation_data = mssqlserver.vegetation.get_furlong_vegetation_conditions(route)
            incident_furlongs = furlong_vegetation_data.join(
                furlong_incidents[['IncidentReported']], on='FurlongID', how='inner')
            incident_furlongs.sort_values(by='StructuredPlantNumber', inplace=True)
            # # incident_furlongs.index = range(len(incident_furlongs))
            save_pickle(incident_furlongs, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_furlongs = None

    return incident_furlongs
