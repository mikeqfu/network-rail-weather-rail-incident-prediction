""" Integrate data of Incidents and Weather """

import datetime
import functools
import itertools
import os
import random

import datetime_truncate
import geopandas as gpd
import geopy.distance
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import scipy.stats
import shapely.geometry
import shapely.ops

import Intermediate.spreadsheet_incidents as isi
import Intermediate.utils as iu
import Intermediate.weather as iw
import mssql_metex
from converters import svg_to_emf, wgs84_to_osgb36
from utils import load_pickle, save_pickle

# ====================================================================================================================
""" 1 """


# Attach Weather conditions for each incident location
def get_incidents_with_weather_1(route_name=None, weather_category='Heat', season='summer',
                                 prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                                 trial=True, illustrate_buf_cir=False, update=False):

    filename = "Incidents-with-Weather-conditions"
    pickle_filename = isi.make_filename(
        filename, route_name, weather_category, "-".join([season] if isinstance(season, str) else season),
        "trial" if trial else "full")
    path_to_pickle = iu.cdd_intermediate(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        iw_data = load_pickle(path_to_pickle)
    else:
        try:
            # Incidents data
            incidents_all = isi.get_schedule8_weather_incidents()
            incidents_all = iu.get_data_by_season(incidents_all, season)
            incidents = isi.subset(incidents_all, route=route_name, weather_category=weather_category)
            # For testing purpose ...
            if trial:
                # import random
                # rand_iloc_idx = random.sample(range(len(Incidents)), 10)
                incidents = incidents.iloc[0:10, :]

            # Weather observations
            weather_obs = iw.get_integrated_daily_gridded_weather_obs()
            weather_obs.reset_index(inplace=True)

            # Radiation observations
            irad_data = iw.get_midas_radtob()
            irad_data.reset_index(inplace=True)

            # --------------------------------------------------------------------------------------------------------
            """ In the spatial context """

            # Weather observation grids
            observation_grids = iw.get_observation_grids()
            obs_cen_geom = shapely.geometry.MultiPoint(list(observation_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(observation_grids.Grid))

            # Find the closest grid centroid and return the corresponding (pseudo) grid id
            def find_closest_obs_grid(x, centroids_geom):
                """
                :param x: e.g. Incidents.StartNE.iloc[0]
                :param centroids_geom: i.e. obs_cen_geom
                :return:
                """
                pseudo_id = np.where(observation_grids.Centroid_XY == shapely.ops.nearest_points(x, centroids_geom)[1])
                return pseudo_id[0][0]

            # Start
            incidents['Start_Pseudo_Grid_ID'] = incidents.StartNE.map(lambda x: find_closest_obs_grid(x, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='Start_Pseudo_Grid_ID')
            # End
            incidents['End_Pseudo_Grid_ID'] = incidents.EndNE.map(lambda x: find_closest_obs_grid(x, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')
            # Modify column names
            for l in ['Start', 'End']:
                a = [c for c in incidents.columns if c.endswith(l)]
                b = [l + '_' + c if c == 'Grid' else l + '_Grid_' + c for c in observation_grids.columns]
                incidents.rename(columns=dict(zip(a, b)), inplace=True)

            # Get midpoint between two points
            def find_midpoint(start, end, as_geom=True):
                """
                :param start: [shapely.geometry.point.Point] e.g. Incidents.StartNE.iloc[0]
                :param end: [shapely.geometry.point.Point] e.g. Incidents.EndNE.iloc[0]
                :param as_geom: [bool] whether to return a shapely.geometry obj
                :return:
                """
                midpoint = (start.x + end.x) / 2, (start.y + end.y) / 2
                return shapely.geometry.Point(midpoint) if as_geom else midpoint

            # Append 'MidpointNE' column
            incidents['MidpointNE'] = incidents.apply(lambda x: find_midpoint(x.StartNE, x.EndNE, as_geom=True), axis=1)

            # Create a circle buffer for start location
            def create_circle_buffer(start, end, midpoint, whisker=500):
                """
                :param start: e.g. Incidents.StartNE[0]
                :param end: e.g. Incidents.EndNE[0]
                :param midpoint: e.g. Incidents.MidpointNE[0]
                :param whisker: An extended length to the diameter (i.e. on both sides of the start and end locations)
                :return:
                """
                if start == end:
                    buffer_circle = start.buffer(2000 + whisker)
                else:
                    # midpoint = find_midpoint(start, end, as_geom=True)
                    radius = (start.distance(end) + whisker) / 2
                    buffer_circle = midpoint.buffer(radius)
                return buffer_circle

            # Make a buffer zone for Weather data aggregation
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: create_circle_buffer(x.StartNE, x.EndNE, x.MidpointNE, whisker=500), axis=1)

            # Find all intersecting geom objects
            def find_intersecting_grid(x, grids_geom, as_grid_id=True):
                """
                :param x: e.g. Incidents.Buffer_Zone.iloc[0]
                :param grids_geom: i.e. obs_grids_geom
                :param as_grid_id: [bool] whether to return grid id number
                :return:
                """
                intxn_grids = [grid for grid in grids_geom if x.intersects(grid)]
                if as_grid_id:
                    intxn_grids = [observation_grids[observation_grids.Grid == g].index[0] for g in intxn_grids]
                return intxn_grids

            # Find all Weather observation grids that intersect with the created buffer zone for each incident location
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(lambda x: find_intersecting_grid(x, obs_grids_geom))

            # Met station locations
            met_stations = iw.get_meteorological_stations()[['E_N', 'E_N_GEOM']]
            met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.E_N_GEOM))

            # Find the closest grid centroid and return the corresponding (pseudo) grid id
            def find_closest_met_stn(x, met_stn_geom):
                """
                :param x: e.g. Incidents.MidpointNE.iloc[0]
                :param met_stn_geom: i.e. met_stations_geom
                :return:
                """
                idx = np.where(met_stations.E_N_GEOM == shapely.ops.nearest_points(x, met_stn_geom)[1])[0]
                src_id = met_stations.index[idx].tolist()
                return src_id

            incidents['Met_SRC_ID'] = incidents.MidpointNE.map(lambda x: find_closest_met_stn(x, met_stations_geom))

            if illustrate_buf_cir:  # Illustration of the buffer circle

                def illustrate_example_buffer_circle():
                    start_point, end_point = incidents.StartNE.iloc[4], incidents.EndNE.iloc[4]
                    midpoint = incidents.MidpointNE.iloc[4]
                    bf_circle = create_circle_buffer(start_point, end_point, midpoint, whisker=500)
                    i_obs_grids = find_intersecting_grid(bf_circle, obs_grids_geom, as_grid_id=False)
                    plt.figure(figsize=(7, 6))
                    ax = plt.subplot2grid((1, 1), (0, 0))
                    for g in i_obs_grids:
                        x, y = g.exterior.xy
                        ax.plot(x, y, color='#433f3f')
                    ax.plot([], 's', label="Weather observation grid", ms=16, color='none', markeredgecolor='#433f3f')
                    x, y = bf_circle.exterior.xy
                    ax.plot(x, y)
                    sx, sy, ex, ey = start_point.xy + end_point.xy
                    if start_point == end_point:
                        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
                    else:
                        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
                        ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')
                    ax.set_xlabel('Easting')
                    ax.set_ylabel('Northing')
                    font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
                    legend = plt.legend(numpoints=1, loc='best', prop=font, fancybox=True, labelspacing=0.5)
                    frame = legend.get_frame()
                    frame.set_edgecolor('k')
                    plt.tight_layout()

                illustrate_example_buffer_circle()

            # --------------------------------------------------------------------------------------------------------
            """ In the temporal context (for the specified IP) """

            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime

            incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
                lambda x: x + pd.Timedelta(days=-1) if x.time() < datetime.time(9) else x)
            incidents.Critical_StartDateTime += datetime.timedelta(hours=prior_ip_start_hrs)
            # Note that the 'Critical_EndDateTime' would be based on the 'Critical_StartDateTime' if we consider the
            # Weather conditions on the day of incident occurrence, and 'StartDateTime' otherwise.
            incidents['Critical_EndDateTime'] = incidents.Critical_StartDateTime
            incidents['Critical_Period'] = incidents.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            # --------------------------------------------------------------------------------------------------------
            """ Integration along both the spatial and temporal dimensions (for the specified prior-IP) """

            # Specify the statistics needed for Weather observations (except radiation)
            def specify_weather_stats_calculations():
                weather_stats_calculations = {'Maximum_Temperature': (max, min, np.average),
                                              'Minimum_Temperature': (max, min, np.average),
                                              'Temperature_Change': np.average,
                                              'Rainfall': (max, min, np.average)}
                return weather_stats_calculations

            # Calculate the statistics for the Weather variables (except radiation)
            def calculate_weather_variables_stats(weather_dat):
                # Specify calculations
                weather_stats_computations = specify_weather_stats_calculations()
                if weather_dat.empty:
                    weather_stats_info = [np.nan] * sum(map(np.count_nonzero, weather_stats_computations.values()))
                else:
                    # Create a pseudo id for groupby() & aggregate()
                    weather_dat['Pseudo_ID'] = 0
                    weather_stats = weather_dat.groupby('Pseudo_ID').aggregate(weather_stats_computations)
                    # a, b = [list(x) for x in weather_stats.columns.levels]
                    # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
                    # if not weather_stats.empty:
                    #     stats_info = weather_stats.values[0].tolist()
                    # else:
                    #     stats_info = [np.nan] * len(weather_stats.columns)
                    weather_stats_info = weather_stats.values[0].tolist()
                return weather_stats_info

            # Gather gridded Weather observations of the given period for each incident record
            def integrate_ip_gridded_weather_obs(grids, period):
                """
                :param grids: e.g. grids = Incidents.Weather_Grid.iloc[0]
                :param period: e.g. period = Incidents.Critical_Period.iloc[0]
                :return:

                if not isinstance(period, list) and isinstance(period, datetime.date):
                    period = [period]
                import itertools
                temp = pd.DataFrame(list(itertools.product(grids, period)), columns=['Pseudo_Grid_ID', 'Date'])

                """
                # Find Weather data for the specified period
                prior_ip_weather = weather_obs[weather_obs.Pseudo_Grid_ID.isin(grids) & weather_obs.Date.isin(period)]
                # Calculate the max/min/avg for Weather parameters during the period
                weather_stats = calculate_weather_variables_stats(prior_ip_weather)

                # Whether "max_temp = weather_stats[0]" is the hottest of year so far
                obs_by_far = weather_obs[
                    (weather_obs.Date < min(period)) &
                    (weather_obs.Date > datetime_truncate.truncate_year(min(period))) &
                    weather_obs.Pseudo_Grid_ID.isin(grids)]  # Or weather_obs.Date > pd.datetime(min(period).year, 6, 1)
                weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

                return weather_stats

            # Get Weather data (except radiation) for the specified IP
            prior_ip_weather_stats = incidents.apply(
                lambda x: pd.Series(integrate_ip_gridded_weather_obs(x.Weather_Grid, x.Critical_Period)), axis=1)

            # Get all Weather variable names
            def specify_weather_variable_names(calc):
                var_stats_names = [[k, [i.__name__ for i in v] if isinstance(v, tuple) else [v.__name__]]
                                   for k, v in calc.items()]
                weather_variable_names = [['_'.join([x, z]) for z in y] for x, y in var_stats_names]
                weather_variable_names = list(itertools.chain.from_iterable(weather_variable_names))
                return weather_variable_names

            w_col_names = specify_weather_variable_names(specify_weather_stats_calculations()) + ['Hottest_Heretofore']
            prior_ip_weather_stats.columns = w_col_names
            prior_ip_weather_stats['Temperature_Change_max'] = \
                abs(prior_ip_weather_stats.Maximum_Temperature_max - prior_ip_weather_stats.Minimum_Temperature_min)
            prior_ip_weather_stats['Temperature_Change_min'] = \
                abs(prior_ip_weather_stats.Maximum_Temperature_min - prior_ip_weather_stats.Minimum_Temperature_max)

            prior_ip_data = incidents.join(prior_ip_weather_stats)

            # Specify the statistics needed for radiation only
            def specify_radtob_stats_calculations():
                radtob_stats_calculations = {'GLBL_IRAD_AMT': (max, scipy.stats.iqr)}
                return radtob_stats_calculations

            # Calculate the statistics for the radiation variables
            def calculate_radtob_variables_stats(radtob_dat):
                # Solar irradiation amount (Kjoules/ sq metre over the observation period)
                radtob_stats_calculations = specify_radtob_stats_calculations()
                if radtob_dat.empty:
                    stats_info = [np.nan] * (sum(map(np.count_nonzero, radtob_stats_calculations.values())) + 1)
                else:
                    if 24 not in list(radtob_dat.OB_HOUR_COUNT):
                        radtob_dat = radtob_dat.append(radtob_dat.iloc[-1, :])
                        radtob_dat.VERSION_NUM.iloc[-1] = 0
                        radtob_dat.OB_HOUR_COUNT.iloc[-1] = radtob_dat.OB_HOUR_COUNT.iloc[0:-1].sum()
                        radtob_dat.GLBL_IRAD_AMT.iloc[-1] = radtob_dat.GLBL_IRAD_AMT.iloc[0:-1].sum()

                    radtob_stats = radtob_dat.groupby('OB_HOUR_COUNT').aggregate(radtob_stats_calculations)
                    stats_info = radtob_stats.values.flatten().tolist()[0:-1]
                    # if len(radtob_stats) != 2:
                    #     stats_info = radtob_stats.values.flatten().tolist() + [np.nan]
                return stats_info

            # Gather solar radiation of the given period for each incident record
            def integrate_ip_midas_radtob(met_stn_id, period):
                """
                :param met_stn_id: e.g. met_stn_id = Incidents.Met_SRC_ID.iloc[1]
                :param period: e.g. period = Incidents.Critical_Period.iloc[1]
                :return:
                """
                prior_ip_radtob = irad_data[irad_data.SRC_ID.isin(met_stn_id) & irad_data.OB_END_DATE.isin(period)]
                radtob_stats = calculate_radtob_variables_stats(prior_ip_radtob)
                return radtob_stats

            prior_ip_radtob_stats = prior_ip_data.apply(
                lambda x: pd.Series(integrate_ip_midas_radtob(x.Met_SRC_ID, x.Critical_Period)), axis=1)

            r_col_names = specify_weather_variable_names(specify_radtob_stats_calculations()) + ['GLBL_IRAD_AMT_total']
            prior_ip_radtob_stats.columns = r_col_names

            prior_ip_data = prior_ip_data.join(prior_ip_radtob_stats)

            prior_ip_data['Incident_Reported'] = 1

            # --------------------------------------------------------------------------------------------------------
            """ Integration along both the spatial and temporal dimensions (for the specified non-IP) """

            # Get Weather data that did not cause any incident
            non_ip_data = incidents.copy(deep=True)
            # non_ip_data[[c for c in non_ip_data.columns if c.startswith('Incident')]] = None
            # non_ip_data[['StartDateTime', 'EndDateTime', 'WeatherCategoryCode', 'WeatherCategory', 'Minutes']] = None

            non_ip_data.Critical_EndDateTime = \
                non_ip_data.Critical_StartDateTime + datetime.timedelta(days=latent_period)
            non_ip_data.Critical_StartDateTime = \
                non_ip_data.Critical_EndDateTime + datetime.timedelta(hours=non_ip_start_hrs)
            non_ip_data.Critical_Period = non_ip_data.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            # Gather gridded Weather observations of the corresponding non-incident period for each incident record
            def integrate_nip_gridded_weather_obs(grids, period, stanox_section):
                """
                :param grids: e.g. grids = non_ip_data.Weather_Grid.iloc[12]
                :param period: e.g. period = non_ip_data.Critical_Period.iloc[12]
                :param stanox_section: e.g. stanox_section = non_ip_data.StanoxSection.iloc[12]
                :return:
                """
                # Get non-IP Weather data about where and when the incident occurred
                nip_weather = weather_obs[(weather_obs.Pseudo_Grid_ID.isin(grids)) & (weather_obs.Date.isin(period))]

                # Get all incident period data on the same section
                ip_overlap = prior_ip_data[
                    (prior_ip_data.StanoxSection == stanox_section) &
                    (((prior_ip_data.Critical_StartDateTime <= min(period)) &
                      (prior_ip_data.Critical_EndDateTime >= min(period))) |
                     ((prior_ip_data.Critical_StartDateTime <= max(period)) &
                      (prior_ip_data.Critical_EndDateTime >= max(period))))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not ip_overlap.empty:
                    nip_weather = nip_weather[
                        (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
                        (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
                # Get the max/min/avg Weather parameters for those incident periods
                weather_stats = calculate_weather_variables_stats(nip_weather)

                # Whether "max_temp = weather_stats[0]" is the hottest of year so far
                obs_by_far = weather_obs[
                    (weather_obs.Date < min(period)) &
                    (weather_obs.Date > datetime_truncate.truncate_year(min(period))) &
                    weather_obs.Pseudo_Grid_ID.isin(grids)]  # Or weather_obs.Date > pd.datetime(min(period).year, 6, 1)
                weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

                return weather_stats

            non_ip_weather_stats = non_ip_data.apply(
                lambda x: pd.Series(integrate_nip_gridded_weather_obs(
                    x.Weather_Grid, x.Critical_Period, x.StanoxSection)), axis=1)

            non_ip_weather_stats.columns = w_col_names
            non_ip_weather_stats['Temperature_Change_max'] = \
                abs(non_ip_weather_stats.Maximum_Temperature_max - non_ip_weather_stats.Minimum_Temperature_min)
            non_ip_weather_stats['Temperature_Change_min'] = \
                abs(non_ip_weather_stats.Maximum_Temperature_min - non_ip_weather_stats.Minimum_Temperature_max)

            non_ip_data = non_ip_data.join(non_ip_weather_stats)

            # Gather solar radiation of the corresponding non-incident period for each incident record
            def integrate_nip_midas_radtob(met_stn_id, period, location):
                """
                :param met_stn_id: e.g. met_stn_id = non_ip_data.Met_SRC_ID.iloc[1]
                :param period: e.g. period = non_ip_data.Critical_Period.iloc[1]
                :param location: e.g. location = non_ip_data.StanoxSection.iloc[0]
                :return:
                """
                non_ip_radtob = irad_data[irad_data.SRC_ID.isin(met_stn_id) & irad_data.OB_END_DATE.isin(period)]

                # Get all incident period data on the same section
                ip_overlap = prior_ip_data[
                    (prior_ip_data.StanoxSection == location) &
                    (((prior_ip_data.Critical_StartDateTime <= min(period)) &
                      (prior_ip_data.Critical_EndDateTime >= min(period))) |
                     ((prior_ip_data.Critical_StartDateTime <= max(period)) &
                      (prior_ip_data.Critical_EndDateTime >= max(period))))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not ip_overlap.empty:
                    non_ip_radtob = non_ip_radtob[
                        (non_ip_radtob.OB_END_DATE < min(ip_overlap.Critical_StartDateTime)) |
                        (non_ip_radtob.OB_END_DATE > max(ip_overlap.Critical_EndDateTime))]

                radtob_stats = calculate_radtob_variables_stats(non_ip_radtob)
                return radtob_stats

            non_ip_radtob_stats = non_ip_data.apply(
                lambda x: pd.Series(integrate_nip_midas_radtob(
                    x.Met_SRC_ID, x.Critical_Period, x.StanoxSection)), axis=1)

            non_ip_radtob_stats.columns = r_col_names

            non_ip_data = non_ip_data.join(non_ip_radtob_stats)

            non_ip_data['Incident_Reported'] = 0

            # Merge "ip_data" and "nip_data" into one DataFrame
            iw_data = pd.concat([prior_ip_data, non_ip_data], axis=0, ignore_index=True, sort=False)

            # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
            iw_data = iw_data.join(iu.categorise_track_orientations(iw_data[['StartNE', 'EndNE']]))

            # Categorise temperature: 25, 26, 27, 28, 29, 30
            iw_data = iw_data.join(iu.categorise_temperatures(iw_data.Maximum_Temperature_max))

            save_pickle(iw_data, path_to_pickle)

        except Exception as e:
            print("Failed to get Incidents with Weather conditions. {}.".format(e))
            iw_data = pd.DataFrame()

    return iw_data


# Describe basic statistics about the main explanatory variables
def describe_explanatory_variables_1(train_set, save_as=".pdf", dpi=None):
    plt.figure(figsize=(13, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    ax1 = plt.subplot2grid((1, 8), (0, 0))
    train_set.Temperature_Change_max.plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
    ax1.set_xticklabels('')
    plt.xlabel('Temp.\nChange', fontsize=13, labelpad=39)
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(0.05, 1.01)

    ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=2)
    train_set.Temperature_Category.value_counts().plot.bar(color='#537979', rot=-90, fontsize=12)
    plt.xticks(range(0, 8), ['< 24°C', '24°C', '25°C', '26°C', '27°C', '28°C', '29°C', '≥ 30°C'], fontsize=12)
    plt.xlabel('Max. Temp.', fontsize=13, labelpad=7)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax2.yaxis.set_label_coords(0.0, 1.01)

    ax3 = plt.subplot2grid((1, 8), (0, 3), colspan=2)
    track_orientation = train_set.Track_Orientation.value_counts()
    track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
    track_orientation.plot.bar(color='#a72a3d', rot=-90, fontsize=12)
    plt.xlabel('Track orientation', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax3.yaxis.set_label_coords(0.0, 1.01)

    ax4 = plt.subplot2grid((1, 8), (0, 5))
    train_set.GLBL_IRAD_AMT_max.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Max.\nirradiation', fontsize=13, labelpad=29)
    plt.ylabel('(KJ/m$\\^2$)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(0.2, 1.01)

    ax5 = plt.subplot2grid((1, 8), (0, 6))
    train_set.Rainfall_max.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
    ax5.set_xticklabels('')
    plt.xlabel('Max.\nPrecip.', fontsize=13, labelpad=29)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(0.0, 1.01)

    ax6 = plt.subplot2grid((1, 8), (0, 7))
    hottest_heretofore = train_set.Hottest_Heretofore.value_counts()
    hottest_heretofore.plot.bar(color='#a72a3d', rot=-90, fontsize=12)
    plt.xlabel('Hottest\nheretofore', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(0.0, 1.01)

    plt.tight_layout()

    path_to_file_weather = iu.cdd_intermediate(0, "Variables" + save_as)
    plt.savefig(path_to_file_weather, dpi=dpi)
    if save_as == ".svg":
        svg_to_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


# ====================================================================================================================
""" 2 """


# Create shapely.points for StartLocations and EndLocations
def create_start_end_shapely_points(incidents):
    data = incidents.copy()
    # Make shapely.geometry.points in longitude and latitude
    data.insert(data.columns.get_loc('StartLatitude') + 1, 'StartLonLat',
                gpd.points_from_xy(data.StartLongitude, data.StartLatitude))
    data.insert(data.columns.get_loc('EndLatitude') + 1, 'EndLonLat',
                gpd.points_from_xy(data.EndLongitude, data.EndLatitude))
    data.insert(data.columns.get_loc('EndLonLat') + 1, 'MidLonLat',
                data[['StartLonLat', 'EndLonLat']].apply(
                    lambda x: shapely.geometry.LineString([x.StartLonLat, x.EndLonLat]).centroid, axis=1))
    # Add Easting and Northing points
    data[['StartEasting', 'StartNorthing']] = data[['StartLongitude', 'StartLatitude']].apply(
        lambda x: pd.Series(wgs84_to_osgb36(x.StartLongitude, x.StartLatitude)), axis=1)
    data['StartEN'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    data[['EndEasting', 'EndNorthing']] = data[['EndLongitude', 'EndLatitude']].apply(
        lambda x: pd.Series(wgs84_to_osgb36(x.EndLongitude, x.EndLatitude)), axis=1)
    data['EndEN'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)
    return data


# Create a circle buffer for an incident location
def create_circle_buffer_for_incident(midpoint, incident_start, incident_end, whisker_km=0.1, as_geom=True):
    """
    Ref: https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python

    :param midpoint: [shapely.geometry.point.Point] e.g. midpoint = incidents.MidLonLat.iloc[0]
    :param incident_start: [shapely.geometry.point.Point] e.g. incident_start = incidents.StartLonLat.iloc[0]
    :param incident_end: [shapely.geometry.point.Point] e.g. incident_end = incidents.EndLonLat.iloc[0]
    :param whisker_km: [num] an extended length to the diameter (i.e. on both sides of the start and end locations)
    :param as_geom: [bool]
    :return: [shapely.geometry.Polygon; list of tuples]
    """
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lon_0={lon} +lat_0={lat} +x_0=0 +y_0=0'
    project = functools.partial(
        pyproj.transform, pyproj.Proj(aeqd_proj.format(lon=midpoint.x, lat=midpoint.y)), pyproj.Proj(init='epsg:4326'))
    if incident_start != incident_end:
        radius_km = geopy.distance.distance(incident_start.coords, incident_end.coords).km / 2 + whisker_km
    else:
        radius_km = 0.5
    buffer = shapely.ops.transform(project, shapely.geometry.Point(0, 0).buffer(radius_km * 1000))
    buffer_circle = buffer if as_geom else buffer.exterior.coords[:]
    return buffer_circle


# Find all intersecting weather cells
def find_intersecting_weather_cells(x, as_geom=False):
    """
    :param x: [shapely.geometry.point.Point] e.g. x = incidents.Buffer_Zone.iloc[0]
    :param as_geom: [bool] whether to return shapely.geometry.Polygon of intersecting weather cells
    :return:
    """
    weather_cell_geoms = mssql_metex.get_weather_cell().poly
    intxn_weather_cells = tuple(cell for cell in weather_cell_geoms if x.intersects(cell))
    if as_geom:
        return intxn_weather_cells
    else:
        intxn_weather_cell_ids = tuple(weather_cell_geoms[weather_cell_geoms == cell].index[0]
                                       for cell in intxn_weather_cells)
        return intxn_weather_cell_ids


# Illustration of the buffer circle
def illustrate_buffer_circle(midpoint, incident_start, incident_end, whisker_km=0.05):
    """
    :param midpoint: e.g. midpoint = incidents.MidLonLat.iloc[4]
    :param incident_start: e.g. incident_start = incidents.StartLonLat.iloc[4]
    :param incident_end: e.g. incident_end = incidents.EndLonLat.iloc[4]
    :param whisker_km:
    """
    buffer_circle = create_circle_buffer_for_incident(midpoint, incident_start, incident_end, whisker_km=whisker_km)
    i_weather_cells = find_intersecting_weather_cells(buffer_circle, as_geom=True)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for g in i_weather_cells:
        x, y = g.exterior.xy
        ax.plot(x, y, color='#433f3f')
        polygons = matplotlib.patches.Polygon(g.exterior.coords[:], fc='#fff68f', ec='b', alpha=0.5)
        plt.gca().add_patch(polygons)
    ax.plot([], 's', label="Weather observation grid", ms=16, color='none', markeredgecolor='#433f3f')
    x, y = buffer_circle.exterior.xy
    ax.plot(x, y)
    sx, sy, ex, ey = incident_start.xy + incident_end.xy
    if incident_start == incident_end:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
    else:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
        ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')
    ax.set_xlabel('Longitude')  # ax.set_xlabel('Easting')
    ax.set_ylabel('Latitude')  # ax.set_ylabel('Northing')
    font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
    legend = plt.legend(numpoints=1, loc='best', prop=font, fancybox=True, labelspacing=0.5)
    frame = legend.get_frame()
    frame.set_edgecolor('k')
    plt.tight_layout()


#
def get_incidents_with_weather_2(route_name=None, weather_category='Heat',
                                 season='summer', lp=5 * 24, non_ip=24,
                                 trial=10, random_select=False, illustrate_buf_cir=False, update=False):
    pickle_filename = mssql_metex.make_filename(
        "Incidents-and-Weather", route_name, weather_category,
        "-".join([season] if isinstance(season, str) else season), "trial" if trial else "", "2")
    path_to_pickle = iu.cdd_intermediate(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incidents_and_weather = load_pickle(path_to_pickle)
    else:
        try:
            # Incidents data
            incidents = mssql_metex.view_schedule8_cost_by_datetime_location_reason(route_name, weather_category)
            incidents = iu.get_data_by_season(incidents, season)
            if trial:  # For initial testing ...
                incidents = incidents.iloc[random.sample(range(len(incidents)), trial), :] \
                    if random_select else incidents.iloc[-trial-100:-100, :]

            # --------------------------------------------------------------------------------------------------------
            """ In the spatial context """

            # Create shapely points
            incidents = create_start_end_shapely_points(incidents)

            # Make a buffer zone for gathering data of weather observations
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: create_circle_buffer_for_incident(x.MidLonLat, x.StartLonLat, x.EndLonLat, whisker_km=0.05),
                axis=1)

            # Find all Weather observation grids that intersect with the created buffer zone for each incident location
            incidents['WeatherCells_Obs'] = incidents.Buffer_Zone.map(lambda x: find_intersecting_weather_cells(x))

            if illustrate_buf_cir:
                x_incident_start, x_incident_end = incidents.StartLonLat.iloc[-1], incidents.EndLonLat.iloc[-1]
                x_midpoint = incidents.MidLonLat.iloc[-1]
                illustrate_buffer_circle(x_midpoint, x_incident_start, x_incident_end, whisker_km=0.01)

            # --------------------------------------------------------------------------------------------------------
            """ In the temporal context """

            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
            incidents['Critical_EndDateTime'] = incidents.StartDateTime
            incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
                lambda x: x.replace(hour=0, minute=0, second=0)) - pd.DateOffset(hours=24)

            # Specify the statistics needed for Weather observations (except radiation)
            def specify_weather_stats_calculations():
                weather_stats_calculations = {'Temperature': (max, min, np.average),
                                              'RelativeHumidity': max,
                                              'WindSpeed': max,
                                              'WindGust': max,
                                              'Snowfall': max,
                                              'TotalPrecipitation': max}
                return weather_stats_calculations

            # Calculate average wind speed and direction
            def calculate_wind_averages(wind_speeds, wind_directions):
                u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
                v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
                uav, vav = np.mean(u), np.mean(v)  # sum up all u and v values and average it
                average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed
                # Calculate average wind direction
                if uav == 0:
                    if vav == 0:
                        average_wind_direction = 0
                    else:
                        average_wind_direction = 360 if vav > 0 else 180
                else:
                    if uav > 0:
                        average_wind_direction = 270 - 180 / np.pi * np.arctan(vav / uav)
                    else:
                        average_wind_direction = 90 - 180 / np.pi * np.arctan(vav / uav)
                return average_wind_speed, average_wind_direction

            # Get all Weather variable names
            def specify_weather_variable_names():
                # var_names = db.colnames_db_table('NR_METEX', table_name='Weather')[2:]
                weather_stats_calculations = specify_weather_stats_calculations()
                stats_names = [k + '_max' for k in weather_stats_calculations.keys()]
                stats_names.insert(stats_names.index('Temperature_max') + 1, 'Temperature_min')
                stats_names.insert(stats_names.index('Temperature_min') + 1, 'Temperature_avg')
                stats_names.insert(stats_names.index('Temperature_avg') + 1, 'Temperature_dif')
                wind_speed_variables = ['WindSpeed_avg', 'WindDirection_avg']
                return stats_names + wind_speed_variables + ['Hottest_by_far']

            # Get the highest temperature of year by far
            def get_highest_temperature_of_year_by_far(weather_cell_id, period_start_dt):
                # Whether "max_temp = weather_stats[0]" is the hottest of year so far
                year_start = datetime_truncate.truncate_year(period_start_dt)
                non_ip_weather_obs = mssql_metex.get_weather_by_id_datetime(weather_cell_id, year_start,
                                                                            period_start_dt)
                weather_obs_by_far = non_ip_weather_obs[
                    (non_ip_weather_obs.DateTime < period_start_dt) & (non_ip_weather_obs.DateTime > year_start)]
                highest_temp = weather_obs_by_far.Temperature.max()
                return highest_temp

            # Calculate the statistics for the weather-related variables (except radiation)
            def calculate_weather_stats(weather_obs, weather_stats_calculations, values_only=True):
                if weather_obs.empty:
                    weather_stats = [np.nan] * (sum(map(np.count_nonzero, weather_stats_calculations.values())) + 4)
                    if not values_only:
                        weather_stats = pd.DataFrame(weather_stats, columns=specify_weather_variable_names())
                else:
                    # Create a pseudo id for groupby() & aggregate()
                    weather_obs['Pseudo_ID'] = 0
                    # Calculate basic statistics
                    weather_stats = weather_obs.groupby('Pseudo_ID').aggregate(weather_stats_calculations)
                    # Calculate average wind speeds and directions
                    weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
                        calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)
                    # Lowest temperature between the time of the highest temperature and 00:00
                    highest_temp_dt = weather_obs[
                        weather_obs.Temperature == weather_stats.Temperature['max'][0]].DateTime.min()
                    weather_stats.Temperature['min'] = weather_obs[
                        weather_obs.DateTime < highest_temp_dt].Temperature.min()
                    # Temperature change between the the highest and lowest temperatures
                    weather_stats.insert(3, 'Temperature_dif',
                                         weather_stats.Temperature['max'] - weather_stats.Temperature['min'])
                    # Whether it is the hottest of the year by far
                    weather_cell_id = tuple(weather_obs.WeatherCell.unique())
                    obs_start_dt = weather_obs.DateTime.min()
                    highest_temp = get_highest_temperature_of_year_by_far(weather_cell_id, obs_start_dt)
                    weather_stats['Hottest_by_far'] = 1 if weather_stats.Temperature['max'][0] >= highest_temp else 0
                    weather_stats.columns = specify_weather_variable_names()
                    weather_stats.index.name = None
                    if values_only:
                        weather_stats = weather_stats.values[0].tolist()
                return weather_stats

            # Calculate weather statistics based on the retrieved weather observation data
            def retrieve_weather_stats_for_incidents(weather_cell_id, start_dt, end_dt):
                # Query weather observations
                weather_obs = mssql_metex.get_weather_by_id_datetime(weather_cell_id, start_dt, end_dt)
                # Calculate basic statistics of the weather observations
                weather_stats_calculations = specify_weather_stats_calculations()
                weather_stats = calculate_weather_stats(weather_obs, weather_stats_calculations, values_only=True)
                return weather_stats

            # Prior-IP ---------------------------------------------------
            incidents[specify_weather_variable_names()] = incidents.apply(
                lambda x: pd.Series(retrieve_weather_stats_for_incidents(
                    x.WeatherCells_Obs, x.Critical_StartDateTime, x.Critical_EndDateTime)), axis=1)

            incidents['Incident_Reported'] = 1

            # Non-IP ---------------------------------------------------
            non_ip_data = incidents.copy(deep=True)

            non_ip_data.Critical_EndDateTime = non_ip_data.Critical_StartDateTime - pd.DateOffset(hours=lp)
            non_ip_data.Critical_StartDateTime = non_ip_data.Critical_EndDateTime - pd.DateOffset(hours=non_ip)

            # Gather gridded Weather observations of the corresponding non-incident period for each incident record
            def retrieve_weather_stats_for_non_incidents(weather_cell_id, start_dt, end_dt, stanox_section):
                """
                :param weather_cell_id: weather_cell_id = non_ip_data.WeatherCells_Obs.iloc[0]
                :param start_dt: e.g. start_dt = non_ip_data.Critical_StartDateTime.iloc[0]
                :param end_dt: e.g. end_dt = non_ip_data.Critical_EndDateTime.iloc[0]
                :param stanox_section: e.g. stanox_section = non_ip_data.StanoxSection.iloc[0]
                :return:
                """
                non_ip_weather = mssql_metex.get_weather_by_id_datetime(weather_cell_id, start_dt, end_dt)
                # Get all incident period data on the same section
                critical_period = pd.date_range(start_dt, end_dt, normalize=True)
                ip_overlap = incidents[
                    (incidents.StanoxSection == stanox_section) &
                    (((incidents.Critical_StartDateTime <= min(critical_period)) &
                      (incidents.Critical_EndDateTime >= min(critical_period))) |
                     ((incidents.Critical_StartDateTime <= max(critical_period)) &
                      (incidents.Critical_EndDateTime >= max(critical_period))))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not ip_overlap.empty:
                    non_ip_weather = non_ip_weather[
                        (non_ip_weather.DateTime < min(ip_overlap.Critical_StartDateTime)) |
                        (non_ip_weather.DateTime > max(ip_overlap.Critical_EndDateTime))]
                # Calculate weather statistics
                weather_stats_calculations = specify_weather_stats_calculations()
                weather_stats = calculate_weather_stats(non_ip_weather, weather_stats_calculations, values_only=True)
                return weather_stats

            non_ip_data[specify_weather_variable_names()] = non_ip_data.apply(
                lambda x: pd.Series(retrieve_weather_stats_for_non_incidents(
                    x.WeatherCells_Obs, x.Critical_StartDateTime, x.Critical_EndDateTime, x.StanoxSection)), axis=1)

            non_ip_data['Incident_Reported'] = 0

            incidents_and_weather = pd.concat([incidents, non_ip_data])

            save_pickle(incidents_and_weather, path_to_pickle)

        except Exception as e:
            print(e)
            incidents_and_weather = None

    return incidents_and_weather
