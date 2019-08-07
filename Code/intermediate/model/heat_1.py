import datetime
import itertools
import os

import datetime_truncate
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import shapely.geometry
import shapely.ops
import sklearn
import sklearn.metrics
import statsmodels.discrete.discrete_model as sm_dcm
import statsmodels.tools as sm_tools
from pyhelpers.store import load_pickle, save_fig, save_pickle

import intermediate.tools
import settings
import spreadsheet.incidents
import weather.midas
import weather.ukcp
from utils import get_subset, make_filename

settings.pd_preferences()

# ====================================================================================================================
""" Change directory """


def cdd_intermediate_heat(*sub_dir):
    path = intermediate.tools.cdd_intermediate("heat", "dat")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Models\\intermediate\\heat\\x" and sub-directories
def cdd_intermediate_heat_mod(trial_id, *sub_dir):
    path = intermediate.tools.cdd_intermediate("heat", "{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Integrate data of Incidents and Weather """


# Attach Weather conditions for each incident location
def get_incident_location_weather(route_name=None, weather_category='Heat', season='summer',
                                  prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                                  trial=True, illustrate_buf_cir=False, update=False):
    """
    Testing parameters:
    e.g.
        route_name=None
        weather_category='Heat'
        season='summer'
        prior_ip_start_hrs=-0
        latent_period=-5
        non_ip_start_hrs=-0
        trial=True
        illustrate_buf_cir=False
        update=False
    """
    pickle_filename = make_filename("incident_location_weather", route_name, weather_category,
                                    "_".join([season] if isinstance(season, str) else season),
                                    "trial" if trial else "full")
    path_to_pickle = cdd_intermediate_heat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_location_weather = load_pickle(path_to_pickle)
    else:
        try:
            # Incidents data
            incidents_all = spreadsheet.incidents.get_schedule8_weather_incidents()
            incidents_all.rename(columns={'Year': 'FinancialYear'}, inplace=True)
            incidents_all_by_season = intermediate.tools.get_data_by_season(incidents_all, season)
            incidents = get_subset(incidents_all_by_season, route_name, weather_category)

            if trial:  # For testing purpose ...
                incidents = incidents.iloc[0:10, :]

            # Weather observations
            weather_obs = weather.ukcp.fetch_integrated_daily_gridded_weather_obs().reset_index()

            # Radiation observations
            irad_obs = weather.midas.fetch_midas_radtob().reset_index()

            # --------------------------------------------------------------------------------------------------------
            """ In the spatial context """

            # Weather observation grids
            observation_grids = weather.ukcp.fetch_observation_grids()
            obs_cen_geom = shapely.geometry.MultiPoint(list(observation_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(observation_grids.Grid))

            # Find the closest grid centroid and return the corresponding (pseudo) grid id
            def find_closest_weather_grid(x, centroids_geom):
                """
                :param x: e.g. Incidents.StartNE.iloc[0]
                :param centroids_geom: i.e. obs_cen_geom
                :return:
                """
                pseudo_id = np.where(observation_grids.Centroid_XY == shapely.ops.nearest_points(x, centroids_geom)[1])
                return pseudo_id[0][0]

            # Start
            incidents['Start_Pseudo_Grid_ID'] = incidents.StartNE.map(
                lambda x: find_closest_weather_grid(x, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='Start_Pseudo_Grid_ID')
            # End
            incidents['End_Pseudo_Grid_ID'] = incidents.EndNE.map(
                lambda x: find_closest_weather_grid(x, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')
            # Modify column names
            for l in ['Start', 'End']:
                a = [c for c in incidents.columns if c.endswith(l)]
                b = [l + '_' + c if c == 'Grid' else l + '_Grid_' + c for c in observation_grids.columns]
                incidents.rename(columns=dict(zip(a, b)), inplace=True)

            # Get midpoint between two points
            def find_midpoint(start, end, as_geom=True):
                """
                :param start: [shapely.geometry.point.Point] e.g. incidents.StartEN.iloc[0]
                :param end: [shapely.geometry.point.Point] e.g. incidents.EndEN.iloc[0]
                :param as_geom: [bool] whether to return a shapely.geometry obj
                :return:
                """
                midpoint = (start.x + end.x) / 2, (start.y + end.y) / 2
                return shapely.geometry.Point(midpoint) if as_geom else midpoint

            # Append 'MidpointNE' column
            incidents['MidpointNE'] = incidents.apply(lambda x: find_midpoint(x.StartNE, x.EndNE, as_geom=True), axis=1)

            # Create a circle buffer for start location
            def create_circle_buffer_upon_weather_grid(start, end, midpoint, whisker=500):
                """
                :param start: e.g. incidents.StartNE[0]
                :param end: e.g. incidents.EndNE[0]
                :param midpoint: e.g. incidents.MidpointNE[0]
                :param whisker: An extended length to the diameter (i.e. on both sides of the start and end locations)
                :return: [shapely.geometry.Polygon]
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
                lambda x: create_circle_buffer_upon_weather_grid(x.StartNE, x.EndNE, x.MidpointNE, whisker=500), axis=1)

            # Find all intersecting geom objects
            def find_intersecting_weather_grid(x, grids_geom, as_grid_id=True):
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
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(
                lambda x: find_intersecting_weather_grid(x, obs_grids_geom))

            # Met station locations
            met_stations = weather.midas.fetch_meteorological_stations_locations()[['E_N', 'E_N_GEOM']]
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
                    bf_circle = create_circle_buffer_upon_weather_grid(start_point, end_point, midpoint, whisker=500)
                    i_obs_grids = find_intersecting_weather_grid(bf_circle, obs_grids_geom, as_grid_id=False)
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
                prior_ip_radtob = irad_obs[irad_obs.SRC_ID.isin(met_stn_id) & irad_obs.OB_END_DATE.isin(period)]
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
                non_ip_radtob = irad_obs[irad_obs.SRC_ID.isin(met_stn_id) & irad_obs.OB_END_DATE.isin(period)]

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
                lambda x: pd.Series(integrate_nip_midas_radtob(x.Met_SRC_ID, x.Critical_Period, x.StanoxSection)),
                axis=1)

            non_ip_radtob_stats.columns = r_col_names

            non_ip_data = non_ip_data.join(non_ip_radtob_stats)

            non_ip_data['Incident_Reported'] = 0

            # Merge "ip_data" and "nip_data" into one DataFrame
            incident_location_weather = pd.concat([prior_ip_data, non_ip_data], axis=0, ignore_index=True, sort=False)

            # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
            incident_location_weather = incident_location_weather.join(
                intermediate.tools.categorise_track_orientations(incident_location_weather[['StartNE', 'EndNE']]))

            # Categorise temperature: 25, 26, 27, 28, 29, 30
            incident_location_weather = incident_location_weather.join(
                intermediate.tools.categorise_temperatures(incident_location_weather.Maximum_Temperature_max))

            save_pickle(incident_location_weather, path_to_pickle)

        except Exception as e:
            print("Failed to get Incidents with Weather conditions. {}.".format(e))
            incident_location_weather = pd.DataFrame()

    return incident_location_weather


# ====================================================================================================================
""" Testing models """


def specify_explanatory_variables():
    return [
        # 'Maximum_Temperature_max',
        # 'Maximum_Temperature_min',
        # 'Maximum_Temperature_average',
        # 'Minimum_Temperature_max',
        # 'Minimum_Temperature_min',
        # 'Minimum_Temperature_average',
        # 'Temperature_Change_average',
        'Rainfall_max',
        # 'Rainfall_min',
        # 'Rainfall_average',
        'Hottest_Heretofore',
        'Temperature_Change_max',
        # 'Temperature_Change_min',
        # 'GLBL_IRAD_AMT_max',
        # 'GLBL_IRAD_AMT_iqr',
        # 'GLBL_IRAD_AMT_total',
        # 'Track_Orientation_E_W',
        'Track_Orientation_NE_SW',
        'Track_Orientation_NW_SE',
        'Track_Orientation_N_S',
        # 'Maximum_Temperature_max [-inf, 24.0)°C',
        'Maximum_Temperature_max [24.0, 25.0)°C',
        'Maximum_Temperature_max [25.0, 26.0)°C',
        'Maximum_Temperature_max [26.0, 27.0)°C',
        'Maximum_Temperature_max [27.0, 28.0)°C',
        'Maximum_Temperature_max [28.0, 29.0)°C',
        'Maximum_Temperature_max [29.0, 30.0)°C',
        'Maximum_Temperature_max [30.0, inf)°C'
    ]


# Describe basic statistics about the main explanatory variables
def describe_explanatory_variables(train_set, save_as=".pdf", dpi=None):
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

    if save_as == ".svg":
        save_fig(intermediate.tools.cd_intermediate_fig_pub("Variables" + save_as), dpi)


#
def logistic_regression_model(trial_id,
                              route=None, weather_category='Heat', season='summer',
                              prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                              outlier_pctl=100,
                              describe_var=False,
                              add_const=True, seed=0, model='logit',
                              plot_roc=False, plot_predicted_likelihood=False,
                              save_as=".svg", dpi=None,
                              verbose=True):
    """
    Testing parameters:
    e.g.
        trial_id
        route=None
        weather_category='Heat'
        season='summer'
        prior_ip_start_hrs=-0
        latent_period=-5
        non_ip_start_hrs=-0
        outlier_pctl=100
        describe_var=False
        add_const=True
        seed=0
        model='logit',
        plot_roc=False
        plot_predicted_likelihood=False
        save_as=".tif"
        dpi=None
        verbose=True

    IncidentReason  IncidentReasonName   IncidentReasonDescription

    IQ              TRACK SIGN           Trackside sign blown down/light out etc.
    IW              COLD                 Non severe - Snow/Ice/Frost affecting infrastructure equipment',
                                         'Takeback Pumps'
    OF              HEAT/WIND            Blanket speed restriction for extreme heat or high wind in accordance with
                                         the Group Standards
    Q1              TKB PUMPS            Takeback Pumps
    X4              BLNK REST            Blanket speed restriction for extreme heat or high wind
    XW              WEATHER              Severe Weather not snow affecting infrastructure the responsibility of NR
    XX              MISC OBS             Msc items on line (incl trees) due to effects of Weather responsibility of RT

    """
    # Get the m_data for modelling
    m_data = get_incident_location_weather(route, weather_category, season,
                                           prior_ip_start_hrs, latent_period, non_ip_start_hrs)

    # temp_data = [load_pickle(cdd_mod_heat_inter("Slices", f)) for f in os.listdir(cdd_mod_heat_inter("Slices"))]
    # m_data = pd.concat(temp_data, ignore_index=True, sort=False)

    m_data.dropna(subset=['GLBL_IRAD_AMT_max', 'GLBL_IRAD_AMT_iqr', 'GLBL_IRAD_AMT_total'], inplace=True)

    # Select features
    explanatory_variables = specify_explanatory_variables()

    for v in explanatory_variables:
        if not m_data[m_data[v].isna()].empty:
            m_data.dropna(subset=[v], inplace=True)

    m_data = m_data[explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime', 'Minutes']]

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        m_data = m_data[m_data.Minutes <= np.percentile(m_data.Minutes, outlier_pctl)]

    # Add the intercept
    if add_const:
        m_data = sm_tools.tools.add_constant(m_data, prepend=True, has_constant='skip')  # data['const'] = 1.0
        explanatory_variables = ['const'] + explanatory_variables

    # Select data before 2014 as training data set, with the rest being test set
    train_set = m_data[m_data.StartDateTime < pd.datetime(2013, 1, 1)]
    test_set = m_data[m_data.StartDateTime >= pd.datetime(2013, 1, 1)]

    if describe_var:
        describe_explanatory_variables(train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)
    try:
        if model == 'logit':
            mod = sm_dcm.Logit(train_set.Incident_Reported, train_set[explanatory_variables])
        else:
            mod = sm_dcm.Probit(train_set.Incident_Reported, train_set[explanatory_variables])
        result = mod.fit(maxiter=1000, full_output=True, disp=True)  # method='newton'
        print(result.summary()) if verbose else print("")

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['OddsRatio'])
        print("\n{}".format(odds_ratios)) if verbose else print("")

        # Prediction
        test_set['incident_prob'] = result.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        fpr, tpr, thr = sklearn.metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = sklearn.metrics.auc(fpr, tpr)
        ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
        threshold = np.min(thr[ind])

        # prediction accuracy
        test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= threshold else 0)
        test = pd.Series(test_set.Incident_Reported == test_set.incident_prediction)
        mod_accuracy = np.divide(test.sum(), len(test))
        print("\nAccuracy: %f" % mod_accuracy) if verbose else print("")

        # incident prediction accuracy
        incident_only = test_set[test_set.Incident_Reported == 1]
        test_acc = pd.Series(incident_only.Incident_Reported == incident_only.incident_prediction)
        incident_accuracy = np.divide(test_acc.sum(), len(test_acc))
        print("Incident accuracy: %f" % incident_accuracy) if verbose else print("")

        if plot_roc:
            plt.figure()
            plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc, color='#6699cc', lw=2.5)
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Random guess")
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False positive rate", fontsize=14, fontweight='bold')
            plt.ylabel("True positive rate", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            # plt.title('Receiver operating characteristic example')
            plt.legend(loc='lower right', fontsize=14)
            plt.fill_between(fpr, tpr, 0, color='#6699cc', alpha=0.2)
            plt.tight_layout()
            if save_as:
                save_fig(cdd_intermediate_heat_mod(trial_id, "ROC" + save_as), dpi=dpi)

        # Plot incident delay minutes against predicted probabilities
        if plot_predicted_likelihood:
            incident_ind = test_set.Incident_Reported == 1
            plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            ax.scatter(test_set[incident_ind].incident_prob, test_set[incident_ind].Minutes,
                       c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
                       label="Heat-related incident (2014/15)")
            plt.axvline(x=threshold, label="Threshold: %.2f" % threshold, color='#e5c100', linewidth=2)
            legend = plt.legend(scatterpoints=1, loc=2, fontsize=14, fancybox=True, labelspacing=0.6)
            frame = legend.get_frame()
            frame.set_edgecolor('k')
            plt.xlim(xmin=0, xmax=1.03)
            plt.ylim(ymin=-15)
            ax.set_xlabel("Likelihood of heat-related incident occurrence", fontsize=14, fontweight='bold')
            ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            if save_as:
                save_fig(cdd_intermediate_heat_mod(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_accuracy, incident_accuracy, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_accuracy, incident_accuracy, threshold
