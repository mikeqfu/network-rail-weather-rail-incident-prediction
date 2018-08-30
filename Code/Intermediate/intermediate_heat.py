""" Integrate data of incidents and weather """

import datetime
import itertools
import os

import datetime_truncate
import numpy as np
import pandas as pd
import scipy.stats
import shapely.geometry
import shapely.ops
import fuzzywuzzy.process

import Intermediate.spreadsheet_incidents as inter_incidents
import Intermediate.weather as inter_weather
from utils import cdd, load_pickle, save_pickle, get_angle_of_line_between


# Change directory to "Model\\Heat-Prototype\\..." and sub-directories
def cdd_mod_heat_inter(*directories):
    path = cdd("Model", "Heat-Intermediate", "dat")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Model\\Heat-Prototype\\Trial_" and sub-directories
def cdd_mod_heat_inter_trial(trial_id=0, *directories):
    path = cdd("Model", "Heat-Intermediate", "Trial_{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
def categorise_track_orientations(attr_dat):
    # Angles in radians, [-pi, pi]
    angles = attr_dat.apply(lambda x: get_angle_of_line_between(x.StartNE, x.EndNE), axis=1)
    track_orientations = pd.DataFrame({'Track_Orientation': None}, index=angles.index)

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(angles >= -np.pi * 2 / 3, angles < -np.pi / 3),
        np.logical_and(angles >= np.pi / 3, angles < np.pi * 2 / 3))
    track_orientations[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(angles >= np.pi / 6, angles < np.pi / 3),
        np.logical_and(angles >= -np.pi * 5 / 6, angles < -np.pi * 2 / 3))
    track_orientations[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(angles >= np.pi * 2 / 3, angles < np.pi * 5 / 6),
        np.logical_and(angles >= -np.pi / 3, angles < -np.pi / 6))
    track_orientations[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(angles >= -np.pi, angles < -np.pi * 5 / 6),
    #     np.logical_and(angles >= -np.pi/6, angles < np.pi/6)),
    #     np.logical_and(angles >= np.pi*5/6, angles < np.pi))
    # track_orientations[e_w] = 'E-W'

    categorical_var = pd.get_dummies(track_orientations, prefix='Track_Orientation')

    return track_orientations.join(categorical_var)


# Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30
def categorise_temperatures(attr_dat):

    temp = pd.cut(attr_dat, [-np.inf] + list(np.arange(24, 31)) + [np.inf], right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp})

    categorical_var = pd.get_dummies(temperature_category, 'Maximum_Temperature_max', prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    return temperature_category.join(categorical_var)


# Get data for specified season(s)
def get_data_by_season(m_data, season):
    """
    :param m_data:
    :param season: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :return:
    """
    if season is None:
        return m_data
    else:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4
        for y in pd.unique(m_data.Year):
            data = m_data[m_data.Year == y]
            # Get data for spring -----------------------------------
            spring_start1 = datetime.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = datetime.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring = ((data.StartDateTime >= spring_start1) & (data.StartDateTime < spring_end1)) | \
                     ((data.StartDateTime >= spring_start2) & (data.StartDateTime < spring_end2))
            spring_data = pd.concat([spring_data, data.loc[spring]])
            # Get data for summer ----------------------------------
            summer_start = datetime.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (data.StartDateTime >= summer_start) & (data.StartDateTime < summer_end)
            summer_data = pd.concat([summer_data, data.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = datetime.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (data.StartDateTime >= autumn_start) & (data.StartDateTime < autumn_end)
            autumn_data = pd.concat([autumn_data, data.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = datetime.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (data.StartDateTime >= winter_start) & (data.StartDateTime < winter_end)
            winter_data = pd.concat([winter_data, data.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [fuzzywuzzy.process.extractOne(s, seasons)[0] for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data


# Attach weather conditions for each incident location
def get_incidents_with_weather(trial=False, route=None, weather_category='Heat',
                               prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                               update=False):

    pickle_filename = inter_incidents.make_filename("incidents-with-weather-conditions", route, weather_category)
    path_to_pickle = cdd_mod_heat_inter(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        iw_data = load_pickle(path_to_pickle)
    else:
        try:
            # Incidents data
            incidents_all = inter_incidents.get_schedule8_weather_incidents()
            incidents_all = get_data_by_season(incidents_all, season='summer')
            incidents = inter_incidents.subset(incidents_all, route=route, weather_category=weather_category)
            # For testing purpose ...
            if trial:
                import random
                rand_iloc_idx = random.sample(range(len(incidents)), 10)
                incidents = incidents.iloc[rand_iloc_idx, :]

            # Weather observations
            weather_obs = inter_weather.get_integrated_daily_gridded_weather_obs()
            weather_obs.reset_index(inplace=True)

            # Radiation observations
            irad_data = inter_weather.get_midas_radtob()
            irad_data.reset_index(inplace=True)

            # --------------------------------------------------------------------------------------------------------
            """ In the spatial context """

            # Weather observation grids
            observation_grids = inter_weather.get_observation_grids()
            obs_cen_geom = shapely.geometry.MultiPoint(list(observation_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(observation_grids.Grid))

            # Find the closest grid centroid and return the corresponding (pseudo) grid id
            def find_closest_obs_grid(x, centroids_geom):
                """
                :param x: e.g. incidents.StartNE.iloc[0]
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
                :param start: [shapely.geometry.point.Point] e.g. incidents.StartNE.iloc[0]
                :param end: [shapely.geometry.point.Point] e.g. incidents.EndNE.iloc[0]
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
                :param start: e.g. incidents.StartNE[0]
                :param end: e.g. incidents.EndNE[0]
                :param midpoint: e.g. incidents.MidpointNE[0]
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

            # Make a buffer zone for weather data aggregation
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: create_circle_buffer(x.StartNE, x.EndNE, x.MidpointNE, whisker=500), axis=1)

            # Find all intersecting geom objects
            def find_intersecting_grid(x, grids_geom, as_grid_id=True):
                """
                :param x: e.g. incidents.Buffer_Zone.iloc[0]
                :param grids_geom: i.e. obs_grids_geom
                :param as_grid_id: [bool] whether to return grid id number
                :return:
                """
                intxn_grids = [grid for grid in grids_geom if x.intersects(grid)]
                if as_grid_id:
                    intxn_grids = [observation_grids[observation_grids.Grid == g].index[0] for g in intxn_grids]
                return intxn_grids

            # Find all weather observation grids that intersect with the created buffer zone for each incident location
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(lambda x: find_intersecting_grid(x, obs_grids_geom))

            # Met station locations
            met_stations = inter_weather.get_meteorological_stations()[['E_N', 'E_N_GEOM']]
            met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.E_N_GEOM))

            # Find the closest grid centroid and return the corresponding (pseudo) grid id
            def find_closest_met_stn(x, met_stn_geom):
                """
                :param x: e.g. incidents.MidpointNE.iloc[0]
                :param met_stn_geom: i.e. met_stations_geom
                :return:
                """
                idx = np.where(met_stations.E_N_GEOM == shapely.ops.nearest_points(x, met_stn_geom)[1])[0]
                src_id = met_stations.index[idx].tolist()
                return src_id

            incidents['Met_SRC_ID'] = incidents.MidpointNE.map(lambda x: find_closest_met_stn(x, met_stations_geom))

            """
            # Illustration of the buffer circle
            import matplotlib.pyplot as plt
            start_point, end_point = incidents.StartNE.iloc[0], incidents.EndNE.iloc[0]
            midpoint = incidents.MidpointNE.iloc[0]
            bf_circle = create_circle_buffer(start_point, end_point, midpoint, whisker=500)
            i_obs_grids = find_intersecting_grid(bf_circle, obs_grids_geom, grid_id=False)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            for g in i_obs_grids:
                x, y = g.exterior.xy
                ax.plot(x, y, 'k')
            x, y = bf_circle.exterior.xy
            ax.plot(x, y)
            sx, sy, ex, ey = start_point.xy + end_point.xy
            ax.plot(sx, sy, 'b', ex, ey, 'g', marker='o', markersize=10)
            """

            # --------------------------------------------------------------------------------------------------------
            """ In the temporal context (for the specified IP) """

            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime

            incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
                lambda x: x + pd.Timedelta(days=-1) if x.time() < datetime.time(9) else x)
            incidents.Critical_StartDateTime += datetime.timedelta(hours=prior_ip_start_hrs)
            # Note that the 'Critical_EndDateTime' would be based on the 'Critical_StartDateTime' if we consider the
            # weather conditions on the day of incident occurrence, and 'StartDateTime' otherwise.
            incidents['Critical_EndDateTime'] = incidents.Critical_StartDateTime
            incidents['Critical_Period'] = incidents.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            # --------------------------------------------------------------------------------------------------------
            """ Integration along both the spatial and temporal dimensions (for the specified prior-IP) """

            # Specify the statistics needed for weather observations (except radiation)
            def specify_weather_stats_calculations():
                weather_stats_calculations = {'Maximum_Temperature': (max, min, np.average),
                                              'Minimum_Temperature': (max, min, np.average),
                                              'Temperature_Change': np.average,
                                              'Rainfall': (max, min, np.average)}
                return weather_stats_calculations

            # Calculate the statistics for the weather variables (except radiation)
            def calculate_weather_variables_stats(weather_dat):
                # Specify calculations
                weather_stats_computations = specify_weather_stats_calculations()
                if weather_dat.empty:
                    weather_stats_info = [np.nan] * sum(map(len, weather_stats_computations.values()))
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

            # Gather gridded weather observations of the given period for each incident record
            def integrate_ip_gridded_weather_obs(grids, period):
                """
                :param grids: e.g. grids = incidents.Weather_Grid.iloc[0]
                :param period: e.g. period = incidents.Critical_Period.iloc[0]
                :return:

                if not isinstance(period, list) and isinstance(period, datetime.date):
                    period = [period]
                import itertools
                temp = pd.DataFrame(list(itertools.product(grids, period)), columns=['Pseudo_Grid_ID', 'Date'])

                """
                # Find weather data for the specified period
                prior_ip_weather = weather_obs[weather_obs.Pseudo_Grid_ID.isin(grids) & weather_obs.Date.isin(period)]
                # Calculate the max/min/avg for weather parameters during the period
                weather_stats = calculate_weather_variables_stats(prior_ip_weather)

                # Whether "max_temp = weather_stats[0]" is the hottest of year so far
                obs_by_far = weather_obs[
                    (weather_obs.Date < min(period)) &
                    (weather_obs.Date > datetime_truncate.truncate_year(min(period))) &
                    weather_obs.Pseudo_Grid_ID.isin(grids)]  # Or weather_obs.Date > pd.datetime(min(period).year, 6, 1)
                weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

                return weather_stats

            # Get weather data (except radiation) for the specified IP
            prior_ip_weather_stats = incidents.apply(
                lambda x: pd.Series(integrate_ip_gridded_weather_obs(x.Weather_Grid, x.Critical_Period)), axis=1)

            # Get all weather variable names
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
                    stats_info = [np.nan] * (sum(map(len, radtob_stats_calculations.values())) + 1)
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
                :param met_stn_id: e.g. met_stn_id = incidents.Met_SRC_ID.iloc[1]
                :param period: e.g. period = incidents.Critical_Period.iloc[1]
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

            # Get weather data that did not cause any incident
            non_ip_data = incidents.copy(deep=True)
            # non_ip_data[[c for c in non_ip_data.columns if c.startswith('Incident')]] = None
            # non_ip_data[['StartDateTime', 'EndDateTime', 'WeatherCategoryCode', 'WeatherCategory', 'Minutes']] = None

            non_ip_data.Critical_EndDateTime = \
                non_ip_data.Critical_StartDateTime + datetime.timedelta(days=latent_period)
            non_ip_data.Critical_StartDateTime = \
                non_ip_data.Critical_EndDateTime + datetime.timedelta(hours=non_ip_start_hrs)
            non_ip_data.Critical_Period = non_ip_data.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            # Gather gridded weather observations of the corresponding non-incident period for each incident record
            def integrate_nip_gridded_weather_obs(grids, period, location):
                """
                :param grids: e.g. grids = non_ip_data.Weather_Grid.iloc[0]
                :param period: e.g. period = non_ip_data.Critical_Period.iloc[0]
                :param location: e.g. location = non_ip_data.StanoxSection.iloc[0]
                :return:
                """
                # Get non-IP weather data about where and when the incident occurred
                nip_weather = weather_obs[(weather_obs.Pseudo_Grid_ID.isin(grids)) & (weather_obs.Date.isin(period))]

                # Get all incident period data on the same section
                ip_overlap = prior_ip_data[
                    (prior_ip_data.StanoxSection == location) &
                    (((prior_ip_data.Critical_StartDateTime <= min(period)) &
                      (prior_ip_data.Critical_EndDateTime >= min(period))) |
                     ((prior_ip_data.Critical_StartDateTime <= max(period)) &
                      (prior_ip_data.Critical_EndDateTime >= max(period))))]
                # Skip data of weather causing incidents at around the same time; but
                if not ip_overlap.empty:
                    nip_weather = nip_weather[
                        (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
                        (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
                # Get the max/min/avg weather parameters for those incident periods
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
                # Skip data of weather causing incidents at around the same time; but
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
            iw_data = iw_data.join(categorise_track_orientations(iw_data[['StartNE', 'EndNE']]))

            # Categorise temperature: 25, 26, 27, 28, 29, 30
            iw_data = iw_data.join(categorise_temperatures(iw_data.Maximum_Temperature_max))

            save_pickle(iw_data, path_to_pickle)

        except Exception as e:
            print("Failed to get incidents with weather conditions. {}.".format(e))
            iw_data = pd.DataFrame()

    return iw_data
