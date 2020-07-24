import itertools

import datetime_truncate
import numpy as np
import scipy.stats
import shapely.geometry
import shapely.ops


def find_closest_weather_grid(x, observation_grids, centroids_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.StartNE.iloc[0]
    :param observation_grids:
    :param centroids_geom: i.e. obs_cen_geom
    :return:
    """

    pseudo_id = np.where(observation_grids.Centroid_XY == shapely.ops.nearest_points(x, centroids_geom)[1])
    return pseudo_id[0][0]


def create_circle_buffer_upon_weather_grid(start, end, midpoint, whisker=500):
    """
    Create a circle buffer for start/end location.

    :param start:
    :type start: shapely.geometry.Point
    :param end:
    :type end: shapely.geometry.Point
    :param midpoint:
    :type midpoint: shapely.geometry.Point
    :param whisker: extended length on both sides of the start and end locations, defaults to ``500``
    :type whisker: int
    :return: a buffer zone
    :rtype: shapely.geometry.Polygon

    **Example**::

        whisker = 0

        start = incidents.StartNE.iloc[0]
        end = incidents.EndNE.iloc[0]
        midpoint = incidents.MidpointNE.iloc[0]
    """

    if start == end:
        buffer_circle = start.buffer(2000 + whisker)
    else:
        radius = (start.distance(end) + whisker) / 2
        buffer_circle = midpoint.buffer(radius)
    return buffer_circle


def find_intersecting_weather_grid(x, observation_grids, grids_geom, as_grid_id=True):
    """
    Find all intersecting geom objects.

    :param x: e.g. Incidents.Buffer_Zone.iloc[0]
    :param observation_grids:
    :param grids_geom: i.e. obs_grids_geom
    :param as_grid_id: [bool] whether to return grid id number
    :return:
    """

    intxn_grids = [grid for grid in grids_geom if x.intersects(grid)]
    if as_grid_id:
        intxn_grids = [observation_grids[observation_grids.Grid == g].index[0] for g in intxn_grids]
    return intxn_grids


def find_closest_met_stn(x, met_stations, met_stations_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.MidpointNE.iloc[0]
    :param met_stations:
    :param met_stations_geom:
    :return:
    """

    idx = np.where(met_stations.E_N_GEOM == shapely.ops.nearest_points(x, met_stations_geom)[1])[0]
    src_id = met_stations.index[idx].tolist()
    return src_id


def specify_weather_stats_calculations():
    """
    Specify the statistics needed for Weather observations (except radiation).

    :return:
    """

    weather_stats_calculations = {'Maximum_Temperature': (max, min, np.average),
                                  'Minimum_Temperature': (max, min, np.average),
                                  'Temperature_Change': np.average,
                                  'Rainfall': (max, min, np.average)}
    return weather_stats_calculations


def specify_weather_variable_names(calc):
    """
    Get all weather variable names.

    :param calc:
    :return:
    """

    var_stats_names = [[k, [i.__name__ for i in v] if isinstance(v, tuple) else [v.__name__]]
                       for k, v in calc.items()]
    weather_variable_names = [['_'.join([x, z]) for z in y] for x, y in var_stats_names]
    weather_variable_names = list(itertools.chain.from_iterable(weather_variable_names))
    return weather_variable_names


def calculate_weather_stats(weather_data):
    """
    Calculate the statistics for the Weather variables (except radiation).

    :param weather_data:
    :return:
    """

    # Specify calculations
    weather_stats_computations = specify_weather_stats_calculations()
    if weather_data.empty:
        weather_stats_info = [np.nan] * sum(map(np.count_nonzero, weather_stats_computations.values()))
    else:
        # Create a pseudo id for groupby() & aggregate()
        weather_data['Pseudo_ID'] = 0
        weather_stats = weather_data.groupby('Pseudo_ID').aggregate(weather_stats_computations)
        # a, b = [list(x) for x in weather_stats.columns.levels]
        # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
        # if not weather_stats.empty:
        #     stats_info = weather_stats.values[0].tolist()
        # else:
        #     stats_info = [np.nan] * len(weather_stats.columns)
        weather_stats_info = weather_stats.values[0].tolist()
    return weather_stats_info


def integrate_pip_gridded_weather_obs(grids, period, weather_obs):
    """
    Gather gridded weather observations of the given period for each incident record.

    :param grids: e.g. grids = Incidents.Weather_Grid.iloc[0]
    :param period: e.g. period = Incidents.Critical_Period.iloc[0]
    :param weather_obs:
    :return:

    if not isinstance(period, list) and isinstance(period, datetime.date):
        period = [period]
    import itertools
    temp = pd.DataFrame(list(itertools.product(grids, period)), columns=['Pseudo_Grid_ID', 'Date'])

    """
    # Find Weather data for the specified period
    prior_ip_weather = weather_obs[weather_obs.Pseudo_Grid_ID.isin(grids) & weather_obs.Date.isin(period)]
    # Calculate the max/min/avg for Weather parameters during the period
    weather_stats = calculate_weather_stats(prior_ip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = weather_obs[
        (weather_obs.Date < min(period)) &
        (weather_obs.Date > datetime_truncate.truncate_year(min(period))) &
        weather_obs.Pseudo_Grid_ID.isin(grids)]  # Or weather_obs.Date > pd.datetime(min(period).year, 6, 1)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


def integrate_nip_gridded_weather_obs(grids, period, stanox_section, weather_obs, prior_ip_data):
    """
    Gather gridded Weather observations of the corresponding non-incident period for each incident record.

    :param grids: e.g. grids = non_ip_data.Weather_Grid.iloc[12]
    :param period: e.g. period = non_ip_data.Critical_Period.iloc[12]
    :param stanox_section: e.g. stanox_section = non_ip_data.StanoxSection.iloc[12]
    :param weather_obs:
    :param prior_ip_data:
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
    weather_stats = calculate_weather_stats(nip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = weather_obs[
        (weather_obs.Date < min(period)) &
        (weather_obs.Date > datetime_truncate.truncate_year(min(period))) &
        weather_obs.Pseudo_Grid_ID.isin(grids)]  # Or weather_obs.Date > pd.datetime(min(period).year, 6, 1)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


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
def integrate_pip_midas_radtob(met_stn_id, period, irad_obs):
    """
    :param met_stn_id: e.g. met_stn_id = Incidents.Met_SRC_ID.iloc[1]
    :param period: e.g. period = Incidents.Critical_Period.iloc[1]
    :param irad_obs:
    :return:
    """
    prior_ip_radtob = irad_obs[irad_obs.SRC_ID.isin(met_stn_id) & irad_obs.OB_END_DATE.isin(period)]
    radtob_stats = calculate_radtob_variables_stats(prior_ip_radtob)
    return radtob_stats


def integrate_nip_midas_radtob(met_stn_id, period, stanox_section, irad_obs, prior_ip_data):
    """
    Gather solar radiation of the corresponding non-incident period for each incident record.

    :param met_stn_id: e.g. met_stn_id = non_ip_data.Met_SRC_ID.iloc[1]
    :param period: e.g. period = non_ip_data.Critical_Period.iloc[1]
    :param stanox_section: e.g. location = non_ip_data.StanoxSection.iloc[0]
    :param irad_obs:
    :param prior_ip_data:
    :return:
    """

    non_ip_radtob = irad_obs[irad_obs.SRC_ID.isin(met_stn_id) & irad_obs.OB_END_DATE.isin(period)]

    # Get all incident period data on the same section
    ip_overlap = prior_ip_data[
        (prior_ip_data.StanoxSection == stanox_section) &
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
