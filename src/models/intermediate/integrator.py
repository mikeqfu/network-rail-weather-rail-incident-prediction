import itertools

import numpy as np
import scipy.stats
import shapely.geometry
import shapely.ops

from weather.midas import query_midas_radtob_by_grid_datetime
from weather.ukcp import query_ukcp09_by_grid_datetime, query_ukcp09_by_grid_datetime_


def find_closest_weather_grid(x, obs_grids, obs_centroid_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x: e.g. Incidents.StartNE.iloc[0]
    :param obs_grids:
    :param obs_centroid_geom:
    :return:

    **Example**::

        import copy

        x = incidents.StartXY.iloc[0]
    """

    x_ = shapely.ops.nearest_points(x, obs_centroid_geom)[1]

    pseudo_id = [i for i, y in enumerate(obs_grids.Centroid_XY) if y.equals(x_)]

    return pseudo_id[0]


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

        start = incidents.StartXY.iloc[0]
        end = incidents.EndXY.iloc[0]
        midpoint = incidents.MidpointXY.iloc[0]
    """

    if start == end:
        buffer_circle = start.buffer(2000 + whisker)
    else:
        radius = (start.distance(end) + whisker) / 2
        buffer_circle = midpoint.buffer(radius)
    return buffer_circle


def find_intersecting_weather_grid(x, obs_grids, obs_grids_geom, as_grid_id=True):
    """
    Find all intersecting geom objects.

    :param x:
    :param obs_grids:
    :param obs_grids_geom:
    :param as_grid_id: whether to return grid id number
    :type as_grid_id: bool
    :return:

    **Example**::

        x = incidents.Buffer_Zone.iloc[0]
        as_grid_id = True
    """

    intxn_grids = [grid for grid in obs_grids_geom if x.intersects(grid)]

    if as_grid_id:
        x_ = shapely.ops.cascaded_union(intxn_grids)
        intxn_grids = [i for i, y in enumerate(obs_grids.Grid) if y.within(x_)]

    return intxn_grids


def find_closest_met_stn(x, met_stations, met_stations_geom):
    """
    Find the closest grid centroid and return the corresponding (pseudo) grid id.

    :param x:
    :param met_stations:
    :param met_stations_geom:
    :return:

    **Example**::

        x = incidents.MidpointXY.iloc[0]
    """

    x_1 = shapely.ops.nearest_points(x, met_stations_geom)[1]

    # rest = shapely.geometry.MultiPoint([p for p in met_stations_geom if not p.equals(x_1)])
    # x_2 = shapely.ops.nearest_points(x, rest)[1]
    # rest = shapely.geometry.MultiPoint([p for p in rest if not p.equals(x_2)])
    # x_3 = shapely.ops.nearest_points(x, rest)[1]

    idx = [i for i, y in enumerate(met_stations.EN_GEOM) if y.equals(x_1)]
    src_id = met_stations.index[idx].to_list()

    return src_id


def specify_weather_stats_calculations():
    """
    Specify the statistics needed for Weather observations (except radiation).

    :return:
    """

    weather_stats_calculations = {'Maximum_Temperature': (max, min, np.average),
                                  'Minimum_Temperature': (max, min, np.average),
                                  'Temperature_Change': np.average,
                                  'Precipitation': (max, min, np.average)}

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


def integrate_pip_ukcp09_data(grids, period):
    """
    Gather gridded weather observations of the given period for each incident record.

    :param grids: e.g. grids = incidents.Weather_Grid.iloc[0]
    :param period: e.g. period = incidents.Critical_Period.iloc[0]
    :return:

    **Example**::

        grids = incidents.Weather_Grid.iloc[0]
        period = incidents.Critical_Period.iloc[0]

    """

    # Find Weather data for the specified period
    prior_ip_weather = query_ukcp09_by_grid_datetime(grids, period, pickle_it=True)
    # Calculate the max/min/avg for Weather parameters during the period
    weather_stats = calculate_weather_stats(prior_ip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = query_ukcp09_by_grid_datetime_(grids, period, pickle_it=True)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


def integrate_nip_ukcp09_data(grids, period, prior_ip_data, stanox_section):
    """
    Gather gridded Weather observations of the corresponding non-incident period for each incident record.

    :param grids:
    :param period:
    :param stanox_section:
    :param prior_ip_data:
    :return:

    **Example**::

        grids = non_ip_data.Weather_Grid.iloc[0]
        period = non_ip_data.Critical_Period.iloc[0]
        stanox_section = non_ip_data.StanoxSection.iloc[0]
    """

    # Get non-IP Weather data about where and when the incident occurred
    nip_weather = query_ukcp09_by_grid_datetime(grids, period, pickle_it=True)

    # Get all incident period data on the same section
    ip_overlap = prior_ip_data[
        (prior_ip_data.StanoxSection == stanox_section) &
        (((prior_ip_data.Critical_StartDateTime <= period.min()) &
          (prior_ip_data.Critical_EndDateTime >= period.min())) |
         ((prior_ip_data.Critical_StartDateTime <= period.max()) &
          (prior_ip_data.Critical_EndDateTime >= period.max())))]
    # Skip data of Weather causing Incidents at around the same time; but
    if not ip_overlap.empty:
        nip_weather = nip_weather[
            (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
            (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
    # Get the max/min/avg Weather parameters for those incident periods
    weather_stats = calculate_weather_stats(nip_weather)

    # Whether "max_temp = weather_stats[0]" is the hottest of year so far
    obs_by_far = query_ukcp09_by_grid_datetime_(grids, period, pickle_it=True)
    weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

    return weather_stats


def specify_radtob_stats_calculations():
    """
    Specify the statistics needed for radiation only.

    :return:
    """

    radtob_stats_calculations = {'GLBL_IRAD_AMT': (max, scipy.stats.iqr)}
    return radtob_stats_calculations


def calculate_radtob_variables_stats(radtob_dat):
    """
    Calculate the statistics for the radiation variables.

    :param radtob_dat:
    :return:
    """

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
def integrate_pip_midas_radtob(met_stn_id, period):
    """
    :param met_stn_id:
    :param period:
    :return:

    **Example**::

        met_stn_id = incidents.Met_SRC_ID.iloc[4]
        period = incidents.Critical_Period.iloc[4]
    """

    # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
    #
    # try:
    #     prior_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
    # except KeyError:
    #     prior_ip_radtob = pd.DataFrame()

    prior_ip_radtob = query_midas_radtob_by_grid_datetime(met_stn_id, period, pickle_it=True)

    radtob_stats = calculate_radtob_variables_stats(prior_ip_radtob)

    return radtob_stats


def integrate_nip_midas_radtob(met_stn_id, period, prior_ip_data, stanox_section):
    """
    Gather solar radiation of the corresponding non-incident period for each incident record.

    :param met_stn_id: e.g. met_stn_id = non_ip_data.Met_SRC_ID.iloc[1]
    :param period: e.g. period = non_ip_data.Critical_Period.iloc[1]
    :param stanox_section: e.g. location = non_ip_data.StanoxSection.iloc[0]
    :param prior_ip_data:
    :return:
    """

    # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
    #
    # try:
    #     non_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
    # except KeyError:
    #     non_ip_radtob = pd.DataFrame()

    non_ip_radtob = query_midas_radtob_by_grid_datetime(met_stn_id, period, pickle_it=True)

    # Get all incident period data on the same section
    ip_overlap = prior_ip_data[
        (prior_ip_data.StanoxSection == stanox_section) &
        (((prior_ip_data.Critical_StartDateTime <= period.min()) &
          (prior_ip_data.Critical_EndDateTime >= period.min())) |
         ((prior_ip_data.Critical_StartDateTime <= period.max()) &
          (prior_ip_data.Critical_EndDateTime >= period.max())))]
    # Skip data of Weather causing Incidents at around the same time; but
    if not ip_overlap.empty:
        non_ip_radtob = non_ip_radtob[
            (non_ip_radtob.OB_END_DATE < min(ip_overlap.Critical_StartDateTime)) |
            (non_ip_radtob.OB_END_DATE > max(ip_overlap.Critical_EndDateTime))]

    radtob_stats = calculate_radtob_variables_stats(non_ip_radtob)

    return radtob_stats
