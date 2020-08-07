import re

import numpy as np
import pandas as pd

from models.tools import calculate_wind_averages


def specify_weather_stats_calculations():
    """
    Specify the weather statistics that need to be computed.

    :return: a dictionary for calculations of weather statistics
    :rtype: dict
    """

    weather_stats_calculations = {'Temperature': (np.nanmax, np.nanmin, np.nanmean),
                                  'RelativeHumidity': (np.nanmax, np.nanmin, np.nanmean),
                                  'WindSpeed': np.nanmax,
                                  'WindGust': np.nanmax,
                                  'Snowfall': (np.nanmax, np.nanmin, np.nanmean),
                                  'TotalPrecipitation': (np.nanmax, np.nanmin, np.nanmean)}

    return weather_stats_calculations


def specify_vegetation_stats_calculations(features):
    """
    Specify the statistics that need to be computed.

    :param features:
    :type features:
    :return:
    :rtype:
    """

    # "CoverPercent..."
    cover_percents = [x for x in features if re.match('^CoverPercent[A-Z]', x)]
    veg_stats_calc = dict(zip(cover_percents, [np.nansum] * len(cover_percents)))
    veg_stats_calc.update({'AssetNumber': np.count_nonzero,
                           'TreeNumber': np.nansum,
                           'TreeNumberUp': np.nansum,
                           'TreeNumberDown': np.nansum,
                           'Electrified': np.any,
                           'DateOfMeasure': lambda x: tuple(x),
                           # 'AssetDesc1': np.all,
                           # 'IncidentReported': np.any
                           'HazardTreeNumber': lambda x: np.nan if np.isnan(x).all() else np.nansum(x)})

    # variables for hazardous trees
    hazard_min = [x for x in features if re.match('^HazardTree.*min$', x)]
    hazard_max = [x for x in features if re.match('^HazardTree.*max$', x)]
    hazard_rest = [x for x in features if re.match('^HazardTree[a-z]((?!_).)*$', x)]
    # Computations for hazardous trees variables
    hazard_calc = [dict(zip(hazard_rest, [lambda x: tuple(x)] * len(hazard_rest))),
                   dict(zip(hazard_min, [np.min] * len(hazard_min))),
                   dict(zip(hazard_max, [np.max] * len(hazard_max)))]

    # Update vegetation_stats_computations
    veg_stats_calc.update({k: v for d in hazard_calc for k, v in d.items()})

    return cover_percents, hazard_rest, veg_stats_calc


def get_weather_variable_names(weather_stats_calculations, temperature_dif=False, supplement=None):
    """
    Get weather variable names.

    :param weather_stats_calculations:
    :type weather_stats_calculations: dict
    :param temperature_dif: whether to include 'Temperature_dif', defaults to ``False``
    :type temperature_dif: bool
    :param supplement: e.g. 'Hottest_Heretofore'
    :type supplement: str, list, None
    :return: a list of names of weather variables
    :rtype: list
    """

    weather_variable_names = []
    for k, v in weather_stats_calculations.items():
        if isinstance(v, tuple):
            for v_ in v:
                weather_variable_names.append('_'.join(
                    [k, v_.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))
        else:
            weather_variable_names.append('_'.join(
                [k, v.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))
    if temperature_dif:
        weather_variable_names.insert(weather_variable_names.index('Temperature_min') + 1, 'Temperature_dif')

    if supplement:
        if isinstance(supplement, str):
            supplement = [supplement]
        wind_variable_names_ = weather_variable_names + ['WindSpeed_avg', 'WindDirection_avg'] + supplement
    else:
        wind_variable_names_ = weather_variable_names + ['WindSpeed_avg', 'WindDirection_avg']

    return wind_variable_names_


def calculate_overall_cover_percent_old(start_and_end_cover_percents, total_yards_adjusted):
    """
    Calculate the cover percents across two neighbouring ELRs.

    :param start_and_end_cover_percents:
    :type start_and_end_cover_percents: tuple
    :param total_yards_adjusted:
    :type total_yards_adjusted: tuple
    :return:
    :rtype:
    """

    # (start * end) / (start + end)
    multiplier = pd.np.prod(total_yards_adjusted) / pd.np.sum(total_yards_adjusted)
    # 1/start, 1/end
    cp_start, cp_end = start_and_end_cover_percents
    s_, e_ = pd.np.divide(1, total_yards_adjusted)
    # numerator
    n = e_ * cp_start + s_ * cp_end
    # denominator
    d = pd.np.sum(start_and_end_cover_percents) if pd.np.all(start_and_end_cover_percents) else 1
    f = multiplier * pd.np.divide(n, d)
    overall_cover_percent = f * d
    return overall_cover_percent


def calculate_prototype_weather_statistics(weather_obs, weather_stats_calculations):
    """
    Compute the statistics for all the Weather variables (except wind).

    :param weather_obs:
    :type weather_obs:
    :param weather_stats_calculations:
    :type weather_stats_calculations: dict
    :return:
    :rtype:

    .. note::

        Note: to get the n-th percentile, use percentile(n)

        This function also returns the Weather dataframe indices. The corresponding Weather conditions in that Weather
        cell might cause wind-related Incidents.
    """

    if not weather_obs.empty:
        # Calculate the statistics
        weather_stats = weather_obs.groupby('WeatherCell').aggregate(weather_stats_calculations)
        weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
            calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)
        stats_info = weather_stats.values[0].tolist()  # + [weather_obs.index.tolist()]
    else:
        stats_info = [np.nan] * 10  # + [[None]]
    return stats_info
