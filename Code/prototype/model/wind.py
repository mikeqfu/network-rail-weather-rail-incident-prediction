""" A prototype model in the context of wind-related Incidents """

import datetime
import itertools
import os
import re
import time

import datetime_truncate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils.extmath
import statsmodels.discrete.discrete_model as sm
from pyhelpers.dir import cdd
from pyhelpers.misc import get_variable_names
from pyhelpers.store import load_pickle, save_pickle, save_svg_as_emf

import mssqlserver.metex
import mssqlserver.vegetation
import prototype.furlong
import prototype.tools
import settings

# Apply the preferences ==============================================================================================
settings.mpl_preferences(use_cambria=True, reset=False)
settings.pd_preferences(reset=False)

# ====================================================================================================================
""" Change directory """


# Change directory to "Models\\prototype\\wind\\data" and sub-directories
def cdd_prototype_wind(*sub_dir):
    path = prototype.tools.cdd_prototype("wind", "data")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Models\\prototype\\wind\\{}" and sub-directories
def cdd_prototype_wind_mod(trial_id, *sub_dir):
    path = prototype.tools.cdd_prototype("wind", "{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Calculations for weather data """


# Specify the statistics that need to be computed
def specify_weather_stats_calculations():
    weather_stats_calculations = {'Temperature': (np.nanmax, np.nanmin, np.nanmean),
                                  'RelativeHumidity': np.nanmax,
                                  'WindSpeed': np.nanmax,
                                  'WindGust': np.nanmax,
                                  'Snowfall': np.nanmax,
                                  'TotalPrecipitation': np.nanmax}
    return weather_stats_calculations


# Get TRUST data and the weather conditions for each incident location
def get_incident_location_weather(route_name='Anglia', weather_category='Wind',
                                  ip_start_hrs=-12, ip_end_hrs=12, nip_start_hrs=-12,
                                  update=False):
    """
    :param route_name: [str] Route name
    :param weather_category: [str] Weather category
    :param ip_start_hrs: [int; float]
    :param ip_end_hrs: [int; float]
    :param nip_start_hrs: [int; float]
    :param update: [bool]
    :return: [DataFrame]

    e.g.

    route_name='Anglia'
    weather_category='Wind'
    ip_start_hrs=-12
    ip_end_hrs=12
    nip_start_hrs=-12
    update=False

    When offset the date and time data, an alternative function for "pd.DateOffset" is "datetime.timedelta"
    """

    pickle_filename = mssqlserver.metex.make_filename("incident_location_weather", route_name, weather_category,
                                                      ip_start_hrs, ip_end_hrs, nip_start_hrs, save_as=".pickle")
    path_to_pickle = cdd_prototype_wind(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)
    else:
        try:
            # Getting Weather data for all incident locations
            incidents = mssqlserver.metex.view_schedule8_costs_by_datetime_location_reason(
                route_name, weather_category, update)
            # Drop non-weather-related incident records
            incidents = incidents[incidents.WeatherCategory != ''] if weather_category is None else incidents
            # Get data for the specified "Incident Periods"
            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
            incidents['Critical_StartDateTime'] = \
                incidents.StartDateTime.apply(datetime_truncate.truncate_hour) + datetime.timedelta(hours=ip_start_hrs)
            incidents['Critical_EndDateTime'] = \
                incidents.EndDateTime.apply(datetime_truncate.truncate_hour) + datetime.timedelta(hours=ip_end_hrs)
            incidents['Critical_Period'] = incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

            weather_stats_calculations = specify_weather_stats_calculations()

            # Processing Weather data for IP - Get data of Weather conditions which led to Incidents for each record
            def get_weather_stats_for_ip(weather_cell_id, ip_start, ip_end) -> list:
                """
                :param weather_cell_id: [int] Weather Cell ID
                :param ip_start: [Timestamp] start of "incident period"
                :param ip_end: [Timestamp] end of "incident period"
                :return: [list] a list of statistics

                e.g.

                weather_cell_id=incidents.WeatherCell[3819210]
                ip_start=incidents.StartDateTime[3819210]
                ip_end=incidents.EndDateTime[3819210]

                """
                # Get Weather data about where and when the incident occurred
                ip_weather_obs = mssqlserver.metex.view_weather_by_id_datetime(weather_cell_id, ip_start, ip_end,
                                                                               pickle_it=False)
                # Get the max/min/avg Weather parameters for those incident periods
                weather_stats = prototype.tools.calculate_statistics_for_weather_variables(
                    ip_weather_obs, weather_stats_calculations)
                return weather_stats

            # Get data for the specified IP
            ip_statistics = incidents.apply(
                lambda x: get_weather_stats_for_ip(x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime),
                axis=1)
            ip_statistics = pd.DataFrame(ip_statistics.to_list(), index=ip_statistics.index,
                                         columns=prototype.tools.get_weather_variable_names(weather_stats_calculations))
            ip_statistics['Temperature_diff'] = ip_statistics.Temperature_max - ip_statistics.Temperature_min

            #
            ip_data = incidents.join(ip_statistics.dropna(), how='inner')
            ip_data['IncidentReported'] = 1

            # Processing Weather data for non-IP
            nip_data = incidents.copy(deep=True)
            nip_data.Critical_EndDateTime = nip_data.Critical_StartDateTime  # + datetime.timedelta(hours=0)
            nip_data.Critical_StartDateTime = nip_data.Critical_StartDateTime + datetime.timedelta(hours=nip_start_hrs)
            nip_data.Critical_Period = nip_data.Critical_EndDateTime - nip_data.Critical_StartDateTime

            # Get data of Weather which did not cause Incidents for each record
            def get_weather_stats_for_non_ip(weather_cell_id, nip_start, nip_end, stanox_section):
                """
                :param weather_cell_id: [int] e.g. weather_cell_id=nip_data.WeatherCell.iloc[500]
                :param nip_start: [Timestamp] e.g. nip_start=nip_data.StartDateTime.iloc[500]
                :param nip_end: [Timestamp] e.g. nip_end=nip_data.EndDateTime.iloc[500]
                :param stanox_section: [str] e.g. stanox_section=nip_data.StanoxSection.iloc[500]
                :return: [list] a list of statistics
                """
                # Get non-IP Weather data about where and when the incident occurred
                non_ip_weather_obs = mssqlserver.metex.view_weather_by_id_datetime(
                    weather_cell_id, nip_start, nip_end, pickle_it=False)
                # Get all incident period data on the same section
                overlaps = ip_data[
                    (ip_data.StanoxSection == stanox_section) &
                    (((ip_data.Critical_StartDateTime <= nip_start) & (ip_data.Critical_EndDateTime >= nip_start)) |
                     ((ip_data.Critical_StartDateTime <= nip_end) & (ip_data.Critical_EndDateTime >= nip_end)))]
                # Skip data of Weather causing Incidents at around the same time but
                if not overlaps.empty:
                    non_ip_weather_obs = non_ip_weather_obs[
                        (non_ip_weather_obs.DateTime < np.min(overlaps.Critical_StartDateTime)) |
                        (non_ip_weather_obs.DateTime > np.max(overlaps.Critical_EndDateTime))]
                # Get the max/min/avg Weather parameters for those incident periods
                non_ip_weather_stats = prototype.tools.calculate_statistics_for_weather_variables(
                    non_ip_weather_obs, weather_stats_calculations)
                return non_ip_weather_stats

            # Get stats data for the specified "Non-Incident Periods"
            nip_stats = nip_data.apply(
                lambda x: get_weather_stats_for_non_ip(
                    x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime, x.StanoxSection), axis=1)
            nip_statistics = pd.DataFrame(nip_stats.tolist(), nip_stats.index,
                                          prototype.tools.get_weather_variable_names(weather_stats_calculations))
            nip_statistics['Temperature_diff'] = nip_statistics.Temperature_max - nip_statistics.Temperature_min

            #
            nip_data = nip_data.join(nip_statistics.dropna(), how='inner')
            nip_data['IncidentReported'] = 0

            # Merge "ip_data" and "nip_data" into one DataFrame
            incident_location_weather = pd.concat([nip_data, ip_data], axis=0, ignore_index=True)

            save_pickle(incident_location_weather, path_to_pickle)

            return incident_location_weather

        except Exception as e:
            print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# ====================================================================================================================
""" Calculations for vegetation data """


# Specify the statistics that need to be computed
def specify_vegetation_stats_calculations(features):
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


# Get Vegetation conditions for incident locations
def get_incident_location_vegetation(route_name='Anglia',
                                     shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                                     update=False):
    """
    Testing parameters:
    e.g.
        route_name='Anglia'
        shift_yards_same_elr=220
        shift_yards_diff_elr=220
        hazard_pctl=50
        update=False

    Note that the "CoverPercent..." in furlong_vegetation_data has been
    amended when furlong_data was read. Check the function get_furlong_data().
    """
    pickle_filename = mssqlserver.metex.make_filename("incident_location_vegetation", route_name, None,
                                                      shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl,
                                                      save_as=".pickle")
    path_to_pickle = cdd_prototype_wind(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)
    else:
        try:
            # # Get data of furlong Vegetation coverage and hazardous trees
            # furlong_vegetation_data = mssqlserver.vegetation.view_vegetation_condition_per_furlong()
            # furlong_vegetation_data.set_index('FurlongID', inplace=True)

            # Get incident_location_furlongs
            incident_furlongs = prototype.furlong.get_incident_furlongs(route_name, None, shift_yards_same_elr,
                                                                        shift_yards_diff_elr)

            # Get all column names as features
            features = incident_furlongs.columns

            # Specify the statistics that need to be computed
            cover_percents, hazard_rest, veg_stats_calc = specify_vegetation_stats_calculations(features)

            # Get features which would be filled with "0" and "inf", respectively
            fill_0 = [x for x in features if re.match('.*height', x)] + ['HazardTreeNumber']
            fill_inf = [x for x in features if re.match('^.*prox|.*diam', x)]

            incident_location_furlongs = prototype.furlong.get_incident_location_furlongs(
                route_name, None, shift_yards_same_elr, shift_yards_diff_elr).dropna()

            # Define a function that computes Vegetation stats for each incident record
            def calculate_vegetation_variables_stats(furlong_ids, start_elr, end_elr, total_yards_adjusted):
                """
                Testing parameters:
                e.g.
                    furlong_ids=incident_location_furlongs.Critical_FurlongIDs[5367]
                    start_elr=incident_location_furlongs.StartELR[5367]
                    end_elr=incident_location_furlongs.EndELR[5367]
                    total_yards_adjusted=incident_location_furlongs.Section_Length_Adj[5367]

                Note: to get the n-th percentile may use percentile(n)

                """
                vegetation_data = incident_furlongs.loc[furlong_ids]

                veg_stats = vegetation_data.groupby('ELR').aggregate(veg_stats_calc)
                veg_stats[cover_percents] = veg_stats[cover_percents].div(veg_stats.AssetNumber, axis=0).values

                if start_elr == end_elr:
                    if np.isnan(veg_stats.HazardTreeNumber[start_elr]):
                        veg_stats[fill_0] = 0.0
                        veg_stats[fill_inf] = 999999.0
                    else:
                        assert 0 <= hazard_pctl <= 100
                        veg_stats[hazard_rest] = veg_stats[hazard_rest].applymap(
                            lambda x: np.nanpercentile(tuple(itertools.chain(*pd.Series(x).dropna())), hazard_pctl))
                else:
                    if np.all(np.isnan(veg_stats.HazardTreeNumber.values)):
                        veg_stats[fill_0] = 0.0
                        veg_stats[fill_inf] = 999999.0
                        calc_further = {k: lambda y: np.nanmean(y) for k in hazard_rest}
                    else:
                        veg_stats[hazard_rest] = veg_stats[hazard_rest].applymap(
                            lambda y: tuple(itertools.chain(*pd.Series(y).dropna())))
                        hazard_rest_func = [lambda y: np.nanpercentile(np.sum(y), hazard_pctl)]
                        calc_further = dict(zip(hazard_rest, hazard_rest_func * len(hazard_rest)))

                    # Specify further calculations
                    calc_further.update({'AssetNumber': np.sum})
                    calc_further.update(dict(DateOfMeasure=lambda y: tuple(itertools.chain(*y))))
                    calc_further.update({k: lambda y: tuple(y) for k in cover_percents})
                    veg_stats_calc_further = veg_stats_calc.copy()
                    veg_stats_calc_further.update(calc_further)

                    # Rename index (by which the dataframe can be grouped)
                    veg_stats.index = pd.Index(data=['-'.join(set(veg_stats.index))] * len(veg_stats.index), name='ELR')
                    veg_stats = veg_stats.groupby(veg_stats.index).aggregate(veg_stats_calc_further)

                    if len(total_yards_adjusted) == 3 and \
                            (total_yards_adjusted[1] == 0 or np.isnan(total_yards_adjusted[1])):
                        total_yards_adjusted = total_yards_adjusted[:1] + total_yards_adjusted[2:]

                    veg_stats[cover_percents] = veg_stats[cover_percents].applymap(
                        lambda x: np.dot(x, total_yards_adjusted) / np.nansum(total_yards_adjusted))

                # Calculate tree densities (number of trees per furlong)
                veg_stats['TreeDensity'] = veg_stats.TreeNumber.div(np.nansum(total_yards_adjusted) / 220.0)
                veg_stats['HazardTreeDensity'] = veg_stats.HazardTreeNumber.div(np.nansum(total_yards_adjusted) / 220.0)

                # Rearrange the order of features
                veg_stats = veg_stats[sorted(veg_stats.columns)]

                return veg_stats.values[0].tolist()

            # Compute Vegetation stats for each incident record
            vegetation_statistics = incident_location_furlongs.apply(
                lambda x: pd.Series(calculate_vegetation_variables_stats(
                    x.Critical_FurlongIDs, x.StartELR, x.EndELR, x.Section_Length_Adj)), axis=1)

            vegetation_statistics.columns = sorted(list(veg_stats_calc.keys()) + ['TreeDensity', 'HazardTreeDensity'])
            veg_percent = [x for x in cover_percents if re.match('^CoverPercent*.[^Open|thr]', x)]
            vegetation_statistics['CoverPercentVegetation'] = vegetation_statistics[veg_percent].apply(np.sum, axis=1)

            hazard_rest_pctl = [''.join([x, '_%s' % hazard_pctl]) for x in hazard_rest]
            rename_features = dict(zip(hazard_rest, hazard_rest_pctl))
            rename_features.update({'AssetNumber': 'AssetCount'})
            vegetation_statistics.rename(columns=rename_features, inplace=True)

            incident_location_vegetation = incident_location_furlongs.join(vegetation_statistics)

            save_pickle(incident_location_vegetation, path_to_pickle)

            return incident_location_vegetation

        except Exception as e:
            print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# ====================================================================================================================
""" Integrate incident data with both the weather and vegetation data """


# Integrate the Weather and Vegetation conditions for incident locations
def integrate_incident_with_weather_and_vegetation(route_name='Anglia', weather_category='Wind',
                                                   ip_start_hrs=-12, ip_end_hrs=12, nip_start_hrs=-12,
                                                   shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                                                   update=False):
    pickle_filename = mssqlserver.metex.make_filename("integrated_data", route_name, weather_category,
                                                      ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                                      shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_file = cdd_prototype_wind(pickle_filename)

    if os.path.isfile(path_to_file) and not update:
        integrated_data = load_pickle(path_to_file)
    else:
        try:
            # Get information of Schedule 8 incident and the relevant weather conditions
            incident_location_weather = get_incident_location_weather(route_name, weather_category,
                                                                      ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                                                      subset_weather_for_nip=False, update=update)
            # Get information of vegetation conditions for the incident locations
            incident_location_vegetation = get_incident_location_vegetation(route_name,
                                                                            shift_yards_same_elr, shift_yards_diff_elr,
                                                                            hazard_pctl)
            # incident_location_vegetation.drop(['IncidentCount', 'DelayCost', 'DelayMinutes'], axis=1, inplace=True)

            common_feats = list(set(incident_location_weather.columns) & set(incident_location_vegetation.columns))
            integrated_weather_vegetation = pd.merge(incident_location_weather, incident_location_vegetation,
                                                     how='inner', on=common_feats)

            # Electrified
            integrated_weather_vegetation.Electrified = integrated_weather_vegetation.Electrified.astype(int)

            # Categorise average wind directions into 4 quadrants
            wind_direction = pd.cut(integrated_weather_vegetation.WindDirection_avg.values, [0, 90, 180, 270, 360],
                                    right=False)
            integrated_data = integrated_weather_vegetation.join(
                pd.DataFrame(wind_direction, columns=['WindDirection_avg_quadrant'])).join(
                pd.get_dummies(wind_direction, prefix='WindDirection_avg'))

            save_pickle(integrated_data, path_to_file)

        except Exception as e:
            print("Failed to get \"{}\". {}".format(pickle_filename, e))
            integrated_data = None

    return integrated_data


# ====================================================================================================================
""" Trials """


# Specify the explanatory variables considered in this prototype model
def specify_explanatory_variables():
    return [
        # 'WindSpeed_max',
        # 'WindSpeed_avg',
        'WindGust_max',
        # 'WindDirection_avg',
        # 'WindDirection_avg_[0, 90)',  # [0°, 90°)
        'WindDirection_avg_[90, 180)',  # [90°, 180°)
        'WindDirection_avg_[180, 270)',  # [180°, 270°)
        'WindDirection_avg_[270, 360)',  # [270°, 360°)
        'Temperature_diff',
        # 'Temperature_avg',
        # 'Temperature_max',
        # 'Temperature_min',
        'RelativeHumidity_max',
        'Snowfall_max',
        'TotalPrecipitation_max',
        # 'Electrified',
        'CoverPercentAlder',
        'CoverPercentAsh',
        'CoverPercentBeech',
        'CoverPercentBirch',
        'CoverPercentConifer',
        'CoverPercentElm',
        'CoverPercentHorseChestnut',
        'CoverPercentLime',
        'CoverPercentOak',
        'CoverPercentPoplar',
        'CoverPercentShrub',
        'CoverPercentSweetChestnut',
        'CoverPercentSycamore',
        'CoverPercentWillow',
        # 'CoverPercentOpenSpace',
        'CoverPercentOther',
        # 'CoverPercentVegetation',
        # 'CoverPercentDiff',
        # 'TreeDensity',
        # 'TreeNumber',
        # 'TreeNumberDown',
        # 'TreeNumberUp',
        # 'HazardTreeDensity',
        # 'HazardTreeNumber',
        # 'HazardTreediameterM_max',
        # 'HazardTreediameterM_min',
        # 'HazardTreeheightM_max',
        # 'HazardTreeheightM_min',
        # 'HazardTreeprox3py_max',
        # 'HazardTreeprox3py_min',
        # 'HazardTreeproxrailM_max',
        # 'HazardTreeproxrailM_min'
    ]


# Describe basic statistics about the main explanatory variables
def describe_explanatory_variables(mdata, save_as=".tif", dpi=None):
    fig = plt.figure(figsize=(12, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    ax1 = fig.add_subplot(161)
    mdata.WindGust_max.plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
    # train_set[['WindGust_max']].boxplot(column='WindGust_max', ax=ax1, boxprops=dict(color='k'))
    ax1.set_xticklabels('')
    plt.xlabel('Max. Gust', fontsize=13, labelpad=16)
    plt.ylabel('($\\times$10 mph)', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(-0.1, 1.02)

    ax2 = fig.add_subplot(162)
    mdata.WindDirection_avg_quadrant.value_counts().sort_index().plot.bar(color='#4c76e1', rot=0, fontsize=12)
    plt.xlabel('Avg. Wind Direction', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax2.set_xticklabels([1, 2, 3, 4])
    ax2.yaxis.set_label_coords(-0.1, 1.02)

    ax3 = fig.add_subplot(163)
    mdata.Temperature_diff.plot.box(color=colour, ax=ax3, widths=0.5, fontsize=12)
    ax3.set_xticklabels('')
    plt.xlabel('Temp. Diff.', fontsize=13, labelpad=16)
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax3.yaxis.set_label_coords(-0.1, 1.02)

    ax4 = fig.add_subplot(164)
    mdata.RelativeHumidity_max.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Max. R.H.', fontsize=13, labelpad=16)
    plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(-0.1, 1.02)

    ax5 = fig.add_subplot(165)
    mdata.Snowfall_max.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
    ax5.set_xticklabels('')
    plt.xlabel('Max. Snowfall', fontsize=13, labelpad=16)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(-0.1, 1.02)

    ax6 = fig.add_subplot(166)
    mdata.TotalPrecipitation_max.plot.box(color=colour, ax=ax6, widths=0.5, fontsize=12)
    ax6.set_xticklabels('')
    plt.xlabel('Max. Total Precip.', fontsize=13, labelpad=16)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(-0.1, 1.02)

    plt.tight_layout()
    path_to_file_weather = cdd(prototype.tools.cd_prototype_fig_pub("Variables", "Weather" + save_as))
    plt.savefig(path_to_file_weather, dpi=dpi)
    if save_as == ".svg":
        save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))

    #
    fig_veg = plt.figure(figsize=(12, 5))
    ax = fig_veg.add_subplot(111)
    colour_veg = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

    cover_percent_cols = [c for c in mdata.columns if c.startswith('CoverPercent')
                          and not (c.endswith('Vegetation') or c.endswith('Diff'))]
    cover_percent_cols += [cover_percent_cols.pop(cover_percent_cols.index('CoverPercentOpenSpace'))]
    cover_percent_cols += [cover_percent_cols.pop(cover_percent_cols.index('CoverPercentOther'))]
    mdata[cover_percent_cols].plot.box(color=colour_veg, ax=ax, widths=0.5, fontsize=12)
    # plt.boxplot([train_set[c] for c in cover_percent_cols])
    # plt.tick_params(axis='x', labelbottom='off')
    ax.set_xticklabels([re.search('(?<=CoverPercent).*', c).group() for c in cover_percent_cols], rotation=45)
    plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
    ax.yaxis.set_label_coords(0, 1.02)

    plt.tight_layout()
    path_to_file_veg = prototype.tools.cd_prototype_fig_pub("Variables", "Vegetation" + save_as)
    plt.savefig(path_to_file_veg, dpi=dpi)
    if save_as == ".svg":
        save_svg_as_emf(path_to_file_veg, path_to_file_veg.replace(save_as, ".emf"))


"""
def save_result_to_excel(result, writer):
    result_file = dbm.make_filename("result", route, Weather, ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                    shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    writer = pd.ExcelWriter(cdd_mod_wind(trial_id, result_file), engine='xlsxwriter')
    info, estimates = pd.read_html(result.summary().as_html().replace(':', ''))
    info_0, info_1 = info.iloc[:, :2].set_index(0), info.iloc[:, 2:].set_index(2)
"""


# A prototype model in the context of wind-related Incidents
def logistic_regression_model(trial_id,
                              route_name='Anglia', weather_category='Wind',
                              ip_start_hrs=-12, ip_end_hrs=12, nip_start_hrs=-12,
                              shift_yards_same_elr=660, shift_yards_diff_elr=220, hazard_pctl=50,
                              in_seasons=None,
                              describe_var=False,
                              outlier_pctl=99,
                              add_const=True, seed=1, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=600,
                              # dig_deeper=False,
                              verbose=True):
    """
    IncidentReason  IncidentReasonName    IncidentReasonDescription

    IQ              TRACK SIGN            Trackside sign blown down/light out etc.
    IW              COLD                  Non severe - Snow/Ice/Frost affecting infrastructure equipment',
                                          'Takeback Pumps'
    OF              HEAT/WIND             Blanket speed restriction for extreme heat or high wind in accordance with
                                          the Group Standards
    Q1              TKB PUMPS             Takeback Pumps
    X4              BLNK REST             Blanket speed restriction for extreme heat or high wind
    XW              WEATHER               Severe Weather not snow affecting infrastructure the responsibility of
                                          Network Rail
    XX              MISC OBS              Msc items on line (incl trees) due to effects of Weather responsibility of RT

    Parameters:
    e.g.

    trial_id=0
    route_name='Anglia'
    weather_category='Wind'
    ip_start_hrs=-12
    ip_end_hrs=12
    nip_start_hrs=-12
    shift_yards_same_elr=660
    shift_yards_diff_elr=220
    hazard_pctl=50
    in_seasons=None
    describe_var=False
    outlier_pctl=99
    add_const=True
    seed=1
    model='logit'
    plot_roc=True
    plot_pred_likelihood=True
    save_as=".tif"
    dpi=600
    dig_deeper=False
    verbose=True

    """
    # Get the mdata for modelling
    integrated_data = integrate_incident_with_weather_and_vegetation(route_name, weather_category,
                                                                     ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                                                     shift_yards_same_elr, shift_yards_diff_elr,
                                                                     hazard_pctl)

    # Select season data: 'Spring', 'Summer', 'Autumn', 'Winter'
    integrated_data = prototype.tools.get_data_by_season(integrated_data, in_seasons, incident_datetime_col='StartDate')

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        integrated_data = integrated_data[
            integrated_data.DelayMinutes <= np.percentile(integrated_data.DelayMinutes, outlier_pctl)]
        # from pyhelpers.misc import get_extreme_outlier_bounds
        # lo, up = get_extreme_outlier_bounds(mod_data.DelayMinutes, k=1.5)
        # mod_data = mod_data[mod_data.DelayMinutes.between(lo, up, inclusive=True)]

    # CoverPercent
    cover_percent_cols = [x for x in integrated_data.columns if re.match('^CoverPercent', x)]
    integrated_data[cover_percent_cols] = integrated_data[cover_percent_cols] / 10.0
    integrated_data['CoverPercentDiff'] = \
        integrated_data.CoverPercentVegetation - integrated_data.CoverPercentOpenSpace - \
        integrated_data.CoverPercentOther
    integrated_data.CoverPercentDiff = integrated_data.CoverPercentDiff * integrated_data.CoverPercentDiff.map(
        lambda x: 1 if x >= 0 else 0)

    # Scale down 'WindGust_max' and 'RelativeHumidity_max'
    integrated_data.WindGust_max = integrated_data.WindGust_max / 10.0
    integrated_data.RelativeHumidity_max = integrated_data.RelativeHumidity_max / 10.0

    # Select features
    explanatory_variables = specify_explanatory_variables()
    explanatory_variables += [
        # 'HazardTreediameterM_%s' % hazard_pctl,
        # 'HazardTreeheightM_%s' % hazard_pctl,
        # 'HazardTreeprox3py_%s' % hazard_pctl,
        # 'HazardTreeproxrailM_%s' % hazard_pctl,
    ]

    # Add an intercept
    if add_const:
        integrated_data['const'] = 1
        explanatory_variables = ['const'] + explanatory_variables

    # Set the outcomes of non-incident records to 0
    integrated_data.loc[integrated_data.IncidentReported == 0, ['DelayMinutes', 'DelayCost', 'IncidentCount']] = 0

    if describe_var:
        describe_explanatory_variables(integrated_data, save_as=save_as, dpi=dpi)

    # Select data before 2014 as training data set, with the rest being test set
    train_set = integrated_data[integrated_data.FinancialYear < 2014]
    test_set = integrated_data[integrated_data.FinancialYear == 2014]

    try:
        np.random.seed(seed)
        if model == 'logit':
            mod = sm.Logit(train_set.IncidentReported, train_set[explanatory_variables])
        else:
            mod = sm.Probit(train_set.IncidentReported, train_set[explanatory_variables])
        result_summary = mod.fit(method='newton', maxiter=1000, full_output=True, disp=False)

        if verbose:
            print(result_summary.summary())

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result_summary.params), columns=['OddsRatio'])
        if verbose:
            print("\nOdds ratio:")
            print(odds_ratios)

        # Prediction
        test_set['incident_prob'] = result_summary.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        fpr, tpr, thr = sklearn.metrics.roc_curve(test_set.IncidentReported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = sklearn.metrics.auc(fpr, tpr)
        ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
        threshold = np.min(thr[ind])

        # prediction accuracy
        test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= threshold else 0)
        test = pd.Series(test_set.IncidentReported == test_set.incident_prediction)
        mod_accuracy = np.divide(test.sum(), len(test))
        if verbose:
            print("\nAccuracy: %f" % mod_accuracy)

        # incident prediction accuracy
        incid_only = test_set[test_set.IncidentReported == 1]
        test_acc = pd.Series(incid_only.IncidentReported == incid_only.incident_prediction)
        incid_accuracy = np.divide(test_acc.sum(), len(test_acc))
        if verbose:
            print("Incident accuracy: %f" % incid_accuracy)

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
            # plt.subplots_adjust(left=0.10, bottom=0.1, right=0.96, top=0.96)
            plt.tight_layout()
            plt.savefig(cdd_prototype_wind_mod(trial_id, "ROC" + save_as), dpi=dpi)
            path_to_file_roc = prototype.tools.cd_prototype_fig_pub("Prediction", "ROC" + save_as)  # Fig. 6.
            plt.savefig(path_to_file_roc, dpi=dpi)
            if save_as == ".svg":
                save_svg_as_emf(path_to_file_roc, path_to_file_roc.replace(save_as, ".emf"))  # Fig. 6.

        # Plot incident delay minutes against predicted probabilities
        if plot_pred_likelihood:
            incid_ind = test_set.IncidentReported == 1
            plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            ax.scatter(test_set[incid_ind].incident_prob, test_set[incid_ind].DelayMinutes,
                       c='#db0101', edgecolors='k', marker='o', linewidths=2, s=80, alpha=.3, label="Incidents")
            plt.axvline(x=threshold, label="Threshold = %.2f" % threshold, color='b')
            legend = plt.legend(scatterpoints=1, loc='best', fontsize=14, fancybox=True)
            frame = legend.get_frame()
            frame.set_edgecolor('k')
            plt.xlim(xmin=0, xmax=1.03)
            plt.ylim(ymin=-15)
            ax.set_xlabel("Predicted probability of incident occurrence (for 2014/15)", fontsize=14, fontweight='bold')
            ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            plt.savefig(cdd_prototype_wind_mod(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)
            path_to_file_pred = prototype.tools.cd_prototype_fig_pub("Prediction", "Likelihood" + save_as)
            plt.savefig(path_to_file_pred, dpi=dpi)  # Fig. 7.
            if save_as == ".svg":
                save_svg_as_emf(path_to_file_pred, path_to_file_pred.replace(save_as, ".emf"))  # Fig. 7.

        # ===================================================================================
        # if dig_deeper:
        #     tmp_cols = explanatory_variables.copy()
        #     tmp_cols.remove('Electrified')
        #     tmps = [np.linspace(train_set[i].min(), train_set[i].max(), 5) for i in tmp_cols]
        #
        #     combos = pd.DataFrame(sklearn.utils.extmath.cartesian(tmps + [np.array([0, 1])]))
        #     combos.columns = explanatory_variables
        #
        #     combos['incident_prob'] = result.predict(combos[explanatory_variables])
        #
        #     def isolate_and_plot(var1='WindGust_max', var2='WindSpeed_max'):
        #         # isolate gre and class rank
        #         grouped = pd.pivot_table(
        #             combos, values=['incident_prob'], index=[var1, var2], aggfunc=np.mean)
        #
        #         # in case you're curious as to what this looks like
        #         # print grouped.head()
        #         #                      admit_pred
        #         # gre        prestige
        #         # 220.000000 1           0.282462
        #         #            2           0.169987
        #         #            3           0.096544
        #         #            4           0.079859
        #         # 284.444444 1           0.311718
        #
        #         # make a plot
        #         colors = 'rbgyrbgy'
        #
        #         lst = combos[var2].unique().tolist()
        #         plt.figure()
        #         for col in lst:
        #             plt_data = grouped.loc[grouped.index.get_level_values(1) == col]
        #             plt.plot(plt_data.index.get_level_values(0), plt_data.incident_prob, color=colors[lst.index(col)])
        #
        #         plt.xlabel(var1)
        #         plt.ylabel("P(wind-related incident)")
        #         plt.legend(lst, loc='best', title=var2)
        #         title0 = 'Pr(wind-related incident) isolating '
        #         plt.title(title0 + var1 + ' and ' + var2)
        #         plt.show()
        #
        #     isolate_and_plot(var1='WindGust_max', var2='TreeNumberUp')
        # ====================================================================================

    except Exception as e:
        print(e)
        result_summary = e
        mod_accuracy, incid_accuracy, threshold = np.nan, np.nan, np.nan

    repo = locals()
    var_names = get_variable_names(integrated_data, train_set, test_set,
                                   result_summary, mod_accuracy, incid_accuracy, threshold)
    resources = {k: repo[k] for k in var_names}
    result_pickle = mssqlserver.metex.make_filename("result", route_name, weather_category,
                                                    ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                                    shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    save_pickle(resources, cdd_prototype_wind_mod(trial_id, result_pickle))

    return integrated_data, train_set, test_set, result_summary, mod_accuracy, incid_accuracy, threshold


# Evaluate the primer model for different settings
def evaluate_prototype_model(season=None):
    start_time = time.time()

    expt = sklearn.utils.extmath.cartesian((range(-12, -11),
                                            range(12, 13),
                                            range(-12, -11),
                                            # range(0, 440, 220),
                                            range(220, 1100, 220),
                                            # range(0, 440, 220),
                                            range(220, 1100, 220),
                                            range(25, 100, 25)))
    # (range(-12, -11), range(12, 13), range(-24, -23), range(220, 221), range(220, 221), range(25, 26)))
    # ((range(-36, 0, 3), range(0, 15, 3), range(-36, 0, 3)))
    # ((range(-24, -11), range(6, 12), range(-12, -6)))

    total_no = len(expt)

    results = []
    nobs = []
    mod_aic = []
    mod_bic = []
    mod_accuracy = []
    incid_accuracy = []
    msg = []
    data_sets = []
    train_sets = []
    test_sets = []
    thresholds = []

    counter = 0
    for h in expt:
        counter += 1
        print("Processing setting %d ... (%d in total)" % (counter, total_no))
        ip_start_hrs, ip_end_hrs, nip_start_hrs, shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl = h
        # Try:
        mdata, train_set, test_set, result, mod_acc, incid_acc, threshold = logistic_regression_model(
            trial_id=counter,
            route_name='ANGLIA', weather_category='Wind',
            ip_start_hrs=int(ip_start_hrs),
            ip_end_hrs=int(ip_end_hrs),
            nip_start_hrs=int(nip_start_hrs),
            shift_yards_same_elr=int(shift_yards_same_elr),
            shift_yards_diff_elr=int(shift_yards_diff_elr),
            hazard_pctl=int(hazard_pctl),
            in_seasons=season,
            describe_var=False,
            outlier_pctl=99,
            add_const=True, seed=123, model='logit',
            plot_roc=False, plot_pred_likelihood=False,
            # dig_deeper=False,
            verbose=False)

        data_sets.append(mdata)
        train_sets.append(train_set)
        test_sets.append(test_set)
        results.append(result)
        mod_accuracy.append(mod_acc)
        incid_accuracy.append(incid_acc)
        thresholds.append(threshold)

        if isinstance(result, sm.BinaryResultsWrapper):
            nobs.append(result.nobs)
            mod_aic.append(result.aic)
            mod_bic.append(result.bic)
            msg.append(result.summary().extra_txt)
        else:
            print("\nProblems may occur given setting %d: {}.\n".format(h) % counter)
            nobs.append(len(train_set))
            mod_aic.append(np.nan)
            mod_bic.append(np.nan)
            msg.append(result.__str__())
        print("\n%d done.\n" % counter)

    # Create a dataframe that summarises the test results
    columns = ['IP_StartHrs', 'IP_EndHrs', 'NIP_StartHrs',
               'YardShift_same_ELR', 'YardShift_diff_ELR', 'hazard_pctl', 'Obs_No',
               'AIC', 'BIC', 'Threshold', 'PredAcc', 'PredAcc_Incid', 'Extra_Info']
    data = [list(x) for x in expt.T] + [nobs, mod_aic, mod_bic, thresholds, mod_accuracy, incid_accuracy, msg]
    trial_summary = pd.DataFrame(dict(zip(columns, data)), columns=columns)
    trial_summary.sort_values(['PredAcc_Incid', 'PredAcc', 'AIC', 'BIC'], ascending=[False, False, True, True],
                              inplace=True)

    save_pickle(results, cdd_prototype_wind("trial_results.pickle"))
    save_pickle(trial_summary, cdd_prototype_wind("trial_summary.pickle"))

    print("Total elapsed time: %.2f hrs." % ((time.time() - start_time) / 3600))

    return trial_summary


# View data
def view_trial_data(trial_id, route='Anglia', weather='Wind',
                    ip_start_hrs=-12, ip_end_hrs=12, nip_start_hrs=-12,
                    shift_yards_same_elr=440, shift_yards_diff_elr=220, hazard_pctl=50):
    result_pickle = mssqlserver.metex.make_filename("result", route, weather, ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                                    shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_pickle = cdd_prototype_wind_mod(trial_id, result_pickle)
    if os.path.isfile(path_to_pickle):
        return load_pickle(path_to_pickle)
    else:
        try:
            result = logistic_regression_model(trial_id, route, weather, ip_start_hrs, ip_end_hrs, nip_start_hrs,
                                               shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
            return result
        except Exception as e:
            print(e)
