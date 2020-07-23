""" A prototype model in the context of heat-related Incidents """

import datetime
import os

import datetime_truncate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import statsmodels.discrete.discrete_model as sm
from pyhelpers.store import load_pickle, save_fig, save_pickle
from sklearn import metrics

from models.prototype.wind import get_incident_location_vegetation
from models.tools import calculate_prototype_weather_statistics, categorise_track_orientations, cdd_prototype
from models.tools import get_data_by_season, get_weather_variable_names
from mssqlserver import metex
from settings import mpl_preferences, pd_preferences
from utils import make_filename

# == Apply the preferences ============================================================================

mpl_preferences(reset=False)
pd_preferences(reset=False)
plt.rc('font', family='Times New Roman')


# == Change directories ===============================================================================

def cdd_prototype_heat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\heat\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\heat\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("heat", *sub_dir, mkdir=mkdir)

    return path


def cdd_prototype_heat_trial(trial_id, *sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\prototype\\heat\\<``trial_id``>" and sub-directories / a file.

    :param trial_id:
    :type trial_id:
    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\prototype\\heat\\data\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_prototype("heat", "{}".format(trial_id), *sub_dir, mkdir=mkdir)

    return path


# == Calculations for weather data ====================================================================


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


def get_incident_location_weather(route_name='Anglia', weather_category='Heat',
                                  ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                  update=False, verbose=False):
    """
    Get TRUST data and the weather conditions for each incident location.

    :param route_name: name of a Route, defaults to ``'Anglia'``
    :type route_name: str, None
    :param weather_category: weather category, defaults to ``'Wind'``
    :type weather_category: str, None
    :param ip_start_hrs:
    :type ip_start_hrs: int, float
    :param nip_ip_gap:
    :type nip_ip_gap: int, float
    :param nip_start_hrs:
    :type nip_start_hrs: int, float
    :param update: whether to check on update and proceed to update the package data, defaults to ``False``
    :type update: bool
    :param verbose: whether to print relevant information in console as the function runs, defaults to ``False``
    :type verbose: bool, int
    :return:
    :rtype: pandas.DataFrame

    **Example**::

        from models.prototype.heat import get_incident_location_weather

        route_name       = 'Anglia'
        weather_category = 'Heat'
        ip_start_hrs     = -24
        nip_ip_gap       = -5
        nip_start_hrs    = -24
        update           = True
        verbose          = True

        incident_location_weather = get_incident_location_weather(route_name, weather_category,
                                                                  ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                                  update, verbose)
    """

    pickle_filename = make_filename("weather", route_name, weather_category, ip_start_hrs, nip_ip_gap, nip_start_hrs)
    path_to_pickle = cdd_prototype_heat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        return load_pickle(path_to_pickle)

    else:
        try:
            # Getting incident data for all incident locations
            incidents = metex.view_schedule8_costs_by_datetime_location_reason(route_name, weather_category)
            # Drop non-Weather-related incident records
            incidents = incidents[incidents.WeatherCategory != ''] if weather_category is None else incidents
            # Get data for the specified "Incident Periods"
            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
            incidents['Critical_StartDateTime'] = \
                incidents.StartDateTime.apply(datetime_truncate.truncate_hour) + datetime.timedelta(hours=ip_start_hrs)
            incidents['Critical_EndDateTime'] = incidents.StartDateTime
            incidents['Critical_Period'] = incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

            if incidents.WeatherCell.dtype != 'int64':
                # Rectify the records for which Weather cell id is empty
                weather_cell = metex.get_weather_cell()
                ll = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ll_Longitude, weather_cell.ll_Latitude)]
                ul = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ul_Longitude, weather_cell.ul_Latitude)]
                ur = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ur_Longitude, weather_cell.ur_Latitude)]
                lr = [shapely.geometry.Point(xy) for xy in zip(weather_cell.lr_Longitude, weather_cell.lr_Latitude)]
                poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]
                cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in range(len(weather_cell))]

                for i in incidents[incidents.WeatherCell == ''].index:
                    pt = shapely.geometry.Point(incidents.StartLongitude.loc[i], incidents.StartLatitude.loc[i])
                    id_set = set(
                        weather_cell.iloc[[i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
                    if len(id_set) == 0:
                        pt_alt = shapely.geometry.Point(incidents.EndLongitude.loc[i], incidents.EndLatitude.loc[i])
                        id_set = set(
                            weather_cell.iloc[
                                [i for i, p in enumerate(cells) if pt_alt.within(p)]].WeatherCellId.tolist())
                    if len(id_set) != 0:
                        incidents.loc[i, 'WeatherCell'] = list(id_set)[0]

            weather_stats_calculations = specify_weather_stats_calculations()

            def get_weather_stats_for_ip(weather_cell_id, ip_start, ip_end):
                """
                Processing Weather data for IP - Get Weather conditions which led to Incidents for each record.

                :param weather_cell_id: [int] Weather Cell ID
                :param ip_start: [Timestamp] start of "incident period"
                :param ip_end: [Timestamp] end of "incident period"
                :return: [list] a list of statistics

                **Example**::

                    weather_cell_id = incidents.WeatherCell[1]
                    ip_start = incidents.StartDateTime[1]
                    ip_end = incidents.EndDateTime[1]
                """

                # Get Weather data about where and when the incident occurred
                ip_weather_obs = metex.query_weather_by_id_datetime(weather_cell_id, ip_start, ip_end, pickle_it=False)
                # Get the max/min/avg Weather parameters for those incident periods
                weather_stats_data = calculate_prototype_weather_statistics(ip_weather_obs, weather_stats_calculations)
                return weather_stats_data

            # Get data for the specified IP
            ip_stats = incidents.apply(
                lambda x: get_weather_stats_for_ip(x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime),
                axis=1)

            ip_statistics = pd.DataFrame(ip_stats.to_list(), index=ip_stats.index,
                                         columns=get_weather_variable_names(weather_stats_calculations))
            ip_statistics['Temperature_dif'] = ip_statistics.Temperature_max - ip_statistics.Temperature_min

            ip_data = incidents.join(ip_statistics.dropna(), how='inner')
            ip_data['IncidentReported'] = 1

            # Processing Weather data for non-IP - Get data of Weather which did not cause Incidents for each record
            nip_data = incidents.copy(deep=True)
            nip_data.Critical_EndDateTime = nip_data.Critical_StartDateTime + datetime.timedelta(days=nip_ip_gap)
            nip_data.Critical_StartDateTime = nip_data.Critical_EndDateTime + datetime.timedelta(hours=nip_start_hrs)
            nip_data.Critical_Period = nip_data.Critical_EndDateTime - nip_data.Critical_StartDateTime

            def get_weather_stats_for_non_ip(weather_cell_id, nip_start, nip_end, stanox_section):
                """
                :param weather_cell_id:
                :param nip_start:
                :param nip_end:
                :param stanox_section:
                :return:
                """

                # Get non-IP Weather data about where and when the incident occurred
                non_ip_weather_obs = metex.query_weather_by_id_datetime(weather_cell_id, nip_start, nip_end,
                                                                        pickle_it=False)
                # Get all incident period data on the same section
                overlaps = ip_data[
                    (ip_data.StanoxSection == stanox_section) &
                    (((ip_data.Critical_StartDateTime <= nip_start) & (ip_data.Critical_EndDateTime >= nip_start)) |
                     ((ip_data.Critical_StartDateTime <= nip_end) & (ip_data.Critical_EndDateTime >= nip_end)))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not overlaps.empty:
                    non_ip_weather_obs = non_ip_weather_obs[
                        (non_ip_weather_obs.DateTime < np.min(overlaps.Critical_StartDateTime)) |
                        (non_ip_weather_obs.DateTime > np.max(overlaps.Critical_EndDateTime))]
                # Get the max/min/avg Weather parameters for those incident periods
                non_ip_weather_stats = calculate_prototype_weather_statistics(non_ip_weather_obs,
                                                                              weather_stats_calculations)
                return non_ip_weather_stats

            # Get stats data for the specified "Non-Incident Periods"
            nip_stats = nip_data.apply(
                lambda x: get_weather_stats_for_non_ip(
                    x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime, x.StanoxSection), axis=1)

            nip_statistics = pd.DataFrame(nip_stats.tolist(), index=nip_stats.index,
                                          columns=get_weather_variable_names(weather_stats_calculations))
            nip_statistics['Temperature_dif'] = nip_statistics.Temperature_max - nip_statistics.Temperature_min

            nip_data = nip_data.join(nip_statistics.dropna(), how='inner')
            nip_data['IncidentReported'] = 0

            # Merge "ip_data" and "nip_data" into one DataFrame
            incident_location_weather = pd.concat([nip_data, ip_data], axis=0, ignore_index=True, sort=False)

            # Categorise average wind directions into 4 quadrants
            wind_direction = pd.cut(incident_location_weather.WindDirection_avg.values, [0, 90, 180, 270, 360],
                                    right=False)
            incident_location_weather = incident_location_weather.join(
                pd.DataFrame(wind_direction, columns=['WindDirection_avg_quadrant'])).join(
                pd.get_dummies(wind_direction, prefix='WindDirection_avg'))

            # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
            track_orientation = categorise_track_orientations(incident_location_weather)
            incident_location_weather = incident_location_weather.join(track_orientation)

            # Categorise temperature: < 24, = 24, 25, 26, 27, 28, 29, >= 30
            labels = ['Temperature_max < 24°C'] + \
                     ['Temperature_max {} {}°C'.format('≥' if x >= 30 else '=', x) for x in range(24, 31)]
            temperature_category = pd.cut(incident_location_weather.Temperature_max.values,
                                          bins=[-np.inf] + list(range(24, 31)) + [np.inf],
                                          right=False, labels=labels)
            incident_location_weather = incident_location_weather.join(
                pd.DataFrame(temperature_category, columns=['Temperature_Category'])).join(
                pd.get_dummies(temperature_category))

            save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

            return incident_location_weather

        except Exception as e:
            print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


def plot_temperature_deviation(route_name='Anglia', nip_ip_gap=-14, add_err_bar=True, update=False, verbose=False,
                               save_as=".tif", dpi=None):
    """
    **Example**::

        from models.prototype.heat import plot_temperature_deviation

        route_name  = 'Anglia'
        nip_ip_gap  = -14
        add_err_bar = True
        update      = False
        verbose     = True
        save_as     = None  # ".tif"
        dpi         = None  # 600
    """

    gap = np.abs(nip_ip_gap)

    incident_location_weather = [
        get_incident_location_weather(route_name, nip_ip_gap=-d, update=update, verbose=verbose)
        for d in range(1, gap + 1)]

    time_and_iloc = ['StartDateTime', 'EndDateTime', 'StanoxSection', 'IncidentDescription']
    selected_cols, data = time_and_iloc + ['Temperature_max'], incident_location_weather[0]
    ip_temperature_max = data[data.IncidentReported == 1][selected_cols]

    diff_means, diff_std = [], []
    for i in range(0, gap):
        data = incident_location_weather[i]
        nip_temperature_max = data[data.IncidentReported == 0][selected_cols]
        temp_diffs = pd.merge(ip_temperature_max, nip_temperature_max, on=time_and_iloc, suffixes=('_ip', '_nip'))
        temp_diff = temp_diffs.Temperature_max_ip - temp_diffs.Temperature_max_nip
        diff_means.append(temp_diff.abs().mean())
        diff_std.append(temp_diff.abs().std())

    plt.figure(figsize=(10, 5))
    if add_err_bar:
        container = plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', yerr=diff_std, capsize=4,
                            width=0.7, color='#9FAFBE')
        connector, cap_lines, (vertical_lines,) = container.errorbar.lines
        vertical_lines.set_color('#666666')
        for cap in cap_lines:
            cap.set_color('#da8067')
    else:
        plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', width=0.7, color='#9FAFBE')
        plt.grid(False)
    plt.xticks(np.arange(1, len(diff_means) + 1), fontsize=14)
    plt.xlabel('Latent period (Number of days)', fontsize=14)
    plt.ylabel('Temperature deviation (°C)', fontsize=14)
    plt.tight_layout()

    if save_as:
        path_to_fig = cdd_prototype_heat_trial(0, "Temp deviation" + save_as)
        save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)


# == Integrate incident data with the weather and vegetation data =====================================

def get_incident_locations_weather_and_vegetation(route_name='Anglia', weather_category='Heat',
                                                  ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                                  shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                                                  update=False, verbose=False):
    """
    Integrate the Weather and Vegetation conditions for incident locations.

    **Example**::

        from models.prototype.heat import get_incident_locations_weather_and_vegetation

        route_name           = 'Anglia'
        weather_category     = 'Heat'
        ip_start_hrs         = -24
        nip_ip_gap           = -5
        nip_start_hrs        = -24
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        hazard_pctl          = 50
        update               = True
        verbose              = True

        integrated_data = get_incident_locations_weather_and_vegetation(
            route_name, weather_category, ip_start_hrs, nip_ip_gap, nip_start_hrs,
            shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl, update, verbose)
    """

    pickle_filename = make_filename("dataset", route_name, weather_category,
                                    ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                    shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_pickle = cdd_prototype_heat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        integrated_data = load_pickle(path_to_pickle)
        return integrated_data

    else:
        try:
            # Get Schedule 8 incident and Weather data for locations
            incident_location_weather = get_incident_location_weather(
                route_name, weather_category, ip_start_hrs, nip_ip_gap, nip_start_hrs, verbose=verbose)
            # Get Vegetation conditions for the locations
            incident_location_vegetation = get_incident_location_vegetation(
                route_name, shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl, verbose=verbose)

            # Merge the above two data sets
            common_features = list(set(incident_location_weather.columns) & set(incident_location_vegetation.columns))
            integrated_data = pd.merge(incident_location_weather, incident_location_vegetation,
                                       how='inner', on=common_features)

            save_pickle(integrated_data, path_to_pickle, verbose=verbose)

            return integrated_data

        except Exception as e:
            print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))


# == Modelling trials =================================================================================

def specify_explanatory_variables_for_model_1():
    """
    Specify the explanatory variables considered in this prototype model - Version 1.

    :return:
    """

    return [
        # 'Temperature_min',
        # 'Temperature_avg',
        # 'Temperature_max',
        'Temperature_dif',
        # 'Temperature_max < 24°C',
        'Temperature_max = 24°C',
        'Temperature_max = 25°C',
        'Temperature_max = 26°C',
        'Temperature_max = 27°C',
        'Temperature_max = 28°C',
        'Temperature_max = 29°C',
        'Temperature_max ≥ 30°C',
        # 'track_orientation_E_W',
        'Track_Orientation_NE_SW',
        'Track_Orientation_NW_SE',
        'Track_Orientation_N_S',
        # 'WindGust_max',
        # 'WindSpeed_avg',
        # 'WindDirection_avg',
        # 'WindSpeed_max',
        # # 'WindDirection_avg_[0, 90)',  # [0°, 90°)
        # 'WindDirection_avg_[90, 180)',  # [90°, 180°)
        # 'WindDirection_avg_[180, 270)',  # [180°, 270°)
        # 'WindDirection_avg_[270, 360)',  # [270°, 360°)
        # 'RelativeHumidity_max',
        # 'RelativeHumidity_avg',
        # 'Snowfall_max',
        # 'TotalPrecipitation_max',
        # 'TotalPrecipitation_avg',
        # 'Electrified'
    ]


def specify_explanatory_variables_for_model_2():
    """
    Specify the explanatory variables considered in this prototype model - Version 2.

    :return:
    """

    return [
        # 'Temperature_min',
        # 'Temperature_avg',
        # 'Temperature_max',
        'Temperature_dif',
        # 'Temperature_max < 24°C',
        'Temperature_max = 24°C',
        'Temperature_max = 25°C',
        'Temperature_max = 26°C',
        'Temperature_max = 27°C',
        'Temperature_max = 28°C',
        'Temperature_max = 29°C',
        'Temperature_max ≥ 30°C',
        # 'track_orientation_E_W',
        'Track_Orientation_NE_SW',
        'Track_Orientation_NW_SE',
        'Track_Orientation_N_S',
        # 'WindGust_max',
        'WindSpeed_avg',
        # 'WindDirection_avg',
        # 'WindSpeed_max',
        # # 'WindDirection_avg_[0, 90)',  # [0°, 90°)
        # 'WindDirection_avg_[90, 180)',  # [90°, 180°)
        # 'WindDirection_avg_[180, 270)',  # [180°, 270°)
        # 'WindDirection_avg_[270, 360)',  # [270°, 360°)
        # 'RelativeHumidity_max',
        'RelativeHumidity_avg',
        # 'Snowfall_max',
        # 'TotalPrecipitation_max',
        'TotalPrecipitation_avg',
        # 'Electrified',
        # 'CoverPercentAlder',
        # 'CoverPercentAsh',
        # 'CoverPercentBeech',
        # 'CoverPercentBirch',
        # 'CoverPercentConifer',
        # 'CoverPercentElm',
        # 'CoverPercentHorseChestnut',
        # 'CoverPercentLime',
        # 'CoverPercentOak',
        # 'CoverPercentPoplar',
        # 'CoverPercentShrub',
        # 'CoverPercentSweetChestnut',
        # 'CoverPercentSycamore',
        # 'CoverPercentWillow',
        'CoverPercentOpenSpace',
        # 'CoverPercentOther',
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


def describe_explanatory_variables(train_set, save_as=".tif", dpi=None, verbose=False):
    """
    Describe basic statistics about the main explanatory variables.

    :param train_set:
    :param save_as:
    :param dpi:
    :param verbose:
    :return:

    **Example**::

        train_set = integrated_data.copy()
    """

    plt.figure(figsize=(14, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    ax1 = plt.subplot2grid((1, 8), (0, 0))
    train_set.Temperature_dif.plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
    ax1.set_xticklabels('')
    plt.xlabel('Temp. Diff.', fontsize=13, labelpad=39)
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(0.05, 1.01)

    ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=2)
    temperature_category = train_set.Temperature_Category.value_counts() / 10
    temperature_category.plot.bar(color='#537979', rot=-45, fontsize=12)
    plt.xticks(range(0, 8), ['< 24°C', '24°C', '25°C', '26°C', '27°C', '28°C', '29°C', '≥ 30°C'], fontsize=12)
    plt.xlabel('Max. Temp.', fontsize=13, labelpad=7)
    plt.ylabel('($\\times$10)', fontsize=12, rotation=0)
    ax2.yaxis.set_label_coords(0.0, 1.01)

    ax3 = plt.subplot2grid((1, 8), (0, 3))
    track_orientation = train_set.Track_Orientation.value_counts() / 100
    track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
    track_orientation.plot.bar(color='#a72a3d', rot=-45, fontsize=12)
    plt.xlabel('Track orientation', fontsize=13)
    plt.ylabel('($\\times$10$^2$)', fontsize=12, rotation=0)
    ax3.yaxis.set_label_coords(0.0, 1.01)

    ax4 = plt.subplot2grid((1, 8), (0, 4))
    train_set.WindSpeed_avg.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Average\nWind speed', fontsize=13, labelpad=29)
    plt.ylabel('($\\times$10 mph)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(0.2, 1.01)

    ax5 = plt.subplot2grid((1, 8), (0, 5))
    train_set.RelativeHumidity_avg.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
    ax5.set_xticklabels('')
    plt.xlabel('Average\nR.H.', fontsize=13, labelpad=29)
    plt.ylabel('(%)', fontsize=12, rotation=0)
    # plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(0.0, 1.01)

    ax6 = plt.subplot2grid((1, 8), (0, 6))
    train_set.TotalPrecipitation_avg.plot.box(color=colour, ax=ax6, widths=0.5, fontsize=12)
    ax6.set_xticklabels('')
    plt.xlabel('Average\nTotal Precip.', fontsize=13, labelpad=29)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(0.0, 1.01)

    ax7 = plt.subplot2grid((1, 8), (0, 7))
    train_set.CoverPercentOpenSpace.plot.box(color=colour, ax=ax7, widths=0.5, fontsize=12)
    ax7.set_xticklabels('')
    plt.xlabel('Open Space\nCoverage', fontsize=13, labelpad=29)
    plt.ylabel('(%)', fontsize=12, rotation=0)
    ax7.yaxis.set_label_coords(0.0, 1.01)

    plt.tight_layout()

    if save_as:
        path_to_file_weather = cdd_prototype_heat_trial(0, "Variables" + save_as)
        save_fig(path_to_file_weather, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)


def logistic_regression_model(trial_id,
                              route_name='Anglia', weather_category='Heat',
                              ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                              shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                              season='summer',
                              describe_var=False,
                              outlier_pctl=100,
                              add_const=True, seed=0, model='logit',
                              plot_roc=False, plot_predicted_likelihood=False,
                              save_as=".tif", dpi=None,
                              update=False, verbose=True):
    """
    Train/test a prototype model in the context of heat-related incidents.

    -------------- | ------------------ | ------------------------------------------------------------------------------
    IncidentReason | IncidentReasonName | IncidentReasonDescription
    -------------- | ------------------ | ------------------------------------------------------------------------------
    IQ             |   TRACK SIGN       | Trackside sign blown down/light out etc.
    IW             |   COLD             | Non severe - Snow/Ice/Frost affecting infr equipment, Takeback Pumps, ...
    OF             |   HEAT/WIND        | Blanket speed restriction for extreme heat or high wind given Group Standards
    Q1             |   TKB PUMPS        | Takeback Pumps
    X4             |   BLNK REST        | Blanket speed restriction for extreme heat or high wind
    XW             |   WEATHER          | Severe Weather not snow affecting infrastructure the responsibility of NR
    XX             |   MISC OBS         | Msc items on line (incl trees) due to effects of Weather responsibility of RT
    -------------- | ------------------ | ------------------------------------------------------------------------------

    **Example**::

        trial_id             = 0
        route_name           = 'Anglia'
        weather_category     = 'Heat'
        ip_start_hrs         = -24
        nip_ip_gap           = -5
        nip_start_hrs        = -24
        shift_yards_same_elr = 220
        shift_yards_diff_elr = 220
        hazard_pctl          = 50
        season               = 'summer'
        describe_var         = False
        outlier_pctl         = 99
        add_const            = True
        seed                 = trial_id
        model                = 'logit'
        plot_roc             = True
        plot_pred_likelihood = True
        save_as              = None
        dpi                  = None
        update               = False
        verbose              = True
    """

    # Get the m_data for Model
    integrated_data = get_incident_locations_weather_and_vegetation(route_name, weather_category,
                                                                    ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                                    shift_yards_same_elr, shift_yards_diff_elr,
                                                                    hazard_pctl, update, verbose)

    # Select season data: 'spring', 'summer', 'autumn', 'winter'
    integrated_data = get_data_by_season(integrated_data, season)

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        integrated_data = integrated_data[
            integrated_data.DelayMinutes <= np.percentile(integrated_data.DelayMinutes, outlier_pctl)]

    # Select features
    explanatory_variables = specify_explanatory_variables_for_model_2()

    # Add the intercept
    if add_const:
        integrated_data['const'] = 1
        explanatory_variables = ['const'] + explanatory_variables

    # Set the outcomes of non-incident records to 0
    integrated_data.loc[integrated_data.IncidentReported == 0, ['DelayMinutes', 'DelayCost', 'IncidentCount']] = 0

    # Select data before 2014 as training data set, with the rest being test set
    train_set = integrated_data[integrated_data.FinancialYear < 2014]
    test_set = integrated_data[integrated_data.FinancialYear == 2014]

    if describe_var:
        describe_explanatory_variables(train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)

    try:
        if model == 'logit':
            mod = sm.Logit(train_set.IncidentReported, train_set[explanatory_variables])
        else:
            mod = sm.Probit(train_set.IncidentReported, train_set[explanatory_variables])
        result = mod.fit(maxiter=10000, full_output=True, disp=False)  # method='newton'
        print(result.summary()) if verbose else print("")

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['OddsRatio'])
        print("\n{}".format(odds_ratios)) if verbose else print("")

        # Prediction
        test_set['incident_prob'] = result.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        fpr, tpr, thr = metrics.roc_curve(test_set.IncidentReported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = metrics.auc(fpr, tpr)
        ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
        threshold = np.min(thr[ind])

        # prediction accuracy
        test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= threshold else 0)
        test = pd.Series(test_set.IncidentReported == test_set.incident_prediction)
        mod_accuracy = np.divide(test.sum(), len(test))
        print("\nAccuracy: %f" % mod_accuracy) if verbose else print("")

        # incident prediction accuracy
        incident_only = test_set[test_set.IncidentReported == 1]
        test_acc = pd.Series(incident_only.IncidentReported == incident_only.incident_prediction)
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
                save_fig(cdd_prototype_heat_trial(trial_id, "ROC" + save_as), dpi=dpi)

        # Plot incident delay minutes against predicted probabilities
        if plot_predicted_likelihood:
            incident_ind = test_set.IncidentReported == 1
            plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            ax.scatter(test_set[incident_ind].incident_prob, test_set[incident_ind].DelayMinutes,
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
                save_fig(cdd_prototype_heat_trial(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_accuracy, incident_accuracy, threshold = np.nan, np.nan, np.nan

    return integrated_data, train_set, test_set, result, mod_accuracy, incident_accuracy, threshold
