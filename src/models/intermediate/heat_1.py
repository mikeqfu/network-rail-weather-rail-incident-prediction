""" A prediction model of heat-related rail incidents (based on the prototype). """

import datetime
import os

import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import statsmodels.discrete.discrete_model as sm_dcm
import statsmodels.tools as sm_tools
from pyhelpers.geom import get_geometric_midpoint
from pyhelpers.store import load_pickle, save_fig, save_pickle
from sklearn import metrics

from models.intermediate import integrator
from models.tools import categorise_temperatures, categorise_track_orientations, get_data_by_season
from models.tools import cd_intermediate_fig_pub, cdd_intermediate_heat, cdd_intermediate_heat_trial
from mssqlserver.metex import view_schedule8_costs_by_datetime_location_reason
from settings import mpl_preferences, pd_preferences
from utils import make_filename
from weather import midas, ukcp

mpl_preferences(reset=False)
pd_preferences(reset=False)
plt.rc('font', family='Times New Roman')


# == Tools ============================================================================================

def define_prior_ip(incidents, prior_ip_start_hrs):
    """

    :param incidents:
    :param prior_ip_start_hrs:
    :return:
    """

    incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
    # End date and time of the prior IP
    incidents['Critical_EndDateTime'] = incidents.StartDateTime.dt.round('H')
    # Start date and time of the prior IP
    critical_start_dt = incidents.Critical_EndDateTime.map(
        lambda x: x + pd.Timedelta(hours=prior_ip_start_hrs if x.time() > datetime.time(9) else prior_ip_start_hrs * 2))
    incidents.insert(incidents.columns.get_loc('Critical_EndDateTime'), 'Critical_StartDateTime', critical_start_dt)
    # Prior-IP dates of each incident
    incidents['Critical_Period'] = incidents.apply(
        lambda x: pd.interval_range(x.Critical_StartDateTime, x.Critical_EndDateTime), axis=1)

    return incidents


def get_prior_ip_ukcp09_stats(incidents):
    """
    Get prior-IP statistics of weather variables for each incident.

    :param incidents: data of incidents
    :type incidents: pandas.DataFrame
    :return: statistics of weather observation data for each incident record during the prior IP
    :rtype: pandas.DataFrame
    """

    prior_ip_weather_stats = incidents.apply(
        lambda x: pd.Series(integrator.integrate_pip_ukcp09_data(x.Weather_Grid, x.Critical_Period)), axis=1)

    w_col_names = integrator.specify_weather_variable_names(
        integrator.specify_weather_stats_calculations()) + ['Hottest_Heretofore']

    prior_ip_weather_stats.columns = w_col_names

    prior_ip_weather_stats['Temperature_Change_max'] = \
        abs(prior_ip_weather_stats.Maximum_Temperature_max - prior_ip_weather_stats.Minimum_Temperature_min)
    prior_ip_weather_stats['Temperature_Change_min'] = \
        abs(prior_ip_weather_stats.Maximum_Temperature_min - prior_ip_weather_stats.Minimum_Temperature_max)

    return prior_ip_weather_stats


def get_prior_ip_radtob_stats(incidents, use_suppl_dat=False):
    """
    Get prior-IP statistics of radiation data for each incident.

    :param incidents: data of incidents
    :type incidents: pandas.DataFrame
    :param use_suppl_dat:
    :type use_suppl_dat:
    :return: statistics of radiation data for each incident record during the prior IP
    :rtype: pandas.DataFrame
    """

    prior_ip_radtob_stats = incidents.apply(
        lambda x: pd.Series(integrator.integrate_pip_midas_radtob(
            x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat)), axis=1)

    # r_col_names = integrator.specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
    # r_col_names += ['GLBL_IRAD_AMT_total']
    prior_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

    return prior_ip_radtob_stats


def define_latent_and_non_ip(incidents, prior_ip_data, non_ip_start_hrs):
    """

    :param incidents:
    :param prior_ip_data:
    :param non_ip_start_hrs:
    :return:
    """

    non_ip_data = incidents.copy(deep=True)  # Get Weather data that did not cause any incident

    def define_latent_period(route_name, ip_max_temp_max, ip_start_dt, non_ip_start_hrs_):
        """
        Determine latent period for a given date/time and the maximum temperature.

        :param route_name:
        :param ip_max_temp_max:
        :param ip_start_dt:
        :param non_ip_start_hrs_:
        :return:
        """

        if route_name == 'Anglia':
            if 24 <= ip_max_temp_max <= 28:
                lp = -20
            elif ip_max_temp_max > 28:
                lp = -13
            else:
                lp = 0
        elif route_name == 'Wessex':
            if 24 <= ip_max_temp_max <= 28:
                lp = -30
            elif ip_max_temp_max > 28:
                lp = -25
            else:
                lp = 0
        elif route_name == 'North and East':
            if 24 <= ip_max_temp_max <= 28:
                lp = -18
            elif ip_max_temp_max > 28:
                lp = -16
            else:
                lp = 0
        else:  # route_name == 'Wales':
            if 24 <= ip_max_temp_max <= 28:
                lp = -19
            elif ip_max_temp_max > 28:
                lp = -5
            else:
                lp = 0

        critical_end_dt = ip_start_dt + datetime.timedelta(days=lp)
        critical_start_dt = critical_end_dt + datetime.timedelta(hours=non_ip_start_hrs_)
        critical_period = pd.interval_range(critical_start_dt, critical_end_dt)

        return critical_start_dt, critical_end_dt, critical_period

    non_ip_data[['Critical_StartDateTime', 'Critical_EndDateTime', 'Critical_Period']] = prior_ip_data.apply(
        lambda x: pd.Series(
            define_latent_period(x.Route, x.Maximum_Temperature_max, x.Critical_StartDateTime, non_ip_start_hrs)),
        axis=1)

    return non_ip_data


def get_non_ip_weather_stats(non_ip_data, prior_ip_data):
    """
    Get prior-IP statistics of weather variables for each incident.

    :param non_ip_data: non-IP data
    :type non_ip_data: pandas.DataFrame
    :param prior_ip_data: prior-IP data
    :type prior_ip_data: pandas.DataFrame
    :return: statistics of weather observation data for each incident record during the non-incident period
    :rtype: pandas.DataFrame
    """

    non_ip_weather_stats = non_ip_data.apply(
        lambda x: pd.Series(integrator.integrate_nip_ukcp09_data(
            x.Weather_Grid, x.Critical_Period, prior_ip_data, x.StanoxSection)), axis=1)

    w_col_names = integrator.specify_weather_variable_names(
        integrator.specify_weather_stats_calculations()) + ['Hottest_Heretofore']

    non_ip_weather_stats.columns = w_col_names

    non_ip_weather_stats['Temperature_Change_max'] = \
        abs(non_ip_weather_stats.Maximum_Temperature_max - non_ip_weather_stats.Minimum_Temperature_min)
    non_ip_weather_stats['Temperature_Change_min'] = \
        abs(non_ip_weather_stats.Maximum_Temperature_min - non_ip_weather_stats.Minimum_Temperature_max)

    return non_ip_weather_stats


def get_non_ip_radtob_stats(non_ip_data, prior_ip_data, use_suppl_dat):
    """
    Get prior-IP statistics of radiation data for each incident.

    :param non_ip_data: non-IP data
    :type non_ip_data: pandas.DataFrame
    :param prior_ip_data: prior-IP data
    :type prior_ip_data: pandas.DataFrame
    :param use_suppl_dat:
    :type use_suppl_dat:
    :return: statistics of radiation data for each incident record during the non-incident period
    :rtype: pandas.DataFrame
    """

    non_ip_radtob_stats = non_ip_data.apply(
        lambda x: pd.Series(integrator.integrate_nip_midas_radtob(
            x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat, prior_ip_data, x.StanoxSection)), axis=1)

    for i in range(len(non_ip_data)):
        try:
            integrator.integrate_nip_midas_radtob(
                non_ip_data.Met_SRC_ID.iloc[i], non_ip_data.Critical_Period.iloc[i], non_ip_data.Route.iloc[i],
                False, prior_ip_data, non_ip_data.StanoxSection.iloc[i])
        except Exception as e:
            print(i, e)
            break

    # r_col_names = integrator.specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
    # r_col_names += ['GLBL_IRAD_AMT_total']
    non_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

    return non_ip_radtob_stats


# == Data of weather conditions =======================================================================

def get_incident_location_weather(route_name=None, weather_category='Heat', season='summer',
                                  prior_ip_start_hrs=-24, non_ip_start_hrs=-24,
                                  trial_only=True, random_state=None, illustrate_buf_cir=False,
                                  update=False, verbose=False):
    """
    Process data of weather conditions for each incident location.

    **Example**::

        route_name         = ['Anglia', 'Wessex', 'Wales', 'North and East']
        weather_category   = 'Heat'
        season             = 'summer'
        prior_ip_start_hrs = -24
        non_ip_start_hrs   = -24
        trial_only         = False
        random_state       = None
        illustrate_buf_cir = False
        update             = False
        verbose            = True

    .. note::

        Note that the 'Critical_EndDateTime' would be based on the 'Critical_StartDateTime' if we consider the weather
        conditions on the day of incident occurrence; 'StartDateTime' otherwise.
    """

    pickle_filename = make_filename("weather", route_name, None,
                                    "_".join([season] if isinstance(season, str) else season),
                                    str(prior_ip_start_hrs) + 'h', str(non_ip_start_hrs) + 'h',
                                    "trial" if trial_only else "", sep="_")
    path_to_pickle = cdd_intermediate_heat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_location_weather = load_pickle(path_to_pickle)

    else:
        try:
            # -- Incidents data -----------------------------------------------------------------------

            metex_incidents = view_schedule8_costs_by_datetime_location_reason(route_name, weather_category)
            # incidents_all.rename(columns={'Year': 'FinancialYear'}, inplace=True)
            incidents_by_season = get_data_by_season(metex_incidents, season)
            incidents = incidents_by_season[
                (incidents_by_season.StartDateTime >= datetime.datetime(2006, 1, 1)) &
                (incidents_by_season.StartDateTime < datetime.datetime(2017, 1, 1))]

            if trial_only:  # For testing purpose only
                incidents = incidents.sample(n=10, random_state=random_state)

            if 'StartXY' not in incidents.columns or 'EndXY' not in incidents.columns:
                incidents['StartLongLat'] = incidents.apply(
                    lambda x: shapely.geometry.Point(x.StartLongitude, x.StartLatitude), axis=1)
                incidents['EndLongLat'] = incidents.apply(
                    lambda x: shapely.geometry.Point(x.EndLongitude, x.EndLatitude), axis=1)
                from pyhelpers.geom import wgs84_to_osgb36
                incidents['StartEasting'], incidents['StartNorthing'] = \
                    wgs84_to_osgb36(incidents.StartLongitude.values, incidents.StartLatitude.values)
                incidents['EndEasting'], incidents['EndNorthing'] = \
                    wgs84_to_osgb36(incidents.EndLongitude.values, incidents.EndLatitude.values)
                incidents['StartXY'] = incidents.apply(
                    lambda x: shapely.geometry.Point(x.StartEasting, x.StartNorthing), axis=1)
                incidents['EndXY'] = incidents.apply(
                    lambda x: shapely.geometry.Point(x.EndEasting, x.EndNorthing), axis=1)

            # Append 'MidpointXY' column
            incidents['MidpointXY'] = incidents.apply(
                lambda x: get_geometric_midpoint(x.StartXY, x.EndXY, as_geom=True), axis=1)

            # Make a buffer zone for Weather data aggregation
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: integrator.create_circle_buffer_upon_weather_grid(
                    x.StartXY, x.EndXY, x.MidpointXY, whisker=0), axis=1)

            # -- Weather data -------------------------------------------------------------------------

            obs_grids = ukcp.get_observation_grids()  # Grids for observing weather conditions
            obs_centroid_geom = shapely.geometry.MultiPoint(list(obs_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

            met_stations = midas.get_radiation_stations_information()  # Met station locations
            met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.EN_GEOM))

            # -- Data integration in the spatial context ----------------------------------------------

            incidents['Start_Pseudo_Grid_ID'] = incidents.StartXY.map(  # Start
                lambda x: integrator.find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
            incidents = incidents.join(obs_grids, on='Start_Pseudo_Grid_ID')

            incidents['End_Pseudo_Grid_ID'] = incidents.EndXY.map(  # End
                lambda x: integrator.find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
            incidents = incidents.join(obs_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')

            # Modify column names
            for p in ['Start', 'End']:
                a = [c for c in incidents.columns if c.endswith(p)]
                b = [p + '_' + c if c == 'Grid' else p + '_Grid_' + c for c in obs_grids.columns]
                incidents.rename(columns=dict(zip(a, b)), inplace=True)

            # Find all Weather observation grids that intersect with the created buffer zone for each incident location
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(
                lambda x: integrator.find_intersecting_weather_grid(x, obs_grids, obs_grids_geom))

            incidents['Met_SRC_ID'] = incidents.MidpointXY.map(
                lambda x: integrator.find_closest_met_stn(x, met_stations, met_stations_geom))

            if illustrate_buf_cir:  # Illustration of the buffer circle
                start_point, end_point, midpoint = incidents[['StartXY', 'EndXY', 'MidpointXY']].iloc[0]
                bf_circle = integrator.create_circle_buffer_upon_weather_grid(
                    start_point, end_point, midpoint, whisker=0)
                i_obs_grids = integrator.find_intersecting_weather_grid(
                    bf_circle, obs_grids, obs_grids_geom, as_grid_id=False)
                plt.figure(figsize=(7, 6))
                ax = plt.subplot2grid((1, 1), (0, 0))
                for g in i_obs_grids:
                    x_, y_ = g.exterior.xy
                    ax.plot(x_, y_, color='#433f3f')
                ax.plot([], 's', label="Weather observation grid", ms=16, color='none', markeredgecolor='#433f3f')
                x_, y_ = bf_circle.exterior.xy
                ax.plot(x_, y_)
                ax.plot([], 'r', marker='o', markersize=15, linestyle='None', fillstyle='none', label='Buffer zone')
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

            # -- Data integration for the specified prior-IP ------------------------------------------

            incidents = define_prior_ip(incidents, prior_ip_start_hrs)

            # Get prior-IP statistics of weather variables for each incident.
            prior_ip_weather_stats = get_prior_ip_ukcp09_stats(incidents)

            # Get prior-IP statistics of radiation data for each incident.
            prior_ip_radtob_stats = get_prior_ip_radtob_stats(incidents, use_suppl_dat=True)

            prior_ip_data = incidents.join(prior_ip_weather_stats).join(prior_ip_radtob_stats)

            prior_ip_data['Incident_Reported'] = 1

            # -- Data integration for the specified non-IP --------------------------------------------

            non_ip_data = define_latent_and_non_ip(incidents, prior_ip_data, non_ip_start_hrs)

            non_ip_weather_stats = get_non_ip_weather_stats(non_ip_data, prior_ip_data)

            non_ip_radtob_stats = get_non_ip_radtob_stats(non_ip_data, prior_ip_data, use_suppl_dat=True)

            non_ip_data = non_ip_data.join(non_ip_weather_stats).join(non_ip_radtob_stats)

            non_ip_data['Incident_Reported'] = 0

            # -- Merge "prior_ip_data" and "non_ip_data" ----------------------------------------------
            incident_location_weather = pd.concat([prior_ip_data, non_ip_data], axis=0, ignore_index=True, sort=False)

            # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
            incident_location_weather = incident_location_weather.join(
                categorise_track_orientations(incident_location_weather))

            # Categorise temperature: 25, 26, 27, 28, 29, 30
            incident_location_weather = incident_location_weather.join(
                categorise_temperatures(incident_location_weather, column_name='Maximum_Temperature_max'))

            # incident_location_weather.dropna(subset=w_col_names, inplace=True)

            save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get weather conditions for the incident locations. {}.".format(e))
            incident_location_weather = pd.DataFrame()

    return incident_location_weather


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
        get_incident_location_weather(route_name, latent_period=-d, update=update, verbose=verbose)
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
        path_to_fig = cdd_intermediate_heat_trial(0, "Temperature deviation" + save_as)
        save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)


# == Modelling trials =================================================================================


def specify_explanatory_variables():
    """
    Specify the explanatory variables considered in this model.
    :return:
    """

    return [
        # 'Maximum_Temperature_max',
        # 'Maximum_Temperature_min',
        # 'Maximum_Temperature_average',
        # 'Minimum_Temperature_max',
        # 'Minimum_Temperature_min',
        # 'Minimum_Temperature_average',
        # 'Temperature_Change_average',
        'Precipitation_max',
        # 'Precipitation_min',
        # 'Precipitation_average',
        'Hottest_Heretofore',
        'Temperature_Change_max',
        # 'Temperature_Change_min',
        'GLBL_IRAD_AMT_total',
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


def describe_explanatory_variables(train_set, save_as=".pdf", dpi=None, verbose=False):
    """
    Describe basic statistics about the main explanatory variables.

    :param train_set:
    :param save_as:
    :param dpi:
    :param verbose:
    :return:

    **Example**::

        train_set = incident_location_weather.dropna().copy()
    """

    plt.figure(figsize=(14, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=3)
    train_set.Temperature_Category.value_counts().plot.bar(color='#537979', rot=0, fontsize=12)
    plt.xticks(range(0, 8), ['<24', '24', '25', '26', '27', '28', '29', '≥30'], rotation=0, fontsize=12)
    ax1.text(7.5, -0.2, '(°C)', fontsize=12)
    plt.xlabel('Maximum temperature', fontsize=13, labelpad=8)
    plt.ylabel('Frequency', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(0.0, 1.01)

    ax2 = plt.subplot2grid((1, 9), (0, 3))
    train_set.Temperature_Change_max.plot.box(color=colour, ax=ax2, widths=0.5, fontsize=12)
    ax2.set_xticklabels('')
    plt.xlabel('Temperature\nchange', fontsize=13, labelpad=10)
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax2.yaxis.set_label_coords(0.05, 1.01)

    ax3 = plt.subplot2grid((1, 9), (0, 4), colspan=2)
    orient_cats = [x.replace('Track_Orientation_', '') for x in train_set.columns if x.startswith('Track_Orientation_')]
    track_orientation = pd.Series([sum(train_set.Track_Orientation == x) for x in orient_cats], index=orient_cats)
    track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
    track_orientation.plot.bar(color='#a72a3d', rot=0, fontsize=12)
    # ax3.set_yticks(range(0, track_orientation.max() + 1, 100))
    plt.xlabel('Track orientation', fontsize=13, labelpad=8)
    plt.ylabel('Count', fontsize=12, rotation=0)
    ax3.yaxis.set_label_coords(0.0, 1.01)

    ax4 = plt.subplot2grid((1, 9), (0, 6))
    train_set.GLBL_IRAD_AMT_total.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Maximum\nirradiation', fontsize=13, labelpad=10)
    plt.ylabel('(KJ/m$^2$)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(0.2, 1.01)

    ax5 = plt.subplot2grid((1, 9), (0, 7))
    train_set.Precipitation_max.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
    ax5.set_xticklabels('')
    plt.xlabel('Maximum\nprecipitation', fontsize=13, labelpad=10)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(0.0, 1.01)

    ax6 = plt.subplot2grid((1, 9), (0, 8))
    hottest_heretofore = train_set.Hottest_Heretofore.value_counts()
    hottest_heretofore.plot.bar(color='#a72a3d', rot=0, fontsize=12)
    plt.xlabel('Hottest\nheretofore', fontsize=13, labelpad=5)
    plt.ylabel('Frequency', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(0.0, 1.01)
    # ax6.set_yticks(range(0, hottest_heretofore.max() + 1, 100))
    ax6.set_xticklabels(['False', 'True'], rotation=0)

    plt.tight_layout()

    if save_as:
        path_to_fig_file = cd_intermediate_fig_pub("Variables" + save_as)
        save_fig(path_to_fig_file, dpi, verbose=verbose, conv_svg_to_emf=True)


def logistic_regression_model(trial_id,
                              route=None, weather_category='Heat', season='summer',
                              prior_ip_start_hrs=-0, non_ip_start_hrs=-0,
                              outlier_pctl=100,
                              describe_var=False,
                              add_const=True, seed=0, model='logit',
                              plot_roc=False, plot_predicted_likelihood=False,
                              save_as=".svg", dpi=None,
                              verbose=True):
    """
    Train/test a logistic regression model for predicting heat-related incidents.

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

        trial_id                  = 0
        route_name                = 'Anglia'
        weather_category          = 'Heat'
        season                    = 'summer'
        prior_ip_start_hrs        = -0
        non_ip_start_hrs          = -0
        outlier_pctl              = 95
        describe_var              = False
        add_const                 = True
        seed                      = 0
        model                     = 'logit',
        plot_roc                  = False
        plot_predicted_likelihood = False
        save_as                   = None
        dpi                       = None
        verbose                   = True

        m_data = incident_location_weather.copy()
    """

    # Get the m_data for modelling
    m_data = get_incident_location_weather(route, weather_category, season, prior_ip_start_hrs, non_ip_start_hrs)

    # temp_data = [load_pickle(cdd_mod_heat_inter("Slices", f)) for f in os.listdir(cdd_mod_heat_inter("Slices"))]
    # m_data = pd.concat(temp_data, ignore_index=True, sort=False)

    m_data.dropna(subset=['GLBL_IRAD_AMT_total'], inplace=True)
    m_data.GLBL_IRAD_AMT_total = m_data.GLBL_IRAD_AMT_total / 1000

    # Select features
    explanatory_variables = specify_explanatory_variables()

    for v in explanatory_variables:
        if not m_data[m_data[v].isna()].empty:
            m_data.dropna(subset=[v], inplace=True)

    m_data = m_data[explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime', 'DelayMinutes']]

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        m_data = m_data[m_data.DelayMinutes <= np.percentile(m_data.DelayMinutes, outlier_pctl)]

    # Add the intercept
    if add_const:
        m_data = sm_tools.tools.add_constant(m_data, prepend=True, has_constant='skip')  # data['const'] = 1.0
        explanatory_variables = ['const'] + explanatory_variables

    # m_data = m_data.loc[:, (m_data != 0).any(axis=0)]
    # explanatory_variables = [x for x in explanatory_variables if x in m_data.columns]

    # Select data before 2014 as training data set, with the rest being test set
    train_set = m_data[m_data.StartDateTime < datetime.datetime(2016, 1, 1)]
    test_set = m_data[m_data.StartDateTime >= datetime.datetime(2016, 1, 1)]

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
        fpr, tpr, thr = metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = metrics.auc(fpr, tpr)
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

        # Plot ROC
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
                path_to_roc_fig = cdd_intermediate_heat_trial(trial_id, "ROC" + save_as)
                save_fig(path_to_roc_fig, dpi=dpi, verbose=verbose)

        # Plot incident delay minutes against predicted probabilities
        if plot_predicted_likelihood:
            incident_ind = test_set.Incident_Reported == 1
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
                path_to_pred_fig = cdd_intermediate_heat_trial(trial_id, "Predicted-likelihood" + save_as)
                save_fig(path_to_pred_fig, dpi=dpi, verbose=verbose)

    except Exception as e:
        print(e)
        result = e
        mod_accuracy, incident_accuracy, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_accuracy, incident_accuracy, threshold
