""" Testing models """

import os
import random

import datetime_truncate
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import statsmodels.discrete.discrete_model as sm_dcm
from pyhelpers.store import load_pickle, save_fig, save_pickle, save_svg_as_emf

import intermediate.utils
import mssqlserver.metex
import settings

settings.pd_preferences()


def cd_prototype_heat(*sub_dir):
    path = intermediate.utils.cdd_intermediate("heat")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


def cdd_prototype_heat(*sub_dir):
    path = cd_prototype_heat("data")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Models\\intermediate\\heat\\x" and sub-directories
def cdd_prototype_heat_mod(trial_id=0, *sub_dir):
    path = cd_prototype_heat("{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Integrate data of Incidents and Weather """


def prep_incidents_with_weather(route_name=None, weather_category='Heat',
                                on_region=True, on_reason=None, on_season='summer',
                                test_only=False, random_select=False, use_buffer_zone=False, illustrate_buf_cir=False,
                                lp=5 * 24, non_ip=24):
    # Incidents data
    incidents_data = mssqlserver.metex.view_schedule8_cost_by_datetime_location_reason(route_name)
    incidents = incidents_data.copy()
    incidents = incidents[incidents.WeatherCategory.isin(['', weather_category])]

    # Investigate only the following incident reasons
    if on_reason is not None:
        reason_codes = mssqlserver.metex.get_incident_reason_metadata().index.tolist()
        assert all(x for x in on_reason if x in reason_codes) and isinstance(on_reason, list)
    else:
        on_reason = ['IR', 'XH', 'IB', 'JH']
    incidents = incidents[incidents.IncidentReasonCode.isin(on_reason)]

    # Select data for specific seasons
    incidents = intermediate.utils.get_data_by_season(incidents, on_season)

    # Select data for specific regions
    if on_region:
        incidents = incidents[incidents.Route.isin(['South East', 'Anglia', 'Wessex'])]

    if test_only:  # For initial testing ...
        if random_select:
            incidents = incidents.iloc[random.sample(range(len(incidents)), test_only), :]
        else:
            incidents = incidents.iloc[-test_only - 1:-1, :]

    # --------------------------------------------------------------------------------------------------------
    """ In the spatial context """

    if use_buffer_zone:
        # Make a buffer zone for gathering data of weather observations
        print("Creating a buffer zone for each incident location ... ", end="")
        incidents['Buffer_Zone'] = incidents.apply(
            lambda x: intermediate.utils.create_circle_buffer_upon_weather_cell(
                x.MidLonLat, x.StartLonLat, x.EndLonLat, whisker_km=0.0), axis=1)
        print("Done.")

        # Find all weather observation grids intersecting with the buffer zone for each incident location
        print("Delimiting zone for calculating weather statistics  ... ", end="")
        incidents['WeatherCell_Obs'] = incidents.Buffer_Zone.map(
            lambda x: intermediate.utils.find_intersecting_weather_cells(x))
        print("Done.")

        if illustrate_buf_cir:
            # Example 1
            x_incident_start, x_incident_end = incidents.StartLonLat.iloc[12], incidents.EndLonLat.iloc[12]
            x_midpoint = incidents.MidLonLat.iloc[12]
            intermediate.utils.illustrate_buffer_circle(x_midpoint, x_incident_start, x_incident_end, whisker_km=0.0,
                                                        legend_loc='upper left')
            save_fig(intermediate.utils.cdd_intermediate("Heat", "Buffer_circle_example_1.png"), dpi=1200)
            # Example 2
            x_incident_start, x_incident_end = incidents.StartLonLat.iloc[16], incidents.EndLonLat.iloc[16]
            x_midpoint = incidents.MidLonLat.iloc[16]
            intermediate.utils.illustrate_buffer_circle(x_midpoint, x_incident_start, x_incident_end, whisker_km=0.0,
                                                        legend_loc='lower right')
            save_fig(intermediate.utils.cdd_intermediate("Heat", "Buffer_circle_example_2.png"), dpi=1200)

    # --------------------------------------------------------------------------------------------------------
    """ In the temporal context """

    incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
    # Incident period (IP)
    incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
        lambda x: x.replace(hour=0, minute=0, second=0)) - pd.DateOffset(hours=24)
    incidents['Critical_EndDateTime'] = incidents.StartDateTime.map(
        lambda x: x.replace(minute=0) + pd.Timedelta(hours=1) if x.minute > 45 else x.replace(minute=0))
    incidents['Critical_Period'] = incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

    # --------------------------------------------------------------------------------------------------------
    """ Calculations """

    # Specify the statistics needed for Weather observations (except radiation)
    def specify_weather_stats_calculations():
        """
        :rtype: dict
        """
        weather_stats_calculations = {'Temperature': (max, min, np.average),
                                      'RelativeHumidity': max,
                                      'WindSpeed': max,
                                      'WindGust': max,
                                      'Snowfall': sum,
                                      'TotalPrecipitation': sum}
        return weather_stats_calculations

    # Calculate average wind speed and direction
    def calculate_wind_averages(wind_speeds, wind_directions):
        u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
        v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
        uav, vav = np.mean(u), np.mean(v)  # sum up all u and v values and average it
        average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed
        # Calculate average wind direction
        if uav == 0:
            average_wind_direction = 0 if vav == 0 else (360 if vav > 0 else 180)
        else:
            average_wind_direction = (270 if uav > 0 else 90) - 180 / np.pi * np.arctan(vav / uav)
        return average_wind_speed, average_wind_direction

    # Get all Weather variable names
    def specify_weather_variable_names():
        weather_stats_calculations = specify_weather_stats_calculations()
        stats_names = [x + '_max' for x in weather_stats_calculations.keys()]
        stats_names[stats_names.index('TotalPrecipitation_max')] = 'TotalPrecipitation_sum'
        stats_names[stats_names.index('Snowfall_max')] = 'Snowfall_sum'
        stats_names.insert(stats_names.index('Temperature_max') + 1, 'Temperature_min')
        stats_names.insert(stats_names.index('Temperature_min') + 1, 'Temperature_avg')
        stats_names.insert(stats_names.index('Temperature_avg') + 1, 'Temperature_dif')
        wind_speed_variables = ['WindSpeed_avg', 'WindDirection_avg']
        weather_variable_names = stats_names + wind_speed_variables + ['Hottest_Heretofore']
        return weather_variable_names

    # Get the highest temperature of year by far
    def get_highest_temperature_of_year_by_far(weather_cell_id, period_start_dt):
        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        yr_start_dt = datetime_truncate.truncate_year(period_start_dt)
        # Specify a directory to pickle slices of weather observation data
        weather_dat_dir = intermediate.utils.cd_intermediate_dat("weather-slices")
        # Get weather observations
        weather_obs = mssqlserver.metex.view_weather_by_id_datetime(
            weather_cell_id, yr_start_dt, period_start_dt, pickle_it=False, dat_dir=weather_dat_dir)
        weather_obs_by_far = weather_obs[
            (weather_obs.DateTime < period_start_dt) & (weather_obs.DateTime > yr_start_dt)]
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
            # Find out weather cell ids
            weather_cell_obs = weather_obs.WeatherCell.unique()
            weather_cell_id = weather_cell_obs[0] if len(weather_cell_obs) == 1 else tuple(weather_cell_obs)
            obs_start_dt = weather_obs.DateTime.min()  # Observation start datetime
            # Whether it is the hottest of the year by far
            highest_temp = get_highest_temperature_of_year_by_far(weather_cell_id, obs_start_dt)
            highest_temp_obs = weather_stats.Temperature['max'][0]
            weather_stats['Hottest_Heretofore'] = 1 if highest_temp_obs >= highest_temp else 0
            weather_stats.columns = specify_weather_variable_names()
            # Scale up variable
            scale_up_vars = ['WindSpeed_max', 'WindGust_max', 'WindSpeed_avg', 'RelativeHumidity_max',
                             'Snowfall_sum']
            weather_stats[scale_up_vars] = weather_stats[scale_up_vars] / 10.0
            weather_stats.index.name = None
            if values_only:
                weather_stats = weather_stats.values[0].tolist()
        return weather_stats

    # Calculate weather statistics based on the retrieved weather observation data
    def get_ip_weather_stats(weather_cell_id, start_dt, end_dt):
        """
        :param weather_cell_id: e.g. weather_cell_id = incidents.WeatherCell.iloc[0]
        :param start_dt: e.g. start_dt = incidents.Critical_StartDateTime.iloc[0]
        :param end_dt: e.g. end_dt = incidents.Critical_EndDateTime.iloc[0]
        :return:
        """
        # Specify a directory to pickle slices of weather observation data
        weather_dat_dir = intermediate.utils.cd_intermediate_dat("weather-slices")
        # Query weather observations
        ip_weather = mssqlserver.metex.view_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                                   pickle_it=False, dat_dir=weather_dat_dir)
        # Calculate basic statistics of the weather observations
        weather_stats_calculations = specify_weather_stats_calculations()
        weather_stats = calculate_weather_stats(ip_weather, weather_stats_calculations, values_only=True)
        return weather_stats

    # Prior-IP ---------------------------------------------------
    print("Calculating weather statistics for IPs ... ", end="")
    incidents[specify_weather_variable_names()] = incidents.apply(
        lambda x: pd.Series(get_ip_weather_stats(
            x.WeatherCell_Obs if use_buffer_zone else x.WeatherCell,
            x.Critical_StartDateTime, x.Critical_EndDateTime)), axis=1)
    print("Done.")

    gc.collect()

    incidents.Hottest_Heretofore = incidents.Hottest_Heretofore.astype(int)
    incidents['Incident_Reported'] = 1

    # Non-IP -----------------------------------------------------------------------------------------------
    non_ip_data = incidents.copy(deep=True)

    non_ip_data.Critical_StartDateTime = incidents.Critical_StartDateTime - pd.DateOffset(hours=non_ip + lp)
    # non_ip_data.Critical_EndDateTime = non_ip_data.Critical_StartDateTime + pd.DateOffset(hours=non_ip)
    non_ip_data.Critical_EndDateTime = non_ip_data.Critical_StartDateTime + incidents.Critical_Period

    # Gather gridded Weather observations of the corresponding non-incident period for each incident record
    def get_non_ip_weather_stats(weather_cell_id, start_dt, end_dt, stanox_section):
        """
        :param weather_cell_id: weather_cell_id = non_ip_data.WeatherCell.iloc[0]
        :param start_dt: e.g. start_dt = non_ip_data.Critical_StartDateTime.iloc[0]
        :param end_dt: e.g. end_dt = non_ip_data.Critical_EndDateTime.iloc[0]
        :param stanox_section: e.g. stanox_section = non_ip_data.StanoxSection.iloc[0]
        :return:
        """
        # Specify a directory to pickle slices of weather observation data
        weather_dat_dir = intermediate.utils.cd_intermediate_dat("weather-slices")
        # Query weather observations
        non_ip_weather = mssqlserver.metex.view_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                                       pickle_it=False, dat_dir=weather_dat_dir)
        # Get all incident period data on the same section
        ip_overlap = incidents[
            (incidents.StanoxSection == stanox_section) &
            (((incidents.Critical_StartDateTime <= start_dt) & (incidents.Critical_EndDateTime >= start_dt)) |
             ((incidents.Critical_StartDateTime <= end_dt) & (incidents.Critical_EndDateTime >= end_dt)))]
        # Skip data of Weather causing Incidents at around the same time; but
        if not ip_overlap.empty:
            non_ip_weather = non_ip_weather[
                (non_ip_weather.DateTime < min(ip_overlap.Critical_StartDateTime)) |
                (non_ip_weather.DateTime > max(ip_overlap.Critical_EndDateTime))]
        # Calculate weather statistics
        weather_stats_calculations = specify_weather_stats_calculations()
        weather_stats = calculate_weather_stats(non_ip_weather, weather_stats_calculations, values_only=True)
        return weather_stats

    print("Calculating weather statistics for Non-IPs ... ", end="")
    non_ip_data[specify_weather_variable_names()] = non_ip_data.apply(
        lambda x: pd.Series(get_non_ip_weather_stats(
            x.WeatherCell_Obs if use_buffer_zone else x.WeatherCell,
            x.Critical_StartDateTime, x.Critical_EndDateTime, x.StanoxSection)), axis=1)
    print("Done.")

    gc.collect()

    non_ip_data['Incident_Reported'] = 0
    # non_ip_data.DelayMinutes = 0.0
    non_ip_data.DelayCost = 0.0

    # Combine IP data and Non-IP data ----------------------------------------------------
    incidents_and_weather = pd.concat([incidents, non_ip_data], axis=0, ignore_index=True)

    # Get track orientation
    incidents_and_weather = intermediate.utils.categorise_track_orientations(incidents_and_weather)

    # Create temperature categories
    incidents_and_weather = intermediate.utils.categorise_temperatures(incidents_and_weather, 'Temperature_max')

    return incidents_and_weather


# Integrate incidents and weather data
def fetch_incidents_with_weather(trial_id=0, route_name=None, weather_category='Heat',
                                 on_region=True, on_reason=None, on_season='summer',
                                 test_only=False, random_select=False, use_buffer_zone=False, illustrate_buf_cir=False,
                                 lp=5 * 24, non_ip=24,
                                 update=False):
    pickle_filename = mssqlserver.metex.make_filename(
        "data", route_name, weather_category, "regional" if on_region else "",
        "-".join([on_season] if isinstance(on_season, str) else on_season), "trial" if test_only else "", sep="-")
    path_to_pickle = cdd_prototype_heat_mod(trial_id, pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        incidents_and_weather = load_pickle(path_to_pickle)
    else:
        try:
            incidents_and_weather = prep_incidents_with_weather(route_name, weather_category,
                                                                on_region, on_reason, on_season,
                                                                test_only, random_select, use_buffer_zone,
                                                                illustrate_buf_cir,
                                                                lp, non_ip)
            save_pickle(incidents_and_weather, path_to_pickle)
        except Exception as e:
            print(e)
            incidents_and_weather = None
    return incidents_and_weather


# ====================================================================================================================
""" Model trials """


def specify_explanatory_variables():
    return [
        # 'Maximum_Temperature_max [-inf, 24.0)°C',
        # 'Maximum_Temperature_max [24.0, 25.0)°C',
        # 'Maximum_Temperature_max [25.0, 26.0)°C',
        # 'Maximum_Temperature_max [26.0, 27.0)°C',
        # 'Maximum_Temperature_max [27.0, 28.0)°C',
        # 'Maximum_Temperature_max [28.0, 29.0)°C',
        # 'Maximum_Temperature_max [29.0, 30.0)°C',
        # 'Maximum_Temperature_max [30.0, inf)°C'
        # 'Temperature_max',
        # 'Temperature_min',
        'Temperature_dif',
        # 'Temperature_max [-inf, 24.0)°C',
        'Temperature_max [24.0, 25.0)°C',
        'Temperature_max [25.0, 26.0)°C',
        'Temperature_max [26.0, 27.0)°C',
        'Temperature_max [27.0, 28.0)°C',
        'Temperature_max [28.0, 29.0)°C',
        'Temperature_max [29.0, 30.0)°C',
        'Temperature_max [30.0, inf)°C',
        # 'Maximum_Temperature_max',
        # 'Maximum_Temperature_min',
        # 'Maximum_Temperature_average',
        # 'Minimum_Temperature_max',
        # 'Minimum_Temperature_min',
        # 'Minimum_Temperature_average',
        # 'Temperature_Change_average',
        # 'Temperature_Change_max',
        # 'Temperature_Change_min',
        # 'Track_Orientation_E_W',
        'Track_Orientation_NE_SW',
        'Track_Orientation_NW_SE',
        'Track_Orientation_N_S',
        # 'WindSpeed_max',
        # 'WindSpeed_avg',
        # 'WindGust_max',
        # 'WindDirection_avg',
        'Hottest_Heretofore',
        # 'RelativeHumidity_max',
        'TotalPrecipitation_sum',
        # 'Snowfall_max',
        # 'Rainfall_max',
        # 'Rainfall_min',
        # 'Rainfall_average',
        # 'GLBL_IRAD_AMT_max',
        # 'GLBL_IRAD_AMT_iqr',
        # 'GLBL_IRAD_AMT_total',
    ]


# Describe basic statistics about the main explanatory variables
def describe_explanatory_variables(trial_id, regional, train_set, save_as=".tif", dpi=None):
    plt.figure(figsize=(13, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    # Temperature change
    ax1 = plt.subplot2grid((1, 8), (0, 0))
    train_set.Temperature_dif.plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
    ax1.set_xticklabels('')
    plt.xlabel('Temperature\nChange\nwithin prior-IP', fontsize=13, labelpad=13)  # labelpad=39
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(0.05, 1.01)

    # Temperature category
    ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=3)
    train_set.Temperature_Category.value_counts().plot.bar(color='#537979', rot=-0, fontsize=12)
    plt.xticks(range(0, 8), ['<24', '24', '25', '26', '27', '28', '29', '≥30'], fontsize=12)
    plt.xlabel('Maximum Temperature (°C)', fontsize=13)  # labelpad=7
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax2.yaxis.set_label_coords(0.0, 1.01)

    # # # Relative humidity (divided by 10)
    # ax3 = plt.subplot2grid((1, 8), (0, 5))
    # train_set.RelativeHumidity_max.plot.box(color=colour, ax=ax3, widths=0.5, fontsize=12)
    # ax3.set_xticklabels('')
    # plt.xlabel('Max.\nRH', fontsize=13, labelpad=29)
    # plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
    # ax3.yaxis.set_label_coords(0.2, 1.01)

    # Total precipitation
    ax4 = plt.subplot2grid((1, 8), (0, 4))
    train_set.TotalPrecipitation_sum.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Total Precipitation', fontsize=13, labelpad=13)  # labelpad=29
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(0.0, 1.01)

    # Track orientation
    ax5 = plt.subplot2grid((1, 8), (0, 5), colspan=2)
    track_orientation = train_set.Track_Orientation.value_counts()
    track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
    track_orientation.plot.bar(color='#565d67', rot=-0, fontsize=12)
    plt.xlabel('Track orientation', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(0.0, 1.01)

    # The hottest day of the year by far
    ax6 = plt.subplot2grid((1, 8), (0, 7))
    hottest_heretofore = train_set.Hottest_Heretofore.value_counts()
    hottest_heretofore.plot.bar(color='#a72a3d', rot=0, fontsize=12)
    plt.xlabel('Hottest\nheretofore', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(0.0, 1.01)

    plt.tight_layout()

    if save_as:
        path_to_file_weather = cdd_prototype_heat_mod("{}".format(trial_id),
                                                      "Variables" + ("-regional" if regional else "") + save_as)
        save_fig(path_to_file_weather, dpi=dpi)
        if save_as == ".svg":
            save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


# Logistic regression model
def logistic_regression_model(trial_id=0, update=True, regional=True, reason=None,
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=100,
                              describe_var=False,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True):
    # Get the m_data for modelling
    incidents_and_weather = fetch_incidents_with_weather(trial_id=trial_id,
                                                         route_name=None, weather_category='Heat',
                                                         on_region=regional, on_reason=reason, on_season=season,
                                                         lp=lp, non_ip=non_ip,
                                                         test_only=False, illustrate_buf_cir=False,
                                                         update=update)

    # Select features
    explanatory_variables = specify_explanatory_variables()

    for v in explanatory_variables:
        if not incidents_and_weather[incidents_and_weather[v].isna()].empty:
            incidents_and_weather.dropna(subset=[v], inplace=True)

    incidents_and_weather = incidents_and_weather[
        explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime', 'DelayMinutes',
                                 'Temperature_Category', 'Track_Orientation']]

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        incidents_and_weather = incidents_and_weather[
            incidents_and_weather.DelayMinutes <= np.percentile(incidents_and_weather.DelayMinutes, outlier_pctl)]

    # Add the intercept
    if add_const:
        # incidents_and_weather = sm_tools.add_constant(incidents_and_weather)
        incidents_and_weather.insert(0, 'const', 1.0)
        explanatory_variables = ['const'] + explanatory_variables

    # Select data before 2017 as training data set, with the rest being test set
    train_set = incidents_and_weather[incidents_and_weather.StartDateTime <= pd.datetime(2018, 3, 31)]
    test_set = incidents_and_weather[incidents_and_weather.StartDateTime >= pd.datetime(2018, 4, 1)]

    if describe_var:
        describe_explanatory_variables(trial_id, regional, train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)
    try:
        if model == 'logit':
            mod = sm_dcm.Logit(train_set.Incident_Reported, train_set[explanatory_variables])
            mod_result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=True)
        else:  # 'probit'
            mod = sm_dcm.Probit(train_set.Incident_Reported, train_set[explanatory_variables])
            mod_result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=True)
        print(mod_result.summary()) if verbose else print("")

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(mod_result.params), columns=['OddsRatio'])
        print("\n{}".format(odds_ratios)) if verbose else print("")

        # Make predictions
        test_set['incident_prob'] = mod_result.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        def compute_roc(show, save_plot):
            """
            False Positive Rate or Type I Error:
                - the number of items wrongly identified as positive out of total true negatives
            True Positive Rate (Recall or Sensitivity)
                - the number of correct positive predictions divided by the total number of positives
                - (i.e. the number of items correctly identified as positive out of total true positives)
            """
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
            auc = sklearn.metrics.auc(fpr, tpr)  # Area under the curve (AUC)
            idx = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
            optimal_thr = np.min(thresholds[idx])
            if show:
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
                if save_plot:
                    save_fig(cdd_prototype_heat_mod("{}".format(trial_id),
                                                    "ROC" + ("-regional" if regional else "") + save_as), dpi=dpi)
            return optimal_thr

        optimal_threshold = compute_roc(show=plot_roc, save_plot=save_as)

        # Test prediction
        def check_predictability(show, save_plot):
            """
            Accuracy
                - the number of correct predictions made by the model by the total number of records
            Precision (Positive predictive value)
                - the number of correct positive predictions divided by the total number of positive predictions
                - (i.e. the number of items correctly identified as positive out of total items identified as positive)
            True Positive Rate (Recall or Sensitivity)
                - the number of correct positive predictions divided by the total number of positives
                - (i.e. the number of items correctly identified as positive out of total true positives)
            True negative rate (Specificity)
                - the number of correct negative predictions divided by the total number of negatives
                - (i.e. the number of items correctly identified as negative out of total negatives)
            """

            test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= optimal_threshold else 0)

            # Accuracy
            test_acc = pd.Series(test_set.Incident_Reported == test_set.incident_prediction)
            accuracy = np.divide(test_acc.sum(), len(test_acc))
            print("\nAccuracy: %f" % accuracy) if verbose else print("")

            # Precision
            pos_pred = test_set[test_set.incident_prediction == 1]
            test_pre = pd.Series(pos_pred.Incident_Reported == pos_pred.incident_prediction)
            precision = np.divide(test_pre.sum(), len(test_pre))
            print("Precision: %f" % precision) if verbose else print("")

            # True Positive Rate (Recall or Sensitivity)
            incident_only = test_set[test_set.Incident_Reported == 1]
            test_tpr = pd.Series(incident_only.Incident_Reported == incident_only.incident_prediction)
            recall = np.divide(test_tpr.sum(), len(test_tpr))
            print("True Positive Rate (Recall or Sensitivity): %f\n" % recall) if verbose else print("\n")

            # Plot incident delay minutes against predicted probabilities
            if show:
                incident_ind = test_set.Incident_Reported == 1
                plt.figure()
                ax = plt.subplot2grid((1, 1), (0, 0))
                ax.scatter(test_set[incident_ind].incident_prob, test_set[incident_ind].DelayMinutes,
                           c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
                           label="Heat-related incidents (2018/19)")
                plt.axvline(x=optimal_threshold, label="Threshold: %.2f" % optimal_threshold,
                            color='#e5c100', linewidth=2)
                legend = plt.legend(scatterpoints=1, loc=2, fontsize=14, fancybox=True, labelspacing=0.6)
                frame = legend.get_frame()
                frame.set_edgecolor('k')
                plt.xlim(left=0, right=1.03)
                # plt.ylim(bottom=-15)
                ax.set_xlabel("Likelihood of heat-related incident occurrence", fontsize=14, fontweight='bold')
                ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
                plt.xticks(fontsize=13)
                plt.yticks(fontsize=13)
                plt.tight_layout()
                if save_plot:
                    path_to_fig = cdd_prototype_heat_mod(
                        "{}".format(trial_id), "Predicted-likelihood" + ("-regional" if regional else "") + save_as)
                    save_fig(path_to_fig, dpi=dpi)

            return accuracy, precision, recall

        mod_accuracy, mod_precision, mod_recall = check_predictability(show=plot_pred_likelihood, save_plot=save_as)

    except Exception as e:
        print(e)
        mod_result = None
        mod_accuracy, mod_precision, mod_recall, optimal_threshold = np.nan, np.nan, np.nan, np.nan

    return incidents_and_weather, train_set, test_set, mod_result, \
        mod_accuracy, mod_precision, mod_recall, optimal_threshold


"""
# 'IR' - Broken/cracked/twisted/buckled/flawed rail
# 'XH' - Severe heat affecting infrastructure the responsibility of Network Rail (excl. Heat related speed restrictions)
# 'IB' - Points failure
# 'JH' - Critical Rail Temperature speeds, (other than buckled rails)
'IZ' - Other infrastructure causes INF OTHER
'XW' - High winds affecting infrastructure the responsibility of Network
'IS' - Track defects (other than rail defects) inc. fish plates, wet beds etc.

0. 'IR'
1. 'XH'
2. 'IB'
3. 'IR', 'XH', 'IB'
4. 'JH'
5. 'IR', 'XH', 'IB', 'JH'
"""

""" 'IR' - Broken/cracked/twisted/buckled/flawed rail ============================================================ """
# Regional
data_11, train_set_11, test_set_11, _, _, _, _, _ = \
    logistic_regression_model(trial_id=1, update=False, regional=True,
                              reason=['IR'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
# Country-wide
data_12, train_12, test_set_12, _, _, _, _, _ = \
    logistic_regression_model(trial_id=1, update=False, regional=False, reason=['IR'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

""" 'XH' - Severe heat affecting infrastructure the responsibility of Network Rail (excl. Heat related speed 
restrictions) ==================================================================================================== """
# Regional
data_21, train_21, test_set_21, _, _, _, _, _ = \
    logistic_regression_model(trial_id=2, update=False, regional=True,
                              reason=['XH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
# Country-wide
data_22, train_22, test_set_22, _, _, _, _, _ = \
    logistic_regression_model(trial_id=2, update=False, regional=False, reason=['XH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

""" 'IB' - Points failure ======================================================================================== """
# Regional
data_31, train_31, test_set_31, _, _, _, _, _ = \
    logistic_regression_model(trial_id=3, update=False, regional=True,
                              reason=['IB'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
# Country-wide
data_32, train_32, test_set_32, _, _, _, _, _ = \
    logistic_regression_model(trial_id=3, update=False, regional=False,
                              reason=['IB'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

""" 'IR', 'XH', 'IB' ============================================================================================= """
# Regional
data_41, train_41, test_set_41, _, _, _, _, _ = \
    logistic_regression_model(trial_id=4, update=False, regional=True, reason=['IR', 'XH', 'IB'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

# Country-wide
data_42, train_42, test_set_42, _, _, _, _, _ = \
    logistic_regression_model(trial_id=4, update=False, regional=False, reason=['IR', 'XH', 'IB'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

""" 'JH' - Critical Rail Temperature speeds, (other than buckled rails) ========================================== """
# Regional
data_51, train_51, test_set_51, _, _, _, _, _ = \
    logistic_regression_model(trial_id=5, update=False, regional=True,
                              reason=['JH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
# Country-wide
data_52, train_52, test_set_52, _, _, _, _, _ = \
    logistic_regression_model(trial_id=5, update=False, regional=False, reason=['JH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)

""" 'IR', 'XH', 'IB', 'JH' ======================================================================================= """
# Regional
data_61, train_61, test_set_61, _, _, _, _, _ = \
    logistic_regression_model(trial_id=6, update=False, regional=True, reason=['IR', 'XH', 'IB', 'JH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
# Country-wide
data_62, train_62, test_set_62, _, _, _, _, _ = \
    logistic_regression_model(trial_id=6, update=False, regional=False,
                              reason=['IR', 'XH', 'IB', 'JH'],
                              season='summer', lp=5 * 24, non_ip=24, outlier_pctl=95,
                              describe_var=True,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True)
