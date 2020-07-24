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
from models.tools import categorise_temperatures, categorise_track_orientations
from models.tools import cd_intermediate_fig_pub, cdd_intermediate
from models.tools import get_data_by_season
from settings import mpl_preferences, pd_preferences
from spreadsheet.incidents import get_schedule8_weather_incidents
from utils import get_subset, make_filename
from weather import midas, ukcp

# == Apply the preferences ============================================================================

mpl_preferences(reset=False)
pd_preferences(reset=False)
plt.rc('font', family='Times New Roman')


# == Change directories ===============================================================================

def cdd_intermediate_heat(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\heat\\dat\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\heat\\dat\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("heat", *sub_dir, mkdir=mkdir)
    return path


def cdd_intermediate_heat_trial(trial_id, *sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\intermediate\\heat\\<``trial_id``>" and sub-directories / a file.

    :param trial_id:
    :type trial_id: int, str
    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\intermediate\\heat\\data\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_intermediate("heat", "{}".format(trial_id), *sub_dir, mkdir=mkdir)
    return path


# == Data of weather conditions =======================================================================

def get_incident_location_weather(route_name=None, weather_category='Heat', season='summer',
                                  prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                                  trial_only=True, random_state=None, illustrate_buf_cir=False, update=False,
                                  verbose=False):
    """
    Process data of weather conditions for each incident location.

    **Example**::

        route_name         = 'Anglia'
        weather_category   = 'Heat'
        season             = 'summer'
        prior_ip_start_hrs = -0
        latent_period      = -5
        non_ip_start_hrs   = -0
        trial_only         = True
        random_state       = 0
        illustrate_buf_cir = False
        update             = False
        verbose            = True

    .. note::

        Note that the 'Critical_EndDateTime' would be based on the 'Critical_StartDateTime' if we consider the weather
        conditions on the day of incident occurrence; 'StartDateTime' otherwise.
    """

    pickle_filename = make_filename("weather", route_name, weather_category,
                                    "_".join([season] if isinstance(season, str) else season),
                                    "trial" if trial_only else "full")
    path_to_pickle = cdd_intermediate_heat(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incident_location_weather = load_pickle(path_to_pickle)

    else:
        try:
            # Incidents data

            incidents_all = get_schedule8_weather_incidents()
            incidents_all.rename(columns={'Year': 'FinancialYear'}, inplace=True)
            incidents_all_by_season = get_data_by_season(incidents_all, season)
            incidents = get_subset(incidents_all_by_season, route_name, weather_category)
            if trial_only:  # For testing purpose only
                incidents = incidents.sample(n=10, random_state=random_state)  # incidents = incidents.iloc[0:10, :]

            # Weather data

            weather_obs = ukcp.fetch_integrated_daily_gridded_weather_obs().reset_index()  # Weather observations
            irad_obs = midas.get_radtob().reset_index()  # Radiation observations

            observation_grids = ukcp.fetch_observation_grids()  # Grids for observing weather conditions
            obs_cen_geom = shapely.geometry.MultiPoint(list(observation_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(observation_grids.Grid))

            met_stations = midas.get_radiation_stations_information()  # Met station locations
            met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.E_N_GEOM))

            # -- Data integration in the spatial context ----------------------------------------------

            # Start
            incidents['Start_Pseudo_Grid_ID'] = incidents.StartNE.map(
                lambda x: integrator.find_closest_weather_grid(x, observation_grids, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='Start_Pseudo_Grid_ID')
            # End
            incidents['End_Pseudo_Grid_ID'] = incidents.EndNE.map(
                lambda x: integrator.find_closest_weather_grid(x, observation_grids, obs_cen_geom))
            incidents = incidents.join(observation_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')
            # Modify column names
            for p in ['Start', 'End']:
                a = [c for c in incidents.columns if c.endswith(p)]
                b = [p + '_' + c if c == 'Grid' else p + '_Grid_' + c for c in observation_grids.columns]
                incidents.rename(columns=dict(zip(a, b)), inplace=True)

            # Append 'MidpointNE' column
            incidents['MidpointNE'] = incidents.apply(
                lambda x: get_geometric_midpoint(x.StartNE, x.EndNE, as_geom=True), axis=1)

            # Make a buffer zone for Weather data aggregation
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: integrator.create_circle_buffer_upon_weather_grid(
                    x.StartNE, x.EndNE, x.MidpointNE, whisker=0), axis=1)

            # Find all Weather observation grids that intersect with the created buffer zone for each incident location
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(
                lambda x: integrator.find_intersecting_weather_grid(x, observation_grids, obs_grids_geom))

            incidents['Met_SRC_ID'] = incidents.MidpointNE.map(
                lambda x: integrator.find_closest_met_stn(x, met_stations, met_stations_geom))

            if illustrate_buf_cir:  # Illustration of the buffer circle
                start_point, end_point, midpoint = incidents[['StartNE', 'EndNE', 'MidpointNE']].iloc[0]
                bf_circle = integrator.create_circle_buffer_upon_weather_grid(
                    start_point, end_point, midpoint, whisker=500)
                i_obs_grids = integrator.find_intersecting_weather_grid(
                    bf_circle, observation_grids, obs_grids_geom, as_grid_id=False)
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

            # -- Calculate critical time periods ------------------------------------------------------
            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
            incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
                lambda x: x + pd.Timedelta(days=-1) if x.time() < datetime.time(9) else x)
            incidents.Critical_StartDateTime += datetime.timedelta(hours=prior_ip_start_hrs)
            incidents['Critical_EndDateTime'] = incidents.Critical_StartDateTime
            incidents['Critical_Period'] = incidents.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            # -- Data integration for the specified prior-IP ------------------------------------------

            prior_ip_weather_stats = incidents.apply(
                lambda x: pd.Series(integrator.integrate_pip_gridded_weather_obs(
                    x.Weather_Grid, x.Critical_Period, weather_obs)), axis=1)

            w_col_names = integrator.specify_weather_variable_names(
                integrator.specify_weather_stats_calculations()) + ['Hottest_Heretofore']
            prior_ip_weather_stats.columns = w_col_names
            prior_ip_weather_stats['Temperature_Change_max'] = \
                abs(prior_ip_weather_stats.Maximum_Temperature_max - prior_ip_weather_stats.Minimum_Temperature_min)
            prior_ip_weather_stats['Temperature_Change_min'] = \
                abs(prior_ip_weather_stats.Maximum_Temperature_min - prior_ip_weather_stats.Minimum_Temperature_max)

            prior_ip_data = incidents.join(prior_ip_weather_stats)

            prior_ip_radtob_stats = prior_ip_data.apply(
                lambda x: pd.Series(integrator.integrate_pip_midas_radtob(
                    x.Met_SRC_ID, x.Critical_Period, irad_obs)), axis=1)

            r_col_names = integrator.specify_weather_variable_names(
                integrator.specify_radtob_stats_calculations()) + ['GLBL_IRAD_AMT_total']
            prior_ip_radtob_stats.columns = r_col_names

            prior_ip_data = prior_ip_data.join(prior_ip_radtob_stats)

            prior_ip_data['Incident_Reported'] = 1

            # -- Data integration for the specified non-IP --------------------------------------------

            non_ip_data = incidents.copy(deep=True)  # Get Weather data that did not cause any incident
            # non_ip_data[[c for c in non_ip_data.columns if c.startswith('Incident')]] = None
            # non_ip_data[['StartDateTime', 'EndDateTime', 'WeatherCategoryCode', 'WeatherCategory', 'Minutes']] = None

            non_ip_data.Critical_EndDateTime = \
                non_ip_data.Critical_StartDateTime + datetime.timedelta(days=latent_period)
            non_ip_data.Critical_StartDateTime = \
                non_ip_data.Critical_EndDateTime + datetime.timedelta(hours=non_ip_start_hrs)
            non_ip_data.Critical_Period = non_ip_data.apply(
                lambda x: pd.date_range(x.Critical_StartDateTime, x.Critical_EndDateTime, normalize=True), axis=1)

            non_ip_weather_stats = non_ip_data.apply(
                lambda x: pd.Series(integrator.integrate_nip_gridded_weather_obs(
                    x.Weather_Grid, x.Critical_Period, x.StanoxSection, weather_obs, prior_ip_data)),
                axis=1)

            non_ip_weather_stats.columns = w_col_names
            non_ip_weather_stats['Temperature_Change_max'] = \
                abs(non_ip_weather_stats.Maximum_Temperature_max - non_ip_weather_stats.Minimum_Temperature_min)
            non_ip_weather_stats['Temperature_Change_min'] = \
                abs(non_ip_weather_stats.Maximum_Temperature_min - non_ip_weather_stats.Minimum_Temperature_max)

            non_ip_data = non_ip_data.join(non_ip_weather_stats)

            non_ip_radtob_stats = non_ip_data.apply(
                lambda x: pd.Series(integrator.integrate_nip_midas_radtob(
                    x.Met_SRC_ID, x.Critical_Period, x.StanoxSection, irad_obs, prior_ip_data)),
                axis=1)

            non_ip_radtob_stats.columns = r_col_names

            non_ip_data = non_ip_data.join(non_ip_radtob_stats)

            non_ip_data['Incident_Reported'] = 0

            # -- Merge "prior_ip_data" and "non_ip_data" ----------------------------------------------
            incident_location_weather = pd.concat([prior_ip_data, non_ip_data], axis=0, ignore_index=True, sort=False)

            # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
            incident_location_weather = incident_location_weather.join(
                categorise_track_orientations(incident_location_weather[['StartNE', 'EndNE']]))

            # Categorise temperature: 25, 26, 27, 28, 29, 30
            incident_location_weather = incident_location_weather.join(
                categorise_temperatures(incident_location_weather, column_name='Maximum_Temperature_max'))

            save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

        except Exception as e:
            print("Failed to get Incidents with Weather conditions. {}.".format(e))
            incident_location_weather = pd.DataFrame()

    return incident_location_weather


# == Modelling trials =================================================================================


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
        save_fig(cd_intermediate_fig_pub("Variables" + save_as), dpi)


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
                save_fig(cdd_intermediate_heat_trial(trial_id, "ROC" + save_as), dpi=dpi)

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
                save_fig(cdd_intermediate_heat_trial(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_accuracy, incident_accuracy, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_accuracy, incident_accuracy, threshold
