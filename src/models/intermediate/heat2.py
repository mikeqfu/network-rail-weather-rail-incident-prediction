""" Testing models

.. todo::

    Find out track orientation through track geometry data.
"""

import statsmodels.discrete.discrete_model as sm_dcm
from pyhelpers.store import save_fig, save_svg_as_emf
from sklearn import metrics

from models.intermediate.heat1 import get_incident_location_weather
from models.intermediate.integrator import *
from models.tools import cdd_intermediate_heat_trial
from settings import mpl_preferences, pd_preferences

pd_preferences()
mpl_preferences(reset=False)
plt.rc('font', family='Times New Roman')


def plot_temperature_deviation(route_name=None, on_region=True, on_reason=None, lp_day_range=14,
                               add_err_bar=True, update=False,
                               save_as=None, dpi=None):
    """
    Testing parameters:
    e.g.
        route_name=None
        on_region=True
        on_reason=None
        lp_day_range=14
        add_err_bar=True
        update=False
        save_as=".tif"
        dpi=None
    """

    incident_location_weather = [
        get_incident_location_weather(route_name, weather_category='Heat', on_region=on_region, on_reason=on_reason,
                                      on_season='summer', ip_hours=24, lp_days=d, non_ip_hours=24, update=update)
        for d in range(1, lp_day_range + 1)]

    time_and_iloc = ['StartDateTime', 'EndDateTime', 'StanoxSection', 'IncidentDescription']
    selected_cols, data = time_and_iloc + ['Temperature_max'], incident_location_weather[0]
    ip_temperature_max = data[data.Incident_Reported == 1][selected_cols]
    diff_means, diff_std = [], []
    for i in range(0, lp_day_range):
        data = incident_location_weather[i]
        nip_temperature_max = data[data.Incident_Reported == 0][selected_cols]
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
        save_fig(cdd_intermediate_heat_trial(0, "Temp deviation" + save_as), dpi=dpi)


plot_temperature_deviation(route_name=None, on_region=True, on_reason=['IR'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['XH'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['IB'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['IR', 'XH', 'IB'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['JH'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['IR', 'XH', 'IB', 'JH'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

plot_temperature_deviation(route_name=None, on_region=True, on_reason=['IR', 'IB'], lp_day_range=14,
                           add_err_bar=True, update=False, save_as=None, dpi=None)

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
        path_to_file_weather = cdd_intermediate_heat_trial(
            "{}".format(trial_id), "Variables" + ("-regional" if regional else "") + save_as)
        save_fig(path_to_file_weather, dpi=dpi)
        if save_as == ".svg":
            save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


def logistic_regression_model(trial_id=0, update=True, regional=True, reason=None,
                              season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=100,
                              describe_var=False,
                              add_const=True, seed=0, model='logit',
                              plot_roc=True, plot_pred_likelihood=True,
                              save_as=".tif", dpi=None, verbose=True):
    """
    Testing parameters:
    e.g.
        trial_id=0
        update=True
        regional=True
        reason=None
        season='summer'
        ip_hours=24
        lp_days=5
        non_ip_hours=24
        outlier_pctl=100
        describe_var=False
        add_const=True
        seed=0
        model='logit'
        plot_roc=True
        plot_pred_likelihood=True
        save_as=".tif"
        dpi=None
        verbose=True

    """
    # Get the m_data for modelling
    incidents_and_weather = get_incident_location_weather(trial_id=trial_id,
                                                          route_name=None, weather_category='Heat',
                                                          on_region=regional, on_reason=reason, on_season=season,
                                                          ip_hours=ip_hours, lp_days=lp_days, non_ip_hours=non_ip_hours,
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
            fpr, tpr, thresholds = metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
            auc = metrics.auc(fpr, tpr)  # Area under the curve (AUC)
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
                    roc_plot_filename = "ROC" + ("-regional" if regional else "") + save_as
                    save_fig(cdd_intermediate_heat_trial("{}".format(trial_id), roc_plot_filename), dpi=dpi)
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
                    path_to_fig = cdd_intermediate_heat_trial(
                        "{}".format(trial_id), "Predicted-likelihood" + ("-regional" if regional else "") + save_as)
                    save_fig(path_to_fig, dpi=dpi)

            return accuracy, precision, recall

        m_accuracy, m_precision, m_recall = check_predictability(show=plot_pred_likelihood, save_plot=save_as)

    except Exception as e:
        print(e)
        mod_result = None
        m_accuracy, m_precision, m_recall, optimal_threshold = np.nan, np.nan, np.nan, np.nan

    return incidents_and_weather, train_set, test_set, mod_result, m_accuracy, m_precision, m_recall, optimal_threshold


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
6. 'IR', 'IB'
"""

""" 'IR' - Broken/cracked/twisted/buckled/flawed rail ============================================================ """


def test_ir():
    # Regional
    data_11, train_set_11, test_set_11, _, _, _, _, _ = \
        logistic_regression_model(trial_id=1, update=False, regional=True,
                                  reason=['IR'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_12, train_12, test_set_12, _, _, _, _, _ = \
        logistic_regression_model(trial_id=1, update=False, regional=False, reason=['IR'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'XH' - Severe heat affecting infrastructure the responsibility of Network Rail (excl. Heat related speed 
restrictions) ==================================================================================================== """


def test_xh():
    # Regional
    data_21, train_21, test_set_21, _, _, _, _, _ = \
        logistic_regression_model(trial_id=2, update=False, regional=True,
                                  reason=['XH'],
                                  season='summer', lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_22, train_22, test_set_22, _, _, _, _, _ = \
        logistic_regression_model(trial_id=2, update=False, regional=False, reason=['XH'],
                                  season='summer', lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'IB' - Points failure ======================================================================================== """


def test_ib():
    # Regional
    data_31, train_31, test_set_31, _, _, _, _, _ = \
        logistic_regression_model(trial_id=3, update=False, regional=True,
                                  reason=['IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_32, train_32, test_set_32, _, _, _, _, _ = \
        logistic_regression_model(trial_id=3, update=False, regional=False,
                                  reason=['IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'IR', 'XH', 'IB' ============================================================================================= """


def test_ir_xh_ib():
    # Regional
    data_41, train_41, test_set_41, _, _, _, _, _ = \
        logistic_regression_model(trial_id=4, update=False, regional=True, reason=['IR', 'XH', 'IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)

    # Country-wide
    data_42, train_42, test_set_42, _, _, _, _, _ = \
        logistic_regression_model(trial_id=4, update=False, regional=False, reason=['IR', 'XH', 'IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'JH' - Critical Rail Temperature speeds, (other than buckled rails) ========================================== """


def test_jh():
    # Regional
    data_51, train_51, test_set_51, _, _, _, _, _ = \
        logistic_regression_model(trial_id=5, update=False, regional=True,
                                  reason=['JH'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_52, train_52, test_set_52, _, _, _, _, _ = \
        logistic_regression_model(trial_id=5, update=False, regional=False, reason=['JH'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'IR', 'XH', 'IB', 'JH' ======================================================================================= """


def test_ir_xh_ib_jh():
    # Regional
    data_61, train_61, test_set_61, _, _, _, _, _ = \
        logistic_regression_model(trial_id=6, update=False, regional=True, reason=['IR', 'XH', 'IB', 'JH'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_62, train_62, test_set_62, _, _, _, _, _ = \
        logistic_regression_model(trial_id=6, update=False, regional=False,
                                  reason=['IR', 'XH', 'IB', 'JH'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)


""" 'IR', 'IB' =================================================================================================== """


def test_ir_ib():
    # Regional
    data_61, train_61, test_set_61, _, _, _, _, _ = \
        logistic_regression_model(trial_id=7, update=False, regional=True, reason=['IR', 'IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
    # Country-wide
    data_62, train_62, test_set_62, _, _, _, _, _ = \
        logistic_regression_model(trial_id=7, update=False, regional=False,
                                  reason=['IR', 'IB'],
                                  season='summer', ip_hours=24, lp_days=5, non_ip_hours=24, outlier_pctl=95,
                                  describe_var=True,
                                  add_const=True, seed=0, model='logit',
                                  plot_roc=True, plot_pred_likelihood=True,
                                  save_as=".tif", dpi=None, verbose=True)
