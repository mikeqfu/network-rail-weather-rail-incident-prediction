""" Testing models """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhelpers.store
import sklearn
import sklearn.metrics
import statsmodels.discrete.discrete_model as sm_dcm
import statsmodels.tools as sm_tools

import Intermediate.Heat.trial_1.processing as itm_processing
import Intermediate.utils as itm_utils
import settings

settings.np_preferences()
settings.pd_preferences()


def specify_explanatory_variables_1():
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
def describe_explanatory_variables_1(train_set, save_as=".pdf", dpi=None):
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

    path_to_file_weather = itm_utils.cdd_intermediate(0, "Variables" + save_as)
    plt.savefig(path_to_file_weather, dpi=dpi)
    if save_as == ".svg":
        pyhelpers.store.save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


#
def logistic_regression_model_1(trial_id=0,
                                route=None, weather_category='Heat', season='summer',
                                prior_ip_start_hrs=-0, latent_period=-5, non_ip_start_hrs=-0,
                                outlier_pctl=100,
                                describe_var=False,
                                add_const=True, seed=0, model='logit',
                                plot_roc=False, plot_predicted_likelihood=False,
                                save_as=".svg", dpi=None,
                                verbose=True):
    """
    :param trial_id:
    :param route: [str]
    :param weather_category: [str]
    :param season: [str] or [list-like]
    :param prior_ip_start_hrs: [int] or [float]
    :param latent_period:
    :param non_ip_start_hrs: [int] or [float]
    :param outlier_pctl: [int]
    :param describe_var: [bool]
    :param add_const:
    :param seed:
    :param model:
    :param plot_roc:
    :param plot_predicted_likelihood:
    :param save_as:
    :param dpi:
    :param verbose:
    :return:

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

    """
    # Get the m_data for modelling
    m_data = itm_processing.get_incidents_with_weather_1(route, weather_category, season,
                                                         prior_ip_start_hrs, latent_period, non_ip_start_hrs)

    # temp_data = [load_pickle(cdd_mod_heat_inter("Slices", f)) for f in os.listdir(cdd_mod_heat_inter("Slices"))]
    # m_data = pd.concat(temp_data, ignore_index=True, sort=False)

    m_data.dropna(subset=['GLBL_IRAD_AMT_max', 'GLBL_IRAD_AMT_iqr', 'GLBL_IRAD_AMT_total'], inplace=True)

    # Select features
    explanatory_variables = specify_explanatory_variables_1()

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
        describe_explanatory_variables_1(train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)
    try:
        if model == 'probit':
            mod = sm_dcm.Probit(train_set.Incident_Reported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=1000, full_output=True, disp=True)
        else:
            mod = sm_dcm.Logit(train_set.Incident_Reported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=1000, full_output=True, disp=True)
        print(result.summary()) if verbose else print("")

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['OddsRatio'])
        print("\n{}".format(odds_ratios)) if verbose else print("")

        # Prediction
        test_set['incident_prob'] = result.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        fpr, tpr, thr = sklearn.metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = sklearn.metrics.auc(fpr, tpr)
        ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
        threshold = np.min(thr[ind])

        # prediction accuracy
        test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= threshold else 0)
        test = pd.Series(test_set.Incident_Reported == test_set.incident_prediction)
        mod_acc = np.divide(test.sum(), len(test))
        print("\nAccuracy: %f" % mod_acc) if verbose else print("")

        # incident prediction accuracy
        incident_only = test_set[test_set.Incident_Reported == 1]
        test_acc = pd.Series(incident_only.Incident_Reported == incident_only.incident_prediction)
        incident_acc = np.divide(test_acc.sum(), len(test_acc))
        print("Incident accuracy: %f" % incident_acc) if verbose else print("")

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
            pyhelpers.store.save_fig(itm_utils.cdd_intermediate(trial_id, "ROC" + save_as), dpi=dpi)

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
            pyhelpers.store.save_fig(itm_utils.cdd_intermediate(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_acc, incident_acc, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_acc, incident_acc, threshold
