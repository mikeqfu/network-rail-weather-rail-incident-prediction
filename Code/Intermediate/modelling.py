import sklearn
import sklearn.metrics
import statsmodels.discrete.discrete_model as sm
import statsmodels.tools.tools as sm_tools

from Intermediate.processing import *
from utils import save_fig


# A prototype model in the context of wind-related Incidents
def logistic_regression_model(trial_id=0,
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
    m_data = get_incidents_with_weather(route, weather_category, season,
                                        prior_ip_start_hrs, latent_period, non_ip_start_hrs)

    # temp_data = [load_pickle(cdd_mod_heat_inter("Slices", f)) for f in os.listdir(cdd_mod_heat_inter("Slices"))]
    # m_data = pd.concat(temp_data, ignore_index=True, sort=False)

    m_data.dropna(subset=['GLBL_IRAD_AMT_max', 'GLBL_IRAD_AMT_iqr', 'GLBL_IRAD_AMT_total'], inplace=True)

    # Select features
    explanatory_variables = specify_explanatory_variables_model()

    for v in explanatory_variables:
        if not m_data[m_data[v].isna()].empty:
            m_data.dropna(subset=[v], inplace=True)

    m_data = m_data[explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime', 'Minutes']]

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        m_data = m_data[m_data.Minutes <= np.percentile(m_data.Minutes, outlier_pctl)]

    # Add the intercept
    if add_const:
        m_data = sm_tools.add_constant(m_data)  # data['const'] = 1.0
        explanatory_variables = ['const'] + explanatory_variables

    # Select data before 2014 as training data set, with the rest being test set
    train_set = m_data[m_data.StartDateTime < pd.datetime(2013, 1, 1)]
    test_set = m_data[m_data.StartDateTime >= pd.datetime(2013, 1, 1)]

    if describe_var:
        describe_explanatory_variables(train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)
    try:
        if model == 'probit':
            mod = sm.Probit(train_set.Incident_Reported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=1000, full_output=True, disp=True)
        else:
            mod = sm.Logit(train_set.Incident_Reported, train_set[explanatory_variables])
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
            save_fig(cdd_mod_heat_inter(trial_id, "ROC" + save_as), dpi=dpi)

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
            save_fig(cdd_mod_heat_inter(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_acc, incident_acc, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_acc, incident_acc, threshold
