""" Weather attribution of Schedule 8 incidents """

import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.model_selection

import database_met as dbm
import workbook_schedule8 as wbs


# ====================================================================================================================
""" Task 1: Broad classification of incidents into weather-related and non-weather-related """


# Get training and test data sets for Task 1
def get_task_1_train_test_data(random_state=0, test_size=0.2):

    dat = dbm.get_schedule8_cost_by_datetime_location_reason()
    dat['weather_related'] = dat.WeatherCategory.map(lambda x: 0 if x == '' else 1)

    features = ['FinancialYear', 'IncidentDescription', 'IncidentCategoryDescription',
                'IncidentReason', 'IncidentReasonName', 'IncidentReasonDescription',
                'IncidentJPIPCategory', 'IncidentCategorySuperGroupCode',
                'IncidentCategoryGroupDescription', 'WeatherCategory']

    data = dat[['weather_related'] + features]
    data['descriptions'] = \
        data.IncidentDescription.astype(str) + ' ' + \
        data.IncidentCategoryDescription + ' ' + \
        data.IncidentReason + ' ' + \
        data.IncidentReasonName + ' ' + \
        data.IncidentReasonDescription + ' ' + \
        data.IncidentJPIPCategory + ' ' + \
        data.IncidentCategorySuperGroupCode + ' ' + \
        data.IncidentCategoryGroupDescription

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    word_counter = vectorizer.fit_transform(np.array(data.descriptions))

    if random_state is None:
        train_data, test_data = data[data.FinancialYear < 2014], data[data.FinancialYear == 2014]
    else:
        # 'random_state' must be an integer
        non_weather_related_dat, weather_related_dat = data[dat.weather_related == 0], data[dat.weather_related == 1]
        train_dat_non, test_dat_non = sklearn.model_selection.train_test_split(non_weather_related_dat,
                                                                               random_state=random_state,
                                                                               test_size=test_size)
        train_dat, test_dat = sklearn.model_selection.train_test_split(weather_related_dat,
                                                                       random_state=random_state,
                                                                       test_size=test_size)
        train_data = pd.concat([train_dat_non, train_dat], axis=0)
        test_data = pd.concat([test_dat_non, test_dat], axis=0)

    idx_train, idx_test = np.array(train_data.index), np.array(test_data.index)

    train_set = dict(zip(['word_counter', 'data_frame'], [word_counter[idx_train], train_data]))
    test_set = dict(zip(['word_counter', 'data_frame'], [word_counter[idx_test], test_data]))

    return train_set, test_set


# Fit model for Task 1
def classification_model_for_identifying_weather_related_incidents(random_state=0, test_size=0.2):
    train_set, test_set = get_task_1_train_test_data(random_state, test_size)
    model = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                                                    intercept_scaling=1, class_weight=None, random_state=random_state,
                                                    solver='saga', max_iter=1000, multi_class='ovr',
                                                    verbose=True,
                                                    warm_start=False, n_jobs=1)
    model.fit(train_set['word_counter'], train_set['data_frame'].weather_related)

    # model.score(test_set['word_counter'], test_set['data_frame'].weather_related)
    # test_set['data_frame']['weather_related_predicted'] = model.predict(test_set['word_counter'])
    return model


# ====================================================================================================================
""" Task 2: Classification of weather-related incidents into different categories """


def get_task_2_train_test_data():
    schedule8_weather_incidents = wbs.get_schedule8_weather_incidents_02062006_31032014()['Data']
    schedule8_weather_incidents.rename(columns={'Year': 'FinancialYear'}, inplace=True)

    dat = dbm.get_schedule8_cost_by_datetime_location_reason()
    dat.WeatherCategory.fillna('', inplace=True)

    features = ['FinancialYear', 'IncidentDescription', 'IncidentCategoryDescription',
                'IncidentReason', 'IncidentReasonName', 'IncidentReasonDescription',
                'IncidentJPIPCategory', 'IncidentCategorySuperGroupCode',
                'IncidentCategoryGroupDescription']

    dat_train = schedule8_weather_incidents[['WeatherCategory'] + features]
    dat_test = dat[(dat.FinancialYear == 2014) & (dat.WeatherCategory != '')][['WeatherCategory'] + features]

    data = pd.DataFrame(pd.concat([dat_train, dat_test], ignore_index=True))

    data['descriptions'] = \
        data.IncidentDescription.astype(str) + ' ' + \
        data.IncidentCategoryDescription + ' ' + \
        data.IncidentReason + ' ' + \
        data.IncidentReasonName + ' ' + \
        data.IncidentReasonDescription + ' ' + \
        data.IncidentJPIPCategory + ' ' + \
        data.IncidentCategorySuperGroupCode + ' ' + \
        data.IncidentCategoryGroupDescription

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    word_counter = vectorizer.fit_transform(np.array(data.descriptions))
    # data['word_count'] = csr_matrix_to_dict(word_counter, vectorizer)

    train_data, test_data = data[data.FinancialYear < 2014], data[data.FinancialYear == 2014]
    idx_train, idx_test = train_data.index, test_data.index

    train_set = dict(zip(['word_counter', 'data_frame'], [word_counter[0:max(idx_train) + 1], train_data]))
    test_set = dict(zip(['word_counter', 'data_frame'], [word_counter[min(idx_test):], test_data]))

    return train_set, test_set


def classification_model_for_weather_related_incidents():
    train_set, test_set = get_task_2_train_test_data()
    model = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                                    intercept_scaling=1, class_weight=None, random_state=None,
                                                    solver='saga', max_iter=1000, multi_class='multinomial',
                                                    verbose=True,
                                                    warm_start=False, n_jobs=1)
    model.fit(train_set['word_counter'], train_set['data_frame'].WeatherCategory)

    # test_set['data_frame']['predicted_weather_category'] = model.predict(test_set['word_counter'])

    return model


# ====================================================================================================================
""" Plot """


def get_stats(data):
    # Calculate the proportions of different types of weather-related incidents
    stats = data. \
        groupby('WeatherCategory').aggregate({'WeatherCategory': 'count', 'DelayMinutes': np.sum, 'DelayCost': np.sum})
    stats.rename(columns={'WeatherCategory': 'Count'}, inplace=True)
    stats['percentage'] = stats.Count / len(data) * 100
    # Sort stats in the ascending order of 'percentage'
    stats.sort_values('percentage', ascending=False, inplace=True)
    return stats


# Pie chart
def proportion_pie_plot(data, save_as='.png'):
    """
    :param data: schedule8_weather_incidents = wbs.get_schedule8_weather_incidents_02062006_31032014()['Data']
    :param save_as:
    :return:
    """
    stats = get_stats(data).reset_index()
    # Set colour array
    colours = matplotlib.cm.get_cmap('Set3')(np.flip(np.linspace(0.0, 1.0, 9), 0))
    # Specify labels
    percentages = ['%1.1f%%' % round(x, 1) for x in stats['percentage']]
    labels = stats.WeatherCategory + ': '
    # labels = [a + b for a, b in zip(labels, total_costs_in_million)]
    labels = [a + b for a, b in zip(labels, percentages)]
    wind_label = ['', 'Most delays\n& Highest costs', '', '', '', '', '', '', '']
    # Specify which part is exploded
    explode_list = np.zeros(len(stats))
    # explode_pos = stats.sort_values(by=['PfPIMinutes', 'percentage'], ascending=False).index[0]
    explode_list[1] = 0.2

    # Create a figure
    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0), aspect='equal')
    # ax.set_rasterization_zorder(1)
    pie_collections = ax.pie(stats.percentage, labels=wind_label, startangle=70, colors=colours,
                             explode=explode_list,
                             labeldistance=0.7)

    # Note that 'pie_collections' includes: patches, texts, autotexts
    patches, texts = pie_collections
    texts[1].set_fontsize(12)
    texts[1].set_fontstyle('italic')
    # texts[1].set_fontweight('bold')
    legend = ax.legend(pie_collections[0], labels, loc='best', fontsize=14, frameon=True, shadow=True, fancybox=True,
                       title='Weather category')
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_facecolor('white')

    # ax.set_title('Reasons for weather-related incidents\n', fontsize=14, weight='bold')

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=0.95)

    plt.savefig(wbs.cdd_schedule8("Exploratory analysis", "Proportions" + save_as), dpi=600)


# Plot 'Total monetary cost incurred by weather-related incidents'
def delay_cost_bar_plot(data, save_as='.png'):
    """
    :param data: schedule8_weather_incidents = wbs.get_schedule8_weather_incidents_02062006_31032014()['Data']
    :param save_as:
    :return:
    """
    stats = get_stats(data).reset_index()
    stats.sort_values(['DelayMinutes', 'DelayCost', 'Count'], inplace=True)
    colour_array = np.sort(np.flip(np.linspace(0.0, 1.0, 9), 0)[stats.index])
    stats.index = range(len(stats))
    plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    # formatter_minutes = FuncFormatter(lambda m, position: format(int(m), ','))
    colours = matplotlib.cm.get_cmap('Set3')(colour_array)
    ax1.barh(stats.index, stats.DelayMinutes, align='center', color=colours)
    plt.yticks(stats.index, stats.WeatherCategory, fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Minutes', fontsize=13, fontweight='bold')
    # plt.ylabel('Weather category', fontsize=12)
    plt.title('Delay', fontsize=15, fontweight='bold')
    # ax1.set_axis_bgcolor('#808080')

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    # plt.barh(range(0, len(stats)), stats['PfPICosts'], align='center', color=colours1)
    plt.barh(stats.index, stats.DelayCost, align='center', color=colours, alpha=1.0, hatch='/')
    plt.yticks(stats.index, [''] * len(stats))
    plt.xticks(fontsize=12)
    # Format labels
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda c, position: '%1.1f' % (c * 1e-7)))
    plt.xlabel('£ millions', fontsize=13, fontweight='bold')
    plt.title('Cost', fontsize=15, fontweight='bold')
    # ax2.set_axis_bgcolor('#dddddd')

    plt.tight_layout()  # plt.subplots_adjust(left=0.16, bottom=0.10, right=0.96, top=0.92, wspace=0.16)
    plt.savefig('./Figures/Delay and costs.%s' % save_as, dpi=1600)
