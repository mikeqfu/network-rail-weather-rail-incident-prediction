""" Weather attribution of Incidents """

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessor import METExLite, Schedule8IncidentReports

metex = METExLite()


# == Task 1: Broad classification of incidents into weather-related and non-weather-related ===========

def get_task_1_train_test_data(random_state=0, test_size=0.2):
    """
    Get training and test data sets for Task 1.

    :param random_state: a random seed number, defaults to ``0``
    :type random_state: int, None
    :param test_size: size of test data set, defaults to ``0.2``
    :type test_size: int, float
    :return: training and test data sets
    :rtype: tuple - (dict, dict)

    **Test**::

        from models.prototype.weather_attr import get_task_1_train_test_data

        random_state = 0
        test_size = 0.2

        train_set, test_set = get_task_1_train_test_data(random_state, test_size)

        print(train_set)
        # {'word_counter': <scipy.sparse.csr_matrix>, 'data_frame': <pandas.DataFrame>}
        print(test_set)
        # {'word_counter': <scipy.sparse.csr_matrix>, 'data_frame': <pandas.DataFrame>}
    """

    dat = metex.view_schedule8_costs_by_datetime_location_reason()
    dat['weather_related'] = dat.WeatherCategory.map(lambda x: 0 if x == '' else 1)

    features = ['FinancialYear',
                'IncidentDescription',
                'IncidentReasonCode',
                'IncidentReasonName',
                'IncidentReasonDescription',
                'IncidentJPIPCategory',
                'IncidentCategory',
                'IncidentCategoryDescription',
                'IncidentCategorySuperGroupCode',
                'WeatherCategory']

    data = dat[['weather_related'] + features]
    data['descriptions'] = \
        data.IncidentDescription.astype(str) + ' ' + \
        data.IncidentReasonCode + ' ' + \
        data.IncidentReasonName + ' ' + \
        data.IncidentReasonDescription + ' ' + \
        data.IncidentJPIPCategory + ' ' + \
        data.IncidentCategory + ' ' + \
        data.IncidentCategoryDescription + ' ' + \
        data.IncidentCategorySuperGroupCode

    vectorizer = CountVectorizer()
    word_counter = vectorizer.fit_transform(np.array(data.descriptions))

    if random_state is None:
        train_data, test_data = data[data.FinancialYear < 2018], data[data.FinancialYear == 2018]
    else:
        # 'random_state' must be an integer
        non_weather_related_dat = data[dat.weather_related == 0]
        weather_related_dat = data[dat.weather_related == 1]

        train_dat_non, test_dat_non = train_test_split(
            non_weather_related_dat, random_state=random_state, test_size=test_size)
        train_dat, test_dat = train_test_split(
            weather_related_dat, random_state=random_state, test_size=test_size)

        train_data = pd.concat([train_dat_non, train_dat], axis=0)
        test_data = pd.concat([test_dat_non, test_dat], axis=0)

    idx_train, idx_test = np.array(train_data.index), np.array(test_data.index)

    train_set = dict(zip(['word_counter', 'data_frame'], [word_counter[idx_train], train_data]))
    test_set = dict(zip(['word_counter', 'data_frame'], [word_counter[idx_test], test_data]))

    return train_set, test_set


def classification_model_for_identifying_weather_related_incidents(random_state=0, test_size=0.2):
    """
    Fit model for Task 1.

    :param random_state: a random seed number, defaults to ``0``
    :type random_state: int, None
    :param test_size: size of test data set, defaults to ``0.2``
    :type test_size: int, float
    :return: trained model
    :rtype: sklearn.linear_model.logistic.LogisticRegression

    Testing e.g.

        from models.prototype.weather_attr import \
            classification_model_for_identifying_weather_related_incidents

        random_state = 0
        test_size = 0.2

        model = classification_model_for_identifying_weather_related_incidents(
            random_state, test_size)
    """

    train_set, test_set = get_task_1_train_test_data(random_state, test_size)
    model = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                               intercept_scaling=1, class_weight=None, random_state=random_state,
                               solver='saga', max_iter=1000, multi_class='ovr',
                               verbose=True,
                               warm_start=False, n_jobs=1)
    model.fit(train_set['word_counter'], train_set['data_frame'].weather_related)

    # model.score(test_set['word_counter'], test_set['data_frame'].weather_related)
    # test_set['data_frame']['weather_related_predicted'] = model.predict(test_set['word_counter'])
    return model


# == Task 2: Classification of weather-related incidents into different categories ====================

def get_task_2_train_test_data():
    """
    Get training and test data sets for Task 2.

    :return: training and test data sets
    :rtype: tuple - (dict, dict)

    **Test**::

        from models.prototype.weather_attr import get_task_2_train_test_data

        train_set, test_set = get_task_2_train_test_data()

        print(train_set)
        # {'word_counter': <scipy.sparse.csr_matrix>, 'data_frame': <pandas.DataFrame>}
        print(test_set)
        # {'word_counter': <scipy.sparse.csr_matrix>, 'data_frame': <pandas.DataFrame>}
    """

    reports = Schedule8IncidentReports()

    schedule8_weather_incidents = reports.get_schedule8_weather_incidents_02062006_31032014()
    schedule8_weather_incidents = schedule8_weather_incidents['Data']
    schedule8_weather_incidents.rename(
        columns={'Year': 'FinancialYear', 'IncidentReason': 'IncidentReasonCode'},
        inplace=True)

    dat = metex.view_schedule8_costs_by_datetime_location_reason()
    dat.WeatherCategory.fillna('', inplace=True)

    features = ['FinancialYear',
                'IncidentDescription',
                'IncidentReasonCode',
                'IncidentReasonName',
                'IncidentReasonDescription',
                'IncidentJPIPCategory',
                'IncidentCategory',
                'IncidentCategoryDescription',
                'IncidentCategorySuperGroupCode',
                'WeatherCategory']

    dat_train = schedule8_weather_incidents[['WeatherCategory'] + features]
    dat_test = dat[
        (dat.FinancialYear == 2014) & (dat.WeatherCategory != '')][['WeatherCategory'] + features]

    data = pd.DataFrame(pd.concat([dat_train, dat_test], ignore_index=True))

    data['descriptions'] = \
        data.IncidentDescription.astype(str) + ' ' + \
        data.IncidentReasonCode + ' ' + \
        data.IncidentReasonName + ' ' + \
        data.IncidentReasonDescription + ' ' + \
        data.IncidentJPIPCategory + ' ' + \
        data.IncidentCategory + ' ' + \
        data.IncidentCategoryDescription + ' ' + \
        data.IncidentCategorySuperGroupCode

    vectorizer = CountVectorizer()
    word_counter = vectorizer.fit_transform(np.array(data.descriptions))
    # data['word_count'] = csr_matrix_to_dict(word_counter, vectorizer)

    train_data, test_data = data[data.FinancialYear < 2014], data[data.FinancialYear == 2014]
    idx_train, idx_test = train_data.index, test_data.index

    train_set = dict(zip(
        ['word_counter', 'data_frame'], [word_counter[0:max(idx_train) + 1], train_data]))
    test_set = dict(zip(['word_counter', 'data_frame'], [word_counter[min(idx_test):], test_data]))

    return train_set, test_set


def classification_model_for_weather_related_incidents(random_state=0):
    """
    Fit model for Task 2.

    :param random_state: a random seed number, defaults to ``0``
    :type random_state: int, None
    :return: trained model
    :rtype: sklearn.linear_model.logistic.LogisticRegression

    Testing e.g.

        from models.prototype.weather_attr import classification_model_for_weather_related_incidents

        model = classification_model_for_weather_related_incidents(random_state, test_size)
    """

    train_set, test_set = get_task_2_train_test_data()
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                               intercept_scaling=1, class_weight=None, random_state=random_state,
                               solver='saga', max_iter=1000, multi_class='multinomial',
                               verbose=True,
                               warm_start=False, n_jobs=1)
    model.fit(train_set['word_counter'], train_set['data_frame'].WeatherCategory)

    # test_set['data_frame']['predicted_weather_category'] = model.predict(test_set['word_counter'])

    return model
