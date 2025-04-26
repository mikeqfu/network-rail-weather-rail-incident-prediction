"""Attribution / classification of different Weather-related Incidents."""

import gc
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessor import METEX, Schedule8IncidentReports


class _Base:

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.metex = METEX()

        self.training_set = None
        self.test_set = None
        self.model = None
        self.score = None

        self.features = [
            'FinancialYear',
            'IncidentDescription',
            'IncidentReasonCode',
            'IncidentReasonName',
            'IncidentReasonDescription',
            'IncidentJPIPCategory',
            'IncidentCategory',
            'IncidentCategoryDescription',
            'IncidentCategorySuperGroupCode',
            'WeatherCategory',
        ]

    @classmethod
    def compile_descriptions(cls, data):
        temp = \
            data['IncidentDescription'].astype(str) + ' ' + \
            data['IncidentReasonCode'] + ' ' + \
            data['IncidentReasonName'] + ' ' + \
            data['IncidentReasonDescription'] + ' ' + \
            data['IncidentJPIPCategory'] + ' ' + \
            data['IncidentCategory'] + ' ' + \
            data['IncidentCategoryDescription'] + ' ' + \
            data['IncidentCategorySuperGroupCode']
        temp.name = 'descriptions'
        data = pd.concat([data, temp], axis=1)

        del temp
        gc.collect()

        vectorizer = CountVectorizer()
        word_counter = vectorizer.fit_transform(data['descriptions'].values)

        # data['word_count'] = csr_matrix_to_dict(word_counter, vectorizer)

        return data, word_counter


class IncidentsIdentification(_Base):
    """Broad classification of Incidents into Weather-related and non-Weather-related."""

    def __init__(self, random_state=0):
        """

        :param random_state:
        :type random_state: int
        """

        super().__init__(random_state=random_state)

        self.test_size = 0.2

    def get_training_test_data(self, random_state=0, ret_data=False):
        # noinspection PyShadowingNames
        """
        Get training and test data sets for Task 1.

        :param random_state: a random seed number, defaults to ``0``
        :type random_state: int | None
        :param ret_data: defaults to ``False``
        :type ret_data: bool
        :return: training and test data sets
        :rtype: tuple[dict, dict]

        **Examples**::

            >>> from src.modeller.attribution import IncidentsIdentification

            >>> incid_ident = IncidentsIdentification()

            >>> incid_ident.get_training_test_data(random_state=0)
            >>> list(incid_ident.training_set.keys())
            ['word_counter', 'data_frame']
            >>> list(incid_ident.test_set.keys())
            ['word_counter', 'data_frame']
        """

        self.metex.view_schedule8_cost_by_day_location_reason()
        dat = self.metex.schedule8_cost_by_day_location_reason.copy()
        dat['weather_related'] = dat['WeatherCategory'].map(lambda x: 0 if x == '' else 1)

        data = dat[['weather_related'] + self.features]
        data, word_counter = self.compile_descriptions(data)

        if random_state == 0:
            training_data = data[data['FinancialYear'] < 2019]
            test_data = data[data['FinancialYear'] == 2019]

        else:
            non_weather_related_dat = data[dat['weather_related'] == 0]
            weather_related_dat = data[dat['weather_related'] == 1]

            training_dat_non, test_dat_non = train_test_split(
                non_weather_related_dat, random_state=self.random_state, test_size=self.test_size)
            training_dat, test_dat = train_test_split(
                weather_related_dat, random_state=self.random_state, test_size=self.test_size)

            training_data = pd.concat([training_dat_non, training_dat], axis=0)
            test_data = pd.concat([test_dat_non, test_dat], axis=0)

        training_idx, test_idx = training_data.index, test_data.index

        keys = ['word_counter', 'data_frame']
        training_set = dict(zip(keys, [word_counter[training_idx], training_data]))
        test_set = dict(zip(keys, [word_counter[test_idx], test_data]))

        self.training_set, self.test_set = training_set, test_set

        if ret_data:
            return self.training_set, self.test_set

    def identify_weather_related_incidents(self, test_size=0.2, random_state=0, verbose=True,
                                           ret_model=False):
        # noinspection PyShadowingNames
        """
        A classification model for identifying Weather-related Incidents.

        :param test_size: Size of test data set; defaults to ``0.2``.
        :type test_size: int | float
        :param random_state: A random seed number; defaults to ``0``.
        :type random_state: int | None
        :param verbose:
        :type verbose:
        :param ret_model:
        :type ret_model:
        :return: trained model
        :rtype: sklearn.linear_model.logistic.LogisticRegression

        Testing e.g.

            >>> from src.modeller.attribution import IncidentsIdentification

            >>> incid_ident = IncidentsIdentification()

            >>> incid_ident.identify_weather_related_incidents()

            >>> incid_ident.score
            0.9993649617654063
        """

        if self.test_size != test_size:
            self.test_size = test_size

        assert isinstance(random_state, int)  # 'random_state' must be an integer
        self.random_state = random_state

        if self.training_set is None or self.test_set is None:
            self.get_training_test_data(random_state=self.random_state)

        model = LogisticRegression(
            penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1,
            class_weight=None, solver='saga', max_iter=1000, multi_class='ovr', verbose=verbose,
            random_state=self.random_state, warm_start=False, n_jobs=os.cpu_count() - 1)

        X_train = self.training_set['word_counter']
        y_train = self.training_set['data_frame']['weather_related']
        model.fit(X_train, y_train)

        X_test = self.test_set['word_counter']
        y_test = self.test_set['data_frame']['weather_related']
        # test_weather_related_predicted = model.predict(X_test)
        self.score = model.score(X_test, y_test)

        self.model = model

        if ret_model:
            return self.model


class WeatherRelatedIncidentsAttribution(_Base):
    """Classification of Weather-related Incidents into different categories."""

    def __init__(self, random_state=0):
        """

        :param random_state:
        :type random_state: int | None
        """

        super().__init__(random_state=random_state)

        self.sir = Schedule8IncidentReports()

    def get_training_test_data(self, ret_data=False):
        # noinspection PyShadowingNames
        """
        Get training and test data sets for Task 2.

        :param ret_data: defaults to ``False``
        :type ret_data: bool
        :return: training and test data sets
        :rtype: tuple - (dict, dict)

        **Examples**::

            >>> from src.modeller.attribution import WeatherRelatedIncidentsAttribution

            >>> wia = WeatherRelatedIncidentsAttribution()

            >>> wia.get_training_test_data()

            >>> list(wia.training_set.keys())
            ['word_counter', 'data_frame']
            >>> list(wia.test_set.keys())
            ['word_counter', 'data_frame']
        """

        self.sir.read_schedule8_weather_incidents_02062006_31032014()
        ref_dat_dict = self.sir.schedule8_weather_incidents_02062006_31032014.copy()
        ref_dat = ref_dat_dict['Schedule8WeatherIncidents_02062006_31032014']
        ref_dat.rename(columns={'IncidentReason': 'IncidentReasonCode'}, inplace=True)

        self.metex.view_schedule8_cost_by_day_location_reason()
        dat = self.metex.schedule8_cost_by_day_location_reason.copy()
        # dat['WeatherCategory'].fillna('', inplace=True)

        dat_train = ref_dat[self.features]

        test_mask = (dat['FinancialYear'] == 2014) & (dat['WeatherCategory'] != '')
        dat_test = dat[test_mask][self.features]

        data = pd.DataFrame(pd.concat([dat_train, dat_test], ignore_index=True))
        data, word_counter = self.compile_descriptions(data)

        training_data = data[data['FinancialYear'] < 2014]
        test_data = data[data['FinancialYear'] == 2014]
        training_idx, test_idx = training_data.index, test_data.index

        keys = ['word_counter', 'data_frame']
        training_set = dict(zip(keys, [word_counter[0:max(training_idx) + 1], training_data]))
        test_set = dict(zip(keys, [word_counter[min(test_idx):], test_data]))

        self.training_set, self.test_set = training_set, test_set

        if ret_data:
            return self.training_set, self.test_set

    def classify_weather_related_incidents(self, random_state=0, verbose=True, ret_model=False):
        # noinspection PyShadowingNames
        """
        Fit model for Task 2.

        :param random_state: a random seed number, defaults to ``0``
        :type random_state: int or None
        :param verbose:
        :type verbose:
        :param ret_model:
        :type ret_model:
        :return: trained model
        :rtype: sklearn.linear_model.logistic.LogisticRegression

        **Examples**::

            >>> from src.modeller.attribution import WeatherRelatedIncidentsAttribution

            >>> wia = WeatherRelatedIncidentsAttribution()

            >>> wia.classify_weather_related_incidents()

            >>> wia.score
            0.9819918796274182
        """

        if self.training_set is None or self.test_set is None:
            self.get_training_test_data()

        if random_state != self.random_state:
            self.random_state = random_state

        model = LogisticRegression(
            penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
            class_weight=None, solver='saga', max_iter=1000, multi_class='multinomial',
            random_state=self.random_state, verbose=verbose, warm_start=False,
            n_jobs=os.cpu_count() - 1)

        X_train = self.training_set['word_counter']
        y_train = self.training_set['data_frame']['WeatherCategory']
        model.fit(X_train, y_train)

        self.model = model

        X_test = self.test_set['word_counter']
        y_test = self.test_set['data_frame']['WeatherCategory']

        self.score = self.model.score(X_test, y_test)
        # test_set['data_frame']['predicted_weather_category'] = model.predict(y_test)

        if ret_model:
            return self.model


if __name__ == '__main__':
    incid_ident = IncidentsIdentification()

    incid_ident.get_training_test_data(random_state=0)
    assert list(incid_ident.training_set.keys()) == ['word_counter', 'data_frame']
    assert list(incid_ident.test_set.keys()) == ['word_counter', 'data_frame']
    incid_ident.identify_weather_related_incidents()
    print(incid_ident.score)

    wia = WeatherRelatedIncidentsAttribution()

    wia.get_training_test_data()
    assert list(wia.training_set.keys()) == ['word_counter', 'data_frame']
    assert list(wia.test_set.keys()) == ['word_counter', 'data_frame']
    wia.classify_weather_related_incidents()
    print(wia.score)
