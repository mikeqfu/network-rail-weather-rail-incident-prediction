""" Utilities - Helper functions """

import numpy as np
import pandas as pd
from pyhelpers import cdd, load_json, find_similar_str


# == Change directories ===============================================================================

def cdd_incidents(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\incidents\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\incidents\\" and sub-directories / a file
    :rtype: str
    """
    path = cdd("incidents", *sub_dir, mkdir=mkdir)
    return path


def cdd_metex(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\metex\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\metex\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd("metex", *sub_dir, mkdir=mkdir)
    return path


def cdd_models(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\models\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\models\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd("models", *sub_dir, mkdir=mkdir)
    return path


def cdd_network(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\network\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\network\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd("network", *sub_dir, mkdir=mkdir)
    return path


def cdd_railway_codes(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\network\\railway codes\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\network\\railway codes\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd_network("railway codes", *sub_dir, mkdir=mkdir)
    return path


def cdd_vegetation(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\vegetation\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\vegetation\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd("vegetation", *sub_dir, mkdir=mkdir)
    return path


def cdd_weather(*sub_dir, mkdir=False):
    """
    Change directory to "..\\data\\weather\\" and sub-directories / a file.

    :param sub_dir: name of directory or names of directories (and/or a filename)
    :type sub_dir: str
    :param mkdir: whether to create a directory, defaults to ``False``
    :type mkdir: bool
    :return: full path to "..\\data\\weather\\" and sub-directories / a file
    :rtype: str
    """

    path = cdd("weather", *sub_dir, mkdir=mkdir)
    return path


# == Misc =============================================================================================

def make_filename(base_name, route_name=None, weather_category=None, *extra_suffixes, sep="-", save_as=".pickle"):
    """
    Make a filename as appropriate.

    :param base_name: base name, defaults to ``None``
    :type base_name: str, None
    :param route_name: name of a Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str, None
    :param extra_suffixes: extra suffixes to the filename
    :type extra_suffixes: str, None
    :param sep: a separator in the filename, defaults to ``"-"``
    :type sep: str, None
    :param save_as: file extension, defaults to ``".pickle"``
    :type save_as: str
    :return: a filename
    :rtype: str

    **Examples**::

        from utils import make_filename

        base_name = "test"  # None
        sep = "-"
        save_as = ".pickle"

        route_name = None
        weather_category = None
        make_filename(base_name, route_name, weather_category)
        # test.pickle

        route_name = None
        weather_category = 'Heat'
        make_filename(None, route_name, weather_category, "test1")
        # test1.pickle

        make_filename(base_name, route_name, weather_category, "test1", "test2")
        # test-test1-test2.pickle

        make_filename(base_name, 'Anglia', weather_category, "test2")
        # test-Anglia-test2.pickle

        make_filename(base_name, 'North and East', 'Heat', "test1", "test2")
        # test-North_and_East-Heat-test1-test2.pickle
    """

    base_name_ = "" if base_name is None else base_name

    if route_name is None:
        route_name_ = ""
    else:
        rts = list(set(load_json(cdd_network("routes", "name-changes.json")).values()))
        route_name_ = (sep if base_name else "") + sep.join(
            [find_similar_str(x, rts).replace(" ", "")
             for x in ([route_name] if isinstance(route_name, str) else list(route_name))])

    if weather_category is None:
        weather_category_ = ""
    else:
        wcs = load_json(cdd_weather("weather-categories.json"))['WeatherCategory']
        weather_category_ = (sep if route_name else "") + sep.join(
            [find_similar_str(x, wcs).replace(" ", "")
             for x in ([weather_category] if isinstance(weather_category, str) else list(weather_category))])

    if base_name_ + route_name_ + weather_category_ == '':
        base_name_ = "data"

    if extra_suffixes:
        extra_suffixes_ = [extra_suffixes] if isinstance(extra_suffixes, str) else extra_suffixes
        suffix_ = ["{}".format(s) for s in extra_suffixes_ if s]
        try:
            suffix = sep + sep.join(suffix_) if len(suffix_) > 1 else sep + suffix_[0]
        except IndexError:
            suffix = ""
        filename = base_name_ + route_name_ + weather_category_ + suffix + save_as

    else:
        filename = base_name_ + route_name_ + weather_category_ + save_as

    return filename


def get_subset(data_frame, route_name=None, weather_category=None, rearrange_index=False):
    """
    Subset of a data set for the given Route and weather category.

    :param data_frame: a data frame (that contains 'Route' and 'WeatherCategory')
    :type data_frame: pandas.DataFrame, None
    :param route_name: name of a Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather category, defaults to ``None``
    :type weather_category: str, None
    :param rearrange_index: whether to rearrange the index of the subset, defaults to ``False``
    :type rearrange_index: bool
    :return: a subset of the ``data_frame`` for the given ``route_name`` and ``weather_category``
    :rtype: pandas.DataFrame, None
    """

    if data_frame is not None:
        assert isinstance(data_frame, pd.DataFrame) and not data_frame.empty
        data_subset = data_frame.copy(deep=True)

        if route_name:
            try:  # assert 'Route' in data_subset.columns
                data_subset.Route = data_subset.Route.astype(str)
                route_lookup = list(set(data_subset.Route))
                route_name_ = [find_similar_str(x, route_lookup)
                               for x in ([route_name] if isinstance(route_name, str) else list(route_name))]
                data_subset = data_subset[data_subset.Route.isin(route_name_)]
            except AttributeError:
                print("Couldn't slice the data by 'Route'. The attribute may not exist in the DataFrame.")
                pass

        if weather_category:
            try:  # assert 'WeatherCategory' in data_subset.columns
                data_subset.WeatherCategory = data_subset.WeatherCategory.astype(str)
                weather_category_lookup = list(set(data_subset.WeatherCategory))
                weather_category_ = [find_similar_str(x, weather_category_lookup)
                                     for x in ([weather_category]
                                               if isinstance(weather_category, str) else list(weather_category))]
                data_subset = data_subset[data_subset.WeatherCategory.isin(weather_category_)]
            except AttributeError:
                print("Couldn't slice the data by 'WeatherCategory'. The attribute may not exist in the DataFrame.")
                pass

        if rearrange_index:
            data_subset.index = range(len(data_subset))  # data_subset.reset_index(inplace=True)

    else:
        data_subset = None

    return data_subset


def remove_list_duplicates(lst):
    """
    Remove duplicates in a list.

    :param lst: a list
    :type lst: list
    :return: a list without duplicated items
    :rtype: list
    """

    output = []
    temp = set()
    for item in lst:
        if item not in temp:
            output.append(item)
            temp.add(item)
    del temp
    return output


def remove_list_duplicated_lists(lst_lst):
    """
    Make each item in a list be unique, where the item is also a list.

    :param lst_lst: a list of lists
    :type lst_lst: list
    :return: a list of lists with each item-list being unique
    :rtype: list
    """

    output = []
    temp = set()
    for lst in lst_lst:
        if any(item not in temp for item in lst):
            # lst[0] not in temp and lst[1] not in temp:
            output.append(lst)
            for item in lst:
                temp.add(item)  # e.g. temp.add(lst[0]); temp.add(lst[1])
    del temp
    return output


def get_index_of_dict_in_list(lst_of_dict, key, val):
    """
    Get the index of a dictionary in a list by a given (key, value).

    :param lst_of_dict: a list of dictionaries
    :param key: key of the queried dictionary
    :param val: value of the queried dictionary
    :return:

    **Example**::

        lst_of_dict = [{'a': 1}, {'b': 2}, {'c': 3}]
        key = 'b'
        value = 2

        get_index_of_dict_in_list(lst_of_dict, key, value)
    """

    return next(idx for (idx, d) in enumerate(lst_of_dict) if d.get(key) == val)


def merge_two_dicts(dict1, dict2):
    """
    Given two dicts, merge them into a new dict as a shallow copy.

    :param dict1: a dictionary
    :type dict1: dict
    :param dict2: another dictionary
    :type dict2: dict
    :return: a merged dictionary of ``dict1`` and ``dict2``
    :rtype: dict
    """

    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict


def merge_dicts(*dictionaries):
    """
    Given any number of dicts, shallow copy and merge into a new dict, precedence goes to key val pairs in latter dicts.

    :param dictionaries: one or a sequence of dictionaries
    :type dictionaries: dict
    :return: a merged dictionary
    :rtype: dict
    """

    new_dict = {}
    for d in dictionaries:
        new_dict.update(d)
    return new_dict


def reset_double_indexes(data_frame):
    """
    Reset double indexes.

    :param data_frame:
    :return:
    """

    levels = list(data_frame.columns)
    column_names = []
    for i in range(len(levels)):
        col_name = levels[i][0] + '_' + levels[i][1]
        column_names += [col_name]
    data_frame.columns = column_names
    data_frame.reset_index(inplace=True)
    return data_frame


def percentile(n):
    """
    Calculate the n-th percentile.
    """

    def np_percentile(x):
        return np.percentile(x, n)

    np_percentile.__name__ = 'percentile_%s' % n
    return np_percentile


def update_nr_route_names(data_set, route_col_name='Route'):
    """
    Update names of NR Routes.

    :param data_set: a given data frame that contains 'Route' column
    :type data_set: pandas.DataFrame
    :param route_col_name: name of the column for 'Route', defaults to ``'Route'``
    :return: updated data frame
    :rtype: pandas.DataFrame
    """

    assert isinstance(data_set, pd.DataFrame)
    assert route_col_name in data_set.columns
    route_names_changes = load_json(cdd_network("Routes", "name-changes.json"))
    new_route_col_name = route_col_name + 'Alias'
    data_set.rename(columns={route_col_name: new_route_col_name}, inplace=True)
    data_set[route_col_name] = data_set[new_route_col_name].replace(route_names_changes)


def get_coefficients(model, feature_names=None):
    """
    Get regression model coefficients presented as a data frame.

    :param model: an instance of regression model
    :type model: sklearn class
    :param feature_names: name of features (i.e. column names of input data); if ``None`` (default), all features
    :return: data frame for model coefficients
    :rtype: pandas.DataFrame
    """

    if feature_names is None:
        feature_names = ['feature_%d' % i for i in range(len(model.coef_))]
    feature_names = ['(intercept)'] + feature_names

    intercept = model.intercept_
    coefficients = model.coef_
    coefficients = [intercept] + list(coefficients)

    coef_dat = pd.DataFrame({'coefficients': coefficients}, index=feature_names)

    return coef_dat
