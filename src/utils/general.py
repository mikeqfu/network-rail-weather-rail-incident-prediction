"""
Provide a range of utility functions for general use.
"""

import functools
import logging
import multiprocessing

import numpy as np
import pandas as pd
import shapely.geometry
from pyhelpers.dirs import cdd
from pyhelpers.store import load_data
from pyhelpers.text import find_similar_str, get_acronym


def make_filename(name, route_name=None, weather_category=None, *suffixes, sep="_", save_as=".pkl"):
    # noinspection PyShadowingNames
    """
    Make a filename.

    :param name: Base filename; defaults to ``None``.
    :type name: str | None
    :param route_name: A specific Route name or a list of Route names; defaults to ``None``.
    :type route_name: str | list | None
    :param weather_category: A specific weather category or a list of weather categories;
        defaults to ``None``.
    :type weather_category: str | list | None
    :param suffixes: Suffixes to the filename.
    :type suffixes: int | str
    :param sep: A separator in the filename; defaults to ``"_"``.
    :type sep: str | None
    :param save_as: File extension; defaults to ``".pkl"``.
    :type save_as: str
    :return: A filename.
    :rtype: str

    **Examples**::

        >>> from src.utils import make_filename
        >>> name = "filename"  # None
        >>> route_name = None
        >>> weather_category = None
        >>> make_filename(name, route_name, weather_category)
        'filename.pkl'
        >>> route_name = None
        >>> weather_category = 'Heat'
        >>> make_filename(None, route_name, weather_category, "test1")
        'Heat_test1.pkl'
        >>> make_filename(name, route_name, weather_category, "test1", "test2")
        'filename_Heat_test1_test2.pkl'
        >>> make_filename(name, 'Anglia', weather_category, "test2")
        'filename_Anglia_Heat_test2.pkl'
        >>> make_filename(name, 'North and East', 'Heat', "test1", "test2")
        'filename_N&E_Heat_test1_test2.pkl'
    """

    base_name = "" if name is None else name

    if route_name is None:
        route_name_ = ""
    else:
        rts = list(set(load_data(cdd("..\\data\\Network\\Routes", "Name-changes.json")).values()))
        route_name_ = sep.join(
            [get_acronym(find_similar_str(x, rts).replace(' and ', '&'), keep_punctuation=True)
             if ' ' in x else x
             for x in ([route_name] if isinstance(route_name, str) else list(route_name))])
        if base_name != "":
            route_name_ = sep + route_name_

    if weather_category is None:
        weather_category_ = ""
    else:
        wcs = load_data(cdd("..\\data\\Weather", "Weather-categories.json"))['WeatherCategory']
        weather_category_ = sep.join(
            [find_similar_str(x, wcs).replace(" ", "") for x in
             ([weather_category] if isinstance(weather_category, str) else list(weather_category))])
        if base_name != "":
            weather_category_ = sep + weather_category_

    if base_name + route_name_ + weather_category_ == "":
        base_name = "data"

    if suffixes:
        extra_suffixes = [suffixes] if isinstance(suffixes, str) else suffixes
        suffix_ = ["{}".format(s) for s in extra_suffixes if s]
        try:
            suffix = sep + sep.join(suffix_) if len(suffix_) > 1 else sep + suffix_[0]
        except IndexError:
            suffix = ""
        filename = base_name + route_name_ + weather_category_ + suffix + save_as

    else:
        filename = base_name + route_name_ + weather_category_ + save_as

    return filename


def _get_subset(subset, column_name, func, arg):
    if arg:
        try:  # assert 'weather_category' in data_subset.columns
            subset[column_name] = subset[column_name].astype(str)
            lookup_list = list(set(subset[column_name]))
            lookup_list_ = func(arg, lookup_list)
            subset = subset[subset[column_name].isin(lookup_list_)]
        except KeyError:
            logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
            logging.warning(
                f"Couldn't slice the data by '{column_name}'. "
                f"The attribute does not appear to exist in the DataFrame.")

    return subset


def _get_route_name(route_name, route_name_lookup_list):
    route_name_ = [
        find_similar_str(x, route_name_lookup_list)
        for x in ([route_name] if isinstance(route_name, str) else list(route_name))]
    return route_name_


def _get_weather_category(weather_category, weather_category_lookup_list):
    weather_category_ = [
        find_similar_str(x, weather_category_lookup_list)
        for x in (
            [weather_category]
            if isinstance(weather_category, str) else list(weather_category))]
    return weather_category_


def get_subset(data, route_name=None, weather_category=None, rearrange_index=False):
    # noinspection PyShadowingNames
    """
    Get a subset or slice of a dataframe for a specific route and weather category.

    :param data: A dataframe (which contains 'Route' and 'weather_category' fields).
    :type data: pandas.DataFrame | None
    :param route_name: Name of a Route; defaults to ``None``.
    :type route_name: str | list | None
    :param weather_category: Weather category; defaults to ``None``.
    :type weather_category: str | list | None
    :param rearrange_index: Whether to rearrange the index of the subset; defaults to ``False``.
    :type rearrange_index: bool
    :return: A subset of the ``data`` for the given ``route_name`` and ``weather_category``.
    :rtype: pandas.DataFrame | None

    **Examples**::

        >>> from src.utils import get_subset
        >>> from pyhelpers._cache import example_dataframe
        >>> data = example_dataframe()
        >>> get_subset(data)
                    Longitude   Latitude
        City
        London      -0.127647  51.507322
        Birmingham  -1.902691  52.479699
        Manchester  -2.245115  53.479489
        Leeds       -1.543794  53.797418
        >>> subset = get_subset(data, route_name='Anglia')
        WARNING:root:Couldn't slice the data by 'Route'. ...
        >>> subset = get_subset(data, weather_category='Wind')
        WARNING:root:Couldn't slice the data by 'WeatherCategory'. ...
        >>> data['Route'] = ['Anglia', 'R1', 'R2', 'R3']
        >>> subset = get_subset(data, route_name='Anglia')
        >>> subset
                Longitude   Latitude   Route
        City
        London  -0.127647  51.507322  Anglia
        >>> data['WeatherCategory'] = ['Wind', 'WC1', 'WC2', 'WC3']
        >>> subset = get_subset(data, weather_category='Wind')
        >>> subset
                Longitude   Latitude   Route WeatherCategory
        City
        London  -0.127647  51.507322  Anglia            Wind
    """

    if data is not None:
        assert isinstance(data, pd.DataFrame) and not data.empty, "`data` must be a dataframe."
        subset = data.copy()

        subset = _get_subset(subset, 'Route', _get_route_name, route_name)
        subset = _get_subset(subset, 'WeatherCategory', _get_weather_category, weather_category)

        if rearrange_index:
            subset.index = range(len(subset))  # data_subset.reset_index(inplace=True)

    else:
        subset = None

    return subset


def remove_list_duplicates(lst):
    """
    Remove duplicates in a list.

    :param lst: A list.
    :type lst: list
    :return: A list without duplicated items.
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

    :param lst_lst: A list of lists.
    :type lst_lst: list
    :return: A list of lists with each item-list being unique.
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


def get_index_of_dict_in_list(list_of_dicts, key, val):
    # noinspection PyShadowingNames
    """
    Get the index of a dictionary in a list by a given (key, value).

    :param list_of_dicts: A list of dictionaries.
    :type list_of_dicts: list
    :param key: Key of the queried dictionary.
    :type key: typing.Any
    :param val: Value of the queried dictionary.
    :type val: typing.Any
    :return: Index of the queried dictionary.
    :rtype: int

    **Examples**::

        >>> from src.utils import get_index_of_dict_in_list
        >>> list_of_dicts = [{'a': 1}, {'b': 2}, {'c': 3}]
        >>> key = 'b'
        >>> value = 2
        >>> get_index_of_dict_in_list(list_of_dicts, key, value)
        1
    """

    index_of_dict = next(idx for (idx, d) in enumerate(list_of_dicts) if d.get(key) == val)

    return index_of_dict


def reset_double_indexes(data_frame):
    """
    Reset double indexes for a given dataframe.

    :param data_frame: A dataframe with double indexes.
    :type data_frame: pandas.DataFrame
    :return: A dataframe with the double indexes being reset.
    :rtype: pandas.DataFrame
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


def update_route_names(data, route_col_name='Route'):
    """
    Update names of Network Rail Routes.

    :param data: A dataframe, of which a column should contain information of Network Rail's Routes.
    :type data: pandas.DataFrame
    :param route_col_name: Name of the column for Network Rail's Routes; defaults to ``'Route'``.
    :return: An updated dataframe.
    :rtype: pandas.DataFrame
    """

    assert isinstance(data, pd.DataFrame)
    assert route_col_name in data.columns

    data_ = data.copy()

    route_names_changes = load_data(cdd("../data/Network", "Routes", "Name-changes.json"))

    new_route_col_name = route_col_name + 'Alias'
    data_.rename(columns={route_col_name: new_route_col_name}, inplace=True)
    data_[route_col_name] = data_[new_route_col_name].replace(route_names_changes)

    return data_


def get_coefficients(model, feature_names=None):
    """
    Get regression model coefficients presented as a data frame.

    :param model: An instance of a model (e.g. a logistic regression model).
    :type model: sklearn.base.BaseEstimator
    :param feature_names: Names of features;
        when ``feature_names=None`` (default), it takes all available features.
    :return: A dataframe containing information of the model coefficients.
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


def points_from_xy(xy_df, as_series=True):
    """
    Get a sequence of points' geographical coordinates from a given dataframe.

    :param xy_df: A dataframe containing geographical coordinates in two separate columns.
    :type xy_df: pandas.DataFrame
    :param as_series: Whether to return the results as a pandas.Series; defaults to ``True``.
    :type as_series: bool
    :return: A sequence of points' geographical coordinates.
    :rtype: pandas.Series | list
    """

    xy_array = xy_df.to_numpy()
    xy_points_ = shapely.geometry.MultiPoint(xy_array)

    if as_series:
        xy_points = pd.Series(x.coords for x in xy_points_.geoms).map(shapely.geometry.Point)
    else:
        xy_points = [x.coords for x in xy_points_.geoms]

    return xy_points


def parallelize(data, func, num_of_processes=None):
    """
    Parallelize some operations on a dataframe.

    :param data: A dataframe.
    :type data: pandas.DataFrame
    :param func: A defined function (or method).
    :type func: typing.Callable
    :param num_of_processes: Number of processors;
        when ``num_of_processes=None`` (default), it takes all but one of the available processors.
    :type num_of_processes: int | None
    :return: A dataframe as a result of the operations by ``func``.
    :rtype: pandas.DataFrame

    Reference: https://stackoverflow.com/questions/26784164/
    """

    if num_of_processes is None:
        num_of_processes = multiprocessing.cpu_count() - 1

    num_of_rows = len(data)
    if num_of_rows == 0:
        return None
    elif num_of_rows < num_of_processes:
        num_of_processes = num_of_rows

    data_partitions = np.array_split(data, indices_or_sections=num_of_processes)

    with multiprocessing.Pool(processes=num_of_processes) as p:
        data_ = pd.concat(p.map(func, data_partitions), axis=0)

    return data_


def _apply_row_wise(func, data_partition):
    dat = data_partition.apply(func, axis=1)

    return dat


def multiprocess_apply_row_wise(data, func, num_of_processes=None):
    """
    Implement the ``.apply()` method on a dataframe via multiple processors.

    :param data: A dataframe.
    :type data: pandas.DataFrame
    :param func: A defined function (or method).
    :type func: typing.Callable
    :param num_of_processes: Number of processors;
        when ``num_of_processes=None`` (default), it takes all but one of the available processors.
    :type num_of_processes: int | None
    :return: A dataframe as a result of the operations by ``func``.
    :rtype: pandas.DataFrame
    """

    data_ = parallelize(
        data=data, func=functools.partial(_apply_row_wise, func), num_of_processes=num_of_processes)

    return data_
