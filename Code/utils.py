""" Utilities - Helper functions """

import os
import re
import shutil

import fuzzywuzzy.fuzz
import fuzzywuzzy.process
import numpy as np
import pandas as pd
from pyhelpers.dir import cd, cdd
from pyhelpers.misc import confirmed
from pyhelpers.store import load_json

# ====================================================================================================================
""" Change/remove directories """


# Change directory to ".\\Data\\METEX"
def cdd_metex(*sub_dir):
    path = cdd("METEX")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to ".\\Data\\Vegetation"
def cdd_vegetation(*sub_dir):
    path = cdd("Vegetation")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to ".\\Data\\RailwayCodes"
def cdd_rc(*sub_dir):
    path = cdd("Network", "Railway Codes")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Publications\\...\\Figures" and sub-directories
def cdd_metex_fig_pub(pid, subject, *sub_dir):
    path = cd("Publications", "{} - {}".format(pid, subject), "Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Remove a directory
def rm_dir(path, confirmation_required=True):
    if os.listdir(path):
        if confirmed("\"{}\" is not empty. Confirmed to continue removing the directory?".format(path),
                     confirmation_required=confirmation_required):
            shutil.rmtree(path)
    else:
        os.rmdir(path)


# ====================================================================================================================
""" Misc """


# Check if a str expresses a float
def is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        try:
            float(re.sub('[()~]', '', text))
            return True
        except ValueError:
            return False


# Reset double indexes
def reset_double_indexes(data):
    levels = list(data.columns)
    column_names = []
    for i in range(len(levels)):
        col_name = levels[i][0] + '_' + levels[i][1]
        column_names += [col_name]
    data.columns = column_names
    data.reset_index(inplace=True)
    return data


# Check whether a string contains digits
def contains_digits(text):
    x = re.compile('\\d').search(text)
    return bool(x)


# Find the closest date of the given 'data' from a list of dates
def find_closest_date(date, dates_list):
    return min(dates_list, key=lambda x: abs(x - date))


# Calculate the n-th percentile
def percentile(n):
    def np_percentile(x):
        return np.percentile(x, n)

    np_percentile.__name__ = 'percentile_%s' % n
    return np_percentile


# Calculate interquartile range (IQR)
def interquartile_range(x):
    """
    Alternative way: using scipy.stats.iqr(x)
    """
    return np.subtract(*np.percentile(x, [75, 25]))


# Form a file name in terms of specific 'Route' and 'Weather' category
def make_filename(base_name, route_name=None, weather_category=None, *extra_suffixes, sep="_", save_as=".pickle"):
    route_lookup = list(set(load_json(cdd("Network\\Routes", "route-names-changes.json")).values()))
    weather_category_lookup = load_json(cdd("Weather", "weather-categories.json"))
    base_name_ = "data" if base_name is None else base_name
    route_name_ = "" if route_name is None \
        else sep + fuzzywuzzy.process.extractOne(route_name, route_lookup, scorer=fuzzywuzzy.fuzz.ratio)[0][0]
    weather_category_ = "" if weather_category is None \
        else sep + fuzzywuzzy.process.extractOne(weather_category, weather_category_lookup['WeatherCategory'],
                                                 scorer=fuzzywuzzy.fuzz.ratio)[0]
    if extra_suffixes:
        suffix = [str(s) for s in extra_suffixes if s]
        suffix = sep + sep.join(suffix) if len(suffix) > 1 else sep + suffix[0]
        filename = base_name_ + route_name_ + weather_category_ + suffix + save_as
    else:
        filename = base_name_ + route_name_ + weather_category_ + save_as
    return filename


# Subset the required data given 'route' and 'Weather'
def get_subset(data, route_name=None, weather_category=None, reset_index=False):
    if data is not None:
        assert isinstance(data, pd.DataFrame) and not data.empty
        data_subset = data.copy(deep=True)

        if route_name:
            try:  # assert 'Route' in data_subset.columns
                data_subset.Route = data_subset.Route.astype(str)
                route_lookup = list(set(data_subset.Route))
                route_name_ = [
                    fuzzywuzzy.process.extractOne(x, route_lookup, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    for x in ([route_name] if isinstance(route_name, str) else list(route_name))]
                data_subset = data_subset[data_subset.Route.isin(route_name_)]
            except AttributeError:
                print("Couldn't slice the data by 'Route'. The attribute may not exist in the DataFrame.")
                pass

        if weather_category:
            try:  # assert 'WeatherCategory' in data_subset.columns
                data_subset.WeatherCategory = data_subset.WeatherCategory.astype(str)
                weather_category_lookup = list(set(data_subset.WeatherCategory))
                weather_category_ = [
                    fuzzywuzzy.process.extractOne(x, weather_category_lookup, scorer=fuzzywuzzy.fuzz.ratio)[0]
                    for x in ([weather_category] if isinstance(weather_category, str) else list(weather_category))]
                data_subset = data_subset[data_subset.WeatherCategory.isin(weather_category_)]
            except AttributeError:
                print("Couldn't slice the data by 'WeatherCategory'. The attribute may not exist in the DataFrame.")
                pass

        if reset_index:
            data_subset.index = range(len(data_subset))  # data_subset.reset_index(inplace=True)

    else:
        data_subset = None

    return data_subset
