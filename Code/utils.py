""" Utilities - Helper functions """

import os
import re
import shutil

import fuzzywuzzy.fuzz
import fuzzywuzzy.process
import numpy as np
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


# Find from a list the closest, case-insensitive, string to the given one
def find_match(x, lookup):
    if x is '' or x is None:
        return None
    else:
        for y in lookup:
            if re.match(x, y, re.IGNORECASE):
                return y


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
def make_filename(base_name, route_name, weather_category, *extra_suffixes, save_as=".pickle"):
    if route_name is not None:
        route_lookup = load_json(cdd("Network\\Routes", "route-names.json"))
        route_name = fuzzywuzzy.process.extractOne(route_name, route_lookup['Route'], scorer=fuzzywuzzy.fuzz.ratio)[0]
    if weather_category is not None:
        weather_category_lookup = load_json(cdd("Weather", "Weather-categories.json"))
        weather_category = fuzzywuzzy.process.extractOne(weather_category, weather_category_lookup['WeatherCategory'],
                                                         scorer=fuzzywuzzy.fuzz.ratio)
    filename_suffix = [s for s in (route_name, weather_category) if s]  # "s" stands for "suffix"
    filename = "_".join([base_name] + filename_suffix + [str(s) for s in extra_suffixes if s]) + save_as
    return filename


# Subset the required data given 'route' and 'Weather'
def subset(data, route=None, weather_category=None, reset_index=False):
    if data is None:
        data_subset = None
    else:
        route_lookup = list(set(data.Route))
        weather_category_lookup = list(set(data.WeatherCategory))
        # Select data for a specific route and Weather category
        if not route and not weather_category:
            data_subset = data.copy()
        elif route and not weather_category:
            data_subset = data[data.Route == fuzzywuzzy.process.extractOne(route, route_lookup, score_cutoff=10)[0]]
        elif not route and weather_category:
            data_subset = data[
                data.WeatherCategory ==
                fuzzywuzzy.process.extractOne(weather_category, weather_category_lookup, score_cutoff=10)[0]]
        else:
            data_subset = data[
                (data.Route == fuzzywuzzy.process.extractOne(route, route_lookup, score_cutoff=10)[0]) &
                (data.WeatherCategory ==
                 fuzzywuzzy.process.extractOne(weather_category, weather_category_lookup, score_cutoff=10)[0])]
        # Reset index
        if reset_index:
            data_subset.reset_index(inplace=True)  # dat.index = range(len(dat))
    return data_subset
