import os

import pandas as pd
from pyhelpers.store import save_fig
from pyhelpers.text import find_similar_str

from utils import cd, cdd

# ====================================================================================================================
""" Change directories """


def cdd_prototype(*sub_dir):
    path = cdd("Models\\prototype")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "Data\\Model" and sub-directories
def cd_prototype_dat(*sub_dir):
    path = cdd_prototype("dat")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "5 - Publications\\...\\Figures"
def cd_prototype_fig_pub(*sub_dir):
    path = cd("Paperwork\\5 - Publications\\1 - Prototype\\0 - Ingredients", "1 - Figures")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# A function for saving the plots
def save_hotpots_fig(fig, keyword, show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees,
                     save_as, dpi):
    """
    :param fig: [matplotlib.figure.Figure]
    :param keyword: [str] a keyword for specifying the filename
    :param show_metex_weather_cells: [bool]
    :param show_osm_landuse_forest: [bool]
    :param show_nr_hazardous_trees: [bool]
    :param save_as: [str]
    :param dpi: [int] or None
    :return:
    """
    if save_as is not None:
        if save_as.lstrip('.') in fig.canvas.get_supported_filetypes():
            suffix = zip([show_metex_weather_cells, show_osm_landuse_forest, show_nr_hazardous_trees],
                         ['cell', 'veg', 'haz'])
            filename = '_'.join([keyword] + [v for s, v in suffix if s is True])
            path_to_file = cd_prototype_fig_pub("Hotspots", filename + save_as)
            save_fig(path_to_file, dpi=dpi)


# Define a function the categorises wind directions into four quadrants
def categorise_wind_directions(degree):
    if (degree >= 0) & (degree < 90):
        return 1
    elif (degree >= 90) & (degree < 180):
        return 2
    elif (degree >= 180) & (degree < 270):
        return 3
    else:  # (degree >= 270) & (degree < 360):
        return 4


# Get data for specified season(s)
def get_data_by_season(incident_data, season='summer'):
    """
    :param incident_data: [pandas.DataFrame]
    :param season: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :return:
    """
    assert season is None or season in ('spring', 'summer', 'autumn', 'winter')
    if season:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4
        for y in pd.unique(incident_data.FinancialYear):
            y_dat = incident_data[incident_data.FinancialYear == y]
            # Get data for spring -----------------------------------
            spring_start1 = pd.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = pd.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring_dates1 = pd.date_range(spring_start1, spring_start1 + pd.DateOffset(months=2))
            spring_dates2 = pd.date_range(spring_start2, spring_start2 + pd.DateOffset(months=2))
            spring_data = pd.concat(
                [spring_data, y_dat[y_dat.StartDateTime.isin(spring_dates1) | y_dat.StartDateTime.isin(spring_dates2)]])

            spring_time = ((y_dat.StartDateTime >= spring_start1) & (y_dat.StartDateTime < spring_end1)) | \
                          ((y_dat.StartDateTime >= spring_start2) & (y_dat.StartDateTime < spring_end2))
            spring_data = pd.concat([spring_data, y_dat.loc[spring_time]])
            # Get data for summer ----------------------------------
            summer_start = pd.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (y_dat.StartDateTime >= summer_start) & (y_dat.StartDateTime < summer_end)
            summer_data = pd.concat([summer_data, y_dat.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = pd.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (y_dat.StartDateTime >= autumn_start) & (y_dat.StartDateTime < autumn_end)
            autumn_data = pd.concat([autumn_data, y_dat.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = pd.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (y_dat.StartDateTime >= winter_start) & (y_dat.StartDateTime < winter_end)
            winter_data = pd.concat([winter_data, y_dat.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [find_similar_str(s, seasons)[0] for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data

    else:
        return incident_data
