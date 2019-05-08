import os

import fuzzywuzzy.process
import numpy as np
import pandas as pd

from utils import cdd

# ====================================================================================================================
""" Change directories """


# Change directory to ".\\modelling\\intermediate\\..." and sub-directories
def cd_intermediate(*sub_dir):
    path = cdd("modelling\\intermediate")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to ".\\modelling\\prototype-Heat\\..." and sub-directories
def cdd_intermediate(*sub_dir):
    path = cd_intermediate("dat")
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Change directory to "modelling\\prototype-Heat\\Trial_" and sub-directories
def cdd_intermediate_trial(trial_id=0, *sub_dir):
    path = cd_intermediate("Trial_{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# ====================================================================================================================
""" Utilities """


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
        season = [fuzzywuzzy.process.extractOne(s, seasons)[0] for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data

    else:
        return incident_data


# Get Angle of Line between Two Points
def get_angle_of_line_between(p1, p2, in_degrees=False):
    x_diff = p2.x - p1.x
    y_diff = p2.y - p1.y
    angle = np.arctan2(y_diff, x_diff)  # in radians
    if in_degrees:
        angle = np.degrees(angle)
    return angle


# Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
def categorise_track_orientations(attr_dat):
    # Angles in radians, [-pi, pi]
    angles = attr_dat.apply(lambda x: get_angle_of_line_between(x.StartEN, x.EndEN), axis=1)
    track_orientation = pd.DataFrame({'Track_Orientation': None}, index=angles.index)

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(angles >= -np.pi * 2 / 3, angles < -np.pi / 3),
        np.logical_and(angles >= np.pi / 3, angles < np.pi * 2 / 3))
    track_orientation[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(angles >= np.pi / 6, angles < np.pi / 3),
        np.logical_and(angles >= -np.pi * 5 / 6, angles < -np.pi * 2 / 3))
    track_orientation[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(angles >= np.pi * 2 / 3, angles < np.pi * 5 / 6),
        np.logical_and(angles >= -np.pi / 3, angles < -np.pi / 6))
    track_orientation[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientation.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(angles >= -np.pi, angles < -np.pi * 5 / 6),
    #     np.logical_and(angles >= -np.pi/6, angles < np.pi/6)),
    #     np.logical_and(angles >= np.pi*5/6, angles < np.pi))
    # track_orientations[e_w] = 'E-W'

    categorical_var = pd.get_dummies(track_orientation, prefix='Track_Orientation')
    track_orientation_data = pd.concat([attr_dat, track_orientation, categorical_var], axis=1)

    return track_orientation_data


# Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30
def categorise_temperatures(attr_dat, column_name='Temperature_max'):

    temp_category = pd.cut(attr_dat[column_name], [-np.inf] + list(np.arange(24, 31)) + [np.inf],
                           right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp_category})

    categorical_var = pd.get_dummies(temperature_category, column_name, prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    temperature_category_data = pd.concat([attr_dat, temperature_category, categorical_var], axis=1)

    return temperature_category_data
