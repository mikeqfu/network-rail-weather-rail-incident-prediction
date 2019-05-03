import os

import fuzzywuzzy.process
import numpy as np
import pandas as pd

from utils import cdd

# ====================================================================================================================
""" Change directories """


# Change directory to "modelling\\Prototype-Heat\\..." and sub-directories
def cdd_mod_heat_inter(*directories):
    path = cdd("modelling", "Intermediate-Heat", "dat")
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "modelling\\Prototype-Heat\\Trial_" and sub-directories
def cdd_mod_heat_inter_trial(trial_id=0, *directories):
    path = cdd("modelling", "Intermediate-Heat", "Trial_{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Utilities """


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
    angles = attr_dat.apply(lambda x: get_angle_of_line_between(x.StartNE, x.EndNE), axis=1)
    track_orientations = pd.DataFrame({'Track_Orientation': None}, index=angles.index)

    # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
    n_s = np.logical_or(
        np.logical_and(angles >= -np.pi * 2 / 3, angles < -np.pi / 3),
        np.logical_and(angles >= np.pi / 3, angles < np.pi * 2 / 3))
    track_orientations[n_s] = 'N_S'

    # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
    ne_sw = np.logical_or(
        np.logical_and(angles >= np.pi / 6, angles < np.pi / 3),
        np.logical_and(angles >= -np.pi * 5 / 6, angles < -np.pi * 2 / 3))
    track_orientations[ne_sw] = 'NE_SW'

    # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
    nw_se = np.logical_or(
        np.logical_and(angles >= np.pi * 2 / 3, angles < np.pi * 5 / 6),
        np.logical_and(angles >= -np.pi / 3, angles < -np.pi / 6))
    track_orientations[nw_se] = 'NW_SE'

    # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
    track_orientations.fillna('E_W', inplace=True)
    # e_w = np.logical_or(np.logical_or(
    #     np.logical_and(angles >= -np.pi, angles < -np.pi * 5 / 6),
    #     np.logical_and(angles >= -np.pi/6, angles < np.pi/6)),
    #     np.logical_and(angles >= np.pi*5/6, angles < np.pi))
    # track_orientations[e_w] = 'E-W'

    categorical_var = pd.get_dummies(track_orientations, prefix='Track_Orientation')

    return track_orientations.join(categorical_var)


# Categorise temperature: <24, 24, 25, 26, 27, 28, 29, >=30
def categorise_temperatures(attr_dat):
    temp = pd.cut(attr_dat, [-np.inf] + list(np.arange(24, 31)) + [np.inf], right=False, include_lowest=False)
    temperature_category = pd.DataFrame({'Temperature_Category': temp})

    categorical_var = pd.get_dummies(temperature_category, 'Maximum_Temperature_max', prefix_sep=' ')
    categorical_var.columns = [c + '°C' for c in categorical_var.columns]

    return temperature_category.join(categorical_var)


# Get data for specified season(s)
def get_data_by_season(m_data, season='summer'):
    """
    :param m_data:
    :param season: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :return:
    """
    if season is None:
        return m_data
    else:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4
        for y in pd.unique(m_data.Year):
            data = m_data[m_data.Year == y]
            # Get data for spring -----------------------------------
            spring_start1 = pd.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = pd.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring = ((data.StartDateTime >= spring_start1) & (data.StartDateTime < spring_end1)) | \
                     ((data.StartDateTime >= spring_start2) & (data.StartDateTime < spring_end2))
            spring_data = pd.concat([spring_data, data.loc[spring]])
            # Get data for summer ----------------------------------
            summer_start = pd.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (data.StartDateTime >= summer_start) & (data.StartDateTime < summer_end)
            summer_data = pd.concat([summer_data, data.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = pd.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (data.StartDateTime >= autumn_start) & (data.StartDateTime < autumn_end)
            autumn_data = pd.concat([autumn_data, data.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = pd.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (data.StartDateTime >= winter_start) & (data.StartDateTime < winter_end)
            winter_data = pd.concat([winter_data, data.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [fuzzywuzzy.process.extractOne(s, seasons)[0] for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data
