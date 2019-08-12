import os

import pandas as pd
from pyhelpers.dir import cdd


# Change directory to "Weather"
def cdd_weather(*sub_dir):
    """
    :param sub_dir:
    :return:
    """
    path = cdd("Weather")
    for x in sub_dir:
        path = os.path.join(path, x)
    return path


# Find coordinates for each corner of the Weather observation grid
def create_grid(centre_point, side_length=5000, rotation=None):
    """
    :param centre_point: (easting, northing)
    :param side_length: [numeric]
    :param rotation: [numeric; None (default)] e.g. rotation=90
    :return: [tuple]

    Easting and northing coordinates are commonly measured in metres from the axes of some horizontal datum.
    However, other units (e.g. survey feet) are also used.
    """
    assert isinstance(centre_point, (tuple, list)) and len(centre_point) == 2
    x, y = centre_point
    if rotation:
        sin_theta, cos_theta = pd.np.sin(rotation), pd.np.cos(rotation)
        lower_left = (x - 1 / 2 * side_length * sin_theta, y - 1 / 2 * side_length * cos_theta)
        upper_left = (x - 1 / 2 * side_length * cos_theta, y + 1 / 2 * side_length * sin_theta)
        upper_right = (x + 1 / 2 * side_length * sin_theta, y + 1 / 2 * side_length * cos_theta)
        lower_right = (x + 1 / 2 * side_length * cos_theta, y - 1 / 2 * side_length * sin_theta)
    else:
        lower_left = (x - 1 / 2 * side_length, y - 1 / 2 * side_length)
        upper_left = (x - 1 / 2 * side_length, y + 1 / 2 * side_length)
        upper_right = (x + 1 / 2 * side_length, y + 1 / 2 * side_length)
        lower_right = (x + 1 / 2 * side_length, y - 1 / 2 * side_length)
    # corners = shapely.geometry.Polygon([lower_left, upper_left, upper_right, lower_right])
    return lower_left, upper_left, upper_right, lower_right
