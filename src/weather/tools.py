"""
Tools for manipulating weather data.
"""

import pandas as pd


def create_grid(centre_point, side_length=5000, rotation=None):
    """
    Find coordinates for each corner of the Weather observation grid.

    :param centre_point: (easting, northing)
    :type centre_point: tuple
    :param side_length: side length
    :type side_length: int, float
    :param rotation: e.g. rotation=90; defaults to ``None``
    :type rotation; int, float, None
    :return: coordinates of four corners of the created grid
    :rtype: tuple

    .. note::

        Easting and northing coordinates are commonly measured in metres from the axes of some
        horizontal datum. However, other units (e.g. survey feet) are also used.
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
