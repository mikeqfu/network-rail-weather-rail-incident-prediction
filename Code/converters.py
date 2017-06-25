"""
Reference:
http://www.hannahfry.co.uk/blog/2012/02/01/converting-british-national-grid-to-latitude-and-longitude-ii
"""

from math import sqrt, pi, sin, cos, tan, atan2, floor

import measurement.measures
import pandas as pd


# Convert british national grid (OSBG36) to latitude and longitude (WGS84) ==========================================
def osgb36_to_wgs84(easting, northing):
    """
    :param easting: X
    :param northing: Y
    :return:

    'easting' and 'northing' are the British national grid coordinates

    """
    # The Airy 180 semi-major and semi-minor axes used for OSGB36 (m)
    a, b = 6377563.396, 6356256.909
    # Scale factor on the central meridian
    f0 = 0.9996012717
    # Latitude of true origin (radians)
    lat0 = 49 * pi / 180
    # Longitude of true origin and central meridian (radians):
    lon0 = -2 * pi / 180
    # Northing and Easting of true origin (m):
    n0, e0 = -100000, 400000
    e2 = 1 - (b * b) / (a * a)  # eccentricity squared
    n = (a - b) / (a + b)

    # Initialise the iterative variables
    lat, m = lat0, 0

    while northing - n0 - m >= 0.00001:  # Accurate to 0.01mm
        lat += (northing - n0 - m) / (a * f0)
        m1 = (1 + n + (5. / 4) * n ** 2 + (5. / 4) * n ** 3) * (lat - lat0)
        m2 = (3 * n + 3 * n ** 2 + (21. / 8) * n ** 3) * sin(lat - lat0) * cos(lat + lat0)
        m3 = ((15. / 8) * n ** 2 + (15. / 8) * n ** 3) * sin(2 * (lat - lat0)) * cos(2 * (lat + lat0))
        m4 = (35. / 24) * n ** 3 * sin(3 * (lat - lat0)) * cos(3 * (lat + lat0))
        # meridional arc
        m = b * f0 * (m1 - m2 + m3 - m4)

    # transverse radius of curvature
    nu = a * f0 / sqrt(1 - e2 * sin(lat) ** 2)

    # meridional radius of curvature
    rho = a * f0 * (1 - e2) * (1 - e2 * sin(lat) ** 2) ** (-1.5)
    eta2 = nu / rho - 1

    sec_lat = 1. / cos(lat)
    vii = tan(lat) / (2 * rho * nu)
    viii = tan(lat) / (24 * rho * nu ** 3) * (5 + 3 * tan(lat) ** 2 + eta2 - 9 * tan(lat) ** 2 * eta2)
    ix = tan(lat) / (720 * rho * nu ** 5) * (61 + 90 * tan(lat) ** 2 + 45 * tan(lat) ** 4)
    x = sec_lat / nu
    xi = sec_lat / (6 * nu ** 3) * (nu / rho + 2 * tan(lat) ** 2)
    xii = sec_lat / (120 * nu ** 5) * (5 + 28 * tan(lat) ** 2 + 24 * tan(lat) ** 4)
    xiia = sec_lat / (5040 * nu ** 7) * (61 + 662 * tan(lat) ** 2 + 1320 * tan(lat) ** 4 + 720 * tan(lat) ** 6)
    de = easting - e0

    # These are on the wrong ellipsoid currently: Airy1830. (Denoted by _1)
    lat_1 = lat - vii * de ** 2 + viii * de ** 4 - ix * de ** 6
    lon_1 = lon0 + x * de - xi * de ** 3 + xii * de ** 5 - xiia * de ** 7

    """ Want to convert to the GRS80 ellipsoid. """
    # First convert to cartesian from spherical polar coordinates
    h = 0  # Third spherical coord.
    x_1 = (nu / f0 + h) * cos(lat_1) * cos(lon_1)
    y_1 = (nu / f0 + h) * cos(lat_1) * sin(lon_1)
    z_1 = ((1 - e2) * nu / f0 + h) * sin(lat_1)

    # Perform Helmut transform (to go between Airy 1830 (_1) and GRS80 (_2))
    s = -20.4894 * 10 ** -6  # The scale factor -1
    # The translations along x,y,z axes respectively
    tx, ty, tz = 446.448, -125.157, + 542.060
    # The rotations along x,y,z respectively, in seconds
    rxs, rys, rzs = 0.1502, 0.2470, 0.8421
    rx, ry, rz = rxs * pi / (180 * 3600.), rys * pi / (180 * 3600.), rzs * pi / (180 * 3600.)  # In radians
    x_2 = tx + (1 + s) * x_1 + (-rz) * y_1 + ry * z_1
    y_2 = ty + rz * x_1 + (1 + s) * y_1 + (-rx) * z_1
    z_2 = tz + (-ry) * x_1 + rx * y_1 + (1 + s) * z_1

    # Back to spherical polar coordinates from cartesian
    # Need some of the characteristics of the new ellipsoid
    # The GSR80 semi-major and semi-minor axes used for WGS84(m)
    a_2, b_2 = 6378137.000, 6356752.3141
    e2_2 = 1 - (b_2 * b_2) / (a_2 * a_2)  # The eccentricity of the GRS80 ellipsoid
    p = sqrt(x_2 ** 2 + y_2 ** 2)

    # Lat is obtained by an iterative procedure:
    lat = atan2(z_2, (p * (1 - e2_2)))  # Initial value
    lat_old = 2 * pi
    while abs(lat - lat_old) > 10 ** -16:
        lat, lat_old = lat_old, lat
        nu_2 = a_2 / sqrt(1 - e2_2 * sin(lat_old) ** 2)
        lat = atan2(z_2 + e2_2 * nu_2 * sin(lat_old), p)

    # Lon and height are then pretty easy
    lon = atan2(y_2, x_2)
    # h = p / cos(lat) - nu_2

    # Uncomment this line if you want to print the results
    # print([(lat - lat_1) * 180 / pi, (lon - lon_1) * 180 / pi])

    # Convert to degrees
    lat = lat * 180 / pi
    lon = lon * 180 / pi

    # Job's a good'n.
    return lat, lon


# Convert latitude and longitude (WGS84) to british national grid (OSBG36) ===========================================
def wgs84_to_osgb36(latitude, longitude):
    """
    :param latitude:
    :param longitude:
    :return:

    This function converts lat lon (WGS84) to british national grid (OSBG36)

    """
    # First convert to radians. These are on the wrong ellipsoid currently: GRS80. (Denoted by _1)
    long_1, lat_1 = longitude * pi / 180, latitude * pi / 180

    # Want to convert to the Airy 1830 ellipsoid, which has the following:
    # The GSR80 semi-major and semi-minor axes used for WGS84(m)
    a_1, b_1 = 6378137.000, 6356752.3141
    e2_1 = 1 - (b_1 * b_1) / (a_1 * a_1)  # The eccentricity of the GRS80 ellipsoid
    nu_1 = a_1 / sqrt(1 - e2_1 * sin(lat_1) ** 2)

    # First convert to cartesian from spherical polar coordinates
    h = 0  # Third spherical coord.
    x_1 = (nu_1 + h) * cos(lat_1) * cos(long_1)
    y_1 = (nu_1 + h) * cos(lat_1) * sin(long_1)
    z_1 = ((1 - e2_1) * nu_1 + h) * sin(lat_1)

    # Perform Helmut transform (to go between GRS80 (_1) and Airy 1830 (_2))
    s = 20.4894 * 10 ** -6  # The scale factor -1
    # The translations along x,y,z axes respectively:
    tx, ty, tz = -446.448, 125.157, -542.060
    # The rotations along x,y,z respectively, in seconds:
    rxs, rys, rzs = -0.1502, -0.2470, -0.8421
    rx, ry, rz = rxs * pi / (180 * 3600.), rys * pi / (180 * 3600.), rzs * pi / (180 * 3600.)  # In radians
    x_2 = tx + (1 + s) * x_1 + (-rz) * y_1 + ry * z_1
    y_2 = ty + rz * x_1 + (1 + s) * y_1 + (-rx) * z_1
    z_2 = tz + (-ry) * x_1 + rx * y_1 + (1 + s) * z_1

    # Back to spherical polar coordinates from cartesian
    # Need some of the characteristics of the new ellipsoid
    # The GSR80 semi-major and semi-minor axes used for WGS84(m)
    a, b = 6377563.396, 6356256.909
    e2 = 1 - (b * b) / (a * a)  # The eccentricity of the Airy 1830 ellipsoid
    p = sqrt(x_2 ** 2 + y_2 ** 2)

    # Lat is obtained by an iterative procedure:
    latitude = atan2(z_2, (p * (1 - e2)))  # Initial value
    lat_old = 2 * pi
    nu = 0
    while abs(latitude - lat_old) > 10 ** -16:
        latitude, lat_old = lat_old, latitude
        nu = a / sqrt(1 - e2 * sin(lat_old) ** 2)
        latitude = atan2(z_2 + e2 * nu * sin(lat_old), p)

    # Lon and height are then pretty easy
    longitude = atan2(y_2, x_2)
    # h = p / cos(lat) - nu

    # e, n are the British national grid coordinates - easting and northing
    # scale factor on the central meridian
    f0 = 0.9996012717
    # Latitude of true origin (radians)
    lat0 = 49 * pi / 180
    # Longitude of true origin and central meridian (radians)
    lon0 = -2 * pi / 180
    # Northing & easting of true origin (m)
    n0, e0 = -100000, 400000
    n = (a - b) / (a + b)

    # meridional radius of curvature
    rho = a * f0 * (1 - e2) * (1 - e2 * sin(latitude) ** 2) ** (-1.5)
    eta2 = nu * f0 / rho - 1

    m1 = (1 + n + (5 / 4) * n ** 2 + (5 / 4) * n ** 3) * (latitude - lat0)
    m2 = (3 * n + 3 * n ** 2 + (21 / 8) * n ** 3) * sin(latitude - lat0) * cos(latitude + lat0)
    m3 = ((15 / 8) * n ** 2 + (15 / 8) * n ** 3) * sin(2 * (latitude - lat0)) * cos(2 * (latitude + lat0))
    m4 = (35 / 24) * n ** 3 * sin(3 * (latitude - lat0)) * cos(3 * (latitude + lat0))

    # meridional arc
    m = b * f0 * (m1 - m2 + m3 - m4)

    i = m + n0
    ii = nu * f0 * sin(latitude) * cos(latitude) / 2
    iii = nu * f0 * sin(latitude) * cos(latitude) ** 3 * (5 - tan(latitude) ** 2 + 9 * eta2) / 24
    iii_a = nu * f0 * sin(latitude) * cos(latitude) ** 5 * (61 - 58 * tan(latitude) ** 2 + tan(latitude) ** 4) / 720
    iv = nu * f0 * cos(latitude)
    v = nu * f0 * cos(latitude) ** 3 * (nu / rho - tan(latitude) ** 2) / 6
    vi = nu * f0 * cos(latitude) ** 5 * (
        5 - 18 * tan(latitude) ** 2 + tan(latitude) ** 4 + 14 * eta2 - 58 * eta2 * tan(latitude) ** 2) / 120

    n = i + ii * (longitude - lon0) ** 2 + iii * (longitude - lon0) ** 4 + iii_a * (longitude - lon0) ** 6
    e = e0 + iv * (longitude - lon0) + v * (longitude - lon0) ** 3 + vi * (longitude - lon0) ** 5

    # Job's a good'n.
    return e, n


# Convert columns ====================================================================================================
def ne_to_latlon(easting_col, northing_col):
    latitudes = []
    longitudes = []
    n, e = northing_col, easting_col
    for i in range(len(n)):
        lat, lon = osgb36_to_wgs84(e.iloc[i], n.iloc[i])
        latitudes.append(lat)
        longitudes.append(lon)
    return pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})


# ====================================================================================================================
def str_to_num_mileage(x):
    return '' if x == '' else round(float(x), 4)


# ====================================================================================================================
def mileage_to_str(x):
    return '%.4f' % round(float(x), 4)


# Convert yards to Network Rail mileages =============================================================================
def yards_to_mileage(yards):
    yd = measurement.measures.Distance(yd=yards)
    mileage_mi = floor(yd.mi)
    mileage_yd = int(yards - measurement.measures.Distance(mi=mileage_mi).yd)
    # Example: "%.2f" % round(2606.89579999999, 2)
    mileage = str('%.4f' % round((mileage_mi + mileage_yd / (10 ** 4)), 4))
    return mileage


# Convert Network Rail mileages to yards =============================================================================
def mileage_to_yards(mileage):
    if isinstance(mileage, float):
        mileage = str('%.4f' % mileage)
    elif isinstance(mileage, str):
        pass
    miles = int(mileage.split('.')[0])
    yards = int(mileage.split('.')[1])
    yards += measurement.measures.Distance(mi=miles).yd
    return yards


# ====================================================================================================================
def num_mileage_shifting(mileage, shift_yards):
    yards = mileage_to_yards(mileage) + shift_yards
    str_mileage = yards_to_mileage(yards)
    return str_to_num_mileage(str_mileage)


# Convert miles to Network Rail mileages =============================================================================
def miles_chains_to_mileage(miles_chains):
    if miles_chains is '':
        return miles_chains
    else:
        miles_chains = str(miles_chains)
        miles = int(miles_chains.split('.')[0])
        chains = float(miles_chains.split('.')[1])
        yards = measurement.measures.Distance(chain=chains).yd
        return '%.4f' % (miles + round(yards / (10 ** 4), 4))


# Convert calendar year to Network Rail financial year ===============================================================
def get_financial_year(date):
    financial_date = date + pd.DateOffset(months=-3)
    return financial_date.year
