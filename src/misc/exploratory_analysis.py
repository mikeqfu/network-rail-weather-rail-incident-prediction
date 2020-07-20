""" Exploratory analysis """

import itertools

import numpy as np
import pandas as pd
from pyhelpers.dir import cdd
from pyhelpers.geom import find_closest_points_between, get_midpoint, wgs84_to_osgb36
from pyhelpers.settings import pd_preferences
from pyhelpers.store import save

from models.prototype.plot_hotspots import get_shp_coordinates
from mssqlserver import metex

pd_preferences()

# Get a collection of coordinates of the railways in GB
railway_coordinates = get_shp_coordinates('Great Britain', osm_layer='railways', osm_feature='rail')


def find_midpoint_of_each_incident_location(incident_data):
    """
    Find the "midpoint" of each incident location.

    :param incident_data: data of incident records, containing information of start/end location coordinates
    :type incident_data: pandas.DataFrame
    :return: midpoints in both (longitude, latitude) and (easting, northing)
    :rtype: pandas.DataFrame
    """

    assert isinstance(incident_data, pd.DataFrame)
    assert all(x in incident_data.columns for x in ['StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude'])

    data = incident_data.copy()

    # Find a pseudo-midpoint location for each incident location
    pseudo_midpoints = get_midpoint(data.StartLongitude.values, data.StartLatitude.values,
                                    data.EndLongitude.values, data.EndLatitude.values,
                                    as_geom=False)

    # Find the "midpoint" of each incident location
    midpoints = find_closest_points_between(pseudo_midpoints, railway_coordinates)

    data[['MidLongitude', 'MidLatitude']] = pd.DataFrame(midpoints)
    data['MidEasting'], data['MidNorthing'] = wgs84_to_osgb36(data.MidLongitude.values, data.MidLatitude.values)

    return data


# == 1st dataset ======================================================================================


def prepare_stats_data(route_name=None, weather_category=None, update=False, verbose=True):
    """
    Prepare data of statistics.

    :param route_name: name of Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather to which an incident is attributed, defaults to ``None``
    :type weather_category: str, None
    :param update: whether to retrieve the source data (in case it has been updated), defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool

    **Example**::

        route_name = None
        weather_category = None
        update = False
        verbose = True

        prepare_stats_data(route_name, weather_category, update, verbose)
    """

    # Get data of Schedule 8 incident locations
    incident_locations = metex.view_schedule8_costs_by_location(route_name, weather_category, update, verbose=verbose)

    # Find the "midpoint" of each incident location
    incident_locations = find_midpoint_of_each_incident_location(incident_locations)

    # Split the data by "region"
    for region in incident_locations.Region.unique():
        region_data = incident_locations[incident_locations.Region == region]
        # Sort data by (frequency of incident occurrences, delay minutes, delay cost)
        sort_by_cols = ['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost']
        region_data.sort_values(sort_by_cols, ascending=False, inplace=True)
        region_data.index = range(len(region_data))
        export_path = cdd("incidents\\exploration\\NC\\01", region.replace(" ", "-").lower() + ".csv")
        save(region_data, export_path, verbose=verbose)

    print("\nCompleted.")


# == 2nd dataset ======================================================================================


def prepare_monthly_stats_data(route_name=None, weather_category=None, update=False, verbose=True):
    """
    Prepare data of monthly statistics.

    :param route_name: name of Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather to which an incident is attributed, defaults to ``None``
    :type weather_category: str, None
    :param update: whether to retrieve the source data (in case it has been updated), defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool

    **Example**::

        route_name = None
        weather_category = None
        update = False
        verbose = True
    """

    # Get data of Schedule 8 incidents by datetime and location
    dat = metex.view_schedule8_costs_by_datetime_location(route_name, weather_category, update, verbose=verbose)

    print("Cleaning data ... ", end="")
    # dat = metex.view_schedule8_data()
    dat.insert(dat.columns.get_loc('EndDateTime') + 1, 'StartYear', dat.StartDateTime.dt.year)
    dat.insert(dat.columns.get_loc('StartYear') + 1, 'StartMonth', dat.StartDateTime.dt.month)

    stats_calc = {'IncidentCount': np.count_nonzero, 'DelayMinutes': np.sum, 'DelayCost': np.sum}
    stats = dat.groupby(list(dat.columns[3:-3])).aggregate(stats_calc)
    stats.reset_index(inplace=True)

    # Find the "midpoint" of each incident location
    data = find_midpoint_of_each_incident_location(stats)
    print("Done.\n")

    sort_by_cols = ['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost']

    print("Processing monthly statistics ... ")
    for m in data.StartMonth.unique():
        m_ = ("0" + str(m)) if m < 10 else m
        print("           \"{}\" ... ".format(m_), end="")
        dat1 = data[data.StartMonth == m]
        if not dat1.empty:
            dat1.sort_values(sort_by_cols, ascending=False, na_position='last', inplace=True)
            dat1.index = range(len(dat1))
            export_path = cdd("incidents\\exploration\\NC\\02\\GB\\Month\\{}.csv".format(m_))
            save(dat1, export_path, sep=',', verbose=False)
        print("Done.")
    print("Completed.\n")

    print("Processing monthly statistics of GB ... ")
    years, months = data.StartYear.unique(), data.StartMonth.unique()
    for y, m in list(itertools.product(years, months)):
        period = ("{}_0{}" if m < 10 else "{}_{}").format(y, m)
        print("           \"{}\" ... ".format(period), end="")
        dat = data[(data.StartYear == y) & (data.StartMonth == m)]
        if not dat.empty:
            dat.sort_values(sort_by_cols, ascending=False, na_position='last', inplace=True)
            dat.index = range(len(dat))
            export_path = cdd("incidents\\exploration\\NC\\02\\GB\\Year_Month\\{}.csv".format(period))
            save(dat, export_path, sep=',', verbose=False)
        print("Done.")
    print("Completed.\n")

    # Split the data by "region"
    print("Processing monthly statistics for each region ... ")
    for region in data.Region.unique():
        region_data = data[data.Region == region]
        years, months = region_data.StartYear.unique(), region_data.StartMonth.unique()
        print("           \"{}\" ... ".format(region), end="")
        for y, m in list(itertools.product(years, months)):
            dat_ = region_data[(region_data.StartYear == y) & (region_data.StartMonth == m)]
            if not dat_.empty:
                dat_.sort_values(sort_by_cols, ascending=False, na_position='last', inplace=True)
                dat_.index = range(len(dat_))
                export_path = cdd("incidents\\exploration\\NC\\02\\Region\\{}".format(
                    region.replace(" ", "-").lower()), ("{}_0{}.csv" if m < 10 else "{}_{}.csv").format(y, m))
                save(dat_, export_path, sep=',', verbose=False)
        print("Done.")
    print("Completed.\n")
