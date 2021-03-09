"""
Exploratory analysis.
"""

import itertools

import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from pyhelpers.geom import find_closest_points, get_midpoint, wgs84_to_osgb36
from pyhelpers.store import save

from coordinator.geometry import get_shp_coordinates
from preprocessor import METExLite
from utils import cdd_exploration


def calc_stats(s8weather_incidents):
    """
    Calculate statistics of different categories of weather-related incidents.

    :param s8weather_incidents: data of weather-related incidents
    :type s8weather_incidents: pandas.DataFrame
    :return: statistics about frequencies of different categories of
        weather-related incidents and their total costs
    :rtype: pandas.DataFrame
    """
    s8weather_incidents.rename(columns={'Minutes': 'DelayMinutes', 'Cost': 'DelayCost'},
                               inplace=True)
    stats = s8weather_incidents.groupby('WeatherCategory').aggregate(
        {'WeatherCategory': 'count', 'DelayMinutes': np.sum, 'DelayCost': np.sum})
    stats.rename(columns={'WeatherCategory': 'Count'}, inplace=True)
    stats['percentage'] = stats.Count / len(s8weather_incidents) * 100
    # Sort stats in the ascending order of 'percentage'
    stats.sort_values('percentage', ascending=False, inplace=True)
    return stats


def create_pie_plot_for_incident_proportions(s8weather_incidents, save_as=".png"):
    """
    Create a pie chart illustrating the proportions of different categories of
    weather-related incidents.

    :param s8weather_incidents: data of Schedule 8 incidents
    :type s8weather_incidents: pandas.DataFrame
    :param save_as: whether to save the pie chart / what format the pie chart is saved as,
        defaults to ``".png"``
    :type save_as: str, None

    **Example**::

        >>> from preprocessor import Schedule8IncidentReports
        >>> from preprocessor.explorer import create_pie_plot_for_incident_proportions

        >>> reports = Schedule8IncidentReports()

        >>> s8_weather_incidents = reports.get_schedule8_weather_incidents_02062006_31032014()
        >>> dat = s8_weather_incidents['Data']

        >>> create_pie_plot_for_incident_proportions(dat, save_as=".png")
    """

    stats = calc_stats(s8weather_incidents).reset_index()
    # Set colour array
    colours = matplotlib.cm.get_cmap('Set3')(np.flip(np.linspace(0.0, 1.0, 9), 0))
    # Specify labels
    percentages = ['%1.1f%%' % round(x, 1) for x in stats['percentage']]
    labels = stats.WeatherCategory + ': '
    # labels = [a + b for a, b in zip(labels, total_costs_in_million)]
    labels = [a + b for a, b in zip(labels, percentages)]
    wind_label = ['', 'Most delays\n& Highest costs', '', '', '', '', '', '', '']
    # Specify which part is exploded
    explode_list = np.zeros(len(stats))
    # explode_pos = stats.sort_values(
    #     by=['PfPIMinutes', 'percentage'], ascending=False).index[0]
    explode_list[1] = 0.2

    # Create a figure
    plt.figure(figsize=(8, 6))
    ax = plt.subplot2grid((1, 1), (0, 0), aspect='equal')
    # ax.set_rasterization_zorder(1)
    pie_collections = ax.pie(stats.percentage, labels=wind_label, startangle=70,
                             colors=colours, explode=explode_list, labeldistance=0.7)

    # Note that 'pie_collections' includes: patches, texts, autotexts
    patches, texts = pie_collections
    texts[1].set_fontsize(12)
    texts[1].set_fontstyle('italic')
    # texts[1].set_fontweight('bold')
    legend = ax.legend(pie_collections[0], labels, loc='best', fontsize=14, frameon=True,
                       shadow=True, fancybox=True, title='Weather category')
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_facecolor('white')

    # ax.set_title('Reasons for Weather-related Incidents\n', fontsize=14, weight='bold')

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=0.95)

    plt.show()

    if save_as:
        plt.savefig(cdd_exploration("proportions" + save_as), dpi=600)


def create_bar_plot_for_delay_cost(s8weather_incidents, save_as=".png"):
    """
    Plot total monetary cost incurred by weather-related incidents.

    :param s8weather_incidents: data of Schedule 8 incidents
    :type s8weather_incidents: pandas.DataFrame
    :param save_as: whether to save the pie chart / what format the pie chart is saved as,
        defaults to ``".png"``
    :type save_as: str, None

    **Example**::

        from spreadsheet.incidents import get_schedule8_weather_incidents_02062006_31032014
        from misc.explor import create_bar_plot_for_delay_cost

        save_as = ".png"

        s8weather_incidents = get_schedule8_weather_incidents_02062006_31032014()['Data']

        create_bar_plot_for_delay_cost(s8weather_incidents, save_as)
    """

    stats = calc_stats(s8weather_incidents).reset_index()
    stats.sort_values(['DelayMinutes', 'DelayCost', 'Count'], inplace=True)
    colour_array = np.sort(np.flip(np.linspace(0.0, 1.0, 9), 0)[stats.index])
    stats.index = range(len(stats))
    plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    # formatter_minutes = FuncFormatter(lambda m, position: format(int(m), ','))
    colours = matplotlib.cm.get_cmap('Set3')(colour_array)
    ax1.barh(stats.index, stats.DelayMinutes, align='center', color=colours)
    plt.yticks(stats.index, stats.WeatherCategory, fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Minutes', fontsize=13, fontweight='bold')
    # plt.ylabel('Weather category', fontsize=12)
    plt.title('Delay', fontsize=15, fontweight='bold')
    # ax1.set_axis_bgcolor('#808080')

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    # plt.barh(range(0, len(stats)), stats['PfPICosts'], align='center', color=colours1)
    plt.barh(stats.index, stats.DelayCost,
             align='center', color=colours, alpha=1.0, hatch='/')
    plt.yticks(stats.index, [''] * len(stats))
    plt.xticks(fontsize=12)
    # Format labels
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda c, position: '%1.1f' % (c * 1e-7)))
    plt.xlabel('£ millions', fontsize=13, fontweight='bold')
    plt.title('Cost', fontsize=15, fontweight='bold')
    # ax2.set_axis_bgcolor('#dddddd')

    # plt.subplots_adjust(left=0.16, bottom=0.10, right=0.96, top=0.92, wspace=0.16)
    plt.tight_layout()

    if save_as:
        plt.savefig(cdd_exploration("delays-and-cost" + save_as), dpi=600)


# == For the paper to Nature Communications ============================================

def find_midpoint_of_each_incident_location(incident_data):
    """
    Find the "midpoint" of each incident location.

    :param incident_data: data of incident records, containing information of
        start/end location coordinates
    :type incident_data: pandas.DataFrame
    :return: midpoints in both (longitude, latitude) and (easting, northing)
    :rtype: pandas.DataFrame
    """

    assert isinstance(incident_data, pd.DataFrame)
    assert all(x in incident_data.columns for x in [
        'StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude'])

    data = incident_data.copy()

    # Find a pseudo-midpoint location for each incident location
    pseudo_midpoints = get_midpoint(data.StartLongitude.values, data.StartLatitude.values,
                                    data.EndLongitude.values, data.EndLatitude.values,
                                    as_geom=False)

    # Find the "midpoint" of each incident location
    railway_coordinates = get_shp_coordinates(osm_subregion='Great Britain', osm_layer='railways',
                                              osm_feature='rail')

    midpoints = find_closest_points(pseudo_midpoints, railway_coordinates)

    data[['MidLongitude', 'MidLatitude']] = pd.DataFrame(midpoints)
    data['MidEasting'], data['MidNorthing'] = \
        wgs84_to_osgb36(data.MidLongitude.values, data.MidLatitude.values)

    return data


# == 1st dataset =======================================================================

def prepare_stats_data(route_name=None, weather_category=None, update=False,
                       verbose=True):
    """
    Prepare data of statistics.

    :param route_name: name of Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather to which an incident is attributed,
        defaults to ``None``
    :type weather_category: str, None
    :param update: whether to retrieve the source data (in case it has been updated),
        defaults to ``False``
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

    metex = METExLite()

    # Get data of Schedule 8 incident locations
    incident_locations = metex.view_schedule8_costs_by_location(
        route_name=route_name, weather_category=weather_category, update=update, verbose=verbose)

    # Find the "midpoint" of each incident location
    incident_locations = find_midpoint_of_each_incident_location(incident_locations)

    # Split the data by "region"
    for region in incident_locations.Region.unique():
        region_data = incident_locations[incident_locations.Region == region]
        # Sort data by (frequency of incident occurrences, delay minutes, delay cost)
        sort_by_cols = ['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost']
        region_data.sort_values(sort_by_cols, ascending=False, inplace=True)
        region_data.index = range(len(region_data))
        export_path = cdd_exploration("NC", "01", region.replace(" ", "-").lower() + ".csv")
        save(region_data, export_path, verbose=verbose)

    print("\nCompleted.")


# == 2nd dataset =======================================================================

def prepare_monthly_stats_data(route_name=None, weather_category=None, update=False, verbose=True):
    """
    Prepare data of monthly statistics.

    :param route_name: name of Route, defaults to ``None``
    :type route_name: str, None
    :param weather_category: weather to which an incident is attributed,
        defaults to ``None``
    :type weather_category: str, None
    :param update: whether to retrieve the source data (in case it has been updated),
        defaults to ``False``
    :type update: bool
    :param verbose: defaults to ``False``
    :type verbose: bool

    **Example**::

        route_name = None
        weather_category = None
        update = False
        verbose = True
    """

    metex = METExLite()

    # Get data of Schedule 8 incidents by datetime and location
    dat = metex.view_schedule8_costs_by_datetime_location(
        route_name=route_name, weather_category=weather_category, update=update, verbose=verbose)

    print("Cleaning data ... ", end="")
    # dat = metex.view_schedule8_data()
    dat.insert(
        dat.columns.get_loc('EndDateTime') + 1, 'StartYear', dat.StartDateTime.dt.year)
    dat.insert(
        dat.columns.get_loc('StartYear') + 1, 'StartMonth', dat.StartDateTime.dt.month)

    stats_calc = {'IncidentCount': np.count_nonzero,
                  'DelayMinutes': np.sum,
                  'DelayCost': np.sum}
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
            dat1.sort_values(
                sort_by_cols, ascending=False, na_position='last', inplace=True)
            dat1.index = range(len(dat1))
            export_path = cdd_exploration("NC\\02\\GB\\Month", "{}.csv".format(m_))
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
            dat.sort_values(
                sort_by_cols, ascending=False, na_position='last', inplace=True)
            dat.index = range(len(dat))
            export_path = cdd_exploration("NC\\02\\GB\\Year_Month", "{}.csv".format(period))
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
            dat_ = region_data[
                (region_data.StartYear == y) & (region_data.StartMonth == m)]
            if not dat_.empty:
                dat_.sort_values(
                    sort_by_cols, ascending=False, na_position='last', inplace=True)
                dat_.index = range(len(dat_))
                export_path = cdd_exploration("NC", "02", "Region", "{}".format(
                    region.replace(" ", "-").lower()),
                    ("{}_0{}.csv" if m < 10 else "{}_{}.csv").format(y, m))
                save(dat_, export_path, sep=',', verbose=False)
        print("Done.")
    print("Completed.\n")
