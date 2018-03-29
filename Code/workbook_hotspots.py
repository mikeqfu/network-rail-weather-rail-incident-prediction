""" Top10Hotspots.xlsx """

import os

import matplotlib.cm
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import numpy as np
import pandas as pd

import database_met as dbm
from utils import cdd, load_pickle, reset_double_indexes, save


# Change directory to "Schedule 8 incidents"
def cdd_schedule8(*directories):
    path = cdd("METEX\\Schedule 8 incidents")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


#
def get_top10hotspots_details(route=None, weather=None, update=False):
    """
    :param route:
    :param weather:
    :param update:
    :return:

    These are for 2006/07 – 2013/14, so don’t include the latest 2014/2015 data

    """
    # filename = dbm.make_filename("Top10Hotspots_Details", route, weather)
    filename = "Top10Hotspots_Details.pickle"
    path_to_file = cdd_schedule8("Top 10 hotspot", filename)
    if os.path.isfile(path_to_file) and not update:
        details_sheet = load_pickle(path_to_file)
    else:
        try:
            # Load excel file 'Top10Hotspots.xlsx'
            workbook = pd.ExcelFile(cdd_schedule8("Top 10 hotspot", "Top10Hotspots.xlsx"))
            # Read the 'Details' worksheet as a data frame
            details_sheet = workbook.parse(sheetname='Details')
            details_sheet.rename(columns={'section': 'StanoxSection',
                                          'cost': 'DelayCost', 'delay': 'DelayMinutes',
                                          'reasonDescription': 'IncidentReasonDescription'},
                                 inplace=True)
            save(details_sheet, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            details_sheet = None
    details_data = dbm.subset(details_sheet, route, weather)
    return details_data


#
def get_top10hotspots_details_by_location(route=None, weather=None, update=False):
    # filename = dbm.make_filename("Top10Hotspots_Details_by_section", route, weather)
    filename = "Top10Hotspots_Details_by_section.pickle"
    path_to_file = cdd_schedule8("Top 10 hotspot", filename)
    if os.path.isfile(path_to_file) and not update:
        dat = load_pickle(path_to_file)
    else:
        try:
            top10hotspots_details = get_top10hotspots_details(route, weather, update)
            dat0 = top10hotspots_details. \
                groupby(['Route', 'StanoxSection', 'WeatherCategory',
                         'StartLatitude', 'StartLongitude', 'EndLatitude', 'EndLongitude']). \
                aggregate({'DelayMinutes': [np.sum, np.count_nonzero, np.mean],
                           'DelayCost': [np.sum, np.count_nonzero, np.mean]})
            dat = reset_double_indexes(dat0)
            dat.drop('DelayCost_count_nonzero', axis=1, inplace=True)
            dat.rename(columns={'DelayMinutes_sum': 'TotalDelays',
                                'DelayMinutes_count_nonzero': 'IncidentCount',
                                'DelayMinutes_mean': 'AverageDelay',
                                'DelayCost_sum': 'TotalCost',
                                'DelayCost_mean': 'AverageCost'},
                       inplace=True)
            dat.sort_values(['TotalCost', 'IncidentCount', 'TotalDelays'], ascending=False, inplace=True)
            dat.index = range(len(dat))
            save(dat, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to '{}'.".format(os.path.splitext(filename)[0], e))
            dat = None
    data = dbm.subset(dat, route, weather)
    return data


#
def plot_top10_hotspots(route='Anglia', weather='Wind', update=False, show_title=False, save_as=".png"):
    data = get_top10hotspots_details_by_location(route, weather, update)

    plt.figure(figsize=(8, 8))
    m = mpl_toolkits.basemap.Basemap(llcrnrlat=51.23622, llcrnrlon=-0.570409,
                                     urcrnrlat=53.062591, urcrnrlon=1.915975,
                                     projection='mill', resolution='i')
    m.drawcoastlines()
    m.drawmapboundary(color='white', fill_color='white')
    m.drawrivers(color='#69cbf5', linewidth=1.5)
    m.fillcontinents(color='#dcdcdc')

    start_lat = data.StartLatitude[0:10]
    start_long = data.StartLongitude[0:10]
    end_lat = data.EndLatitude[0:10]
    end_long = data.EndLongitude[0:10]
    # Note the order of lat. and long. when plotting the point!
    # colours = cm.get_cmap('Set3')(np.arange(10) / 10)

    # Annotations
    sections = data.StanoxSection

    for i in data.index:
        xs, ys = [], []
        x_start_pt, y_start_pt = m(start_long[i], start_lat[i])
        m.plot(x_start_pt, y_start_pt, 'r^', markersize=9)
        x_end_pt, y_end_pt = m(end_long[i], end_lat[i])
        m.plot(x_end_pt, y_end_pt, 'ro')
        xs.append(x_start_pt)
        ys.append(y_start_pt)
        xs.append(x_end_pt)
        ys.append(y_end_pt)
        m.plot(xs, ys, label=data.StanoxSection[i], color='r', linewidth=2)
        # Add annotation
        x = x_start_pt + (x_end_pt - x_start_pt) / 2
        y = y_start_pt + (y_end_pt - y_start_pt) / 2
        plt.annotate(sections[i], xy=(x + 7000, y - 1000), fontsize=12, color='#00004d')

    if show_title:
        plt.suptitle('Top 10 hotspots for %s-related delays/cost on %s Route\n' % (weather.lower(), route),
                     fontsize=16, weight='bold')
        plt.subplots_adjust(left=0.02, bottom=0.00, right=0.98, top=0.95)
    else:
        plt.tight_layout()

    if save_as:
        filename = dbm.make_filename("Top10Hotspots", route, weather)
        plt.savefig(cdd_schedule8("Exploratory analysis", filename.replace(".pickle", save_as)), facecolor='white',
                    dpi=600)


#
def top10_pie(data, add_title=False):
    # Create labels for pie chart
    incidents_no = ['No. of incidents: %s' % str(int(i)) for i in data.IncidentCount]
    # costs = ['Cost: £%s' % format(int(i), ',') % i for i in data['Cost']]
    minutes = ['Delay (minutes): %s' % format(int(i), ',') % i for i in data.TotalDelays]

    labels = data.StanoxSection + '\n' + incidents_no + '\n' + minutes
    data.index = range(len(data))
    exp_pos1 = data.sort_values(by='TotalDelays', ascending=False).index[0]
    exp_pos2 = data.sort_values(by='IncidentCount', ascending=False).index[0]
    explode_list = np.zeros(10)
    explode_list[[exp_pos1, exp_pos2]] = [0.2, 0.2]
    colours = matplotlib.cm.get_cmap('Set3')(np.arange(10) / 10)

    plt.figure(figsize=[10, 6])
    ax = plt.subplot2grid((1, 1), (0, 0), aspect='equal')
    pie_collections = ax.pie(data.TotalCost, labels=labels, shadow=True,
                             startangle=45, colors=colours,
                             autopct='%1.1f%%', explode=explode_list)
    # pie_collections includes: patches, texts, autotexts
    for pie_text in pie_collections[1]:
        pie_text.set_fontsize(13)
        # pie_text.set_fontweight('normal')

    if add_title:
        ax.set_title('Top 10 hotspots for wind-related delays on Anglia Route\n', fontsize=16, weight='bold')

    plt.subplots_adjust(left=0.25, bottom=0.05, right=0.78)
