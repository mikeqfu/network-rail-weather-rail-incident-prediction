""" Incidents data available in spreadsheets """

import datetime
import glob
import os

import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from pyhelpers.dir import cdd
from pyhelpers.store import load_pickle, save, save_fig
from pyrcs.line_data import LineData

import mssql_metex as dbm
import mssql_vegetation as dbv


# Change directory to "Incidents"
def cdd_incidents(*directories):
    path = cdd("Incidents")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Incidents\\Spreadsheets"
def cdd_spreadsheet(*directories):
    path = cdd_incidents("Spreadsheets")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Schedule8WeatherCostReport.xlsx """


#
def get_schedule8_weather_cost_report(route=None, weather=None, update=False):
    """
    Summary report for Schedule 8 Weather costs covering the UK.
    """
    pickle_filename = "Schedule8WeatherCostReport.pickle"
    path_to_pickle = cdd_spreadsheet(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        all_data = load_pickle(path_to_pickle)
    else:
        try:
            spreadsheet_filename = "Schedule8WeatherCostReport.xlsx"
            sheet_name = 'AllData'
            workbook = pd.ExcelFile(cdd_incidents("Spreadsheets", spreadsheet_filename))
            # 'AllData'
            all_data = workbook.parse(sheet_name=sheet_name)
            all_data.columns = [c.replace(' ', '') for c in all_data.columns]
            all_data.rename(columns={'Weather': 'WeatherCategoryCode',
                                     'Cost': 'DelayCost',
                                     'WeatherDescription': 'WeatherCategory'}, inplace=True)
            all_data.sort_values(by=['Year', 'Route', 'WeatherCategory'], inplace=True)
            all_data = all_data.round({'DelayCost': 2, 'DelayMinutes': 4})

            workbook.close()

            save(all_data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))
            all_data = pd.DataFrame()

    s8_cost_report_all_data = dbm.subset(all_data, route, weather)
    s8_cost_report_all_data.index = range(len(s8_cost_report_all_data))

    return s8_cost_report_all_data


#
def plot_schedule8_weather_cost_report(route=None, weather=None, update=False, show_title=False, save_as=".png"):
    # Get data
    report_data = get_schedule8_weather_cost_report(update=update)
    colours = matplotlib.cm.get_cmap('Set2')(np.flip(np.linspace(0.0, 1.0, 9), 0))
    colour = colours[list(set(report_data.WeatherCategory)).index(weather)] if weather else '#c0bc7c'
    report_data = dbm.subset(report_data, route, weather)
    # Total delay minutes and costs for all Weather-related Incidents
    data = report_data.groupby('Year').agg({'DelayCost': np.sum, 'DelayMinutes': np.sum})
    data.reset_index(inplace=True)

    # A bar chart of total delay minutes by year
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.grid(False)
    ax1.set_facecolor('none')
    # Delay minutes (left-hand side)
    ax1.bar(data.Year, data.DelayMinutes, align='center', color=colour, alpha=0.9,
            label='Total delay minutes{}'.format(' (' + weather + '-related)' if weather else ''))
    # Format labels against the y-ticks (The two args in lambda func are the value and tick position)
    formatter1 = matplotlib.ticker.FuncFormatter(lambda x, position: format(int(x), ','))
    ax1.get_yaxis().set_major_formatter(formatter1)
    ax1.tick_params(axis='y', labelsize=12)
    # Set labels for the y-axis
    ax1.set_ylabel('Delay (minutes)', fontweight='bold', fontsize=13)
    # Set labels for the x-axis
    ax1.set_xlabel('Financial year (1 April - 31 March)', fontweight='bold', fontsize=13, labelpad=10)
    ax1.tick_params(axis='x', labelsize=12)
    fig1, label1 = ax1.get_legend_handles_labels()  # Legend

    ax2 = ax1.twinx()
    ax2.grid(False)
    for spine in ['left', 'right', 'top', 'bottom']:
        ax2.spines[spine].set_color('k')
    ax2.plot(data.Year, data.DelayCost, color='#800000', alpha=0.6,
             marker='o', markersize=15, markeredgecolor='None',  # markeredgewidth=1,
             linewidth=3, label='Total delay cost{}'.format(' (' + weather + '-related)' if weather else ''))
    # Set fontsize of labels against the x-ticks
    ax2.set_xticks(data.Year)
    ax2.set_xticklabels(['/'.join([str(x), str(x + 1)[-2:]]) for x in data.Year])
    formatter2 = matplotlib.ticker.FuncFormatter(lambda x, position: '£%1.1fM' % (x * 1e-6))
    ax2.yaxis.set_major_formatter(formatter2)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylabel('Delay cost (£)', fontweight='bold', fontsize=13)
    fig2, label2 = ax2.get_legend_handles_labels()

    # Show legend
    plt.legend(fig1 + fig2, label1 + label2, numpoints=1, loc=2, frameon=False, prop={'size': 15}, labelspacing=0.8)

    if show_title:
        title = 'Total delays/cost attributed to {}-related Incidents{}'.format(
            weather.lower() if weather else 'Weather', ' on the %s Route' % route if route else '')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.subplots_adjust(left=.13, bottom=.1, right=0.89, top=.89)
    else:
        plt.tight_layout()

    if save_as:
        filename = "-".join(["Schedule8WeatherCostReport"] + [s for s in (route, weather) if s is not None])
        save_fig(cdd_incidents("Exploratory analysis", filename + save_as), dpi=600)


#
def plot_schedule8_weather_cost_report_by_weather(route=None, update=False, show_title=False, save_as=".png"):
    # Get data
    report_data = get_schedule8_weather_cost_report(route, weather=None, update=update)
    # Total delay minutes and costs for all Weather-related Incidents
    data = report_data.groupby(['Year', 'WeatherCategory']).aggregate({'DelayCost': np.sum, 'DelayMinutes': np.sum})
    data.reset_index(inplace=True)
    data.set_index('Year', inplace=True)

    # A bar chart of total delays/cost by year and Weather category
    plt.figure(figsize=(10, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    ax.grid(False)
    ax.set_facecolor('none')
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('k')
    # Colours
    colours = matplotlib.cm.get_cmap('Set2')(np.flip(np.linspace(0.0, 1.0, 9), 0))
    xdata1 = pd.unique(data.index)  # data for x-axis
    bottom = None  # Bottom
    for w in np.flip(data.WeatherCategory.unique(), 0):
        dat = data[data.WeatherCategory == w]
        if len(dat) < 9:
            diff = list(set(xdata1) - set(dat.index))
            for y in diff:
                dat.loc[y] = 0
            dat.WeatherCategory = w
        ydata1 = np.array(dat.DelayMinutes)  # data for y-axis (left-hand side)
        ax.bar(xdata1, ydata1, bottom=bottom, align='center', width=0.7, edgecolor='black',
               color=colours[list(set(data.WeatherCategory)).index(w)], alpha=0.9, label=w)
        bottom = dat.DelayMinutes if bottom is None else bottom + dat.DelayMinutes

    formatter = matplotlib.ticker.FuncFormatter(lambda x, position: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(formatter)
    ax.set_ylabel('Delay (minutes)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Financial year (1 April - 31 March)', fontweight='bold', fontsize=13, labelpad=10)
    # ax.set_xticks(xdata1)
    ax.set_xticklabels(['/'.join([str(x), str(x + 1)[-2:]]) for x in xdata1])
    ax.tick_params(labelsize=12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=2, frameon=False, prop={'size': 11})

    if show_title:
        title = 'Total delays/cost attributed to Weather-related Incidents{}'.format(
            ' on the %s Route' % route if route else '')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.subplots_adjust(left=.12, bottom=.10, right=.98, top=.90)
    else:
        plt.tight_layout()

    if save_as:
        filename = "-".join(["Schedule8WeatherCostReport-by-Weather"] + [s for s in [route] if route is not None])
        save_fig(cdd_incidents("Exploratory analysis", filename + save_as), dpi=600)


# ====================================================================================================================
""" S8WeatherIncidentsByDay.xlsx """


# Read Schedule 8 Weather Incidents data
def get_schedule8_weather_incidents_by_day(route=None, weather=None, update=False):
    """

    Description:
    "Schedule 8 delay minutes partitioned by DU / Day / Weather Category"
    "The report includes summaries of incident counts and delay minutes per day for each Weather category together
    with detailed data for each incident." (Retrieved directly from the Database)

    """
    pickle_filename = "S8WeatherIncidentsByDay.pickle"
    path_to_pickle = cdd_spreadsheet(pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        workbook_data = load_pickle(path_to_pickle)
    else:
        try:
            path_to_spreadsheet = path_to_pickle.replace(".pickle", ".xlsx")
            workbook = pd.ExcelFile(path_to_spreadsheet)  # Open the workbook
            # Read the first worksheet: "Summary"
            summary = workbook.parse('Summary', parse_cols='A:G', parse_dates=False, dayfirst=True)

            # Read the second worksheet: "Details"
            details = workbook.parse('Details', parse_dates=False, dayfirst=True)
            details.rename(columns={'DU': 'IMDM', 'delayMinutes': 'DelayMinutes'}, inplace=True)
            workbook_data = dict(zip(workbook.sheet_names, [summary, details]))
            workbook.close()
            save(workbook_data, cdd_spreadsheet(pickle_filename))
        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.basename(pickle_filename), e))
            workbook_data = pd.DataFrame()

    workbook_data['Summary'] = dbm.subset(workbook_data['Summary'], route, weather)
    workbook_data['Details'] = dbm.subset(workbook_data['Details'], route, weather)

    return workbook_data


#
def summarise_delays(route=None, weather=None, update=False, feature='Year'):
    # Read the second worksheet: 'Details'
    workbook_data = get_schedule8_weather_incidents_by_day(route, weather, update)
    details_data = workbook_data['Details']
    details_summary = details_data.groupby(feature).agg({'DelayMinutes': np.sum})
    return details_summary.reset_index()


#
def plot_schedule8_weather_delays_by_season(route=None, weather=None, update=False, show_title=False, save_as=".png"):
    """
    The meteorological seasons (instead of the astronomical seasons) are defined as

    Spring (March, April, May), from March 1 to May 31
    Summer (June, July, August), from June 1 to August 31
    Autumn (September, October, November), from September 1 to November 30
    Winter (December, January, February), from December 1 to February 28/29

    Leap year: 2008 and 2012

    """
    # Read the second worksheet: 'Details'
    details = get_schedule8_weather_incidents_by_day(route, weather, update)['Details']

    spring_data = pd.DataFrame()
    summer_data = pd.DataFrame()
    autumn_data = pd.DataFrame()
    winter_data = pd.DataFrame()
    for y in pd.unique(details.Year):
        df = details[details.Year == y]
        # Get data for spring
        spring_start1 = datetime.datetime(year=y, month=4, day=1)
        spring_end1 = datetime.datetime(year=y, month=5, day=31)
        spring_start2 = datetime.datetime(year=y + 1, month=3, day=1)
        spring_end2 = datetime.datetime(year=y + 1, month=3, day=31)
        spring = ((df.Date >= spring_start1) & (df.Date <= spring_end1)) | \
                 ((df.Date >= spring_start2) & (df.Date <= spring_end2))
        spring_data = pd.concat([spring_data, df.loc[spring]])
        # Get data for summer
        summer_start = datetime.datetime(year=y, month=6, day=1)
        summer_end = datetime.datetime(year=y, month=8, day=31)
        summer = (df.Date >= summer_start) & (df.Date <= summer_end)
        summer_data = pd.concat([summer_data, df.loc[summer]])
        # Get data for autumn
        autumn_start = datetime.datetime(year=y, month=9, day=1)
        autumn_end = datetime.datetime(year=y, month=11, day=30)
        autumn = (df.Date >= autumn_start) & (df.Date <= autumn_end)
        autumn_data = pd.concat([autumn_data, df.loc[autumn]])
        # Get data for winter
        winter_start = datetime.datetime(year=y, month=12, day=1)
        if (y != 2007) & (y != 2011):
            winter_end = datetime.datetime(year=y + 1, month=2, day=28)
        else:
            winter_end = datetime.datetime(year=y + 1, month=2, day=29)
        winter = (df.Date >= winter_start) & (df.Date <= winter_end)
        winter_data = pd.concat([winter_data, df.loc[winter]])

    spring_delay = spring_data.groupby('Year').aggregate({'DelayMinutes': sum})
    summer_delay = summer_data.groupby('Year').aggregate({'DelayMinutes': sum})
    autumn_delay = autumn_data.groupby('Year').aggregate({'DelayMinutes': sum})
    winter_delay = winter_data.groupby('Year').aggregate({'DelayMinutes': sum})

    bottom1 = spring_delay.DelayMinutes
    bottom2 = bottom1 + summer_delay.DelayMinutes
    bottom3 = bottom2 + autumn_delay.DelayMinutes

    plt.figure(figsize=(10, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    ax.grid(False)
    ax.set_facecolor('none')
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('k')
    bar1 = ax.bar(spring_delay.index, spring_delay.DelayMinutes, align='center', color='#AFCE45')
    bar2 = ax.bar(summer_delay.index, summer_delay.DelayMinutes, align='center', color='#EF4F39', bottom=bottom1)
    bar3 = ax.bar(autumn_delay.index, autumn_delay.DelayMinutes, align='center', color='#FEBE68', bottom=bottom2)
    bar4 = ax.bar(winter_delay.index, winter_delay.DelayMinutes, align='center', color='#279DD6', bottom=bottom3)

    ax.set_xticks(list(spring_delay.index))

    plt.xticks(list(spring_delay.index), fontsize=13)
    ylabel_formatter = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(ylabel_formatter)
    ax.set_xlabel('Financial year (1 April - 31 March)', fontweight='bold', fontsize=13, labelpad=10)
    ax.set_ylabel('Delay (minutes)', fontweight='bold', fontsize=13)
    ax.set_xticklabels(['/'.join([str(x), str(x + 1)[-2:]]) for x in spring_delay.index])
    ax.tick_params(labelsize=12)
    ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]),
              ('Spring\t\t\t\t\t(1 Mar\t\t- 31 May)',
               'Summer (1 Jun\t\t\t- 31 Aug)',
               'Autumn\t\t(1 Sept - 30 Nov)',
               'Winter\t\t\t\t(1 Dec\t\t- 28/29 Feb)'),
              loc='best', frameon=False, prop={'size': 13})

    if show_title:
        title = 'Total delays/cost attributed to {}-related Incidents{}'.format(
            weather.lower() if weather else 'Weather', ' on the %s Route' % route if route else '')
        plt.suptitle(title, fontsize=15, fontweight='bold')
        plt.subplots_adjust(left=0.12, bottom=0.10, right=0.98, top=0.92)
    else:
        plt.tight_layout()

    if save_as:
        filename = "-".join(["S8WeatherIncidentsByDay-delays-by-season"] +
                            [s for s in (route, weather) if s is not None])
        save_fig(cdd_incidents("Exploratory analysis", filename + save_as), dpi=600)


# ====================================================================================================================
""" Schedule8WeatherIncidents-02062006-31032014.xlsm """  # Need to be modified


def get_schedule8_weather_incidents_02062006_31032014(route=None, weather=None, update=False):
    """
    Description:
    "Details of schedule 8 Incidents together with Weather leading up to the incident. Although this file contains
    other Weather categories, the main focus of this prototype is adhesion.

    * WORK IN PROGRESS *  MET-9 - Report of Schedule 8 adhesion Incidents vs Weather conditions Done."

    """
    # Path to the file
    pickle_filename = "Schedule8WeatherIncidents-02062006-31032014.pickle"
    path_to_pickle = cdd_spreadsheet(pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        workbook_data = load_pickle(path_to_pickle)
    else:
        try:
            # Open the original file
            spreadsheet_filename = pickle_filename.replace(".pickle", ".xlsm")
            workbook = pd.ExcelFile(cdd_incidents("Spreadsheets", spreadsheet_filename))

            # 'Thresholds' ================================================================
            thresholds = workbook.parse(sheet_name='Thresholds', parse_cols='A:F').dropna()
            thresholds.index = range(len(thresholds))
            thresholds.columns = [col.replace(' ', '') for col in thresholds.columns]
            thresholds.WeatherHazard = thresholds.WeatherHazard.map(lambda x: x.upper().strip())

            # 'Data' =========================================================================================
            data = workbook.parse('Data', parse_dates=False, dayfirst=True, converters={'stanoxSection': str})
            data.columns = [c.replace('(C)', '(degrees Celsius)').replace(' ', '') for c in data.columns]
            data.rename(columns={'imdm': 'IMDM',
                                 'stanoxSection': 'StanoxSection',
                                 'Minutes': 'DelayMinutes',
                                 'Cost': 'DelayCost',
                                 'Reason': 'IncidentReason',
                                 'CategoryDescription': 'IncidentCategoryDescription',
                                 'WeatherHazard(pdmint)': 'PreviousDayMinTemperature_WeatherHazard',
                                 'WeatherHazard(pdmaxt)': 'PreviousDayMaxTemperature_WeatherHazard',
                                 'WeatherHazard(pddt)': 'PreviousDayDeltaTemperature_WeatherHazard',
                                 'WeatherHazard(pdrh)': 'PreviousDayRelativeHumidity_WeatherHazard',
                                 '3-HourRain(mm)': 'PreviousThreeHourRain(mm)',
                                 'WeatherHazard(thr)': 'PreviousThreeHourRain_WeatherHazard',
                                 'DailyRain(mm)': 'PreviousDailyRain(mm)',
                                 'WeatherHazard(pdr)': 'PreviousDailyRain_WeatherHazard',
                                 '15-DayRain(mm)': 'PreviousFifteenDayRain(mm)',
                                 'WeatherHazard(pfdr)': 'PreviousFifteenDayRain_WeatherHazard',
                                 'DailySnow(cm)': 'PreviousDailySnow(cm)',
                                 'WeatherHazard(pds)': 'PreviousDailySnow_WeatherHazard'}, inplace=True)
            data.WeatherCategory = data.WeatherCategory.replace('Heat Speed/Buckle', 'Heat')

            stanox_section = data.StanoxSection.str.split(' : ', expand=True)
            stanox_section.columns = ['StartLocation', 'EndLocation']
            stanox_section.EndLocation.fillna(stanox_section.StartLocation, inplace=True)

            stanox_dict_1 = dbm.get_stanox_location().Location.to_dict()

            line_data = LineData()
            stanox_dict_2 = line_data.LocationIdentifiers.make_location_codes_dictionary('STANOX')

            stanox_section.StartLocation = stanox_section.StartLocation.replace(stanox_dict_1).replace(stanox_dict_2)
            stanox_section.EndLocation = stanox_section.EndLocation.replace(stanox_dict_1).replace(stanox_dict_2)

            stanme_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('STANME')
            tiploc_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('TIPLOC')
            loc_name_replacement_dict = dbm.location_names_replacement_dict()
            loc_name_regexp_replacement_dict = dbm.location_names_regexp_replacement_dict()
            # Processing 'StartStanox'
            stanox_section.StartLocation = stanox_section.StartLocation. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)
            # Processing 'EndStanox_loc'
            stanox_section.EndLocation = stanox_section.EndLocation. \
                replace(stanme_dict).replace(tiploc_dict). \
                replace(loc_name_replacement_dict).replace(loc_name_regexp_replacement_dict)

            # Form new STANOX sections
            stanox_section['StanoxSection'] = stanox_section.StartLocation + ' - ' + stanox_section.EndLocation
            point_idx = stanox_section.StartLocation == stanox_section.EndLocation
            stanox_section.StanoxSection[point_idx] = stanox_section.StartLocation[point_idx]

            # Resort column order
            col_names = list(data.columns)
            col_names.insert(col_names.index('StanoxSection') + 1, 'StartLocation')
            col_names.insert(col_names.index('StartLocation') + 1, 'EndLocation')
            data = stanox_section.join(data.drop('StanoxSection', axis=1))[col_names]

            # Add incident reason metadata --------------------------
            incident_reason_metadata = dbm.get_incident_reason_metadata()
            data = pd.merge(data, incident_reason_metadata.reset_index(),
                            on=['IncidentReason', 'IncidentCategoryDescription'], how='inner')

            # Weather'CategoryLookup' ===========================================
            weather_category_lookup = workbook.parse(sheet_name='CategoryLookup')
            weather_category_lookup.columns = ['WeatherCategoryCode', 'WeatherCategory']

            # Make a dictionary
            workbook_data = dict(zip(workbook.sheet_names, [thresholds, data, weather_category_lookup]))
            workbook.close()
            # Save the workbook data
            save(workbook_data, path_to_pickle)

        except Exception as e:
            print("Failed to get \"{}\". {}.".format(os.path.basename(pickle_filename), e))
            workbook_data = None

        # Retain data for specific Route and Weather category
        workbook_data['Data'] = dbm.subset(workbook_data['Data'], route, weather)

    return workbook_data


# ====================================================================================================================
""" TRUST """  # Need to be modified


# Get a reference for matching DU with Route
def get_du_route_dict(as_dict=True, update=False):
    filename = "DU-Route"
    path_to_pickle = cdd_incidents("TRUST", filename + ".pickle")

    if as_dict:
        path_to_pickle = path_to_pickle.replace(filename, filename + "-dict")

    if os.path.isfile(path_to_pickle) and not update:
        du_route = load_pickle(path_to_pickle)
    else:
        try:
            dr1 = dbm.get_imdm().reset_index()
            dr2 = dbv.get_du_route().reset_index()[['DUNameGIS', 'Route']]
            dr2.rename(columns={'DUNameGIS': 'IMDM'}, inplace=True)

            du_route = pd.DataFrame(pd.concat([dr1, dr2], ignore_index=True)).drop_duplicates()

            du = du_route.IMDM.apply(lambda x: x.replace('IMDM ', '').strip().upper())
            du_route['DU'] = du
            if as_dict:
                du_route = dict(zip(du_route.DU, du_route.Route))
            else:
                du_route.set_index('IMDM', inplace=True)
            save(du_route, path_to_pickle)
        except Exception as e:
            print("Failed to get \"DU-Route reference\". {}.".format(e))
            du_route = None
    return du_route


#
def get_trust_schedule8_incidents_details(update=False):
    """

    Description:
    "Output from TRUST database containing details of Incidents."
    (from 01-Apr-2006 to 31-Mar-2015, i.e. financial year 2006-2014)

    """
    pickle_filename = "Schedule-8-Incidents-details.pickle"
    path_to_pickle = cdd_incidents("TRUST", pickle_filename)
    if os.path.isfile(path_to_pickle) and not update:
        schedule8_incidents_details = load_pickle(path_to_pickle)
    else:
        try:
            # Get a reference for matching DU with Route (See also utils.py)
            du_rte_dict = get_du_route_dict(as_dict=True, update=update)
            # Get information of Performance Event Code
            prfm_event_code = dbm.get_performance_event_code()

            line_data = LineData()
            loc_dict = line_data.LocationIdentifiers.make_location_codes_dictionary('STANOX', drop_duplicates=True)
            # Get location data
            location = dbm.get_location()
            stanox_location = dbm.get_stanox_location()
            location_ref = stanox_location.join(location, on='LocationId', how='inner')
            loc_imdm = location_ref[['IMDM']].join(get_du_route_dict(as_dict=False, update=update), on='IMDM')

            # A function that reads, and pre-processes, a single archive file
            def read_trust_schedule8_incidents_details_archive(zip_archive):
                # Read raw data
                data = pd.read_csv(zip_archive,
                                   parse_dates=['Date', 'Incident Start Datetime', 'Incident End Datetime'],
                                   dayfirst=True)
                # Rename columns
                data.columns = [c.replace(' ', '') for c in data.columns]
                data.rename(columns={'Route': 'RouteCode', 'SectionCode': 'StanoxSectionCode'}, inplace=True)
                data['Route'] = data.DeliveryUnitName.replace(du_rte_dict)
                data['IMDM'] = ['IMDM '] + data['DeliveryUnitName'].str.title()

                data = data.join(prfm_event_code, on='PerformanceEventCode').drop('SectionName', axis=1)

                stanox_section_code = data.StanoxSectionCode.str.split(':', expand=True)
                stanox_section_code.columns = ['StartStanox', 'EndStanox']
                stanox_section_code.EndStanox.fillna(stanox_section_code.StartStanox, inplace=True)

                # for start locations
                start_location = stanox_section_code[['StartStanox']].join(stanox_location, on='StartStanox')
                start_location.Location[start_location.Location.isnull()] = \
                    start_location.StartStanox[start_location.Location.isnull()].replace(loc_dict)
                start_location.columns = ['Start' + c if 'Start' not in c else c for c in start_location.columns]

                # for end locations
                end_location = stanox_section_code[['EndStanox']].join(stanox_location, on='EndStanox')
                end_location.Location[end_location.Location.isnull()] = \
                    end_location.EndStanox[end_location.Location.isnull()].replace(loc_dict)
                end_location.columns = ['End' + c if 'End' not in c else c for c in end_location.columns]

                start_end = start_location.StartLocation + ' - ' + end_location.EndLocation

                # Find point location
                start_end[start_location.StartLocation == end_location.EndLocation] = \
                    start_location.StartLocation[start_location.StartLocation == end_location.EndLocation].tolist()
                stanox_section = start_end.to_frame('StanoxSection')

                data = pd.DataFrame(pd.concat([data, stanox_section, start_location, end_location], axis=1))

                dat_without_route_idx = data.Route == 'None'
                temp = data[dat_without_route_idx].join(loc_imdm, on='StartStanox', rsuffix='_start')
                temp = temp.join(loc_imdm, on='EndStanox', rsuffix='_end')
                temp.Route_start.fillna(temp.Route_end, inplace=True)
                data.Route[dat_without_route_idx] = temp.Route_start
                data.IMDM[dat_without_route_idx] = temp.IMDM_start
                data.DeliveryUnitName[dat_without_route_idx] = temp.DU
                temp_du_dict = dict(zip(data.DeliveryUnitName, data.DeliveryUnit))
                data.DeliveryUnit[dat_without_route_idx] = \
                    data.DeliveryUnitName[dat_without_route_idx].replace(temp_du_dict)
                # data.fillna('', inplace=True)
                print("{} ... processed.".format(zip_archive[zip_archive.index('(') + 1:zip_archive.index(')')]))
                return data

            # To load the data
            schedule8_incidents_details_data = \
                [read_trust_schedule8_incidents_details_archive(zip_archive) for zip_archive in
                 glob.glob(cdd_incidents("TRUST", "CLowe PSS Snapshot v1.4 Orig*.zip"))]

            schedule8_incidents_details = pd.concat(schedule8_incidents_details_data, ignore_index=True, axis=0)

            save(schedule8_incidents_details, path_to_pickle)

        except Exception as e:
            print("Failed to get \"TRUST Schedule 8 Incidents details\". {}.".format(e))
            schedule8_incidents_details = None

    return schedule8_incidents_details
