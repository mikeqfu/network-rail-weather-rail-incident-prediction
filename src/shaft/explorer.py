"""Exploratory analysis."""

import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from pyhelpers.dirs import cdd
from pyhelpers.geom import find_closest_points, get_midpoint, wgs84_to_osgb36
from pyhelpers.settings import mpl_preferences
from pyhelpers.store import save_data

from src.shaft.geometry import get_shp_coordinates


class BasicStats:

    def __init__(self):
        mpl_preferences(backend='TkAgg', font_name='Times New Roman')

    @classmethod
    def calculate_statistics(cls, s8_weather_incidents):
        """
        Calculate statistics of different categories of Weather-related Incidents.

        :param s8_weather_incidents: data of Weather-related Incidents
        :type s8_weather_incidents: pandas.DataFrame
        :return: statistics about frequencies of different categories of
            Weather-related Incidents and their total costs
        :rtype: pandas.DataFrame

        **Example**::

            >>> from src.shaft.explorer import BasicStats
            >>> from src.preprocessor import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()
            >>> sir.read_schedule8_weather_incidents_02062006_31032014()
            >>> dat_ = sir.incidents_02062006_31032014.copy()
            >>> dat = dat_['Schedule8WeatherIncidents_02062006_31032014']

            >>> bs = BasicStats()
            >>> stats_dat = bs.calculate_statistics(dat)
            >>> stats_dat.index.to_list()
            ['Adhesion',
             'Wind',
             'Snow',
             'Flood',
             'Cold',
             'Heat Speed/Buckle',
             'Subsidence',
             'Lightning',
             'Fog']
        """

        data = s8_weather_incidents.rename(columns={'Minutes': 'DelayMinutes', 'Cost': 'DelayCost'})

        stats = data.groupby('WeatherCategory').aggregate(
            {'WeatherCategory': 'count', 'DelayMinutes': np.sum, 'DelayCost': np.sum})
        stats.rename(columns={'WeatherCategory': 'Count'}, inplace=True)
        stats['percentage'] = stats.Count / len(data) * 100
        # Sort stats in the ascending order of 'percentage'
        stats.sort_values('percentage', ascending=False, inplace=True)

        return stats

    def pie_chart_for_incident_proportions(self, s8_weather_incidents, save_as=None):
        """
        Create a pie chart illustrating the proportions of different categories of
        Weather-related Incidents.

        :param s8_weather_incidents: data of Schedule 8 Incidents
        :type s8_weather_incidents: pandas.DataFrame
        :param save_as: whether to save the pie chart / what format the pie chart is saved as,
            defaults to ``None``
        :type save_as: str or None

        **Example**::

            >>> from src.shaft.explorer import BasicStats
            >>> from src.preprocessor import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()
            >>> sir.read_schedule8_weather_incidents_02062006_31032014()
            >>> dat_ = sir.incidents_02062006_31032014.copy()
            >>> dat = dat_['Schedule8WeatherIncidents_02062006_31032014']

            >>> bs = BasicStats()
            >>> bs.pie_chart_for_incident_proportions(dat)
        """

        stats = self.calculate_statistics(s8_weather_incidents).reset_index()
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
        # explode_pos = stats.sort_values(by=['PfPIMinutes', 'percentage'], ascending=False).index[0]
        explode_list[1] = 0.2

        # Create a figure
        plt.figure(figsize=(8, 6))
        ax = plt.subplot2grid((1, 1), (0, 0), aspect='equal')
        # ax.set_rasterization_zorder(1)
        pie_collections = ax.pie(
            stats.percentage, labels=wind_label, startangle=70, colors=colours, explode=explode_list,
            labeldistance=0.7)

        # Note that 'pie_collections' includes: patches, texts, autotexts
        patches, texts = pie_collections
        texts[1].set_fontsize(12)
        texts[1].set_fontstyle('italic')
        # texts[1].set_fontweight('bold')
        legend = ax.legend(
            pie_collections[0], labels, loc='best', fontsize=14, frameon=True, shadow=True,
            fancybox=True, title='Weather category')
        frame = legend.get_frame()
        frame.set_edgecolor('black')
        frame.set_facecolor('white')

        # ax.set_title('Reasons for Weather-related Incidents\n', fontsize=14, weight='bold')

        plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=0.95)

        plt.show()

        if save_as:
            plt.savefig(cdd("../../data/Exploration", "proportions" + save_as), dpi=600)

    def bar_chart_for_delay_cost(self, s8_weather_incidents, save_as=".png"):
        """
        Plot total monetary cost incurred by Weather-related Incidents.

        :param s8_weather_incidents: data of Schedule 8 Incidents
        :type s8_weather_incidents: pandas.DataFrame
        :param save_as: whether to save the pie chart / what format the pie chart is saved as,
            defaults to ``".png"``
        :type save_as: str or None

        **Example**::

            >>> from src.shaft.explorer import BasicStats
            >>> from src.preprocessor import Schedule8IncidentReports

            >>> sir = Schedule8IncidentReports()
            >>> sir.read_schedule8_weather_incidents_02062006_31032014()
            >>> dat_ = sir.incidents_02062006_31032014.copy()
            >>> dat = dat_['Schedule8WeatherIncidents_02062006_31032014']

            >>> bs = BasicStats()
            >>> bs.bar_chart_for_delay_cost(dat)
        """

        stats = self.calculate_statistics(s8_weather_incidents).reset_index()
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
        plt.barh(stats.index, stats.DelayCost, align='center', color=colours, alpha=1.0, hatch='/')
        plt.yticks(stats.index, [''] * len(stats))
        plt.xticks(fontsize=12)
        # Format labels
        ax2.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda c, position: '%1.1f' % (c * 1e-7)))
        plt.xlabel('£ millions', fontsize=13, fontweight='bold')
        plt.title('Cost', fontsize=15, fontweight='bold')
        # ax2.set_axis_bgcolor('#dddddd')

        # plt.subplots_adjust(left=0.16, bottom=0.10, right=0.96, top=0.92, wspace=0.16)
        plt.tight_layout()

        if save_as:
            plt.savefig(cdd("../../data/Exploration", "delays-and-cost" + save_as), dpi=600)


class ExtraDataPrep:
    """Preprocess data for the paper co-authored by Y. Zhang et al."""

    DATA_DIRNAME = "extra_prep_data"

    def __init__(self):
        from src.preprocessor.metex import METEX

        self.metex = METEX()

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change to the data directory.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "data\\Exploration\\extra_prep_data" and subdirectories / a file
        :rtype: str

        **Examples**::

            >>> from src.shaft.explorer import ExtraDataPrep
            >>> import os

            >>> edp = ExtraDataPrep()

            >>> os.path.relpath(edp.cdd())
            'data\\Exploration\\extra_prep_data'
        """

        path = cdd("../../data/Exploration", self.DATA_DIRNAME, *sub_dir, mkdir=mkdir)

        return path

    @staticmethod
    def find_midpoint_of_each_incident_location(incident_data):
        """
        Find the "midpoint" of each incident location.

        :param incident_data: data of incident records, containing information of
            start/end location coordinates
        :type incident_data: pandas.DataFrame
        :return: midpoints in both (longitude, latitude) and (easting, northing)
        :rtype: pandas.DataFrame

        **Example**::

            >>> from src.shaft.explorer import ExtraDataPrep
            >>> from src.preprocessor import METEX

            >>> mt = METEX()

            >>> mt.view_schedule8_cost_by_day_location()
            >>> dat = mt.schedule8_cost_by_day_location

            >>> edp = ExtraDataPrep()

            >>> incident_location_midpoints = edp.find_midpoint_of_each_incident_location(dat)
            >>> incident_location_midpoints
        """

        assert isinstance(incident_data, pd.DataFrame)
        lon_lat_col_names = ['StartLongitude', 'StartLatitude', 'EndLongitude', 'EndLatitude']
        assert all(x in incident_data.columns for x in lon_lat_col_names)

        data = incident_data.copy()

        # Find a pseudo-midpoint location for each incident location
        pseudo_midpoints = get_midpoint(
            data['StartLongitude'], data['StartLatitude'], data['EndLongitude'], data['EndLatitude'])

        # Find the "midpoint" of each incident location
        railway_coordinates = get_shp_coordinates(
            osm_subregion='Great Britain', osm_layer='railways', osm_feature='rail')

        midpoints = find_closest_points(pts=pseudo_midpoints, ref_pts=railway_coordinates, k=1)

        data[['MidLongitude', 'MidLatitude']] = pd.DataFrame(midpoints)
        data['MidEasting'], data['MidNorthing'] = wgs84_to_osgb36(
            data['MidLongitude'], data['MidLatitude'])

        return data

    # == 1st dataset ===============================================================================

    def prepare_stats_data(self, route_name=None, weather_category=None, verbose=True):
        """
        Prepare data of statistics.

        :param route_name: name of Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: Weather to which an incident is attributed, defaults to ``None``
        :type weather_category: str or None
        :param verbose: defaults to ``False``
        :type verbose: bool

        **Example**::

            >>> from src.shaft.explorer import ExtraDataPrep

            >>> edp = ExtraDataPrep()

            >>> edp.prepare_stats_data(verbose=True)
        """

        # Get data of Schedule 8 incident locations
        self.metex.view_schedule8_cost_by_location(
            route_name=route_name, weather_category=weather_category)
        incident_locations_ = self.metex.schedule8_cost_by_location.copy()

        # Find the "midpoint" of each incident location
        incident_locations = self.find_midpoint_of_each_incident_location(incident_locations_)

        # Split the data by "region"
        for region_name, region_data in incident_locations.groupby('Region'):
            # Sort data by (frequency of incident occurrences, delay minutes, delay cost)
            region_data.sort_values(
                ['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost'], ascending=False,
                ignore_index=True, inplace=True)
            export_path = self.cdd(str(region_name).replace(" ", "-").lower() + ".csv")
            save_data(region_data, export_path, verbose=verbose)

        print("\nCompleted.")

    # == 2nd dataset ===================================================================================

    def prepare_monthly_stats_data(self, route_name=None, weather_category=None):
        """
        Prepare data of monthly statistics.

        :param route_name: name of Route, defaults to ``None``
        :type route_name: str or None
        :param weather_category: Weather to which an incident is attributed,
            defaults to ``None``
        :type weather_category: str or None

        **Example**::

            >>> from src.shaft.explorer import ExtraDataPrep

            >>> edp = ExtraDataPrep()

            >>> edp.prepare_monthly_stats_data()
        """

        # Get data of Schedule 8 Incidents by datetime and location
        self.metex.view_schedule8_cost_by_day_location(
            route_name=route_name, weather_category=weather_category)
        dat = self.metex.schedule8_cost_by_day_location.copy()

        print("Cleaning data ... ", end="")
        datetime_cols = ['StartDateTime', 'EndDateTime']
        dat[datetime_cols] = dat[datetime_cols].apply(pd.to_datetime)
        dat.insert(dat.columns.get_loc('EndDateTime') + 1, 'StartYear', dat.StartDateTime.dt.year)
        dat.insert(dat.columns.get_loc('StartYear') + 1, 'StartMonth', dat.StartDateTime.dt.month)

        stats_calc = {'IncidentCount': np.count_nonzero, 'DelayMinutes': np.sum, 'DelayCost': np.sum}
        stats = dat.groupby(list(dat.columns[3:-3])).aggregate(stats_calc)
        stats.reset_index(inplace=True)

        # Find the "midpoint" of each incident location
        data = self.find_midpoint_of_each_incident_location(stats)
        print("Done.\n")

        sort_by_cols = ['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost']

        print("Processing monthly statistics ... ")
        for m, dat1 in data.groupby('StartMonth'):
            m_ = str(m).zfill(2)
            print(f"\t\"{m_}\"", end=" ... ")
            if not dat1.empty:
                dat1.sort_values(
                    sort_by_cols, ascending=False, na_position='last', ignore_index=True, inplace=True)
                export_path = self.cdd("GB", "Month", f"{m_}.csv")
                save_data(dat1, export_path, verbose=False)
            print("Done.")
        print("Completed.\n")

        print("Processing monthly statistics of GB ... ")
        for (y, m), dat2 in data.groupby(['StartYear', 'StartMonth']):
            period = f"{y}_{str(m).zfill(2)}"
            print(f"\t\"{period}\"", end=" ... ")
            if not dat2.empty:
                dat2.sort_values(
                    sort_by_cols, ascending=False, na_position='last', ignore_index=True, inplace=True)
                export_path = self.cdd("GB", "Year_Month", f"{period}.csv")
                save_data(dat2, export_path, verbose=False)
            print("Done.")
        print("Completed.\n")

        # Split the data by "region"
        print("Processing monthly statistics for each region ... ")
        for region_name, region_data in data.groupby('Region'):
            print(f"\t\"{region_name}\"", end=" ... ")
            for (y, m), dat3 in region_data.groupby(['StartYear', 'StartMonth']):
                if not dat3.empty:
                    dat3.sort_values(
                        sort_by_cols, ascending=False, na_position='last', ignore_index=True,
                        inplace=True)
                    subdir_name = str(region_name).replace(" ", "-").lower()
                    filename = f"{y}_{str(m).zfill(2)}.csv"
                    export_path = self.cdd("Region", subdir_name, filename)
                    save_data(dat3, export_path, verbose=False)
            print("Done.")
        print("Completed.\n")
