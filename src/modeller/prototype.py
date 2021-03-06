""" Prototype """

import itertools
import os
import re
import time

import datetime_truncate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import statsmodels.discrete.discrete_model as sm_dcm
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_pickle, save_pickle, save_svg_as_emf
from sklearn import metrics
from sklearn.utils import extmath

from integrator.furlong import get_furlongs_data, get_incident_location_furlongs
from preprocessor import METExLite
from utils import categorise_track_orientations, cd_models, get_data_by_season_, make_filename


class WindAttributedIncidents:
    """

    :param shift_yards_same_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR == EndELR, defaults to ``220``
    :type shift_yards_same_elr: int, float
    :param shift_yards_diff_elr: yards by which the start/end mileage is shifted for adjustment,
        given that StartELR != EndELR, defaults to ``220``
    :param hazard_pctl:
    :type hazard_pctl: defaults to ``50``
    """

    def __init__(self, trial_id, model_type='logit', seed=0, in_seasons=None,
                 ip_start_hrs=-12, ip_end_hrs=12, nip_start_hrs=-12,
                 shift_yards_same_elr=220, shift_yards_diff_elr=220,
                 hazard_pctl=50, outlier_pctl=99):

        self.Name = 'A prototype data model of predicting wind-related incidents.'

        self.TrialID = "{}".format(trial_id)

        self.METEx = METExLite()

        self.Route = 'Anglia'
        self.WeatherCategory = 'Wind'

        self.Seasons = in_seasons

        self.IP_StartHrs = ip_start_hrs
        self.IP_EndHrs = ip_end_hrs
        self.NIP_StartHrs = nip_start_hrs

        self.ShiftYardsForSameELRs = shift_yards_same_elr
        self.ShiftYardsForDiffELRs = shift_yards_diff_elr

        self.HazardsPercentile = hazard_pctl

        self.ExplanatoryVariables = [
            # 'WindSpeed_max',
            # 'WindSpeed_avg',
            'WindGust_max',
            # 'WindDirection_avg',
            # 'WindDirection_avg_[0, 90)',  # [0°, 90°)
            'WindDirection_avg_[90, 180)',  # [90°, 180°)
            'WindDirection_avg_[180, 270)',  # [180°, 270°)
            'WindDirection_avg_[270, 360)',  # [270°, 360°)
            'Temperature_dif',
            # 'Temperature_avg',
            # 'Temperature_max',
            # 'Temperature_min',
            'RelativeHumidity_max',
            'Snowfall_max',
            'TotalPrecipitation_max',
            # 'Electrified',
            'CoverPercentAlder',
            'CoverPercentAsh',
            'CoverPercentBeech',
            'CoverPercentBirch',
            'CoverPercentConifer',
            'CoverPercentElm',
            'CoverPercentHorseChestnut',
            'CoverPercentLime',
            'CoverPercentOak',
            'CoverPercentPoplar',
            'CoverPercentShrub',
            'CoverPercentSweetChestnut',
            'CoverPercentSycamore',
            'CoverPercentWillow',
            # 'CoverPercentOpenSpace',
            'CoverPercentOther',
            # 'CoverPercentVegetation',
            # 'CoverPercentDiff',
            # 'TreeDensity',
            # 'TreeNumber',
            # 'TreeNumberDown',
            # 'TreeNumberUp',
            # 'HazardTreeDensity',
            # 'HazardTreeNumber',
            # 'HazardTreediameterM_max',
            # 'HazardTreediameterM_min',
            # 'HazardTreeheightM_max',
            # 'HazardTreeheightM_min',
            # 'HazardTreeprox3py_max',
            # 'HazardTreeprox3py_min',
            # 'HazardTreeproxrailM_max',
            # 'HazardTreeproxrailM_min'
        ]

        self.OutlierPercentile = outlier_pctl

        self.RandomSeed = seed
        self.ModelType = model_type

        self.WeatherStatsCalc = {'Temperature': (np.nanmax, np.nanmin, np.nanmean),
                                 'RelativeHumidity': (np.nanmax, np.nanmin, np.nanmean),
                                 'WindSpeed': np.nanmax,
                                 'WindGust': np.nanmax,
                                 'Snowfall': (np.nanmax, np.nanmin, np.nanmean),
                                 'TotalPrecipitation': (np.nanmax, np.nanmin, np.nanmean)}

        # Get incident_location_furlongs
        self.Furlongs = get_furlongs_data(route_name=self.Route, weather_category=None,
                                          shift_yards_same_elr=self.ShiftYardsForSameELRs,
                                          shift_yards_diff_elr=self.ShiftYardsForDiffELRs)

        def specify_veg_stats_calc():
            """
            Specify the statistics that need to be computed.
            """

            features = self.Furlongs.columns

            # "CoverPercent..."
            cover_percents = [x for x in features if re.match('^CoverPercent[A-Z]', x)]
            veg_stats_calc = dict(zip(cover_percents, [np.nansum] * len(cover_percents)))
            veg_stats_calc.update({'AssetNumber': np.count_nonzero,
                                   'TreeNumber': np.nansum,
                                   'TreeNumberUp': np.nansum,
                                   'TreeNumberDown': np.nansum,
                                   'Electrified': np.any,
                                   'DateOfMeasure': lambda x: tuple(x),
                                   # 'AssetDesc1': np.all,
                                   # 'IncidentReported': np.any
                                   'HazardTreeNumber':
                                       lambda x: np.nan if np.isnan(x).all() else np.nansum(x)})

            # variables for hazardous trees
            hazard_min = [x for x in features if re.match('^HazardTree.*min$', x)]
            hazard_max = [x for x in features if re.match('^HazardTree.*max$', x)]
            hazard_others = [x for x in features if re.match('^HazardTree[a-z]((?!_).)*$', x)]
            # Computations for hazardous trees variables
            hazard_calc = [dict(zip(hazard_others, [lambda x: tuple(x)] * len(hazard_others))),
                           dict(zip(hazard_min, [np.min] * len(hazard_min))),
                           dict(zip(hazard_max, [np.max] * len(hazard_max)))]

            # Update vegetation_stats_computations
            veg_stats_calc.update({k: v for d in hazard_calc for k, v in d.items()})

            return cover_percents, hazard_others, veg_stats_calc

        self.CoverPercents, self.HazardsOthers, self.VegStatsCalc_ = specify_veg_stats_calc()

        mpl_preferences(font_name='Cambria', reset=False)
        pd_preferences(reset=False)

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype\\wind" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype\\wind" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> os.path.relpath(wind_attributed_incidents.cdd())
            'models\\prototype\\wind'
        """

        path = cd_models("prototype", "wind", *sub_dir, mkdir=mkdir)

        return path

    def cdd_trial(self, *sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype\\wind\\<trial_id>" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype\\wind\\<trial_id>" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> os.path.relpath(wind_attributed_incidents.cdd_trial())
            'models\\prototype\\wind\\0'
        """

        path = self.cdd(self.TrialID, *sub_dir, mkdir=mkdir)

        return path

    def get_weather_variable_names(self, temperature_dif=False, supplement=None):
        """
        Get weather variable names.

        :param temperature_dif: whether to include ``'Temperature_dif'``, defaults to ``False``
        :type temperature_dif: bool
        :param supplement: e.g. ``'Hottest_Heretofore'``
        :type supplement: str or list or None
        :return: a list of names of weather variables
        :rtype: list

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> wind_attributed_incidents.get_weather_variable_names()
            ['Temperature_max',
             'Temperature_min',
             'Temperature_avg',
             'RelativeHumidity_max',
             'RelativeHumidity_min',
             'RelativeHumidity_avg',
             'WindSpeed_max',
             'WindGust_max',
             'Snowfall_max',
             'Snowfall_min',
             'Snowfall_avg',
             'TotalPrecipitation_max',
             'TotalPrecipitation_min',
             'TotalPrecipitation_avg',
             'WindSpeed_avg',
             'WindDirection_avg']
        """

        weather_var_names = []
        for k, v in self.WeatherStatsCalc.items():
            if isinstance(v, tuple):
                for v_ in v:
                    weather_var_names.append('_'.join([
                        k, v_.__name__.replace('mean', 'avg').replace('median', 'med')]).replace(
                        '_nan', '_'))

            else:
                weather_var_names.append('_'.join([
                    k, v.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))

        if temperature_dif:
            weather_var_names.insert(weather_var_names.index('Temperature_min') + 1, 'Temperature_dif')

        if supplement:
            if isinstance(supplement, str):
                supplement = [supplement]
            wind_variable_names = weather_var_names + ['WindSpeed_avg', 'WindDirection_avg'] + supplement

        else:
            wind_variable_names = weather_var_names + ['WindSpeed_avg', 'WindDirection_avg']

        return wind_variable_names

    # == Calculators ==================================================================================

    @staticmethod
    def calc_average_wind(wind_speeds, wind_directions):
        """
        Calculate average wind speed and direction.

        :param wind_speeds: wind speed
        :type wind_speeds: float or int
        :param wind_directions: wind direction
        :type wind_directions: float or int
        :return: average wind speed and average wind direction
        :rtype: tuple
        """

        u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
        v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
        uav, vav = np.nanmean(u), np.nanmean(v)  # sum up all u and v values and average it

        average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed

        # Calculate average wind direction
        if uav == 0:
            average_wind_direction = 0 if vav == 0 else (360 if vav > 0 else 180)
        else:
            average_wind_direction = (270 if uav > 0 else 90) - 180 / np.pi * np.arctan(vav / uav)

        return average_wind_speed, average_wind_direction

    def calc_weather_stats(self, weather_obs):
        """
        Compute the statistics for all the Weather variables (except wind).

        :param weather_obs: observed data of weather conditions
        :type weather_obs: pandas.DataFrame
        :return: statistics for weather conditions
        :rtype: list

        .. note::

            Note: to get the n-th percentile, use percentile(n)

            This function also returns the Weather dataframe indices.
            The corresponding Weather conditions in that WeatherCell might cause wind-related Incidents.
        """

        if not weather_obs.empty:
            # Calculate the statistics
            weather_obs.fillna(value=np.nan, inplace=True)
            stats = weather_obs.fillna(np.nan).groupby('WeatherCell').aggregate(self.WeatherStatsCalc)
            stats['WindSpeed_avg'], stats['WindDirection_avg'] = \
                self.calc_average_wind(weather_obs.WindSpeed, weather_obs.WindDirection)

            weather_stats = stats.values[0].tolist()  # + [weather_obs.index.tolist()]

        else:
            weather_stats = [np.nan] * 10  # + [[None]]

        return weather_stats

    @staticmethod
    def calc_overall_cover_percent_old(start_and_end_cover_percents, total_yards_adjusted):
        """
        Calculate the cover percents across two neighbouring ELRs.

        :param start_and_end_cover_percents: vegetation cover percents of a start and an end ELR
        :type start_and_end_cover_percents: tuple
        :param total_yards_adjusted: adjusted total yards
        :type total_yards_adjusted: tuple
        :return: overall vegetation cover percent across two neighbouring ELRs
        :rtype: float or int
        """

        # (start * end) / (start + end)
        multiplier = pd.np.prod(total_yards_adjusted) / pd.np.sum(total_yards_adjusted)
        # 1/start, 1/end
        cp_start, cp_end = start_and_end_cover_percents
        s_, e_ = pd.np.divide(1, total_yards_adjusted)
        # numerator
        n = e_ * cp_start + s_ * cp_end
        # denominator
        d = pd.np.sum(start_and_end_cover_percents) if pd.np.all(start_and_end_cover_percents) else 1

        f = multiplier * pd.np.divide(n, d)

        overall_cover_percent = f * d

        return overall_cover_percent

    def calc_vegetation_stats(self, furlong_ids, start_elr, end_elr, total_yards_adjusted):
        """
        Calculate stats of vegetation variables for each incident record

        **Test**::

            i = 337

            furlong_ids = incident_location_furlongs.loc[i, 'Critical_FurlongIDs']
            start_elr = incident_location_furlongs.loc[i, 'StartELR']
            end_elr = incident_location_furlongs.loc[i, 'EndELR']
            total_yards_adjusted = incident_location_furlongs.loc[i, 'Section_Length_Adj']

        Note: to get the n-th percentile may use percentile(n)

        """

        # Get all column names as features
        veg_feats = self.Furlongs.columns

        # Get features which would be filled with "0" and "inf", respectively
        fill_0 = [x for x in veg_feats if re.match('.*height', x)] + ['HazardTreeNumber']
        fill_inf = [x for x in veg_feats if re.match('^.*prox|.*diam', x)]

        furlong_ids_ = [fid for fid in furlong_ids if fid in self.Furlongs.index]

        if not furlong_ids_:
            veg_stats = list(np.empty(len(self.VegStatsCalc_) + 2) * np.nan)

        else:
            vegetation_data = self.Furlongs.loc[furlong_ids_]

            veg_stats = vegetation_data.groupby('ELR').aggregate(self.VegStatsCalc_)
            veg_stats[self.CoverPercents] = \
                veg_stats[self.CoverPercents].div(veg_stats.AssetNumber, axis=0).values

            if start_elr == end_elr:
                elr = veg_stats.index[0]

                if np.isnan(veg_stats.HazardTreeNumber[elr]):
                    veg_stats[fill_0] = 0.0
                    veg_stats[fill_inf] = 999999.0
                else:
                    assert 0 <= self.HazardsPercentile <= 100

                    def calc_percentile(x):
                        temp = tuple(itertools.chain(*pd.Series(x).dropna()))
                        if not temp:
                            pctl = np.nan
                        else:
                            pctl = np.nanpercentile(temp, self.HazardsPercentile)
                        return pctl

                    veg_stats[self.HazardsOthers] = \
                        veg_stats[self.HazardsOthers].applymap(calc_percentile)
                    # lambda x: np.nanpercentile(
                    #     tuple(itertools.chain(*pd.Series(x).dropna())), self.HazardsPercentile))

            else:
                if np.all(np.isnan(veg_stats.HazardTreeNumber.values)):
                    veg_stats[fill_0] = 0.0
                    veg_stats[fill_inf] = 999999.0
                    calc_further = {k: lambda y: np.nanmean(y) for k in self.HazardsOthers}
                else:
                    veg_stats[self.HazardsOthers] = veg_stats[self.HazardsOthers].applymap(
                        lambda y: tuple(itertools.chain(*pd.Series(y).dropna())))
                    hazard_others_func = [
                        lambda y: np.nanpercentile(np.sum(y), self.HazardsPercentile)]
                    calc_further = dict(
                        zip(self.HazardsOthers, hazard_others_func * len(self.HazardsOthers)))

                # Specify further calculations
                calc_further.update({'AssetNumber': np.sum})
                calc_further.update(dict(DateOfMeasure=lambda y: tuple(itertools.chain(*y))))
                calc_further.update({k: lambda y: tuple(y) for k in self.CoverPercents})

                # noinspection PyAttributeOutsideInit
                self.VegStatsCalc = self.VegStatsCalc_.copy()

                self.VegStatsCalc.update(calc_further)

                # Rename index (by which the dataframe can be grouped)
                veg_stats.index = pd.Index(
                    data=['-'.join(set(veg_stats.index))] * len(veg_stats.index), name='ELR')
                veg_stats = veg_stats.groupby(veg_stats.index).aggregate(self.VegStatsCalc)

                if isinstance(total_yards_adjusted, tuple):
                    # if (len(total_yards_adjusted) == 3) and
                    if total_yards_adjusted[1] == 0 or np.isnan(total_yards_adjusted[1]):
                        total_yards_adjusted = total_yards_adjusted[:1] + total_yards_adjusted[2:]

                def calc_cp(x):
                    cp = np.dot(x, total_yards_adjusted) / np.nansum(total_yards_adjusted)
                    if isinstance(cp, (np.ndarray, tuple, list)):
                        cp = cp[0]
                    return cp

                veg_stats[self.CoverPercents] = veg_stats[self.CoverPercents].applymap(calc_cp)

            # Calculate tree densities (number of trees per furlong)
            veg_stats['TreeDensity'] = veg_stats.TreeNumber.div(
                np.nansum(total_yards_adjusted) / 220.0)
            veg_stats['HazardTreeDensity'] = veg_stats.HazardTreeNumber.div(
                np.nansum(total_yards_adjusted) / 220.0)

            # Rearrange the order of features
            veg_stats = veg_stats[sorted(veg_stats.columns)].values[0].tolist()

        return veg_stats

    # == Data integration =============================================================================

    def get_incident_location_weather(self, update=False, pickle_it=False, verbose=False):
        """
        Get TRUST data and the weather conditions for each incident location.

        :param update: whether to do an update check, defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: weather conditions of incident locations
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)
            
            >>> incid_loc_weather = wind_attributed_incidents.get_incident_location_weather()

            >>> incid_loc_weather.tail()
                  FinancialYear       StartDateTime  ... Temperature_dif IncidentReported
            3318           2018 2019-01-27 20:04:00  ...             3.0                1
            3319           2018 2019-01-27 20:08:00  ...             4.0                1
            3320           2018 2019-01-27 23:13:00  ...             6.0                1
            3321           2018 2019-01-29 23:00:00  ...             6.0                1
            3322           2018 2019-01-30 05:21:00  ...             6.0                1
            [5 rows x 54 columns]
        """

        pickle_filename = make_filename(
            "weather", self.Route, self.WeatherCategory,
            self.IP_StartHrs, self.IP_EndHrs, self.NIP_StartHrs, save_as=".pickle")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_weather = load_pickle(path_to_pickle)

        else:
            try:
                # Getting Weather data for all incident locations
                incidents = self.METEx.view_schedule8_costs_by_datetime_location_reason(
                    self.Route, self.WeatherCategory)
                # Drop non-weather-related incident records
                if self.WeatherCategory is None:
                    incidents = incidents[incidents.WeatherCategory != '']
                # Get data for the specified "Incident Periods"
                incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
                incidents['Critical_StartDateTime'] = \
                    incidents.StartDateTime.map(datetime_truncate.truncate_hour) + \
                    pd.Timedelta(hours=self.IP_StartHrs)
                incidents['Critical_EndDateTime'] = \
                    incidents.EndDateTime.apply(datetime_truncate.truncate_hour) + \
                    pd.Timedelta(hours=self.IP_EndHrs)
                incidents['Critical_Period'] = \
                    incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

                def get_ip_weather_stats(weather_cell_id, ip_start, ip_end):
                    """
                    Processing weather data for IP.
                    (Get data of weather conditions that led to Incidents for each record.)

                    :param weather_cell_id: weather cell ID
                    :type weather_cell_id: int
                    :param ip_start: start of an incident period
                    :type ip_start: pandas.Timestamp
                    :param ip_end: end of an incident period
                    :type ip_end: pandas.Timestamp
                    :return: a list of statistics
                    :rtype: list

                    **Test**::

                        i = 1

                        weather_cell_id = incidents.WeatherCell[i]
                        ip_start = incidents.StartDateTime[i]
                        ip_end = incidents.EndDateTime[i]
                    """

                    # Get Weather data about where and when the incident occurred
                    ip_weather_obs = self.METEx.query_weather_by_id_datetime(
                        weather_cell_id, ip_start, ip_end, pickle_it=False)

                    # Get the max/min/avg Weather parameters for those incident periods
                    weather_stats = self.calc_weather_stats(ip_weather_obs)

                    return weather_stats

                # Get data for the specified IP
                # noinspection PyTypeChecker
                ip_stats = incidents.apply(
                    lambda x: get_ip_weather_stats(
                        x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime),
                    axis=1)

                ip_statistics = pd.DataFrame(
                    ip_stats.to_list(), index=ip_stats.index, columns=self.get_weather_variable_names())

                ip_statistics['Temperature_dif'] = \
                    ip_statistics.Temperature_max - ip_statistics.Temperature_min

                #
                ip_data = incidents.join(ip_statistics.dropna(), how='inner')
                ip_data['IncidentReported'] = 1

                # Processing Weather data for non-IP
                nip_data = incidents.copy(deep=True)
                nip_data.Critical_EndDateTime = nip_data.Critical_StartDateTime  # + .timedelta(hours=0)
                nip_data.Critical_StartDateTime = \
                    nip_data.Critical_StartDateTime + pd.Timedelta(hours=self.NIP_StartHrs)
                nip_data.Critical_Period = \
                    nip_data.Critical_EndDateTime - nip_data.Critical_StartDateTime

                # Get data of Weather which did not cause Incidents for each record
                def get_non_ip_weather_stats(weather_cell_id, nip_start, nip_end, stanox_section):
                    """
                    Processing weather data for non-IP.
                    (Get data of weather conditions that were less likely to lead to incidents.)

                    :param weather_cell_id: weather cell ID
                    :type weather_cell_id: int
                    :param nip_start: start of a non-incident period
                    :type nip_start: pandas.Timestamp
                    :param nip_end: end of a non-incident period
                    :type nip_end: pandas.Timestamp
                    :param stanox_section: STANOX section
                    :type stanox_section: str
                    :return: a list of statistics
                    :rtype: list

                    **Test**::

                        i = 1000

                        weather_cell_id = nip_data.WeatherCell.iloc[i]
                        nip_start = nip_data.StartDateTime.iloc[i]
                        nip_end = nip_data.EndDateTime.iloc[i]
                        stanox_section = nip_data.StanoxSection.iloc[i]
                    """

                    # Get non-IP Weather data about where and when the incident occurred
                    non_ip_weather_obs = self.METEx.query_weather_by_id_datetime(
                        weather_cell_id, nip_start, nip_end, pickle_it=False)

                    # Get all incident period data on the same section
                    overlaps = ip_data[
                        (ip_data.StanoxSection == stanox_section) &
                        (((ip_data.Critical_StartDateTime <= nip_start) & (
                                ip_data.Critical_EndDateTime >= nip_start)) |
                         ((ip_data.Critical_StartDateTime <= nip_end) & (
                                 ip_data.Critical_EndDateTime >= nip_end)))]

                    # Skip data of Weather causing Incidents at around the same time but
                    if not overlaps.empty:
                        non_ip_weather_obs = non_ip_weather_obs[
                            (non_ip_weather_obs.DateTime < np.min(overlaps.Critical_StartDateTime)) |
                            (non_ip_weather_obs.DateTime > np.max(overlaps.Critical_EndDateTime))]

                    # Get the max/min/avg Weather parameters for those incident periods
                    non_ip_weather_stats = self.calc_weather_stats(non_ip_weather_obs)

                    return non_ip_weather_stats

                # Get stats data for the specified "Non-Incident Periods"
                # noinspection PyTypeChecker
                nip_stats = nip_data.apply(
                    lambda x: get_non_ip_weather_stats(
                        x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime,
                        x.StanoxSection),
                    axis=1)
                nip_statistics = pd.DataFrame(
                    nip_stats.tolist(), index=nip_stats.index, columns=self.get_weather_variable_names())
                nip_statistics['Temperature_dif'] = \
                    nip_statistics.Temperature_max - nip_statistics.Temperature_min

                #
                nip_data = nip_data.join(nip_statistics.dropna(), how='inner')
                nip_data['IncidentReported'] = 0

                # Merge "ip_data" and "nip_data" into one DataFrame
                incident_location_weather = pd.concat([nip_data, ip_data], axis=0, ignore_index=True)

                if pickle_it:
                    save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))
                incident_location_weather = None

        return incident_location_weather

    def get_incident_location_vegetation(self, update=False, pickle_it=False, verbose=False):
        """
        Get vegetation conditions of incident locations.

        :param update: whether to do an update check, defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: vegetation conditions of incident locations
        :rtype: pandas.DataFrame or None

        .. note::

            Note that the "CoverPercent..." in ``furlong_vegetation_data`` has been amended
            when furlong_data was read. Check the function ``get_furlong_data()``.

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> incid_loc_vegetation = wind_attributed_incidents.get_incident_location_vegetation()

            >>> incid_loc_vegetation.tail()
                  Route            IMDM  ... TreeNumberUp CoverPercentVegetation
            915  Anglia  IMDM Tottenham  ...          783              31.377019
            916  Anglia  IMDM Tottenham  ...          783              31.377019
            917  Anglia  IMDM Tottenham  ...           35               0.441664
            918  Anglia  IMDM Tottenham  ...           34              19.756064
            919  Anglia  IMDM Tottenham  ...          321              16.048627
            [5 rows x 60 columns]
        """

        pickle_filename = make_filename(
            "vegetation", self.Route, None,
            self.ShiftYardsForSameELRs, self.ShiftYardsForDiffELRs, self.HazardsPercentile,
            save_as=".pickle")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_vegetation = load_pickle(path_to_pickle)

        else:
            try:
                """
                # Get data of furlong Vegetation coverage and hazardous trees
                from mssqlserver.vegetation import view_vegetation_condition_per_furlong
                furlong_vegetation_data = view_vegetation_condition_per_furlong()
                furlong_vegetation_data.set_index('FurlongID', inplace=True)
                """

                incident_location_furlongs = get_incident_location_furlongs(
                    route_name=self.Route, weather_category=None,
                    shift_yards_same_elr=self.ShiftYardsForSameELRs,
                    shift_yards_diff_elr=self.ShiftYardsForDiffELRs)
                incident_location_furlongs.dropna(inplace=True)

                # Compute Vegetation stats for each incident record
                # noinspection PyTypeChecker
                vegetation_statistics = incident_location_furlongs.apply(
                    lambda x: pd.Series(self.calc_vegetation_stats(
                        x.Critical_FurlongIDs, x.StartELR, x.EndELR, x.Section_Length_Adj)),
                    axis=1)

                vegetation_statistics.columns = sorted(
                    list(self.VegStatsCalc_.keys()) + ['TreeDensity', 'HazardTreeDensity'])
                veg_percent = [
                    x for x in self.CoverPercents if re.match('^CoverPercent*.[^Open|thr]', x)]
                vegetation_statistics['CoverPercentVegetation'] = \
                    vegetation_statistics[veg_percent].apply(np.sum, axis=1)

                hazard_others_pctl = [
                    ''.join([x, '_%s' % self.HazardsPercentile]) for x in self.HazardsOthers]
                rename_features = dict(zip(self.HazardsOthers, hazard_others_pctl))
                rename_features.update({'AssetNumber': 'AssetCount'})
                vegetation_statistics.rename(columns=rename_features, inplace=True)

                incident_location_vegetation = incident_location_furlongs.join(vegetation_statistics)

                if pickle_it:
                    save_pickle(incident_location_vegetation, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}.\" {}.".format(os.path.splitext(pickle_filename)[0], e))
                incident_location_vegetation = None

        return incident_location_vegetation

    def integrate_data(self, update=False, pickle_it=False, verbose=False):
        """
        Integrate the weather and vegetation conditions for incident locations.

        :param update: whether to do an update check, defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: integrated data set for modelling
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> integrated_data_set = wind_attributed_incidents.integrate_data()

            >>> integrated_data_set.tail()
                  FinancialYear  ... WindDirection_avg_[270, 360)
            3188           2018  ...                            0
            3189           2018  ...                            0
            3190           2018  ...                            0
            3191           2018  ...                            1
            3192           2018  ...                            1
            [5 rows x 103 columns]
        """

        pickle_filename = make_filename("dataset", self.Route, self.WeatherCategory,
                                        self.IP_StartHrs, self.IP_EndHrs, self.NIP_StartHrs,
                                        self.ShiftYardsForSameELRs, self.ShiftYardsForDiffELRs,
                                        self.HazardsPercentile)
        path_to_file = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_file) and not update:
            integrated_data = load_pickle(path_to_file)

        else:
            try:
                # Get information of Schedule 8 incident and the relevant weather conditions
                incident_location_weather = self.get_incident_location_weather()
                # Get information of vegetation conditions for the incident locations
                incident_location_vegetation = self.get_incident_location_vegetation()
                # incident_location_vegetation.drop(['IncidentCount', 'DelayCost', 'DelayMinutes'],
                #                                   axis=1, inplace=True)

                common_feats = list(
                    set(incident_location_weather.columns) & set(incident_location_vegetation.columns))
                integrated_weather_vegetation = pd.merge(
                    incident_location_weather, incident_location_vegetation,
                    how='inner', on=common_feats)

                # Electrified
                integrated_weather_vegetation.Electrified = \
                    integrated_weather_vegetation.Electrified.astype(int)

                # Categorise average wind directions into 4 quadrants
                wind_direction = pd.cut(
                    integrated_weather_vegetation.WindDirection_avg.values,
                    [0, 90, 180, 270, 360], right=False)

                integrated_data = integrated_weather_vegetation.join(
                    pd.DataFrame(wind_direction, columns=['WindDirection_avg_quadrant'])).join(
                    pd.get_dummies(wind_direction, prefix='WindDirection_avg'))

                if pickle_it:
                    save_pickle(integrated_data, path_to_file, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}".format(pickle_filename, e))
                integrated_data = None

        return integrated_data

    # == Model training ===============================================================================

    def prep_training_and_test_sets(self):
        """
        Further process the integrated data set and split it into a training set and a test set.

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> _, training_data, test_data = wind_attributed_incidents.prep_training_and_test_sets()
        """

        # Get the mdata for modelling
        integrated_dat = self.integrate_data()

        # Select season data: 'Spring', 'Summer', 'Autumn', 'Winter'
        integrated_data = get_data_by_season_(integrated_dat, self.Seasons,
                                              incident_datetime_col='StartDate')

        # Remove outliers
        if 95 <= self.OutlierPercentile <= 100:
            integrated_data = integrated_data[
                integrated_data.DelayMinutes <= np.percentile(integrated_data.DelayMinutes,
                                                              self.OutlierPercentile)]
            # from pyhelpers.misc import get_extreme_outlier_bounds
            # lo, up = get_extreme_outlier_bounds(mod_data.DelayMinutes, k=1.5)
            # mod_data = mod_data[mod_data.DelayMinutes.between(lo, up, inclusive=True)]

        # CoverPercent
        cover_percent_cols = [x for x in integrated_data.columns if re.match('^CoverPercent', x)]
        integrated_data.loc[:, cover_percent_cols] = integrated_data[cover_percent_cols] / 10.0
        integrated_data.loc[:, 'CoverPercentDiff'] = \
            integrated_data.CoverPercentVegetation - integrated_data.CoverPercentOpenSpace - \
            integrated_data.CoverPercentOther
        integrated_data.loc[:, 'CoverPercentDiff'] = \
            integrated_data.CoverPercentDiff * integrated_data.CoverPercentDiff.map(
                lambda x: 1 if x >= 0 else 0)

        # Scale down 'WindGust_max' and 'RelativeHumidity_max'
        integrated_data.loc[:, 'WindGust_max'] = integrated_data.WindGust_max / 10.0
        integrated_data.loc[:, 'RelativeHumidity_max'] = integrated_data.RelativeHumidity_max / 10.0

        # Select features
        explanatory_variables = self.ExplanatoryVariables.copy()
        explanatory_variables += [
            # 'HazardTreediameterM_%s' % hazard_pctl,
            # 'HazardTreeheightM_%s' % hazard_pctl,
            # 'HazardTreeprox3py_%s' % hazard_pctl,
            # 'HazardTreeproxrailM_%s' % hazard_pctl,
        ]

        # Add an intercept
        integrated_data['const'] = 1
        explanatory_variables = ['const'] + explanatory_variables

        # Set the outcomes of non-incident records to 0
        integrated_data.loc[
            integrated_data.IncidentReported == 0, ['DelayMinutes', 'DelayCost', 'IncidentCount']] = 0

        integrated_dat = integrated_data[['FinancialYear', 'IncidentReported'] + explanatory_variables]

        # Select data before 2014 as training data set, with the rest being test set
        train_set = integrated_data[integrated_dat.FinancialYear < 2014]
        test_set = integrated_data[integrated_dat.FinancialYear == 2014]

        return integrated_data, train_set, test_set

    def illustrate_explanatory_variables(self, save_as=".tif", dpi=None):
        """
        Describe basic statistics about the main explanatory variables.

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> wind_attributed_incidents.illustrate_explanatory_variables(save_as=None)
        """

        processed_data, _, _ = self.prep_training_and_test_sets()

        fig = plt.figure(figsize=(12, 5))
        colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

        ax1 = fig.add_subplot(161)
        processed_data['WindGust_max'].plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
        # train_set[['WindGust_max']].boxplot(column='WindGust_max', ax=ax1, boxprops=dict(color='k'))
        ax1.set_xticklabels('')
        plt.xlabel('Max. Gust', fontsize=13, labelpad=16)
        plt.ylabel('($\\times$10 mph)', fontsize=12, rotation=0)
        ax1.yaxis.set_label_coords(-0.1, 1.02)

        ax2 = fig.add_subplot(162)
        processed_data['WindDirection_avg_quadrant'].value_counts().sort_index().plot.bar(
            color='#4c76e1', rot=0, fontsize=12)
        plt.xlabel('Avg. Wind Direction', fontsize=13)
        plt.ylabel('No.', fontsize=12, rotation=0)
        ax2.set_xticklabels([1, 2, 3, 4])
        ax2.yaxis.set_label_coords(-0.1, 1.02)

        ax3 = fig.add_subplot(163)
        processed_data['Temperature_dif'].plot.box(color=colour, ax=ax3, widths=0.5, fontsize=12)
        ax3.set_xticklabels('')
        plt.xlabel('Temp. Diff.', fontsize=13, labelpad=16)
        plt.ylabel('(°C)', fontsize=12, rotation=0)
        ax3.yaxis.set_label_coords(-0.1, 1.02)

        ax4 = fig.add_subplot(164)
        processed_data['RelativeHumidity_max'].plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
        ax4.set_xticklabels('')
        plt.xlabel('Max. R.H.', fontsize=13, labelpad=16)
        plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
        ax4.yaxis.set_label_coords(-0.1, 1.02)

        ax5 = fig.add_subplot(165)
        processed_data['Snowfall_max'].plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
        ax5.set_xticklabels('')
        plt.xlabel('Max. Snowfall', fontsize=13, labelpad=16)
        plt.ylabel('(mm)', fontsize=12, rotation=0)
        ax5.yaxis.set_label_coords(-0.1, 1.02)

        ax6 = fig.add_subplot(166)
        processed_data['TotalPrecipitation_max'].plot.box(color=colour, ax=ax6, widths=0.5, fontsize=12)
        ax6.set_xticklabels('')
        plt.xlabel('Max. Total Precip.', fontsize=13, labelpad=16)
        plt.ylabel('(mm)', fontsize=12, rotation=0)
        ax6.yaxis.set_label_coords(-0.1, 1.02)

        plt.tight_layout()
        if save_as:
            path_to_file_weather = self.cdd("figures" + "weather_variables" + save_as)
            plt.savefig(path_to_file_weather, dpi=dpi)
            if save_as == ".svg":
                save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))

        #
        fig_veg = plt.figure(figsize=(12, 5))
        ax = fig_veg.add_subplot(111)
        colour_veg = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

        cover_percent_cols = [c for c in processed_data.columns if c.startswith('CoverPercent')
                              and not (c.endswith('Vegetation') or c.endswith('Diff'))]
        cover_percent_cols += [cover_percent_cols.pop(cover_percent_cols.index('CoverPercentOpenSpace'))]
        cover_percent_cols += [cover_percent_cols.pop(cover_percent_cols.index('CoverPercentOther'))]
        processed_data[cover_percent_cols].plot.box(color=colour_veg, ax=ax, widths=0.5, fontsize=12)
        # plt.boxplot([train_set[c] for c in cover_percent_cols])
        # plt.tick_params(axis='x', labelbottom='off')
        ax.set_xticklabels([re.search('(?<=CoverPercent).*', c).group() for c in cover_percent_cols],
                           rotation=45)
        plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
        ax.yaxis.set_label_coords(0, 1.02)

        plt.tight_layout()
        if save_as:
            path_to_file_veg = self.cdd("figures", "vegetation_variables" + save_as)
            plt.savefig(path_to_file_veg, dpi=dpi)
            if save_as == ".svg":
                save_svg_as_emf(path_to_file_veg, path_to_file_veg.replace(save_as, ".emf"))

    def logistic_regression_model(self, pickle_it=True, verbose=True):
        """

        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: defaults to ``True``
        :param verbose: bool or int
        :return: estimated model and relevant results
        :rtype: tuple

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> output = wind_attributed_incidents.logistic_regression_model()
        """

        _, train_set, test_set = self.prep_training_and_test_sets()

        self.__setattr__('TrainingSet', train_set)
        self.__setattr__('TestSet', test_set)

        try:
            np.random.seed(self.RandomSeed)
            if self.ModelType == 'logit':
                mod = sm_dcm.Logit(train_set.IncidentReported, train_set[self.ExplanatoryVariables])
            else:
                mod = sm_dcm.Probit(train_set.IncidentReported, train_set[self.ExplanatoryVariables])
            result_summary = mod.fit(method='newton', maxiter=1000, full_output=True, disp=False)

            if verbose:
                print(result_summary.summary2())

            # Odds ratios
            odds_ratios = pd.DataFrame(np.exp(result_summary.params), columns=['OddsRatio'])
            if verbose:
                print("\nOdds ratio:")
                print(odds_ratios)

            # Prediction
            test_set['incident_prob'] = result_summary.predict(test_set[self.ExplanatoryVariables])

            # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
            fpr, tpr, thr = metrics.roc_curve(test_set.IncidentReported, test_set.incident_prob)
            # Area under the curve (AUC)
            auc = metrics.auc(fpr, tpr)
            ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
            threshold = np.min(thr[ind])

            self.__setattr__('FPR', fpr)
            self.__setattr__('TPR', tpr)
            self.__setattr__('AUC', auc)
            self.__setattr__('Threshold', threshold)

            # prediction accuracy
            test_set['incident_prediction'] = test_set.incident_prob.apply(
                lambda x: 1 if x >= threshold else 0)
            test = pd.Series(test_set.IncidentReported == test_set.incident_prediction)
            mod_accuracy = np.divide(sum(test), len(test))
            if verbose:
                print("\nAccuracy: %f" % mod_accuracy)

            # incident prediction accuracy
            incid_only = test_set[test_set.IncidentReported == 1]
            test_acc = pd.Series(incid_only.IncidentReported == incid_only.incident_prediction)
            incid_accuracy = np.divide(sum(test_acc), len(test_acc))
            if verbose:
                print("Incident accuracy: %f" % incid_accuracy)

            # ===================================================================================
            # if dig_deeper:
            #     tmp_cols = explanatory_variables.copy()
            #     tmp_cols.remove('Electrified')
            #     tmps = [np.linspace(train_set[i].min(), train_set[i].max(), 5) for i in tmp_cols]
            #
            #     combos = pd.DataFrame(sklearn.utils.extmath.cartesian(tmps + [np.array([0, 1])]))
            #     combos.columns = explanatory_variables
            #
            #     combos['incident_prob'] = result.predict(combos[explanatory_variables])
            #
            #     def isolate_and_plot(var1='WindGust_max', var2='WindSpeed_max'):
            #         # isolate gre and class rank
            #         grouped = pd.pivot_table(
            #             combos, values=['incident_prob'], index=[var1, var2], aggfunc=np.mean)
            #
            #         # in case you're curious as to what this looks like
            #         # print grouped.head()
            #         #                      admit_pred
            #         # gre        prestige
            #         # 220.000000 1           0.282462
            #         #            2           0.169987
            #         #            3           0.096544
            #         #            4           0.079859
            #         # 284.444444 1           0.311718
            #
            #         # make a plot
            #         colors = 'rbgyrbgy'
            #
            #         lst = combos[var2].unique().tolist()
            #         plt.figure()
            #         for col in lst:
            #             plt_data = grouped.loc[grouped.index.get_level_values(1) == col]
            #             plt.plot(plt_data.index.get_level_values(0), plt_data.incident_prob,
            #                      color=colors[lst.index(col)])
            #
            #         plt.xlabel(var1)
            #         plt.ylabel("P(wind-related incident)")
            #         plt.legend(lst, loc='best', title=var2)
            #         title0 = 'Pr(wind-related incident) isolating '
            #         plt.title(title0 + var1 + ' and ' + var2)
            #         plt.show()
            #
            #     isolate_and_plot(var1='WindGust_max', var2='TreeNumberUp')
            # ====================================================================================

        except Exception as e:
            print(e)
            result_summary = e
            mod_accuracy, incid_accuracy, threshold = np.nan, np.nan, np.nan

        if pickle_it:
            repo = locals()
            var_names = ['train_set', 'test_set',
                         'result_summary', 'mod_accuracy', 'incid_accuracy', 'threshold']
            resources = {k: repo[k] for k in list(var_names)}
            result_pickle = make_filename("result", self.Route, self.WeatherCategory,
                                          self.IP_StartHrs, self.IP_EndHrs, self.NIP_StartHrs,
                                          self.ShiftYardsForSameELRs, self.ShiftYardsForDiffELRs,
                                          self.HazardsPercentile)

            save_pickle(resources, self.cdd_trial(result_pickle), verbose=verbose)

        return result_summary, mod_accuracy, incid_accuracy, threshold

    def plot_roc(self, save_as=".tif", dpi=600):
        """

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> _ = wind_attributed_incidents.logistic_regression_model(pickle_it=False)
            >>> wind_attributed_incidents.plot_roc(save_as=None)
        """

        fpr = self.__getattribute__('FPR')
        tpr = self.__getattribute__('TPR')
        auc = self.__getattribute__('AUC')

        plt.figure()

        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc, color='#6699cc', lw=2.5)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Random guess")

        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel("False positive rate", fontsize=14, fontweight='bold')
        plt.ylabel("True positive rate", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # plt.title('Receiver operating characteristic example')
        plt.legend(loc='lower right', fontsize=14)
        plt.fill_between(fpr, tpr, 0, color='#6699cc', alpha=0.2)

        # plt.subplots_adjust(left=0.10, bottom=0.1, right=0.96, top=0.96)
        plt.tight_layout()

        if save_as:
            path_to_file_roc = self.cdd_trial("ROC" + save_as)  # Fig. 6.
            plt.savefig(path_to_file_roc, dpi=dpi)
            if save_as == ".svg":
                save_svg_as_emf(path_to_file_roc,
                                path_to_file_roc.replace(save_as, ".emf"))  # Fig. 6.

    def plot_pred_likelihood(self, save_as=".tif", dpi=600):
        """

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> _ = wind_attributed_incidents.logistic_regression_model(pickle_it=False)
            >>> wind_attributed_incidents.plot_pred_likelihood(save_as=None)
        """

        test_set = self.__getattribute__('TestSet')
        threshold = self.__getattribute__('Threshold')

        # Plot incident delay minutes against predicted probabilities
        incid_ind = test_set.IncidentReported == 1

        plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        ax.scatter(test_set[incid_ind].incident_prob, test_set[incid_ind].DelayMinutes,
                   c='#db0101', edgecolors='k', marker='o', linewidths=2, s=80, alpha=.3,
                   label="Incidents")
        plt.axvline(x=threshold, label="Threshold = %.2f" % threshold, color='b')

        legend = plt.legend(scatterpoints=1, loc='best', fontsize=14, fancybox=True)
        frame = legend.get_frame()
        frame.set_edgecolor('k')

        plt.xlim(xmin=0, xmax=1.03)
        plt.ylim(ymin=-15)
        ax.set_xlabel("Predicted probability of incident occurrence (for 2014/15)", fontsize=14,
                      fontweight='bold')
        ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.tight_layout()

        if save_as:
            path_to_file_pred = cd_models("figures", self.TrialID, "predicted-likelihood" + save_as)
            plt.savefig(path_to_file_pred, dpi=dpi)  # Fig. 7.
            if save_as == ".svg":
                # Fig. 7.
                save_svg_as_emf(path_to_file_pred, path_to_file_pred.replace(save_as, ".emf"))

    def evaluate_prototype_model(self, pickle_each_run=False, verbose=True):
        """
        Evaluate the primer model given different settings.

        :return: summary of the evaluation results
        :rtype: pandas.DataFrame

        **Test**::

            >>> from modeller.prototype import WindAttributedIncidents

            >>> wind_attributed_incidents = WindAttributedIncidents(trial_id=0)

            >>> eval_summary = wind_attributed_incidents.evaluate_prototype_model()
        """

        start_time = time.time()

        params_set = extmath.cartesian((range(-12, -5, 3),
                                        range(6, 13, 3),
                                        range(-12, -5, 3),
                                        range(220, 880, 220),  # range(0, 440, 220),
                                        range(220, 880, 220),  # range(0, 440, 220),
                                        range(50, 75, 25)))  # range(25, 75, 25)

        total_no = len(params_set)

        results = []
        nobs = []
        mod_aic = []
        mod_bic = []
        mod_accuracy = []
        incid_accuracy = []
        msg = []
        train_sets = []
        test_sets = []
        thresholds = []

        if verbose:
            print("Evaluation starts ... ")

        counter = 0
        for params in params_set:
            counter += 1

            if verbose:
                print("\tParameter set {} / {}".format(counter, total_no), end=" ... ")

            (self.IP_StartHrs,
             self.IP_EndHrs,
             self.NIP_StartHrs,
             self.ShiftYardsForSameELRs,
             self.ShiftYardsForDiffELRs,
             self.HazardsPercentile) = params

            result, mod_acc, incid_acc, threshold = self.logistic_regression_model(
                pickle_it=pickle_each_run, verbose=False)

            train_set = self.__getattribute__('TrainingSet')
            test_set = self.__getattribute__('TestSet')

            train_sets.append(train_set)
            test_sets.append(test_set)
            results.append(result)
            mod_accuracy.append(mod_acc)
            incid_accuracy.append(incid_acc)
            thresholds.append(threshold)

            if isinstance(result, sm_dcm.BinaryResultsWrapper):
                if verbose:
                    print("Done.")
                nobs.append(result.nobs)
                mod_aic.append(result.aic)
                mod_bic.append(result.bic)
                msg.append(result.summary().extra_txt)
            else:
                if verbose:
                    print("Problems may occur given the parameter set {}: {}.".format(counter, params))
                nobs.append(len(train_set))
                mod_aic.append(np.nan)
                mod_bic.append(np.nan)
                msg.append(result.__str__())

        # Create a dataframe that summarises the test results
        columns = ['IP_StartHrs', 'IP_EndHrs', 'NIP_StartHrs',
                   'YardShift_same_ELR', 'YardShift_diff_ELR', 'HazardsPercentile', 'Obs_No',
                   'AIC', 'BIC', 'Threshold', 'PredAcc', 'PredAcc_Incid', 'Extra_Info']

        data = [list(x) for x in params_set.T]
        data += [nobs, mod_aic, mod_bic, thresholds, mod_accuracy, incid_accuracy, msg]

        evaluation_summary = pd.DataFrame(dict(zip(columns, data)), columns=columns)
        evaluation_summary.sort_values(
            ['PredAcc_Incid', 'PredAcc', 'AIC', 'BIC'], ascending=[False, False, True, True],
            inplace=True)

        save_pickle(results, self.cdd_trial("evaluation_results.pickle"), verbose=verbose)
        save_pickle(evaluation_summary, self.cdd_trial("evaluation_summary.pickle"), verbose=verbose)

        if verbose:
            print("Total elapsed time: %.2f hrs." % ((time.time() - start_time) / 3600))

        return evaluation_summary

    def view_trial_results(self, pickle_it=True):
        """
        View data.

        :return:
        :rtype:
        """

        result_pickle = make_filename(
            "result", self.Route, self.WeatherCategory, self.IP_StartHrs, self.IP_EndHrs,
            self.NIP_StartHrs, self.ShiftYardsForSameELRs, self.ShiftYardsForDiffELRs,
            self.HazardsPercentile)

        path_to_pickle = self.cdd_trial(result_pickle)

        if os.path.isfile(path_to_pickle):
            results = load_pickle(path_to_pickle)

        else:
            try:
                results = self.logistic_regression_model(pickle_it=pickle_it, verbose=True)
            except Exception as e:
                print(e)
                results = None

        return results


class HeatAttributedIncidents:

    def __init__(self, trial_id, model_type='logit', seed=0, in_seasons=None,
                 ip_start_hrs=-24, lp=-5, nip_start_hrs=-24,
                 shift_yards_same_elr=220, shift_yards_diff_elr=220,
                 hazard_pctl=50, outlier_pctl=99):
        self.Name = 'A prototype data model of predicting heat-related incidents.'

        self.TrialID = "{}".format(trial_id)

        self.METEx = METExLite()

        self.Route = 'Anglia'
        self.WeatherCategory = 'Heat'

        self.Seasons = in_seasons

        self.IP_StartHrs = ip_start_hrs
        self.LP = lp
        self.NIP_StartHrs = nip_start_hrs

        self.ShiftYardsForSameELRs = shift_yards_same_elr
        self.ShiftYardsForDiffELRs = shift_yards_diff_elr

        self.HazardsPercentile = hazard_pctl

        self.ExplanatoryVariables = [
            # 'WindSpeed_max',
            # 'WindSpeed_avg',
            'WindGust_max',
            # 'WindDirection_avg',
            # 'WindDirection_avg_[0, 90)',  # [0°, 90°)
            'WindDirection_avg_[90, 180)',  # [90°, 180°)
            'WindDirection_avg_[180, 270)',  # [180°, 270°)
            'WindDirection_avg_[270, 360)',  # [270°, 360°)
            'Temperature_dif',
            # 'Temperature_avg',
            # 'Temperature_max',
            # 'Temperature_min',
            'RelativeHumidity_max',
            'Snowfall_max',
            'TotalPrecipitation_max',
            # 'Electrified',
            'CoverPercentAlder',
            'CoverPercentAsh',
            'CoverPercentBeech',
            'CoverPercentBirch',
            'CoverPercentConifer',
            'CoverPercentElm',
            'CoverPercentHorseChestnut',
            'CoverPercentLime',
            'CoverPercentOak',
            'CoverPercentPoplar',
            'CoverPercentShrub',
            'CoverPercentSweetChestnut',
            'CoverPercentSycamore',
            'CoverPercentWillow',
            # 'CoverPercentOpenSpace',
            'CoverPercentOther',
            # 'CoverPercentVegetation',
            # 'CoverPercentDiff',
            # 'TreeDensity',
            # 'TreeNumber',
            # 'TreeNumberDown',
            # 'TreeNumberUp',
            # 'HazardTreeDensity',
            # 'HazardTreeNumber',
            # 'HazardTreediameterM_max',
            # 'HazardTreediameterM_min',
            # 'HazardTreeheightM_max',
            # 'HazardTreeheightM_min',
            # 'HazardTreeprox3py_max',
            # 'HazardTreeprox3py_min',
            # 'HazardTreeproxrailM_max',
            # 'HazardTreeproxrailM_min'
        ]

        self.OutlierPercentile = outlier_pctl

        self.RandomSeed = seed
        self.ModelType = model_type

        self.WeatherStatsCalc = {'Temperature': (np.nanmax, np.nanmin, np.nanmean),
                                 'RelativeHumidity': (np.nanmax, np.nanmin, np.nanmean),
                                 'WindSpeed': np.nanmax,
                                 'WindGust': np.nanmax,
                                 'Snowfall': (np.nanmax, np.nanmin, np.nanmean),
                                 'TotalPrecipitation': (np.nanmax, np.nanmin, np.nanmean)}

        # Get incident_location_furlongs
        self.Furlongs = get_furlongs_data(route_name=self.Route, weather_category=None,
                                          shift_yards_same_elr=self.ShiftYardsForSameELRs,
                                          shift_yards_diff_elr=self.ShiftYardsForDiffELRs)

        def specify_veg_stats_calc():
            """
            Specify the statistics that need to be computed.
            """

            features = self.Furlongs.columns

            # "CoverPercent..."
            cover_percents = [x for x in features if re.match('^CoverPercent[A-Z]', x)]
            veg_stats_calc = dict(zip(cover_percents, [np.nansum] * len(cover_percents)))
            veg_stats_calc.update({'AssetNumber': np.count_nonzero,
                                   'TreeNumber': np.nansum,
                                   'TreeNumberUp': np.nansum,
                                   'TreeNumberDown': np.nansum,
                                   'Electrified': np.any,
                                   'DateOfMeasure': lambda x: tuple(x),
                                   # 'AssetDesc1': np.all,
                                   # 'IncidentReported': np.any
                                   'HazardTreeNumber':
                                       lambda x: np.nan if np.isnan(x).all() else np.nansum(x)})

            # variables for hazardous trees
            hazard_min = [x for x in features if re.match('^HazardTree.*min$', x)]
            hazard_max = [x for x in features if re.match('^HazardTree.*max$', x)]
            hazard_others = [x for x in features if re.match('^HazardTree[a-z]((?!_).)*$', x)]
            # Computations for hazardous trees variables
            hazard_calc = [dict(zip(hazard_others, [lambda x: tuple(x)] * len(hazard_others))),
                           dict(zip(hazard_min, [np.min] * len(hazard_min))),
                           dict(zip(hazard_max, [np.max] * len(hazard_max)))]

            # Update vegetation_stats_computations
            veg_stats_calc.update({k: v for d in hazard_calc for k, v in d.items()})

            return cover_percents, hazard_others, veg_stats_calc

        self.CoverPercents, self.HazardsOthers, self.VegStatsCalc_ = specify_veg_stats_calc()

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype\\heat" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype\\heat" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype import HeatAttributedIncidents

            >>> heat_attributed_incidents = HeatAttributedIncidents(trial_id=0)

            >>> os.path.relpath(heat_attributed_incidents.cdd())
            'models\\prototype\\heat'
        """

        path = cd_models("prototype", "heat", *sub_dir, mkdir=mkdir)

        return path

    def cdd_trial(self, *sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype\\heat\\<trial_id>" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype\\heat\\<trial_id>" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype import HeatAttributedIncidents

            >>> heat_attributed_incidents = HeatAttributedIncidents(trial_id=0)

            >>> os.path.relpath(heat_attributed_incidents.cdd_trial())
            'models\\prototype\\heat\\0'
        """

        path = self.cdd(self.TrialID, *sub_dir, mkdir=mkdir)

        return path

    def get_weather_variable_names(self, temperature_dif=False, supplement=None):
        """
        Get weather variable names.

        :param temperature_dif: whether to include 'Temperature_dif', defaults to ``False``
        :type temperature_dif: bool
        :param supplement: e.g. 'Hottest_Heretofore'
        :type supplement: str, list, None
        :return: a list of names of weather variables
        :rtype: list
        """

        weather_var_names = []
        for k, v in self.WeatherStatsCalc.items():
            if isinstance(v, tuple):
                for v_ in v:
                    weather_var_names.append('_'.join([
                        k, v_.__name__.replace('mean', 'avg').replace('median', 'med')]).replace(
                        '_nan', '_'))

            else:
                weather_var_names.append('_'.join([
                    k, v.__name__.replace('mean', 'avg').replace('median', 'med')]).replace('_nan', '_'))

        if temperature_dif:
            weather_var_names.insert(weather_var_names.index('Temperature_min') + 1, 'Temperature_dif')

        if supplement:
            if isinstance(supplement, str):
                supplement = [supplement]
            wind_variable_names = weather_var_names + ['WindSpeed_avg', 'WindDirection_avg'] + supplement

        else:
            wind_variable_names = weather_var_names + ['WindSpeed_avg', 'WindDirection_avg']

        return wind_variable_names

    # == Calculators ==================================================================================

    @staticmethod
    def calc_average_wind(wind_speeds, wind_directions):
        """
        Calculate average wind speed and direction.

        :param wind_speeds: wind speed
        :type wind_speeds: float or int
        :param wind_directions: wind direction
        :type wind_directions: float or int
        :return: average wind speed and average wind direction
        :rtype: tuple
        """

        u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
        v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
        uav, vav = np.nanmean(u), np.nanmean(v)  # sum up all u and v values and average it

        average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed

        # Calculate average wind direction
        if uav == 0:
            average_wind_direction = 0 if vav == 0 else (360 if vav > 0 else 180)
        else:
            average_wind_direction = (270 if uav > 0 else 90) - 180 / np.pi * np.arctan(vav / uav)

        return average_wind_speed, average_wind_direction

    def calc_weather_stats(self, weather_obs):
        """
        Compute the statistics for all the Weather variables (except wind).

        :param weather_obs: observed data of weather conditions
        :type weather_obs: pandas.DataFrame
        :return: statistics for weather conditions
        :rtype: list

        .. note::

            Note: to get the n-th percentile, use percentile(n)

            This function also returns the Weather dataframe indices.
            The corresponding Weather conditions in that WeatherCell might cause wind-related Incidents.
        """

        if not weather_obs.empty:
            # Calculate the statistics
            weather_obs.fillna(value=np.nan, inplace=True)
            stats = weather_obs.fillna(np.nan).groupby('WeatherCell').aggregate(self.WeatherStatsCalc)
            stats['WindSpeed_avg'], stats['WindDirection_avg'] = \
                self.calc_average_wind(weather_obs.WindSpeed, weather_obs.WindDirection)

            weather_stats = stats.values[0].tolist()  # + [weather_obs.index.tolist()]

        else:
            weather_stats = [np.nan] * 10  # + [[None]]

        return weather_stats

    @staticmethod
    def calc_overall_cover_percent_old(start_and_end_cover_percents, total_yards_adjusted):
        """
        Calculate the cover percents across two neighbouring ELRs.

        :param start_and_end_cover_percents: vegetation cover percents of a start and an end ELR
        :type start_and_end_cover_percents: tuple
        :param total_yards_adjusted: adjusted total yards
        :type total_yards_adjusted: tuple
        :return: overall vegetation cover percent across two neighbouring ELRs
        :rtype: float or int
        """

        # (start * end) / (start + end)
        multiplier = pd.np.prod(total_yards_adjusted) / pd.np.sum(total_yards_adjusted)
        # 1/start, 1/end
        cp_start, cp_end = start_and_end_cover_percents
        s_, e_ = pd.np.divide(1, total_yards_adjusted)
        # numerator
        n = e_ * cp_start + s_ * cp_end
        # denominator
        d = pd.np.sum(start_and_end_cover_percents) if pd.np.all(start_and_end_cover_percents) else 1

        f = multiplier * pd.np.divide(n, d)

        overall_cover_percent = f * d

        return overall_cover_percent

    def calc_vegetation_stats(self, furlong_ids, start_elr, end_elr, total_yards_adjusted):
        """
        Calculate stats of vegetation variables for each incident record

        **Test**::

            i = 337

            furlong_ids = incident_location_furlongs.loc[i, 'Critical_FurlongIDs']
            start_elr = incident_location_furlongs.loc[i, 'StartELR']
            end_elr = incident_location_furlongs.loc[i, 'EndELR']
            total_yards_adjusted = incident_location_furlongs.loc[i, 'Section_Length_Adj']

        Note: to get the n-th percentile may use percentile(n)

        """

        # Get all column names as features
        veg_feats = self.Furlongs.columns

        # Get features which would be filled with "0" and "inf", respectively
        fill_0 = [x for x in veg_feats if re.match('.*height', x)] + ['HazardTreeNumber']
        fill_inf = [x for x in veg_feats if re.match('^.*prox|.*diam', x)]

        furlong_ids_ = [fid for fid in furlong_ids if fid in self.Furlongs.index]

        if not furlong_ids_:
            veg_stats = list(np.empty(len(self.VegStatsCalc_) + 2) * np.nan)

        else:
            vegetation_data = self.Furlongs.loc[furlong_ids_]

            veg_stats = vegetation_data.groupby('ELR').aggregate(self.VegStatsCalc_)
            veg_stats[self.CoverPercents] = \
                veg_stats[self.CoverPercents].div(veg_stats.AssetNumber, axis=0).values

            if start_elr == end_elr:
                elr = veg_stats.index[0]

                if np.isnan(veg_stats.HazardTreeNumber[elr]):
                    veg_stats[fill_0] = 0.0
                    veg_stats[fill_inf] = 999999.0
                else:
                    assert 0 <= self.HazardsPercentile <= 100

                    def calc_percentile(x):
                        temp = tuple(itertools.chain(*pd.Series(x).dropna()))
                        if not temp:
                            pctl = np.nan
                        else:
                            pctl = np.nanpercentile(temp, self.HazardsPercentile)
                        return pctl

                    veg_stats[self.HazardsOthers] = \
                        veg_stats[self.HazardsOthers].applymap(calc_percentile)
                    # lambda x: np.nanpercentile(
                    #     tuple(itertools.chain(*pd.Series(x).dropna())), self.HazardsPercentile))

            else:
                if np.all(np.isnan(veg_stats.HazardTreeNumber.values)):
                    veg_stats[fill_0] = 0.0
                    veg_stats[fill_inf] = 999999.0
                    calc_further = {k: lambda y: np.nanmean(y) for k in self.HazardsOthers}
                else:
                    veg_stats[self.HazardsOthers] = veg_stats[self.HazardsOthers].applymap(
                        lambda y: tuple(itertools.chain(*pd.Series(y).dropna())))
                    hazard_others_func = [
                        lambda y: np.nanpercentile(np.sum(y), self.HazardsPercentile)]
                    calc_further = dict(
                        zip(self.HazardsOthers, hazard_others_func * len(self.HazardsOthers)))

                # Specify further calculations
                calc_further.update({'AssetNumber': np.sum})
                calc_further.update(dict(DateOfMeasure=lambda y: tuple(itertools.chain(*y))))
                calc_further.update({k: lambda y: tuple(y) for k in self.CoverPercents})

                # noinspection PyAttributeOutsideInit
                self.VegStatsCalc = self.VegStatsCalc_.copy()

                self.VegStatsCalc.update(calc_further)

                # Rename index (by which the dataframe can be grouped)
                veg_stats.index = pd.Index(
                    data=['-'.join(set(veg_stats.index))] * len(veg_stats.index), name='ELR')
                veg_stats = veg_stats.groupby(veg_stats.index).aggregate(self.VegStatsCalc)

                if isinstance(total_yards_adjusted, tuple):
                    # if (len(total_yards_adjusted) == 3) and
                    if total_yards_adjusted[1] == 0 or np.isnan(total_yards_adjusted[1]):
                        total_yards_adjusted = total_yards_adjusted[:1] + total_yards_adjusted[2:]

                def calc_cp(x):
                    cp = np.dot(x, total_yards_adjusted) / np.nansum(total_yards_adjusted)
                    if isinstance(cp, (np.ndarray, tuple, list)):
                        cp = cp[0]
                    return cp

                veg_stats[self.CoverPercents] = veg_stats[self.CoverPercents].applymap(calc_cp)

            # Calculate tree densities (number of trees per furlong)
            veg_stats['TreeDensity'] = veg_stats.TreeNumber.div(
                np.nansum(total_yards_adjusted) / 220.0)
            veg_stats['HazardTreeDensity'] = veg_stats.HazardTreeNumber.div(
                np.nansum(total_yards_adjusted) / 220.0)

            # Rearrange the order of features
            veg_stats = veg_stats[sorted(veg_stats.columns)].values[0].tolist()

        return veg_stats

    # == Data integration =============================================================================

    def get_incident_location_weather(self, update=False, pickle_it=False, verbose=False):
        """
        Get TRUST data and the weather conditions for each incident location.

        :param update: whether to do an update check, defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :return: weather conditions of incident locations
        :rtype: pandas.DataFrame or None

        **Test**::

            >>> from modeller.prototype import HeatAttributedIncidents

            >>> heat_attributed_incidents = HeatAttributedIncidents(trial_id=0)

            >>> incid_loc_weather = heat_attributed_incidents.get_incident_location_weather()

            >>> incid_loc_weather.tail()
                  FinancialYear  ... Temperature_max ≥ 30°C
            1807           2018  ...                      0
            1808           2018  ...                      0
            1809           2018  ...                      0
            1810           2018  ...                      0
            1811           2018  ...                      0
            [5 rows x 73 columns]
        """

        pickle_filename = make_filename(
            "weather", self.Route, self.WeatherCategory, self.IP_StartHrs, self.LP, self.NIP_StartHrs,
            save_as=".pickle")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_weather = load_pickle(path_to_pickle)

        else:
            try:
                # Getting incident data for all incident locations
                incidents = self.METEx.view_schedule8_costs_by_datetime_location_reason(
                    self.Route, self.WeatherCategory)
                # Drop non-Weather-related incident records
                if self.WeatherCategory is None:
                    incidents = incidents[incidents.WeatherCategory != '']
                # Get data for the specified "Incident Periods"
                incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
                incidents['Critical_StartDateTime'] = \
                    incidents.StartDateTime.apply(datetime_truncate.truncate_hour) + pd.Timedelta(
                        hours=self.IP_StartHrs)
                incidents['Critical_EndDateTime'] = incidents.StartDateTime
                incidents['Critical_Period'] = \
                    incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

                if incidents.WeatherCell.dtype != 'int64':
                    # Rectify the records for which Weather cell id is empty
                    weather_cell = self.METEx.get_weather_cell()
                    ll = [shapely.geometry.Point(xy) for xy in
                          zip(weather_cell.ll_Longitude, weather_cell.ll_Latitude)]
                    ul = [shapely.geometry.Point(xy) for xy in
                          zip(weather_cell.ul_Longitude, weather_cell.ul_Latitude)]
                    ur = [shapely.geometry.Point(xy) for xy in
                          zip(weather_cell.ur_Longitude, weather_cell.ur_Latitude)]
                    lr = [shapely.geometry.Point(xy) for xy in
                          zip(weather_cell.lr_Longitude, weather_cell.lr_Latitude)]
                    poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]
                    cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in
                             range(len(weather_cell))]

                    for i in incidents[incidents.WeatherCell == ''].index:
                        pt = shapely.geometry.Point(incidents.StartLongitude.loc[i],
                                                    incidents.StartLatitude.loc[i])
                        id_set = set(
                            weather_cell.iloc[
                                [i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
                        if len(id_set) == 0:
                            pt_alt = shapely.geometry.Point(incidents.EndLongitude.loc[i],
                                                            incidents.EndLatitude.loc[i])
                            id_set = set(
                                weather_cell.iloc[
                                    [i for i, p in enumerate(cells)
                                     if pt_alt.within(p)]].WeatherCellId.tolist())
                        if len(id_set) != 0:
                            incidents.loc[i, 'WeatherCell'] = list(id_set)[0]

                def get_ip_weather_stats(weather_cell_id, ip_start, ip_end):
                    """
                    Processing weather data for IP.
                    (Get data of weather conditions that led to Incidents for each record.)

                    :param weather_cell_id: weather cell ID
                    :type weather_cell_id: int
                    :param ip_start: start of an incident period
                    :type ip_start: pandas.Timestamp
                    :param ip_end: end of an incident period
                    :type ip_end: pandas.Timestamp
                    :return: a list of statistics
                    :rtype: list

                    **Test**::

                        i = 100

                        weather_cell_id = incidents.WeatherCell.iloc[i]
                        ip_start = incidents.StartDateTime.iloc[i]
                        ip_end = incidents.EndDateTime.iloc[i]

                        get_ip_weather_stats(weather_cell_id, ip_start, ip_end)
                    """

                    # Get Weather data about where and when the incident occurred
                    ip_weather_obs = self.METEx.query_weather_by_id_datetime(
                        weather_cell_id, ip_start, ip_end, pickle_it=False)

                    # Get the max/min/avg Weather parameters for those incident periods
                    weather_stats_data = self.calc_weather_stats(ip_weather_obs)

                    return weather_stats_data

                # Get data for the specified IP
                # noinspection PyTypeChecker
                ip_stats = incidents.apply(
                    lambda x: get_ip_weather_stats(
                        x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime),
                    axis=1)

                ip_statistics = pd.DataFrame(
                    ip_stats.to_list(), index=ip_stats.index, columns=self.get_weather_variable_names())
                ip_statistics['Temperature_dif'] = \
                    ip_statistics.Temperature_max - ip_statistics.Temperature_min

                ip_data = incidents.join(ip_statistics.dropna(), how='inner')
                ip_data['IncidentReported'] = 1

                # Processing Weather data for non-IP -
                # Get data of Weather which did not cause Incidents for each record
                nip_data = incidents.copy(deep=True)
                nip_data.Critical_EndDateTime = \
                    nip_data.Critical_StartDateTime + pd.Timedelta(days=self.LP)
                nip_data.Critical_StartDateTime = \
                    nip_data.Critical_EndDateTime + pd.Timedelta(hours=self.NIP_StartHrs)
                nip_data.Critical_Period = \
                    nip_data.Critical_EndDateTime - nip_data.Critical_StartDateTime

                def get_non_ip_weather_stats(weather_cell_id, nip_start, nip_end, stanox_section):
                    """
                    Processing weather data for non-IP.
                    (Get data of weather conditions that were less likely to lead to incidents.)

                    :param weather_cell_id: weather cell ID
                    :type weather_cell_id: int
                    :param nip_start: start of a non-incident period
                    :type nip_start: pandas.Timestamp
                    :param nip_end: end of a non-incident period
                    :type nip_end: pandas.Timestamp
                    :param stanox_section: STANOX section
                    :type stanox_section: str
                    :return: a list of statistics
                    :rtype: list

                    **Test**::

                        i = 100

                        weather_cell_id = nip_data.WeatherCell.iloc[i]
                        nip_start = nip_data.StartDateTime.iloc[i]
                        nip_end = nip_data.EndDateTime.iloc[i]
                        stanox_section = nip_data.StanoxSection.iloc[i]

                        get_non_ip_weather_stats(weather_cell_id, nip_start, nip_end, stanox_section)
                    """

                    # Get non-IP Weather data about where and when the incident occurred
                    non_ip_weather_obs = self.METEx.query_weather_by_id_datetime(
                        weather_cell_id, nip_start, nip_end, pickle_it=False)

                    # Get all incident period data on the same section
                    overlaps = ip_data[
                        (ip_data.StanoxSection == stanox_section) &
                        (((ip_data.Critical_StartDateTime <= nip_start) & (
                                    ip_data.Critical_EndDateTime >= nip_start)) |
                         ((ip_data.Critical_StartDateTime <= nip_end) & (
                                     ip_data.Critical_EndDateTime >= nip_end)))]

                    # Skip data of Weather causing Incidents at around the same time; but
                    if not overlaps.empty:
                        non_ip_weather_obs = non_ip_weather_obs[
                            (non_ip_weather_obs.DateTime < np.min(overlaps.Critical_StartDateTime)) |
                            (non_ip_weather_obs.DateTime > np.max(overlaps.Critical_EndDateTime))]

                    # Get the max/min/avg Weather parameters for those incident periods
                    non_ip_weather_stats = self.calc_weather_stats(non_ip_weather_obs)

                    return non_ip_weather_stats

                # Get stats data for the specified "Non-Incident Periods"
                # noinspection PyTypeChecker
                nip_stats = nip_data.apply(
                    lambda x: get_non_ip_weather_stats(
                        x.WeatherCell, x.Critical_StartDateTime, x.Critical_EndDateTime,
                        x.StanoxSection),
                    axis=1)

                nip_statistics = pd.DataFrame(
                    nip_stats.tolist(), index=nip_stats.index, columns=self.get_weather_variable_names())
                nip_statistics['Temperature_dif'] = \
                    nip_statistics.Temperature_max - nip_statistics.Temperature_min

                nip_data = nip_data.join(nip_statistics.dropna(), how='inner')
                nip_data['IncidentReported'] = 0

                # Merge "ip_data" and "nip_data" into one DataFrame
                incident_location_weather = pd.concat(
                    [nip_data, ip_data], axis=0, ignore_index=True, sort=False)

                # Categorise average wind directions into 4 quadrants
                wind_direction = pd.cut(
                    incident_location_weather.WindDirection_avg.values,
                    bins=[0, 90, 180, 270, 360], right=False)
                incident_location_weather = incident_location_weather.join(
                    pd.DataFrame(wind_direction, columns=['WindDirection_avg_quadrant'])).join(
                    pd.get_dummies(wind_direction, prefix='WindDirection_avg'))

                # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
                track_orientation = categorise_track_orientations(incident_location_weather)
                incident_location_weather = incident_location_weather.join(track_orientation)

                # Categorise temperature: < 24, = 24, 25, 26, 27, 28, 29, >= 30
                labels = \
                    ['Temperature_max < 24°C'] + \
                    ['Temperature_max {} {}°C'.format('≥' if x >= 30 else '=', x) for x in range(24, 31)]
                temperature_category = pd.cut(
                    incident_location_weather.Temperature_max.values,
                    bins=[-np.inf] + list(range(24, 31)) + [np.inf], right=False, labels=labels)
                incident_location_weather = incident_location_weather.join(
                    pd.DataFrame(temperature_category, columns=['Temperature_Category'])).join(
                    pd.get_dummies(temperature_category))

                if pickle_it:
                    save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get \"{}\". {}.".format(os.path.splitext(pickle_filename)[0], e))
                incident_location_weather = None

        return incident_location_weather
