""" A prediction model of heat-related rail incidents (based on the prototype). """

import datetime
import itertools
import os

import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import statsmodels.discrete.discrete_model as sm_dcm
import statsmodels.tools as sm_tools
from pyhelpers import get_geometric_midpoint, wgs84_to_osgb36
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_pickle, save_fig, save_pickle
from sklearn import metrics

from preprocessor import METExLite, MIDAS, UKCP09
from utils import categorise_temperatures, categorise_track_orientations, cd_models, \
    get_data_by_season, make_filename


class HeatAttributedIncidentsPlus:

    def __init__(self, trial_id, route_name=None, season='summer',
                 prior_ip_start_hrs=-24, non_ip_start_hrs=-24, trial_only=True, outlier_pctl=100,
                 model_type='logit'):

        pd_preferences()
        mpl_preferences(font_name='Times New Roman')

        self.Name = ''

        self.TrialID = "{}".format(trial_id)

        self.METEx = METExLite(database_name='NR_METEx_20190203')
        self.UKCP = UKCP09()
        self.MIDAS = MIDAS()

        self.Route = route_name
        self.WeatherCategory = 'Heat'
        self.Season = season
        self.PIP_StartHrs = prior_ip_start_hrs
        self.NIP_StartHrs = non_ip_start_hrs

        self.LP_Anglia = lambda x: -20 if x in range(24, 29) else (-13 if x > 28 else 0)
        self.LP_Wessex = lambda x: -30 if x in range(24, 29) else (-25 if x > 28 else 0)
        self.LP_NE = lambda x: -18 if x in range(24, 29) else (-16 if x > 28 else 0)
        self.LP_Wales = lambda x: -19 if x in range(24, 29) else (-5 if x > 28 else 0)

        self.OnlyTrial = trial_only

        self.WeatherStatsCalc = {'Maximum_Temperature': (max, min, np.average),
                                 'Minimum_Temperature': (max, min, np.average),
                                 'Temperature_Change': np.average,
                                 'Precipitation': (max, min, np.average)}

        self.RADTOBStatsCalc = {'GLBL_IRAD_AMT': sum}

        self.ExplanatoryVariables = [
            # 'Maximum_Temperature_max',
            # 'Maximum_Temperature_min',
            # 'Maximum_Temperature_average',
            # 'Minimum_Temperature_max',
            # 'Minimum_Temperature_min',
            # 'Minimum_Temperature_average',
            # 'Temperature_Change_average',
            'Precipitation_max',
            # 'Precipitation_min',
            # 'Precipitation_average',
            'Hottest_Heretofore',
            'Temperature_Change_max',
            # 'Temperature_Change_min',
            'GLBL_IRAD_AMT_total',
            # 'Track_Orientation_E_W',
            'Track_Orientation_NE_SW',
            'Track_Orientation_NW_SE',
            'Track_Orientation_N_S',
            # 'Maximum_Temperature_max [-inf, 24.0)°C',
            'Maximum_Temperature_max [24.0, 25.0)°C',
            'Maximum_Temperature_max [25.0, 26.0)°C',
            'Maximum_Temperature_max [26.0, 27.0)°C',
            'Maximum_Temperature_max [27.0, 28.0)°C',
            'Maximum_Temperature_max [28.0, 29.0)°C',
            'Maximum_Temperature_max [29.0, 30.0)°C',
            'Maximum_Temperature_max [30.0, inf)°C'
        ]

        self.OutlierPercentile = outlier_pctl
        self.ModelType = model_type

    def cdd(self, *sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype_ext\\heat" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype_ext\\heat" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_plus = HeatAttributedIncidentsPlus(trial_id=0)

            >>> os.path.relpath(h_plus.cdd())
            'models\\prototype_ext\\heat'
        """

        path = cd_models("prototype_ext", self.WeatherCategory.lower(), *sub_dir, mkdir=mkdir)

        return path

    def cdd_trial(self, *sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype_ext\\heat\\<trial_id>" and sub-directories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype_ext\\heat\\<trial_id>" and sub-directories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_plus = HeatAttributedIncidentsPlus(trial_id=0)

            >>> os.path.relpath(h_plus.cdd_trial())
            'models\\prototype_ext\\heat\\0'
        """

        path = self.cdd(self.TrialID, *sub_dir, mkdir=mkdir)

        return path

    # == Set Prior-IP, LP and Non-IP ==================================================================

    def set_prior_ip(self, incidents):
        """

        :param incidents:
        :return:
        """

        incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
        # End date and time of the prior IP
        incidents['Critical_EndDateTime'] = incidents.StartDateTime.dt.round('H')
        # Start date and time of the prior IP
        critical_start_dt = incidents.Critical_EndDateTime.map(
            lambda x: x + pd.Timedelta(
                hours=self.PIP_StartHrs if x.time() > datetime.time(9) else self.PIP_StartHrs * 2))
        incidents.insert(incidents.columns.get_loc('Critical_EndDateTime'), 'Critical_StartDateTime',
                         critical_start_dt)
        # Prior-IP dates of each incident
        incidents['Critical_Period'] = incidents.apply(
            lambda x: pd.interval_range(x.Critical_StartDateTime, x.Critical_EndDateTime), axis=1)

        return incidents

    def set_lp_and_non_ip(self, route_name, ip_max_temp_max, ip_start_dt):
        """
        Determine latent period for a given date/time and the maximum temperature.

        :param route_name:
        :param ip_max_temp_max:
        :param ip_start_dt:
        :return:
        """

        if route_name == 'Anglia':
            lp = self.LP_Anglia(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -20
            # elif ip_max_temp_max > 28:
            #     lp = -13
            # else:
            #     lp = 0
        elif route_name == 'Wessex':
            lp = self.LP_Wessex(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -30
            # elif ip_max_temp_max > 28:
            #     lp = -25
            # else:
            #     lp = 0
        elif route_name == 'North and East':
            lp = self.LP_NE(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -18
            # elif ip_max_temp_max > 28:
            #     lp = -16
            # else:
            #     lp = 0
        else:  # route_name == 'Wales':
            lp = self.LP_Wales(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -19
            # elif ip_max_temp_max > 28:
            #     lp = -5
            # else:
            #     lp = 0

        critical_end_dt = ip_start_dt + datetime.timedelta(days=lp)
        critical_start_dt = critical_end_dt + datetime.timedelta(hours=self.NIP_StartHrs)
        critical_period = pd.interval_range(critical_start_dt, critical_end_dt)

        return critical_start_dt, critical_end_dt, critical_period

    def get_non_ip_data(self, incidents, prior_ip_data):
        """

        :param incidents:
        :param prior_ip_data:
        :return:
        """

        non_ip_data = incidents.copy(deep=True)  # Get weather data that did not cause any incident

        col_names = ['Critical_StartDateTime', 'Critical_EndDateTime', 'Critical_Period']
        non_ip_data[col_names] = prior_ip_data.apply(
            lambda x: pd.Series(
                self.set_lp_and_non_ip(
                    x.Route, x.Maximum_Temperature_max, x.Critical_StartDateTime)),
            axis=1)

        return non_ip_data

    # == Integrators ==================================================================================

    def specify_weather_variable_names(self):
        """
        Get all weather variable names.

        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_plus = HeatAttributedIncidentsPlus(trial_id=0)

            >>> h_plus.specify_weather_variable_names()
        """

        var_stats_names = [
            [k, [i.__name__ for i in v] if isinstance(v, tuple) else [v.__name__]]
            for k, v in self.WeatherStatsCalc.items()]

        weather_variable_names = [['_'.join([x, z]) for z in y] for x, y in var_stats_names]

        weather_variable_names = list(itertools.chain.from_iterable(weather_variable_names))

        return weather_variable_names

    def calculate_weather_stats(self, weather_data):
        """
        Calculate the statistics for the Weather variables (except radiation).

        :param weather_data:
        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_plus = HeatAttributedIncidentsPlus(trial_id=0)

            # >>> h_plus.calculate_radtob_variables_stats()
        """

        if weather_data.empty:
            weather_stats_info = [np.nan] * sum(map(np.count_nonzero, self.WeatherStatsCalc.values()))

        else:
            # Create a pseudo id for groupby() & aggregate()
            weather_data['Pseudo_ID'] = 0
            weather_stats = weather_data.groupby('Pseudo_ID').aggregate(self.WeatherStatsCalc)
            # a, b = [list(x) for x in weather_stats.columns.levels]
            # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
            # if not weather_stats.empty:
            #     stats_info = weather_stats.values[0].tolist()
            # else:
            #     stats_info = [np.nan] * len(weather_stats.columns)

            weather_stats_info = weather_stats.values[0].tolist()

        return weather_stats_info

    def calculate_radtob_variables_stats(self, midas_radtob):
        """
        Calculate the statistics for the radiation variables.

        :param midas_radtob:
        :return:

        **Test**::

            midas_radtob = prior_ip_radtob.copy()
        """

        # Solar irradiation amount (Kjoules/ sq metre over the observation period)
        if midas_radtob.empty:
            stats_info = [np.nan] * (sum(map(np.count_nonzero, self.RADTOBStatsCalc.values())))

        else:
            # if 24 not in midas_radtob.OB_HOUR_COUNT:
            #     midas_radtob = midas_radtob.append(midas_radtob.iloc[-1, :])
            #     midas_radtob.VERSION_NUM.iloc[-1] = 0
            #     midas_radtob.OB_HOUR_COUNT.iloc[-1] = midas_radtob.OB_HOUR_COUNT.iloc[0:-1].sum()
            #     midas_radtob.GLBL_IRAD_AMT.iloc[-1] = midas_radtob.GLBL_IRAD_AMT.iloc[0:-1].sum()

            if 24 in midas_radtob.OB_HOUR_COUNT.to_list():
                temp = midas_radtob[midas_radtob.OB_HOUR_COUNT == 24]
                midas_radtob = pd.concat([temp, midas_radtob.loc[temp.last_valid_index() + 1:]])

            radtob_stats = midas_radtob.groupby('SRC_ID').aggregate(self.RADTOBStatsCalc)
            stats_info = radtob_stats.values.flatten().tolist()

        return stats_info

    def integrate_pip_midas_radtob(self, met_stn_id, period, route_name, use_suppl_dat):
        """
        Gather solar radiation of the given period for each incident record.

        :param met_stn_id:
        :param period:
        :param route_name:
        :param use_suppl_dat:
        :return:

        **Test**::

            met_stn_id = incidents.Met_SRC_ID.iloc[1]
            period = incidents.Critical_Period.iloc[1]
            use_suppl_dat = False
        """

        # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
        #
        # try:
        #     prior_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
        # except KeyError:
        #     prior_ip_radtob = pd.DataFrame()

        prior_ip_radtob = self.MIDAS.query_radtob_by_grid_datetime(
            met_stn_id, period, route_name, use_suppl_dat, pickle_it=True)

        radtob_stats = self.calculate_radtob_variables_stats(prior_ip_radtob)

        return radtob_stats

    def integrate_nip_midas_radtob(self, met_stn_id, period, route_name, use_suppl_dat, prior_ip_data,
                                   stanox_section):
        """
        Gather solar radiation of the corresponding non-incident period for each incident record.

        :param met_stn_id: e.g. met_stn_id = non_ip_data.Met_SRC_ID.iloc[1]
        :param period: e.g. period = non_ip_data.Critical_Period.iloc[1]
        :param route_name:
        :param use_suppl_dat:
        :param stanox_section: e.g. location = non_ip_data.StanoxSection.iloc[0]
        :param prior_ip_data:
        :return:
        """

        # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
        #
        # try:
        #     non_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
        # except KeyError:
        #     non_ip_radtob = pd.DataFrame()

        non_ip_radtob = self.MIDAS.query_radtob_by_grid_datetime(met_stn_id, period, route_name,
                                                                 use_suppl_dat,
                                                                 pickle_it=True)

        # Get all incident period data on the same section
        ip_overlap = prior_ip_data[
            (prior_ip_data.StanoxSection == stanox_section) &
            (((prior_ip_data.Critical_StartDateTime <= period.left.to_pydatetime()[0]) &
              (prior_ip_data.Critical_EndDateTime >= period.left.to_pydatetime()[0])) |
             ((prior_ip_data.Critical_StartDateTime <= period.right.to_pydatetime()[0]) &
              (prior_ip_data.Critical_EndDateTime >= period.right.to_pydatetime()[0])))]
        # Skip data of Weather causing Incidents at around the same time; but
        if not ip_overlap.empty:
            non_ip_radtob = non_ip_radtob[
                (non_ip_radtob.OB_END_DATE < min(ip_overlap.Critical_StartDateTime)) |
                (non_ip_radtob.OB_END_DATE > max(ip_overlap.Critical_EndDateTime))]

        radtob_stats = self.calculate_radtob_variables_stats(non_ip_radtob)

        return radtob_stats

    def get_prior_ip_radtob_stats(self, incidents, use_suppl_dat=False):
        """
        Get prior-IP statistics of radiation data for each incident.

        :param incidents: data of incidents
        :type incidents: pandas.DataFrame
        :param use_suppl_dat:
        :type use_suppl_dat:
        :return: statistics of radiation data for each incident record during the prior IP
        :rtype: pandas.DataFrame

        **Test**::

            incidents
        """

        # noinspection PyTypeChecker
        prior_ip_radtob_stats = incidents.apply(
            lambda x: pd.Series(self.integrate_pip_midas_radtob(
                x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat)), axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        prior_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return prior_ip_radtob_stats

    def integrate_pip_ukcp09_data(self, grids, period):
        """
        Gather gridded weather observations of the given period for each incident record.

        :param grids: e.g. grids = incidents.Weather_Grid.iloc[0]
        :param period: e.g. period = incidents.Critical_Period.iloc[0]
        :return:

        **Test**::

            grids = incidents.Weather_Grid.iloc[0]
            period = incidents.Critical_Period.iloc[0]

        """

        # Find Weather data for the specified period
        prior_ip_weather = self.UKCP.query_by_grid_datetime(grids, period, pickle_it=True)
        # Calculate the max/min/avg for Weather parameters during the period
        weather_stats = self.calculate_weather_stats(prior_ip_weather)

        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        obs_by_far = self.UKCP.query_by_grid_datetime_(grids, period, pickle_it=True)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_prior_ip_ukcp09_stats(self, incidents):
        """
        Get prior-IP statistics of weather variables for each incident.

        :param incidents: data of incidents
        :type incidents: pandas.DataFrame
        :return: statistics of weather observation data for each incident record during the prior IP
        :rtype: pandas.DataFrame
        """

        # noinspection PyTypeChecker
        prior_ip_weather_stats = incidents.apply(
            lambda x: pd.Series(self.integrate_pip_ukcp09_data(x.Weather_Grid, x.Critical_Period)),
            axis=1)

        w_col_names = self.specify_weather_variable_names() + ['Hottest_Heretofore']

        prior_ip_weather_stats.columns = w_col_names

        prior_ip_weather_stats['Temperature_Change_max'] = \
            abs(prior_ip_weather_stats.Maximum_Temperature_max -
                prior_ip_weather_stats.Minimum_Temperature_min)
        prior_ip_weather_stats['Temperature_Change_min'] = \
            abs(prior_ip_weather_stats.Maximum_Temperature_min -
                prior_ip_weather_stats.Minimum_Temperature_max)

        return prior_ip_weather_stats

    def integrate_nip_ukcp09_data(self, grids, period, prior_ip_data, stanox_section):
        """
        Gather gridded Weather observations of the corresponding non-incident period
        for each incident record.

        :param grids:
        :param period:
        :param stanox_section:
        :param prior_ip_data:
        :return:

        **Test**::

            grids = non_ip_data.Weather_Grid.iloc[0]
            period = non_ip_data.Critical_Period.iloc[0]
            stanox_section = non_ip_data.StanoxSection.iloc[0]
        """

        # Get non-IP Weather data about where and when the incident occurred
        nip_weather = self.UKCP.query_by_grid_datetime(grids, period, pickle_it=True)

        # Get all incident period data on the same section
        ip_overlap = prior_ip_data[
            (prior_ip_data.StanoxSection == stanox_section) &
            (((prior_ip_data.Critical_StartDateTime <= period.left.to_pydatetime()[0]) &
              (prior_ip_data.Critical_EndDateTime >= period.left.to_pydatetime()[0])) |
             ((prior_ip_data.Critical_StartDateTime <= period.right.to_pydatetime()[0]) &
              (prior_ip_data.Critical_EndDateTime >= period.right.to_pydatetime()[0])))]
        # Skip data of Weather causing Incidents at around the same time; but
        if not ip_overlap.empty:
            nip_weather = nip_weather[
                (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
                (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
        # Get the max/min/avg Weather parameters for those incident periods
        weather_stats = self.calculate_weather_stats(nip_weather)

        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        obs_by_far = self.UKCP.query_by_grid_datetime_(grids, period, pickle_it=True)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_non_ip_weather_stats(self, non_ip_data, prior_ip_data):
        """
        Get prior-IP statistics of weather variables for each incident.

        :param non_ip_data: non-IP data
        :type non_ip_data: pandas.DataFrame
        :param prior_ip_data: prior-IP data
        :type prior_ip_data: pandas.DataFrame
        :return: stats of weather observation data for each incident record
            during the non-incident period
        :rtype: pandas.DataFrame
        """

        # noinspection PyTypeChecker
        non_ip_weather_stats = non_ip_data.apply(
            lambda x: pd.Series(self.integrate_nip_ukcp09_data(
                x.Weather_Grid, x.Critical_Period, prior_ip_data, x.StanoxSection)), axis=1)

        w_col_names = self.specify_weather_variable_names() + ['Hottest_Heretofore']

        non_ip_weather_stats.columns = w_col_names

        non_ip_weather_stats['Temperature_Change_max'] = \
            abs(non_ip_weather_stats.Maximum_Temperature_max -
                non_ip_weather_stats.Minimum_Temperature_min)
        non_ip_weather_stats['Temperature_Change_min'] = \
            abs(non_ip_weather_stats.Maximum_Temperature_min -
                non_ip_weather_stats.Minimum_Temperature_max)

        return non_ip_weather_stats

    def get_non_ip_radtob_stats(self, non_ip_data, prior_ip_data, use_suppl_dat):
        """
        Get prior-IP statistics of radiation data for each incident.

        :param non_ip_data: non-IP data
        :type non_ip_data: pandas.DataFrame
        :param prior_ip_data: prior-IP data
        :type prior_ip_data: pandas.DataFrame
        :param use_suppl_dat:
        :type use_suppl_dat:
        :return: statistics of radiation data for each incident record during the non-incident period
        :rtype: pandas.DataFrame
        """

        # noinspection PyTypeChecker
        non_ip_radtob_stats = non_ip_data.apply(
            lambda x: pd.Series(
                self.integrate_nip_midas_radtob(
                    x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat, prior_ip_data,
                    x.StanoxSection)),
            axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        non_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return non_ip_radtob_stats

    @staticmethod
    def find_closest_weather_grid(x, obs_grids, obs_centroid_geom):
        """
        Find the closest grid centroid and return the corresponding (pseudo) grid id.

        :param x: e.g. Incidents.StartNE.iloc[0]
        :param obs_grids:
        :param obs_centroid_geom:
        :return:

        **Test**::

            import copy

            x = incidents.StartXY.iloc[0]
        """

        x_ = shapely.ops.nearest_points(x, obs_centroid_geom)[1]

        pseudo_id = [i for i, y in enumerate(obs_grids.Centroid_XY) if y.equals(x_)]

        return pseudo_id[0]

    @staticmethod
    def create_circle_buffer_upon_weather_grid(start, end, midpoint, whisker=500):
        """
        Create a circle buffer for start/end location.

        :param start:
        :type start: shapely.geometry.Point
        :param end:
        :type end: shapely.geometry.Point
        :param midpoint:
        :type midpoint: shapely.geometry.Point
        :param whisker: extended length on both sides of the start and end locations, defaults to ``500``
        :type whisker: int
        :return: a buffer zone
        :rtype: shapely.geometry.Polygon

        **Test**::

            whisker = 0

            start = incidents.StartXY.iloc[0]
            end = incidents.EndXY.iloc[0]
            midpoint = incidents.MidpointXY.iloc[0]
        """

        if start == end:
            buffer_circle = start.buffer(2000 + whisker)
        else:
            radius = (start.distance(end) + whisker) / 2
            buffer_circle = midpoint.buffer(radius)
        return buffer_circle

    @staticmethod
    def find_intersecting_weather_grid(x, obs_grids, obs_grids_geom, as_grid_id=True):
        """
        Find all intersecting geom objects.

        :param x:
        :param obs_grids:
        :param obs_grids_geom:
        :param as_grid_id: whether to return grid id number
        :type as_grid_id: bool
        :return:

        **Test**::

            x = incidents.Buffer_Zone.iloc[0]
            as_grid_id = True
        """

        intxn_grids = [grid for grid in obs_grids_geom if x.intersects(grid)]

        if as_grid_id:
            x_ = shapely.ops.cascaded_union(intxn_grids)
            intxn_grids = [i for i, y in enumerate(obs_grids.Grid) if y.within(x_)]

        return intxn_grids

    @staticmethod
    def find_closest_met_stn(x, met_stations, met_stations_geom):
        """
        Find the closest grid centroid and return the corresponding (pseudo) grid id.

        :param x:
        :param met_stations:
        :param met_stations_geom:
        :return:

        **Test**::

            x = incidents.MidpointXY.iloc[0]
        """

        x_1 = shapely.ops.nearest_points(x, met_stations_geom)[1]

        # rest = shapely.geometry.MultiPoint([p for p in met_stations_geom if not p.equals(x_1)])
        # x_2 = shapely.ops.nearest_points(x, rest)[1]
        # rest = shapely.geometry.MultiPoint([p for p in rest if not p.equals(x_2)])
        # x_3 = shapely.ops.nearest_points(x, rest)[1]

        idx = [i for i, y in enumerate(met_stations.EN_GEOM) if y.equals(x_1)]
        src_id = met_stations.index[idx].to_list()

        return src_id

    # == Data of weather conditions ===================================================================

    def get_processed_incident_records(self, random_state=0):
        """

        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_plus = HeatAttributedIncidentsPlus(trial_id=0)

            >>> h_plus.get_processed_incident_records()
        """

        metex_incident_records = self.METEx.view_schedule8_costs_by_datetime_location_reason(
            route_name=self.Route, weather_category=self.WeatherCategory)

        # incidents_all.rename(columns={'Year': 'FinancialYear'}, inplace=True)
        incidents_by_season = get_data_by_season(metex_incident_records, self.Season)
        incidents = incidents_by_season[
            (incidents_by_season.StartDateTime >= datetime.datetime(2006, 1, 1)) &
            (incidents_by_season.StartDateTime < datetime.datetime(2017, 1, 1))]

        if self.OnlyTrial:  # For testing purpose only
            incidents = incidents.sample(n=10, random_state=random_state)

        if 'StartXY' not in incidents.columns or 'EndXY' not in incidents.columns:
            incidents['StartLongLat'] = incidents.apply(
                lambda x: shapely.geometry.Point(x.StartLongitude, x.StartLatitude), axis=1)
            incidents['EndLongLat'] = incidents.apply(
                lambda x: shapely.geometry.Point(x.EndLongitude, x.EndLatitude), axis=1)

            incidents['StartEasting'], incidents['StartNorthing'] = \
                wgs84_to_osgb36(incidents.StartLongitude.values, incidents.StartLatitude.values)
            incidents['EndEasting'], incidents['EndNorthing'] = \
                wgs84_to_osgb36(incidents.EndLongitude.values, incidents.EndLatitude.values)
            incidents['StartXY'] = incidents.apply(
                lambda x: shapely.geometry.Point(x.StartEasting, x.StartNorthing), axis=1)
            incidents['EndXY'] = incidents.apply(
                lambda x: shapely.geometry.Point(x.EndEasting, x.EndNorthing), axis=1)

        # Append 'MidpointXY' column
        incidents['MidpointXY'] = incidents.apply(
            lambda x: get_geometric_midpoint(x.StartXY, x.EndXY, as_geom=True), axis=1)

        # Make a buffer zone for Weather data aggregation
        incidents['Buffer_Zone'] = incidents.apply(
            lambda x: self.create_circle_buffer_upon_weather_grid(
                x.StartXY, x.EndXY, x.MidpointXY, whisker=0),
            axis=1)

        return incidents

    def get_incident_location_weather(self, update=False, pickle_it=False, verbose=False):
        """
        Process data of weather conditions for each incident location.

        **Test**::

            route_name         = ['Anglia', 'Wessex', 'Wales', 'North and East']
            weather_category   = 'Heat'
            season             = 'summer'
            prior_ip_start_hrs = -24
            non_ip_start_hrs   = -24
            trial_only         = False
            random_state       = None
            illustrate_buf_cir = False
            update             = False
            verbose            = True

        .. note::

            Note that the 'Critical_EndDateTime' would be based on the 'Critical_StartDateTime'
            if we consider the weather conditions on the day of incident occurrence;
            'StartDateTime' otherwise.
        """

        pickle_filename = make_filename(
            "weather", self.Route, None,
            "_".join([self.Season] if isinstance(self.Season, str) else self.Season),
            str(self.PIP_StartHrs) + 'h', str(self.NIP_StartHrs) + 'h',
            "trial" if self.OnlyTrial else "", sep="_")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_weather = load_pickle(path_to_pickle)

        else:
            try:
                # -- Incidents data -------------------------------------------------------------------

                incidents = self.get_processed_incident_records()

                # -- Weather data ---------------------------------------------------------------------

                obs_grids = self.UKCP.get_observation_grids()  # Grids for observing weather conditions
                obs_centroid_geom = shapely.geometry.MultiPoint(list(obs_grids.Centroid_XY))
                obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

                met_stations = self.MIDAS.get_radiation_stations()  # Met station locations
                met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.EN_GEOM))

                # -- Data integration in the spatial context ------------------------------------------

                incidents['Start_Pseudo_Grid_ID'] = incidents.StartXY.map(  # Start
                    lambda x: self.find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
                incidents = incidents.join(obs_grids, on='Start_Pseudo_Grid_ID')

                incidents['End_Pseudo_Grid_ID'] = incidents.EndXY.map(  # End
                    lambda x: self.find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
                incidents = incidents.join(
                    obs_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')

                # Modify column names
                for p in ['Start', 'End']:
                    a = [c for c in incidents.columns if c.endswith(p)]
                    b = [p + '_' + c if c == 'Grid' else p + '_Grid_' + c for c in obs_grids.columns]
                    incidents.rename(columns=dict(zip(a, b)), inplace=True)

                # Find all Weather observation grids that intersect with the created buffer zone
                # for each incident location
                incidents['Weather_Grid'] = incidents.Buffer_Zone.map(
                    lambda x: self.find_intersecting_weather_grid(x, obs_grids, obs_grids_geom))

                incidents['Met_SRC_ID'] = incidents.MidpointXY.map(
                    lambda x: self.find_closest_met_stn(x, met_stations, met_stations_geom))

                # -- Data integration for the specified prior-IP --------------------------------------

                incidents = self.set_prior_ip(incidents)

                # Get prior-IP statistics of weather variables for each incident.
                prior_ip_weather_stats = self.get_prior_ip_ukcp09_stats(incidents)

                # Get prior-IP statistics of radiation data for each incident.
                prior_ip_radtob_stats = self.get_prior_ip_radtob_stats(incidents, use_suppl_dat=True)

                prior_ip_data = incidents.join(prior_ip_weather_stats).join(prior_ip_radtob_stats)

                prior_ip_data['Incident_Reported'] = 1

                # -- Data integration for the specified non-IP ----------------------------------------

                non_ip_data = self.get_non_ip_data(incidents, prior_ip_data)

                non_ip_weather_stats = self.get_non_ip_weather_stats(non_ip_data, prior_ip_data)

                non_ip_radtob_stats = self.get_non_ip_radtob_stats(
                    non_ip_data, prior_ip_data, use_suppl_dat=True)

                non_ip_data = non_ip_data.join(non_ip_weather_stats).join(non_ip_radtob_stats)

                non_ip_data['Incident_Reported'] = 0

                # -- Merge "prior_ip_data" and "non_ip_data" ------------------------------------------
                incident_location_weather = pd.concat(
                    [prior_ip_data, non_ip_data], axis=0, ignore_index=True, sort=False)

                # Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE)
                incident_location_weather = incident_location_weather.join(
                    categorise_track_orientations(incident_location_weather))

                # Categorise temperature: 25, 26, 27, 28, 29, 30
                incident_location_weather = incident_location_weather.join(
                    categorise_temperatures(
                        incident_location_weather, column_name='Maximum_Temperature_max'))

                # incident_location_weather.dropna(subset=w_col_names, inplace=True)

                if pickle_it:
                    save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get weather conditions for the incident locations. {}.".format(e))
                incident_location_weather = pd.DataFrame()

        return incident_location_weather

    def illustrate_buffer_circle(self):

        incidents = self.get_processed_incident_records()

        obs_grids = self.UKCP.get_observation_grids()  # Grids for observing weather conditions
        obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

        # Illustration of the buffer circle
        start_point, end_point, midpoint = incidents.iloc[0, ['StartXY', 'EndXY', 'MidpointXY']]

        bf_circle = self.create_circle_buffer_upon_weather_grid(
            start_point, end_point, midpoint, whisker=0)
        i_obs_grids = self.find_intersecting_weather_grid(
            bf_circle, obs_grids, obs_grids_geom, as_grid_id=False)

        plt.figure(figsize=(7, 6))

        ax = plt.subplot2grid((1, 1), (0, 0))

        for g in i_obs_grids:
            x_, y_ = g.exterior.xy
            ax.plot(x_, y_, color='#433f3f')

        ax.plot([], 's', label="Weather observation grid", ms=16, color='none',
                markeredgecolor='#433f3f')
        x_, y_ = bf_circle.exterior.xy

        ax.plot(x_, y_)
        ax.plot(
            [], 'r', marker='o', markersize=15, linestyle='None', fillstyle='none', label='Buffer zone')

        sx, sy, ex, ey = start_point.xy + end_point.xy
        if start_point == end_point:
            ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
        else:
            ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
            ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
        legend = plt.legend(numpoints=1, loc='best', prop=font, fancybox=True, labelspacing=0.5)
        frame = legend.get_frame()
        frame.set_edgecolor('k')

        plt.tight_layout()

    def plot_temperature_deviation(self, lp_days=-14, add_err_bar=True, update=False,
                                   verbose=False, save_as=".tif", dpi=None):
        """

        **Test**::

            from models.prototype.heat import plot_temperature_deviation

            route_name  = 'Anglia'
            lp_days  = -14
            add_err_bar = True
            update      = False
            verbose     = True
            save_as     = None  # ".tif"
            dpi         = None  # 600
        """

        default_lp = self.LP

        data_sets = []
        for d in range(1, lp_days + 1):
            self.__setattr__('LP', -d)
            data_sets.append(self.get_incident_location_weather(update=update, pickle_it=True))

        self.__setattr__('LP', default_lp)

        incident_location_weather = [
            self.get_incident_location_weather(update=update, verbose=verbose)
            for d in range(1, gap + 1)]

        time_and_iloc = ['StartDateTime', 'EndDateTime', 'StanoxSection', 'IncidentDescription']
        selected_cols, data = time_and_iloc + ['Temperature_max'], incident_location_weather[0]
        ip_temperature_max = data[data.IncidentReported == 1][selected_cols]

        diff_means, diff_std = [], []
        for i in range(0, gap):
            data = incident_location_weather[i]
            nip_temperature_max = data[data.IncidentReported == 0][selected_cols]
            temp_diffs = pd.merge(
                ip_temperature_max, nip_temperature_max, on=time_and_iloc, suffixes=('_ip', '_nip'))
            temp_diff = temp_diffs.Temperature_max_ip - temp_diffs.Temperature_max_nip
            diff_means.append(temp_diff.abs().mean())
            diff_std.append(temp_diff.abs().std())

        plt.figure(figsize=(10, 5))
        if add_err_bar:
            container = plt.bar(
                np.arange(1, len(diff_means) + 1), diff_means, align='center', yerr=diff_std, capsize=4,
                width=0.7, color='#9FAFBE')
            connector, cap_lines, (vertical_lines,) = container.errorbar.lines
            vertical_lines.set_color('#666666')
            for cap in cap_lines:
                cap.set_color('#da8067')
        else:
            plt.bar(
                np.arange(1, len(diff_means) + 1), diff_means, align='center', width=0.7,
                color='#9FAFBE')
            plt.grid(False)
        plt.xticks(np.arange(1, len(diff_means) + 1), fontsize=14)
        plt.xlabel('Latent period (Number of days)', fontsize=14)
        plt.ylabel('Temperature deviation (°C)', fontsize=14)
        plt.tight_layout()

        if save_as:
            path_to_fig = self.cdd_trial("temperature-deviation" + save_as)
            save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    def illustrate_explanatory_variables(self, save_as=".pdf", dpi=None, verbose=False):
        """
        Describe basic statistics about the main explanatory variables.

        :param save_as:
        :param dpi:
        :param verbose:
        :return:

        **Test**::

            train_set = incident_location_weather.dropna().copy()
        """

        train_set = self.__getattribute__('TrainingSet')

        plt.figure(figsize=(14, 5))
        colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

        ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=3)
        train_set.Temperature_Category.value_counts().plot.bar(color='#537979', rot=0, fontsize=12)
        plt.xticks(range(0, 8), ['<24', '24', '25', '26', '27', '28', '29', '≥30'], rotation=0,
                   fontsize=12)
        ax1.text(7.5, -0.2, '(°C)', fontsize=12)
        plt.xlabel('Maximum temperature', fontsize=13, labelpad=8)
        plt.ylabel('Frequency', fontsize=12, rotation=0)
        ax1.yaxis.set_label_coords(0.0, 1.01)

        ax2 = plt.subplot2grid((1, 9), (0, 3))
        train_set.Temperature_Change_max.plot.box(color=colour, ax=ax2, widths=0.5, fontsize=12)
        ax2.set_xticklabels('')
        plt.xlabel('Temperature\nchange', fontsize=13, labelpad=10)
        plt.ylabel('(°C)', fontsize=12, rotation=0)
        ax2.yaxis.set_label_coords(0.05, 1.01)

        ax3 = plt.subplot2grid((1, 9), (0, 4), colspan=2)
        orient_cats = [
            x.replace('Track_Orientation_', '') for x in train_set.columns
            if x.startswith('Track_Orientation_')]
        track_orientation = pd.Series(
            [np.sum(train_set.Track_Orientation == x) for x in orient_cats], index=orient_cats)
        track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
        track_orientation.plot.bar(color='#a72a3d', rot=0, fontsize=12)
        # ax3.set_yticks(range(0, track_orientation.max() + 1, 100))
        plt.xlabel('Track orientation', fontsize=13, labelpad=8)
        plt.ylabel('Count', fontsize=12, rotation=0)
        ax3.yaxis.set_label_coords(0.0, 1.01)

        ax4 = plt.subplot2grid((1, 9), (0, 6))
        train_set.GLBL_IRAD_AMT_total.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
        ax4.set_xticklabels('')
        plt.xlabel('Maximum\nirradiation', fontsize=13, labelpad=10)
        plt.ylabel('(KJ/m$^2$)', fontsize=12, rotation=0)
        ax4.yaxis.set_label_coords(0.2, 1.01)

        ax5 = plt.subplot2grid((1, 9), (0, 7))
        train_set.Precipitation_max.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
        ax5.set_xticklabels('')
        plt.xlabel('Maximum\nprecipitation', fontsize=13, labelpad=10)
        plt.ylabel('(mm)', fontsize=12, rotation=0)
        ax5.yaxis.set_label_coords(0.0, 1.01)

        ax6 = plt.subplot2grid((1, 9), (0, 8))
        hottest_heretofore = train_set.Hottest_Heretofore.value_counts()
        hottest_heretofore.plot.bar(color='#a72a3d', rot=0, fontsize=12)
        plt.xlabel('Hottest\nheretofore', fontsize=13, labelpad=5)
        plt.ylabel('Frequency', fontsize=12, rotation=0)
        ax6.yaxis.set_label_coords(0.0, 1.01)
        # ax6.set_yticks(range(0, hottest_heretofore.max() + 1, 100))
        ax6.set_xticklabels(['False', 'True'], rotation=0)

        plt.tight_layout()

        if save_as:
            path_to_fig_file = self.cdd_trial("variables" + save_as)
            save_fig(path_to_fig_file, dpi, verbose=verbose, conv_svg_to_emf=True)

    def logistic_regression(self, add_intercept=True, random_state=0, pickle_it=True, verbose=True):
        """
        Train/test a logistic regression model for predicting heat-related incidents.

        -------------- | ------------------ | -----------------------------------------------------------
        IncidentReason | IncidentReasonName | IncidentReasonDescription
        -------------- | ------------------ | -----------------------------------------------------------
        IQ             |   TRACK SIGN       | Trackside sign blown down/light out etc.
        IW             |   COLD             | Non severe - Snow/Ice/Frost affecting infr equipment, ...
        OF             |   HEAT/WIND        | Blanket speed restriction for extreme heat or high wind ...
        Q1             |   TKB PUMPS        | Takeback Pumps
        X4             |   BLNK REST        | Blanket speed restriction for extreme heat or high wind
        XW             |   WEATHER          | Severe Weather not snow affecting infrastructure, resp. ...
        XX             |   MISC OBS         | Msc items on line (incl. trees) due to weather, resp. of...
        -------------- | ------------------ | -----------------------------------------------------------

        **Test**::

            trial_id                  = 0
            route_name                = 'Anglia'
            weather_category          = 'Heat'
            season                    = 'summer'
            prior_ip_start_hrs        = -0
            non_ip_start_hrs          = -0
            outlier_pctl              = 95
            describe_var              = False
            add_const                 = True
            seed                      = 0
            model                     = 'logit'
            plot_roc                  = False
            plot_predicted_likelihood = False
            save_as                   = None
            dpi                       = None
            verbose                   = True

            m_data = incident_location_weather.copy()
        """

        # Get the m_data for modelling
        m_data = self.get_incident_location_weather()

        # temp_data = [load_pickle(cdd_mod_heat_inter("Slices", f))
        #              for f in os.listdir(cdd_mod_heat_inter("Slices"))]
        # m_data = pd.concat(temp_data, ignore_index=True, sort=False)

        m_data.dropna(subset=['GLBL_IRAD_AMT_total'], inplace=True)
        m_data.GLBL_IRAD_AMT_total = m_data.GLBL_IRAD_AMT_total / 1000

        # Select features
        explanatory_variables = self.ExplanatoryVariables.copy()

        for v in explanatory_variables:
            if not m_data[m_data[v].isna()].empty:
                m_data.dropna(subset=[v], inplace=True)

        m_data = m_data[
            explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime',
                                     'DelayMinutes']]

        # Remove outliers
        if 95 <= self.OutlierPercentile <= 100:
            m_data = m_data[
                m_data.DelayMinutes <= np.percentile(m_data.DelayMinutes, self.OutlierPercentile)]

        # Add the intercept
        if add_intercept:
            # data['const'] = 1.0
            m_data = sm_tools.tools.add_constant(m_data, prepend=True, has_constant='skip')
            explanatory_variables = ['const'] + explanatory_variables

        # m_data = m_data.loc[:, (m_data != 0).any(axis=0)]
        # explanatory_variables = [x for x in explanatory_variables if x in m_data.columns]

        # Select data before 2014 as training data set, with the rest being test set
        train_set = m_data[m_data.StartDateTime < datetime.datetime(2016, 1, 1)]
        test_set = m_data[m_data.StartDateTime >= datetime.datetime(2016, 1, 1)]

        np.random.seed(random_state)

        try:
            if self.ModelType == 'logit':
                mod = sm_dcm.Logit(train_set.Incident_Reported, train_set[explanatory_variables])
            else:
                mod = sm_dcm.Probit(train_set.Incident_Reported, train_set[explanatory_variables])
            result_summary = mod.fit(maxiter=1000, full_output=True, disp=True)  # method='newton'
            print(result_summary.summary()) if verbose else print("")

            # Odds ratios
            odds_ratios = pd.DataFrame(np.exp(result_summary.params), columns=['OddsRatio'])
            print("\n{}".format(odds_ratios)) if verbose else print("")

            # Prediction
            test_set['incident_prob'] = result_summary.predict(test_set[explanatory_variables])

            # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
            fpr, tpr, thr = metrics.roc_curve(test_set.Incident_Reported, test_set.incident_prob)
            # Area under the curve (AUC)
            auc = metrics.auc(fpr, tpr)
            ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
            threshold = np.min(thr[ind])

            # prediction accuracy
            test_set['incident_prediction'] = test_set.incident_prob.apply(
                lambda x: 1 if x >= threshold else 0)
            test = pd.Series(test_set.Incident_Reported == test_set.incident_prediction)
            model_accuracy = np.divide(sum(test), len(test))
            print("\nAccuracy: %f" % model_accuracy) if verbose else print("")

            # incident prediction accuracy
            incident_only = test_set[test_set.Incident_Reported == 1]
            test_acc = pd.Series(incident_only.Incident_Reported == incident_only.incident_prediction)
            incident_accuracy = np.divide(sum(test_acc), len(test_acc))
            print("Incident accuracy: %f" % incident_accuracy) if verbose else print("")

        except Exception as e:
            print(e)
            result_summary = e
            model_accuracy, incident_accuracy, threshold = np.nan, np.nan, np.nan

        if pickle_it:
            repo = locals()
            var_names = ['train_set', 'test_set',
                         'result_summary', 'model_accuracy', 'incident_accuracy', 'threshold']
            resources = {k: repo[k] for k in list(var_names)}
            result_pickle = make_filename(
                "result", self.Route, self.WeatherCategory,
                self.PIP_StartHrs, self.LP, self.NIP_StartHrs)

            save_pickle(resources, self.cdd_trial(result_pickle), verbose=verbose)

        return m_data, train_set, test_set, result_summary, model_accuracy, incident_accuracy, threshold

    def plot_roc(self, save_as=".tif", dpi=600, verbose=True):
        """
        Plot ROC.

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None
        :param verbose: whether to print relevant information in console, defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype import HeatAttributedIncidents

            >>> heat_attributed_incidents = HeatAttributedIncidents(trial_id=0)

            >>> _ = heat_attributed_incidents.logistic_regression()
            >>> heat_attributed_incidents.plot_roc(save_as=None)
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

        plt.tight_layout()

        if save_as:
            path_to_roc_fig = self.cdd_trial("roc" + save_as)
            save_fig(path_to_roc_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    def plot_pred_likelihood(self, save_as=".tif", dpi=600, verbose=True):
        """
        Plot incident delay minutes against predicted probabilities

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None
        :param verbose: whether to print relevant information in console, defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype import HeatAttributedIncidents

            >>> heat_attributed_incidents = HeatAttributedIncidents(trial_id=0)

            >>> _ = heat_attributed_incidents.logistic_regression()
            >>> heat_attributed_incidents.plot_pred_likelihood(save_as=None)
        """

        test_set = self.__getattribute__('TestSet')
        threshold = self.__getattribute__('Threshold')

        incident_ind = test_set.IncidentReported == 1

        plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))

        ax.scatter(
            test_set[incident_ind].incident_prob, test_set[incident_ind].DelayMinutes,
            c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
            label="Heat-related incident (2014/15)")
        plt.axvline(
            x=threshold, label="Threshold: %.2f" % threshold, color='#e5c100', linewidth=2)

        legend = plt.legend(scatterpoints=1, loc=2, fontsize=14, fancybox=True, labelspacing=0.6)
        frame = legend.get_frame()
        frame.set_edgecolor('k')

        plt.xlim(xmin=0, xmax=1.03)
        plt.ylim(ymin=-15)

        ax.set_xlabel("Likelihood of heat-related incident occurrence", fontsize=14, fontweight='bold')
        ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.tight_layout()

        if save_as:
            path_to_pred_fig = self.cdd_trial("predicted_likelihood" + save_as)
            save_fig(path_to_pred_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)
