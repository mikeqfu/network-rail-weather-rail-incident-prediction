"""
An extended version of the prototype model in the context of heat-related rail incidents.
"""

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
from pyhelpers.geom import get_geometric_midpoint, wgs84_to_osgb36
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_pickle, save_fig, save_pickle
from sklearn import metrics

from coordinator.feature import categorise_temperatures, categorise_track_orientations, \
    get_data_by_meteorological_seasons
from coordinator.geometry import create_weather_grid_buffer, find_closest_met_stn, \
    find_closest_weather_grid, find_intersecting_weather_grid
from preprocessor import METExLite, MIDAS, UKCP09
from utils import cd_models, make_filename


class HeatAttributedIncidentsPlus:

    def __init__(self, trial_id, route_name=None, weather_category='Heat',
                 in_seasons=None, incident_reasons=None,
                 pip_start_hrs=-24, nip_start_hrs=-24, lp_days=None,
                 samples_only=False, outlier_pctl=100, model_type='logit'):

        pd_preferences()
        mpl_preferences(font_name='Times New Roman')

        self.Name = ''

        self.TrialID = "{}".format(trial_id)

        self.METEx = METExLite(database_name='NR_METEx_20190203')
        self.UKCP = UKCP09()
        self.MIDAS = MIDAS()

        if route_name is None:
            route_name = ['Anglia', 'Wessex', 'Wales', 'North and East']
        self.Route = route_name if isinstance(route_name, list) else [route_name]

        self.WeatherCategory = weather_category

        if in_seasons is None:
            in_seasons = ['summer']
        self.Seasons = in_seasons if isinstance(in_seasons, list) else [in_seasons]

        if incident_reasons is None:
            incident_reasons = ['IR', 'XH', 'IB', 'JH']
        self.ReasonCodes = incident_reasons if isinstance(incident_reasons, list) else [incident_reasons]

        self.PIP_StartHrs = pip_start_hrs
        self.NIP_StartHrs = nip_start_hrs
        self.LP = lp_days

        self.LP_Anglia = lambda x: -20 if x in range(24, 29) else (-13 if x > 28 else 0)
        self.LP_Wessex = lambda x: -30 if x in range(24, 29) else (-25 if x > 28 else 0)
        self.LP_NE = lambda x: -18 if x in range(24, 29) else (-16 if x > 28 else 0)
        self.LP_Wales = lambda x: -19 if x in range(24, 29) else (-5 if x > 28 else 0)

        self.SamplesOnly = samples_only
        if isinstance(self.SamplesOnly, bool):
            self.SampleSize = 10
        elif isinstance(self.SamplesOnly, int):
            self.SampleSize = self.SamplesOnly

        self.UKCP09StatsCalc = {'Maximum_Temperature': (max, min, np.average),
                                'Minimum_Temperature': (max, min, np.average),
                                'Temperature_Change': np.average,
                                'Precipitation': (max, min, np.average)}
        self.RADTOBStatsCalc = {'GLBL_IRAD_AMT': sum}

        var_stats_names = [
            [k, [i.__name__ for i in v] if isinstance(v, tuple) else [v.__name__]]
            for k, v in self.UKCP09StatsCalc.items()]
        weather_variable_names = [['_'.join([x, z]) for z in y] for x, y in var_stats_names]
        self.WeatherVariableNames = list(itertools.chain.from_iterable(weather_variable_names))

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

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> os.path.relpath(h_model_plus.cdd())
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

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> os.path.relpath(h_model_plus.cdd_trial())
            'models\\prototype_ext\\heat\\0'
        """

        path = self.cdd(self.TrialID, *sub_dir, mkdir=mkdir)

        return path

    # == Set Prior-IP, LP and Non-IP ==================================================================

    def get_pip_records(self, incidents):
        """

        :param incidents:
        :return:
        """

        data = incidents.copy()

        data['Incident_Duration'] = data.EndDateTime - data.StartDateTime

        # End date and time of the prior IP
        data['Critical_EndDateTime'] = data.StartDateTime.dt.round('H')

        # Start date and time of the prior IP
        critical_start_dt = data.Critical_EndDateTime.map(
            lambda x: x + pd.Timedelta(
                hours=self.PIP_StartHrs if x.time() > datetime.time(9) else self.PIP_StartHrs * 2))
        data.insert(
            data.columns.get_loc('Critical_EndDateTime'), 'Critical_StartDateTime', critical_start_dt)

        # Prior-IP dates of each incident
        data['Critical_Period'] = data.apply(
            lambda x: pd.interval_range(x.Critical_StartDateTime, x.Critical_EndDateTime), axis=1)

        return data

    def set_lp_and_nip(self, route_name, ip_max_temp_max, ip_start_dt):
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

        critical_end_dt = ip_start_dt + pd.Timedelta(days=lp)
        critical_start_dt = critical_end_dt + pd.Timedelta(hours=self.NIP_StartHrs)
        critical_period = pd.interval_range(critical_start_dt, critical_end_dt)

        return critical_start_dt, critical_end_dt, critical_period

    def get_nip_records(self, incidents, prior_ip_data):
        """

        :param incidents:
        :param prior_ip_data:
        :return:
        """

        non_ip_data = incidents.copy(deep=True)  # Get weather data that did not cause any incident

        col_names = ['Critical_StartDateTime', 'Critical_EndDateTime', 'Critical_Period']
        non_ip_data[col_names] = prior_ip_data.apply(
            lambda x: pd.Series(
                self.set_lp_and_nip(
                    x.Route, x.Maximum_Temperature_max, x.Critical_StartDateTime)),
            axis=1)

        return non_ip_data

    # == Calculators ==================================

    def calculate_ukcp09_stats(self, weather_data):
        """
        Calculate the statistics for the Weather variables (except radiation).

        :param weather_data:
        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)
        """

        if weather_data.empty:
            weather_stats_info = [np.nan] * sum(map(np.count_nonzero, self.UKCP09StatsCalc.values()))

        else:
            # Create a pseudo id for groupby() & aggregate()
            weather_data['Pseudo_ID'] = 0
            weather_stats = weather_data.groupby('Pseudo_ID').aggregate(self.UKCP09StatsCalc)
            # a, b = [list(x) for x in weather_stats.columns.levels]
            # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
            # if not weather_stats.empty:
            #     stats_info = weather_stats.values[0].tolist()
            # else:
            #     stats_info = [np.nan] * len(weather_stats.columns)

            weather_stats_info = weather_stats.values[0].tolist()

        return weather_stats_info

    def calculate_radtob_stats(self, midas_radtob):
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

    # == Prior-incident period ========================

    def integrate_pip_ukcp09_data(self, grids, period, pickle_it=False):
        """
        Gather gridded weather observations of the given period for each incident record.

        :param grids: e.g. grids = incidents.Weather_Grid.iloc[0]
        :param period: e.g. period = incidents.Critical_Period.iloc[0]
        :param pickle_it:
        :return:

        **Test**::

            grids = incidents.Weather_Grid.iloc[0]
            period = incidents.Critical_Period.iloc[0]

        """

        # Find Weather data for the specified period
        prior_ip_weather = self.UKCP.query_by_grid_datetime(grids, period, pickle_it=pickle_it)
        # Calculate the max/min/avg for Weather parameters during the period
        weather_stats = self.calculate_ukcp09_stats(prior_ip_weather)

        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        obs_by_far = self.UKCP.query_by_grid_datetime_(grids, period, pickle_it=pickle_it)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_pip_ukcp09_stats(self, incidents):
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

        w_col_names = self.WeatherVariableNames + ['Hottest_Heretofore']

        prior_ip_weather_stats.columns = w_col_names

        prior_ip_weather_stats['Temperature_Change_max'] = \
            abs(prior_ip_weather_stats.Maximum_Temperature_max -
                prior_ip_weather_stats.Minimum_Temperature_min)
        prior_ip_weather_stats['Temperature_Change_min'] = \
            abs(prior_ip_weather_stats.Maximum_Temperature_min -
                prior_ip_weather_stats.Minimum_Temperature_max)

        return prior_ip_weather_stats

    def integrate_pip_radtob(self, met_stn_id, period, route_name, use_suppl_dat, pickle_it=False):
        """
        Gather solar radiation of the given period for each incident record.

        :param met_stn_id:
        :param period:
        :param route_name:
        :param use_suppl_dat:
        :param pickle_it:
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
            met_stn_id, period, route_name, use_suppl_dat, pickle_it=pickle_it)

        radtob_stats = self.calculate_radtob_stats(prior_ip_radtob)

        return radtob_stats

    def get_pip_radtob_stats(self, incidents, use_suppl_dat=True):
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
            lambda x: pd.Series(self.integrate_pip_radtob(
                x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat)), axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        prior_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return prior_ip_radtob_stats

    # == Non-incident period ==========================

    def integrate_nip_ukcp09_data(self, grids, period, pip_data, stanox_section, pickle_it=False):
        """
        Gather gridded Weather observations of the corresponding non-incident period
        for each incident record.

        :param grids:
        :param period:
        :param stanox_section:
        :param pip_data:
        :param pickle_it:
        :return:

        **Test**::

            grids = nip_data_.Weather_Grid.iloc[0]
            period = nip_data_.Critical_Period.iloc[0]
            stanox_section = nip_data_.StanoxSection.iloc[0]
        """

        # Get non-IP Weather data about where and when the incident occurred
        nip_weather = self.UKCP.query_by_grid_datetime(grids, period, pickle_it=pickle_it)

        # Get all incident period data on the same section
        ip_overlap = pip_data[
            (pip_data.StanoxSection == stanox_section) &
            (((pip_data.Critical_StartDateTime <= period.left.to_pydatetime()[0]) &
              (pip_data.Critical_EndDateTime >= period.left.to_pydatetime()[0])) |
             ((pip_data.Critical_StartDateTime <= period.right.to_pydatetime()[0]) &
              (pip_data.Critical_EndDateTime >= period.right.to_pydatetime()[0])))]
        # Skip data of Weather causing Incidents at around the same time; but
        if not ip_overlap.empty:
            nip_weather = nip_weather[
                (nip_weather.Date < min(ip_overlap.Critical_StartDateTime)) |
                (nip_weather.Date > max(ip_overlap.Critical_EndDateTime))]
        # Get the max/min/avg Weather parameters for those incident periods
        weather_stats = self.calculate_ukcp09_stats(nip_weather)

        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        obs_by_far = self.UKCP.query_by_grid_datetime_(grids, period, pickle_it=pickle_it)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_nip_ukcp09_stats(self, nip_data_, pip_data):
        """
        Get prior-IP statistics of weather variables for each incident.

        :param nip_data_: non-IP data
        :type nip_data_: pandas.DataFrame
        :param pip_data: prior-IP data
        :type pip_data: pandas.DataFrame
        :return: stats of weather observation data for each incident record
            during the non-incident period
        :rtype: pandas.DataFrame
        """

        # noinspection PyTypeChecker
        non_ip_weather_stats = nip_data_.apply(
            lambda x: pd.Series(self.integrate_nip_ukcp09_data(
                x.Weather_Grid, x.Critical_Period, pip_data, x.StanoxSection)), axis=1)

        w_col_names = self.WeatherVariableNames + ['Hottest_Heretofore']

        non_ip_weather_stats.columns = w_col_names

        non_ip_weather_stats['Temperature_Change_max'] = \
            abs(non_ip_weather_stats.Maximum_Temperature_max -
                non_ip_weather_stats.Minimum_Temperature_min)
        non_ip_weather_stats['Temperature_Change_min'] = \
            abs(non_ip_weather_stats.Maximum_Temperature_min -
                non_ip_weather_stats.Minimum_Temperature_max)

        return non_ip_weather_stats

    def integrate_nip_radtob(self, met_stn_id, period, route_name, use_suppl_dat, prior_ip_data,
                             stanox_section, pickle_it=False):
        """
        Gather solar radiation of the corresponding non-incident period for each incident record.

        :param met_stn_id: e.g. met_stn_id = nip_data_.Met_SRC_ID.iloc[1]
        :param period: e.g. period = nip_data_.Critical_Period.iloc[1]
        :param route_name:
        :param use_suppl_dat:
        :param stanox_section: e.g. location = nip_data_.StanoxSection.iloc[0]
        :param prior_ip_data:
        :param pickle_it:
        :return:
        """

        # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
        #
        # try:
        #     non_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
        # except KeyError:
        #     non_ip_radtob = pd.DataFrame()

        non_ip_radtob = self.MIDAS.query_radtob_by_grid_datetime(
            met_stn_id, period, route_name, use_suppl_dat, pickle_it=pickle_it)

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

        radtob_stats = self.calculate_radtob_stats(non_ip_radtob)

        return radtob_stats

    def get_nip_radtob_stats(self, non_ip_data, prior_ip_data, use_suppl_dat=True):
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
                self.integrate_nip_radtob(
                    x.Met_SRC_ID, x.Critical_Period, x.Route, use_suppl_dat, prior_ip_data,
                    x.StanoxSection)),
            axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        non_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return non_ip_radtob_stats

    # == Data of weather conditions ===================================================================

    def get_processed_incident_records(self, update=False, random_state=0):
        """

        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> # h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, trial_only=False)
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, samples_only=False)

            >>> incid_rec = h_model_plus.get_processed_incident_records()
        """

        self.__setattr__('RandomState', random_state)

        pickle_filename = make_filename(
            "incidents", self.Route, self.WeatherCategory,
            # "" if self.Seasons is None else "_".join(self.Seasons),
            "_".join(self.ReasonCodes),
            "sample_rs{}".format(self.__getattribute__('RandomState')) if self.SamplesOnly else "",
            sep="_")

        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incidents = load_pickle(path_to_pickle)

        else:
            metex_incident_records = self.METEx.view_schedule8_costs_by_datetime_location_reason(
                route_name=self.Route, weather_category=self.WeatherCategory, update=update)

            metex_incident_records = metex_incident_records[
                metex_incident_records.IncidentReasonCode.isin(self.ReasonCodes) &
                ~metex_incident_records.WeatherCategory.isin(['Cold'])]

            # incidents_all.rename(columns={'Year': 'FinancialYear'}, inplace=True)
            incidents_by_season = get_data_by_meteorological_seasons(
                incident_records=metex_incident_records, in_seasons=self.Seasons,
                datetime_col='StartDateTime')
            # incidents = incidents_by_season[
            #     (incidents_by_season.StartDateTime >= datetime.datetime(2006, 1, 1)) &
            #     (incidents_by_season.StartDateTime < datetime.datetime(2017, 3, 31))]

            incidents = incidents_by_season[
                (incidents_by_season.StartDateTime >= datetime.datetime(2006, 4, 1))]

            if self.SamplesOnly:  # For testing purpose only
                incidents = incidents.sample(n=self.SampleSize, random_state=random_state)

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

            # Make a buffer zone for weather data aggregation
            incidents['Buffer_Zone'] = incidents.apply(
                lambda x: create_weather_grid_buffer(x.StartXY, x.EndXY, x.MidpointXY, whisker=0),
                axis=1)

            # Weather observation grids
            obs_grids = self.UKCP.get_observation_grids()  # Grids for observing weather conditions
            obs_centroid_geom = shapely.geometry.MultiPoint(list(obs_grids.Centroid_XY))
            obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

            met_stations = self.MIDAS.get_radiation_stations()  # Met station locations
            met_stations_geom = shapely.geometry.MultiPoint(list(met_stations.EN_GEOM))

            incidents['Start_Pseudo_Grid_ID'] = incidents.StartXY.map(  # Start
                lambda x: find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
            incidents = incidents.join(obs_grids, on='Start_Pseudo_Grid_ID')

            incidents['End_Pseudo_Grid_ID'] = incidents.EndXY.map(  # End
                lambda x: find_closest_weather_grid(x, obs_grids, obs_centroid_geom))
            incidents = incidents.join(
                obs_grids, on='End_Pseudo_Grid_ID', lsuffix='_Start', rsuffix='_End')

            # Modify column names
            for p in ['Start', 'End']:
                a = [c for c in incidents.columns if c.endswith(p)]
                b = [p + '_' + c if c == 'Grid' else p + '_Grid_' + c for c in obs_grids.columns]
                incidents.rename(columns=dict(zip(a, b)), inplace=True)

            # Find weather obs. grids intersecting with the buffer zone for each incident location
            incidents['Weather_Grid'] = incidents.Buffer_Zone.map(
                lambda x: find_intersecting_weather_grid(x, obs_grids, obs_grids_geom))

            incidents['Met_SRC_ID'] = incidents.MidpointXY.map(
                lambda x: find_closest_met_stn(x, met_stations, met_stations_geom))

            save_pickle(incidents, path_to_pickle)

        return incidents

    def get_incident_location_weather(self, random_state=0, update=False, pickle_it=False, verbose=True):
        """
        Process data of weather conditions for each incident location.

        :param random_state:
        :param update: whether to do an update check, defaults to ``False``
        :type update: bool
        :param pickle_it: whether to save the result as a pickle file
        :type pickle_it: bool
        :param verbose: whether to print relevant information in console, defaults to ``True``
        :type verbose: bool or int
        :return:

        .. note::

            Note that the ``'Critical_EndDateTime'`` would be based on the ``'Critical_StartDateTime'``
            if we consider the weather conditions on the day of incident occurrence;
            ``'StartDateTime'`` otherwise.

        **Test**::

            pip_start_hrs = -24
            nip_start_hrs = -24
            random_state  = 0
            update = False
            verbose = True

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> # h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, samples_only=True)
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> incid_loc_weather = h_model_plus.get_incident_location_weather(pickle_it=True)
        """

        self.__setattr__('RandomState', random_state)

        pickle_filename = make_filename(
            "weather", self.Route, self.WeatherCategory,
            # "_".join([self.Seasons] if isinstance(self.Seasons, str) else self.Seasons),
            str(self.PIP_StartHrs) + 'h', str(self.LP) + 'd' if self.LP else '-xd',
            str(self.NIP_StartHrs) + 'h',
            "sample-rs{}".format(random_state) if self.SamplesOnly else "", sep="_")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_weather = load_pickle(path_to_pickle)

        else:
            try:
                # -- Incidents data -------------------------------------------------------------------

                incidents = self.get_processed_incident_records(update=update, random_state=random_state)

                # -- Data integration for the specified prior-IP --------------------------------------

                incidents = self.get_pip_records(incidents)

                # Get prior-IP statistics of weather variables for each incident.
                pip_ukcp09_stats = self.get_pip_ukcp09_stats(incidents)

                # Get prior-IP statistics of radiation data for each incident.
                pip_radtob_stats = self.get_pip_radtob_stats(incidents)

                pip_data = incidents.join(pip_ukcp09_stats).join(pip_radtob_stats)

                pip_data['Incident_Reported'] = 1

                # -- Data integration for the specified non-IP ----------------------------------------

                if self.LP is None:
                    nip_data_ = self.get_nip_records(incidents, pip_data)
                else:
                    nip_data_ = incidents.copy(deep=True)
                    nip_data_.Critical_EndDateTime = \
                        nip_data_.Critical_StartDateTime + pd.Timedelta(days=self.LP)
                    nip_data_.Critical_StartDateTime = \
                        nip_data_.Critical_EndDateTime + pd.Timedelta(hours=self.NIP_StartHrs)
                    nip_data_.Critical_Period = nip_data_.apply(
                        lambda x: pd.interval_range(x.Critical_StartDateTime, x.Critical_EndDateTime),
                        axis=1)

                nip_ukcp09_stats = self.get_nip_ukcp09_stats(nip_data_, pip_data)

                nip_radtob_stats = self.get_nip_radtob_stats(nip_data_, pip_data)

                nip_data = nip_data_.join(nip_ukcp09_stats).join(nip_radtob_stats)

                nip_data['Incident_Reported'] = 0

                # -- Merge "pip_data" and "nip_data_" ------------------------------------------
                incident_location_weather = pd.concat(
                    [pip_data, nip_data], axis=0, ignore_index=True, sort=False)

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
                incident_location_weather = None

        return incident_location_weather

    def illustrate_weather_grid_buffer_circle(self, single_point=True, save_as=".tif", dpi=600,
                                              verbose=False):
        """

        :param single_point:
        :type single_point: bool
        :param save_as:
        :type save_as: str or None
        :param dpi:
        :type dpi: int or None
        :param verbose:
        :type verbose: bool or int
        :return:

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> h_model_plus.illustrate_weather_grid_buffer_circle(save_as=None)

            >>> h_model_plus.illustrate_weather_grid_buffer_circle(single_point=False, save_as=None)
        """

        incidents = self.get_processed_incident_records()

        obs_grids = self.UKCP.get_observation_grids()  # Grids for observing weather conditions
        obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

        # Illustration of the buffer circle
        if single_point:
            idx = incidents[incidents.StartLocation == incidents.EndLocation].index[-1]
        else:
            idx = incidents[incidents.StartLocation != incidents.EndLocation].index[-1]
        start_point, end_point, midpoint = incidents.loc[idx, ['StartXY', 'EndXY', 'MidpointXY']]

        bf_circle = create_weather_grid_buffer(start_point, end_point, midpoint, whisker=0)
        i_obs_grids = find_intersecting_weather_grid(
            bf_circle, obs_grids, obs_grids_geom, as_grid_id=False)

        plt.figure(figsize=(7, 6))

        ax = plt.subplot2grid((1, 1), (0, 0))

        for g in i_obs_grids:
            x_, y_ = g.exterior.xy
            ax.plot(x_, y_, color='#433f3f')

        ax.plot(
            [], 's', label="Weather observation grid", ms=16, color='none', markeredgecolor='#433f3f')
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

        if save_as:
            path_to_fig = self.cdd_trial("weather-grid-buffer-circle-{}".format(idx) + save_as)
            save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    # noinspection DuplicatedCode
    def plot_temperature_deviation(self, lp_range=14, add_err_bar=True, update=False,
                                   save_as=".tif", dpi=600, verbose=False):
        """
        Plot temperature deviation.

        :param lp_range:
        :param add_err_bar:
        :param update:
        :param save_as:
        :type save_as: str or None
        :param dpi:
        :type dpi: int or None
        :param verbose:
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> h_model_plus.plot_temperature_deviation(save_as=None, verbose=True)
        """

        default_lp = self.LP

        data_sets = []

        if verbose:
            print("Preparing datasets ... ")

        for d in range(1, lp_range + 1):
            self.__setattr__('LP', -d)

            if verbose:
                print("\t{} / {}".format(-d, -lp_range), end=" ... ")

            data_sets.append(
                self.get_incident_location_weather(update=update, pickle_it=True, verbose=False))

            if verbose:
                print("Done.")

        self.__setattr__('LP', default_lp)

        time_and_iloc = ['StartDateTime', 'EndDateTime', 'StanoxSection', 'IncidentDescription']
        selected_cols = time_and_iloc + ['Maximum_Temperature_max']

        base_data = data_sets[0]
        ip_temperature_max = base_data[base_data.Incident_Reported == 1][selected_cols]

        diff_means, diff_std = [], []
        for i in range(0, lp_range):

            data = data_sets[i]

            nip_temperature_max = data[data.Incident_Reported == 0][selected_cols]

            temp_diffs = pd.merge(
                ip_temperature_max, nip_temperature_max, on=time_and_iloc, suffixes=('_pip', '_nip'))
            temp_diff = temp_diffs.Maximum_Temperature_max_pip - temp_diffs.Maximum_Temperature_max_nip

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

    # noinspection DuplicatedCode
    def prep_training_and_test_sets(self, add_intercept=True):
        """
        Further process the integrated data set and split it into a training set and a test set.

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> _, training_data, test_data = h_model_plus.prep_training_and_test_sets()

            >>> training_data.tail()
                  FinancialYear  ... const
            2103           2015  ...   1.0
            2106           2015  ...   1.0
            2107           2015  ...   1.0
            2108           2015  ...   1.0
            2109           2015  ...   1.0
            [5 rows x 88 columns]

            >>> test_data.tail()
                  FinancialYear  ... const
            2255           2016  ...   1.0
            2258           2016  ...   1.0
            2260           2016  ...   1.0
            2264           2016  ...   1.0
            2270           2016  ...   1.0
            [5 rows x 88 columns]
        """

        # Get the mdata for modelling
        processed_data = self.get_incident_location_weather()

        processed_data.dropna(subset=['GLBL_IRAD_AMT_total'], inplace=True)
        processed_data.GLBL_IRAD_AMT_total = processed_data.GLBL_IRAD_AMT_total / 1000

        # Select features
        explanatory_variables = self.ExplanatoryVariables.copy()

        for v in explanatory_variables:
            if not processed_data[processed_data[v].isna()].empty:
                processed_data.dropna(subset=[v], inplace=True)

        # processed_data = processed_data[
        #     explanatory_variables + ['Incident_Reported', 'StartDateTime', 'EndDateTime',
        #                              'DelayMinutes']]

        # Remove outliers
        if 95 <= self.OutlierPercentile <= 100:
            upper_limit = np.percentile(processed_data.DelayMinutes, self.OutlierPercentile)
            processed_data = processed_data[processed_data.DelayMinutes <= upper_limit]
        # from pyhelpers.ops import get_extreme_outlier_bounds
        # l, u = get_extreme_outlier_bounds(processed_data.DelayMinutes, k=1.5)
        # processed_data = processed_data[
        #     processed_data.DelayMinutes.between(l, u, inclusive=True)]

        # Add the intercept
        if add_intercept:
            processed_data['const'] = 1.0

        # Set the outcomes of non-incident records to 0
        outcome_columns = ['DelayMinutes', 'DelayCost', 'IncidentCount']
        processed_data.loc[processed_data.Incident_Reported == 0, outcome_columns] = 0

        # Select data before 2014 as training data set, with the rest being test set
        training_set = processed_data[processed_data.StartDateTime < datetime.datetime(2016, 1, 1)]
        test_set = processed_data[processed_data.StartDateTime >= datetime.datetime(2016, 1, 1)]

        self.__setattr__('TrainingSet', training_set)
        self.__setattr__('TestSet', test_set)

        return processed_data, training_set, test_set

    # noinspection DuplicatedCode
    def describe_training_set(self, save_as=".tif", dpi=600, verbose=False):
        """
        Describe basic statistics about the main explanatory variables.

        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> h_model_plus.describe_training_set(save_as=None)
        """

        _, training_set, _ = self.prep_training_and_test_sets()

        plt.figure(figsize=(14, 5))

        colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

        ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=3)
        training_set.Temperature_Category.value_counts().plot.bar(color='#537979', rot=0, fontsize=12)
        plt.xticks(range(0, 8), ['<24', '24', '25', '26', '27', '28', '29', '≥30'], rotation=0,
                   fontsize=12)
        ax1.text(7.5, -0.2, '(°C)', fontsize=12)
        plt.xlabel('Maximum temperature', fontsize=13, labelpad=8)
        plt.ylabel('Frequency', fontsize=12, rotation=0)
        ax1.yaxis.set_label_coords(0.0, 1.01)

        ax2 = plt.subplot2grid((1, 9), (0, 3))
        training_set.Temperature_Change_max.plot.box(color=colour, ax=ax2, widths=0.5, fontsize=12)
        ax2.set_xticklabels('')
        plt.xlabel('Temperature\nchange', fontsize=13, labelpad=10)
        plt.ylabel('(°C)', fontsize=12, rotation=0)
        ax2.yaxis.set_label_coords(0.05, 1.01)

        ax3 = plt.subplot2grid((1, 9), (0, 4), colspan=2)
        orient_cats = [
            x.replace('Track_Orientation_', '') for x in training_set.columns
            if x.startswith('Track_Orientation_')]
        track_orientation = pd.Series(
            [np.sum(training_set.Track_Orientation == x) for x in orient_cats], index=orient_cats)
        track_orientation.index = [i.replace('_', '-') for i in track_orientation.index]
        track_orientation.plot.bar(color='#a72a3d', rot=0, fontsize=12)
        # ax3.set_yticks(range(0, track_orientation.max() + 1, 100))
        plt.xlabel('Track orientation', fontsize=13, labelpad=8)
        plt.ylabel('Count', fontsize=12, rotation=0)
        ax3.yaxis.set_label_coords(0.0, 1.01)

        ax4 = plt.subplot2grid((1, 9), (0, 6))
        training_set.GLBL_IRAD_AMT_total.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
        ax4.set_xticklabels('')
        plt.xlabel('Maximum\nirradiation', fontsize=13, labelpad=10)
        plt.ylabel('(KJ/m$^2$)', fontsize=12, rotation=0)
        ax4.yaxis.set_label_coords(0.2, 1.01)

        ax5 = plt.subplot2grid((1, 9), (0, 7))
        training_set.Precipitation_max.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
        ax5.set_xticklabels('')
        plt.xlabel('Maximum\nprecipitation', fontsize=13, labelpad=10)
        plt.ylabel('(mm)', fontsize=12, rotation=0)
        ax5.yaxis.set_label_coords(0.0, 1.01)

        ax6 = plt.subplot2grid((1, 9), (0, 8))
        hottest_heretofore = training_set.Hottest_Heretofore.value_counts()
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

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> model_results = h_model_plus.logistic_regression()
        """

        # Get data for modelling
        _, training_set, test_set = self.prep_training_and_test_sets()

        if add_intercept:
            explanatory_variables = ['const'] + self.ExplanatoryVariables
        else:
            explanatory_variables = self.ExplanatoryVariables.copy()

        np.random.seed(random_state)

        try:
            if self.ModelType == 'logit':
                mod = sm_dcm.Logit(training_set.Incident_Reported, training_set[explanatory_variables])
            else:
                mod = sm_dcm.Probit(training_set.Incident_Reported, training_set[explanatory_variables])
            result_summary = mod.fit(maxiter=1000, full_output=True, disp=True)  # method='newton'
            print(result_summary.summary2()) if verbose else print("")

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

            self.__setattr__('FPR', fpr)
            self.__setattr__('TPR', tpr)
            self.__setattr__('AUC', auc)
            self.__setattr__('Threshold', threshold)

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
            var_names = ['training_set', 'test_set',
                         'result_summary', 'model_accuracy', 'incident_accuracy', 'threshold']
            resources = {k: repo[k] for k in list(var_names)}
            result_pickle = make_filename(
                "result", self.Route, self.WeatherCategory,
                str(self.PIP_StartHrs) + 'h', str(self.LP) + 'd' if self.LP else '-xd',
                str(self.NIP_StartHrs) + 'h', sep="_")

            save_pickle(resources, self.cdd_trial(result_pickle), verbose=verbose)

        return training_set, test_set, result_summary, model_accuracy, incident_accuracy, threshold

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

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> _ = h_model_plus.logistic_regression()
            >>> h_model_plus.plot_roc(save_as=None)
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

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> _ = h_model_plus.logistic_regression()
            >>> h_model_plus.plot_pred_likelihood(save_as=None)
        """

        test_set = self.__getattribute__('TestSet')
        threshold = self.__getattribute__('Threshold')

        incident_ind = test_set.Incident_Reported == 1

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


"""
# 'IR' - Broken/cracked/twisted/buckled/flawed rail
# 'XH' - Severe heat affecting infrastructure the responsibility of Network Rail 
#        (excl. Heat related speed restrictions)
# 'IB' - Points failure
# 'JH' - Critical Rail Temperature speeds, (other than buckled rails)
'IZ' - Other infrastructure causes INF OTHER
'XW' - High winds affecting infrastructure the responsibility of Network
'IS' - Track defects (other than rail defects) inc. fish plates, wet beds etc.

0. 'IR'
1. 'XH'
2. 'IB'
3. 'IR', 'XH', 'IB'
4. 'JH'
5. 'IR', 'XH', 'IB', 'JH'
6. 'IR', 'IB'
"""

# == 'IR' - Broken/cracked/twisted/buckled/flawed rail ==


# == 'IB' - Points failure ==


# == 'IR', 'XH', 'IB' ==


# == 'JH' - Critical Rail Temperature speeds, (other than buckled rails) ==


# == 'IR', 'XH', 'IB', 'JH' ==


# == 'IR', 'IB' ==
