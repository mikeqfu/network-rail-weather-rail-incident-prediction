"""
An extended version of the prototype model in the context of heat-related rail Incidents.

'IR' - Broken/cracked/twisted/buckled/flawed rail
'XH' - Severe heat affecting infrastructure the responsibility of Network Rail
       (excl. Heat related speed restrictions)
'IB' - Points failure
'JH' - Critical Rail Temperature speeds, (other than buckled rails)
# 'IZ' - Other infrastructure causes INF OTHER
# 'XW' - High winds affecting infrastructure the responsibility of Network
# 'IS' - Track defects (other than rail defects) incl. fishplates, wet beds etc.

0. 'IR'
1. 'XH'
2. 'IB'
3. 'IR', 'XH', 'IB'
4. 'JH'
5. 'IR', 'XH', 'IB', 'JH'
6. 'IR', 'IB'

More:
-------------- | ------------------ | -----------------------------------------------------------
IncidentReason | IncidentReasonName | IncidentReasonDescription
-------------- | ------------------ | -----------------------------------------------------------
IQ             |   TRACK SIGN       | Trackside sign blown down/light out etc.
IW             |   COLD             | Non-severe - Snow/Ice/Frost affecting infr equipment, ...
OF             |   HEAT/WIND        | Blanket speed restriction for extreme heat or high wind ...
Q1             |   TKB PUMPS        | Takeback Pumps
X4             |   BLNK REST        | Blanket speed restriction for extreme heat or high wind
XW             |   WEATHER          | Severe Weather not snow affecting infrastructure, resp. ...
XX             |   MISC OBS         | Msc items on line (incl. trees) due to Weather, resp. of...
-------------- | ------------------ | -----------------------------------------------------------
"""

import datetime
import functools
import itertools
import multiprocessing
import os

import descartes
import matplotlib.cbook
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import shapely.geometry
import shapely.geometry
import shapely.ops
from pyhelpers.dirs import cd
from pyhelpers.geom import wgs84_to_osgb36
from pyhelpers.settings import mpl_preferences, pd_preferences
from pyhelpers.store import load_pickle, save_fig, save_pickle
from sklearn import metrics

from src.preprocessor.metex import METEX
from src.preprocessor.weather import MIDAS, UKCP09
from src.shaft.feature import (categorise_temperatures, categorise_track_orientations,
                               get_data_by_meteorological_seasons)
from src.shaft.geometry import (create_weather_grid_buffer, find_closest_met_stn,
                                find_intersecting_weather_grid)
from src.utils import make_filename, points_from_xy


class HeatAttributedIncidentsPlus:
    """A data model for heat-attributed rail Incidents."""

    NAME = ''

    METEx = METEX()
    UKCP09 = UKCP09()
    MIDAS = MIDAS()

    def __init__(self, trial_id, route_names=None, weather_category='Heat',
                 seasons=None, reason_codes=None, pip_start_hrs=-24, nip_start_hrs=-24, lp_days=None,
                 sample_only=False, outlier_pctl=100, model_type='LogisticRegression',
                 random_state=0):
        """
        A data model for heat-attributed rail Incidents.

        :param trial_id: ID number of a trial to be run
        :type trial_id: int or str
        :param route_names: name of a NR Route
        :type route_names: str or list or None
        :param weather_category: Weather category
        :type weather_category: str or None
        :param seasons: season(s)
        :type seasons: str or list or None
        :param reason_codes: incident reason code(s)
        :type reason_codes: str or list or None
        :param pip_start_hrs: how many hours prior to the recorded start of an incident
        :type pip_start_hrs: int or float
        :param nip_start_hrs: how many hours prior to the defined start of a prior-incident period
        :type nip_start_hrs: int or float
        :param lp_days: number of days of a latent period between prior-incident and non-incident period
        :type lp_days: int or float or None
        :param sample_only: whether to test on a subset only
        :type sample_only: bool or int
        :param outlier_pctl: percentile threshold to exclude those incident records regarded as outliters
        :type outlier_pctl: int
        :param model_type: 'logit' or 'probit'
        :type model_type: str

        **Test**::

            >>> from modeller import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=0)

        """

        # Settings
        pd_preferences()
        mpl_preferences(backend='TkAgg', font_name='Times New Roman')
        # warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

        self.trial_id = f"{trial_id}"

        if route_names is None:
            route_names = ['Anglia', 'Wessex', 'Wales', 'North and East']
        self.route_names = [route_names] if isinstance(route_names, str) else route_names

        self.weather_category = weather_category

        if seasons is None:
            seasons = ['summer']
        self.seasons = [seasons] if isinstance(seasons, str) else seasons

        if reason_codes is None:
            reason_codes = ['IR', 'XH', 'IB', 'JH']
        self.reason_codes = reason_codes if isinstance(reason_codes, list) else [reason_codes]

        self.pip_start_hrs = pip_start_hrs
        self.nip_start_hrs = nip_start_hrs
        self.lp_days = lp_days

        self.lp_Anglia = lambda x: -20 if x in range(24, 29) else (-13 if x > 28 else 0)
        self.lp_Wessex = lambda x: -30 if x in range(24, 29) else (-25 if x > 28 else 0)
        self.lp_NorthAndEast = lambda x: -18 if x in range(24, 29) else (-16 if x > 28 else 0)
        self.lp_Wales = lambda x: -19 if x in range(24, 29) else (-5 if x > 28 else 0)

        self.samples_only = sample_only
        if isinstance(self.samples_only, bool):
            self.sample_size = 10
        elif isinstance(self.samples_only, int):
            self.sample_size = self.samples_only

        self.outlier_percentile = outlier_pctl

        def mode(x):
            # noinspection PyUnresolvedReferences
            return scipy.stats.mode(np.around(x))[0]

        self.UKCP09_stats_calc = {
            'Maximum_Temperature': (
                np.nanmax, np.nanmin, np.nanmedian, np.nanmean, np.nanstd, mode),
            'Minimum_Temperature': (
                np.nanmax, np.nanmin, np.nanmedian, np.nanmean, np.nanstd, mode),
            'Temperature_Change': (
                np.nanmax, np.nanmedian, np.nanmean, np.nanstd, mode),
            'Precipitation': (
                np.nansum, np.nanmax, np.nanmin, np.nanmedian, np.nanmean, np.nanstd, mode),
        }

        self.RADTOB_stats_calc = {
            'GLBL_IRAD_AMT': np.nansum,
        }

        ukcp09_variable_names_ = [
            [k, [i.__name__.replace('nan', '') for i in v] if isinstance(v, tuple) else [
                v.__name__.replace('nan', '')]]
            for k, v in self.UKCP09_stats_calc.items()]
        ukcp09_variable_names = [['_'.join([x, z]) for z in y] for x, y in ukcp09_variable_names_]
        self.ukcp09_variable_names = list(itertools.chain.from_iterable(ukcp09_variable_names))

        self.variable_names = [
            # 'Maximum_Temperature_max',
            # 'Maximum_Temperature_min',
            # 'Maximum_Temperature_median',
            # 'Maximum_Temperature_mean',
            # 'Maximum_Temperature_std',
            # 'Maximum_Temperature_mode',
            # 'Minimum_Temperature_max',
            # 'Minimum_Temperature_min',
            # 'Minimum_Temperature_median',
            # 'Minimum_Temperature_mean',
            # 'Minimum_Temperature_std',
            # 'Minimum_Temperature_mode',
            'Temperature_Change_max',
            # 'Temperature_Change_median',
            # 'Temperature_Change_mean',
            # 'Temperature_Change_std',
            # 'Temperature_Change_mode',
            # 'Precipitation_sum',
            # 'Precipitation_max',
            # 'Precipitation_min',
            # 'Precipitation_median',
            'Precipitation_mean',
            # 'Precipitation_std',
            # 'Precipitation_mode',
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
            'Maximum_Temperature_max [30.0, inf)°C',
            # # 'Maximum_Temperature_median [-inf, 24.0)°C',
            # 'Maximum_Temperature_median [24.0, 25.0)°C',
            # 'Maximum_Temperature_median [25.0, 26.0)°C',
            # 'Maximum_Temperature_median [26.0, 27.0)°C',
            # 'Maximum_Temperature_median [27.0, 28.0)°C',
            # 'Maximum_Temperature_median [28.0, 29.0)°C',
            # 'Maximum_Temperature_median [29.0, 30.0)°C',
            # 'Maximum_Temperature_median [30.0, inf)°C',
        ]

        self.model_type = model_type

        self.random_state = random_state

    @staticmethod
    def cdd(*sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype_ext\\heat" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype_ext\\heat" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> os.path.relpath(h_model_plus.cdd())
            'models\\prototype_ext\\heat'
        """

        path = cd("models", "prototype_ext", "heat", *sub_dir, mkdir=mkdir)

        return path

    def cdd_trial(self, *sub_dir, mkdir=False):
        """
        Change directory to "models\\prototype_ext\\heat\\<trial_id>" and subdirectories / a file.

        :param sub_dir: name of directory or names of directories (and/or a filename)
        :type sub_dir: str
        :param mkdir: whether to create a directory, defaults to ``False``
        :type mkdir: bool
        :return: absolute path to "models\\prototype_ext\\heat\\<trial_id>" and subdirectories / a file
        :rtype: str

        **Test**::

            >>> import os
            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> os.path.relpath(h_model_plus.cdd_trial())
            'models\\prototype_ext\\heat\\0'
        """

        path = self.cdd(self.trial_id, *sub_dir, mkdir=mkdir)

        return path

    # == Set Prior-IP, LP and Non-IP ===============================================================

    def get_pip_records(self, incident_data):
        """
        Prior-incident periods.

        :param incident_data: data of incident records
        :type incident_data: pandas.DataFrame
        :return: Incidents records together with defined prior-incident periods
        :rtype: pandas.DataFrame

        **Test**::

            >>> from modeller import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> incident_records = h_model_plus.get_processed_incident_data()

            >>> dat = h_model_plus.get_pip_records(incident_records)
        """

        data = incident_data.copy()

        data['Incident_Duration'] = data['EndDateTime'] - data['StartDateTime']

        # End date and time of the prior IP
        data['Critical_EndDateTime'] = data['StartDateTime'].dt.round('H')

        # Start date and time of the prior IP
        critical_start_dt = data['Critical_EndDateTime'].map(
            lambda x: x + pd.Timedelta(
                hours=self.pip_start_hrs if x.time() > datetime.time(9) else self.pip_start_hrs * 2))
        data.insert(
            data.columns.get_loc('Critical_EndDateTime'), 'Critical_StartDateTime', critical_start_dt)

        # Prior-IP dates of each incident
        data['Critical_Period'] = data[['Critical_StartDateTime', 'Critical_EndDateTime']].apply(
            lambda x: pd.interval_range(x[0], x[1]), axis=1)

        return data

    def set_lp_and_nip(self, route_name, ip_max_temp_max, ip_start_dt):
        """
        Determine latent period for a given date/time and the maximum temperature.

        :param route_name:
        :type route_name:
        :param ip_max_temp_max:
        :type ip_max_temp_max:
        :param ip_start_dt:
        :type ip_start_dt:
        :return:
        :rtype:
        """

        if route_name == 'Anglia':
            lp = self.lp_Anglia(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -20
            # elif ip_max_temp_max > 28:
            #     lp = -13
            # else:
            #     lp = 0
        elif route_name == 'Wessex':
            lp = self.lp_Wessex(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -30
            # elif ip_max_temp_max > 28:
            #     lp = -25
            # else:
            #     lp = 0
        elif route_name == 'North and East':
            lp = self.lp_NorthAndEast(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -18
            # elif ip_max_temp_max > 28:
            #     lp = -16
            # else:
            #     lp = 0
        else:  # route_name == 'Wales':
            lp = self.lp_Wales(ip_max_temp_max)
            # if 24 <= ip_max_temp_max <= 28:
            #     lp = -19
            # elif ip_max_temp_max > 28:
            #     lp = -5
            # else:
            #     lp = 0

        critical_end_dt = ip_start_dt + pd.Timedelta(days=lp)

        critical_start_dt = critical_end_dt + pd.Timedelta(hours=self.nip_start_hrs)

        critical_period = pd.interval_range(critical_start_dt, critical_end_dt)

        return critical_start_dt, critical_end_dt, critical_period

    def get_nip_records(self, incidents, prior_ip_data):
        """
        Non-incident periods.

        :param incidents: data of incident records
        :type incidents: pandas.DataFrame
        :param prior_ip_data:
        :type prior_ip_data:
        :return: incident records and their associated prior-incident periods
        :rtype: pandas.DataFrame

        **Test**::

            >>> from modeller import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> incident_data = h_model_plus.get_processed_incident_data()

            >>> dat = h_model_plus.get_pip_records(incident_data)
        """

        non_ip_data = incidents.copy()  # Get Weather data that did not cause any incident

        if self.lp_days is None:
            col_names = ['Critical_StartDateTime', 'Critical_EndDateTime', 'Critical_Period']
            non_ip_data[col_names] = prior_ip_data.apply(
                lambda x: pd.Series(self.set_lp_and_nip(
                    x['route_names'], x['Maximum_Temperature_max'], x['Critical_StartDateTime'])),
                axis=1)

        else:
            non_ip_data.Critical_EndDateTime = \
                non_ip_data.Critical_StartDateTime + pd.Timedelta(days=self.lp_days)
            non_ip_data.Critical_StartDateTime = \
                non_ip_data.Critical_EndDateTime + pd.Timedelta(hours=self.nip_start_hrs)
            non_ip_data.Critical_Period = \
                non_ip_data[['Critical_StartDateTime', 'Critical_EndDateTime']].apply(
                    lambda x: pd.interval_range(x[0], x[1]), axis=1)

        return non_ip_data

    # == UKCP09 ====================================================================================

    def calculate_ukcp09_stats(self, weather_data):
        """
        Calculate the statistics for the Weather variables (except radiation).

        :param weather_data: data set of Weather observations (for a certain period)
        :type weather_data: pandas.DataFrame
        :return: some statistics of the UKCP09 data in the data set of Weather observations
        :rtype: list

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)
        """

        if weather_data.empty:
            weather_stats_info = [np.nan] * sum(map(np.count_nonzero, self.UKCP09_stats_calc.values()))

        else:
            # Create a pseudo id for groupby() & aggregate()
            weather_data['Pseudo_ID'] = 0
            weather_stats = weather_data.groupby('Pseudo_ID').aggregate(self.UKCP09_stats_calc)
            # a, b = [list(x) for x in weather_stats.columns.levels]
            # weather_stats.columns = ['_'.join(x) for x in itertools.product(a, b)]
            # if not weather_stats.empty:
            #     stats_info = weather_stats.values[0].tolist()
            # else:
            #     stats_info = [np.nan] * len(weather_stats.columns)

            weather_stats_info = weather_stats.values[0].tolist()

        return weather_stats_info

    def integrate_pip_ukcp09_data(self, grids, period, pickle_it=True):
        """
        Gather gridded Weather observations of the given period for each incident record.

        :param grids: e.g. grids = Incidents.Weather_Grid.iloc[0]
        :param period: e.g. period = Incidents.Critical_Period.iloc[0]
        :param pickle_it:
        :return:

        **Test**::

            grids = Incidents.Weather_Grid.iloc[1]
            period = Incidents.Critical_Period.iloc[1]

        """

        # Find Weather data for the specified period
        pip_weather = self.UKCP09.query_by_datetime_grid(period=period, grids=grids, pickle_it=pickle_it)
        # Calculate the max/min/avg for Weather parameters during the period
        weather_stats = self.calculate_ukcp09_stats(pip_weather)

        # Whether "max_temp = weather_stats[0]" is the hottest of year so far
        obs_by_far = self.UKCP09.query_by_grid_datetime_heretofore(
            period=period, grids=grids, pickle_it=pickle_it)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_pip_ukcp09_stats(self, incidents, weather_grid_col='Weather_Grid',
                             critical_period_col='Critical_Period'):
        """
        Get prior-IP statistics of Weather variables for each incident.

        :param incidents: data of Incidents
        :type incidents: pandas.DataFrame
        :param weather_grid_col:
        :param critical_period_col:
        :return: statistics of Weather observation data for each incident record during the prior IP
        :rtype: pandas.DataFrame

        **Test**::

            weather_grid_col = 'Weather_Grid'
            critical_period_col = 'Critical_Period'
        """

        pip_weather_stats = incidents[[weather_grid_col, critical_period_col]].apply(
            lambda x: pd.Series(self.integrate_pip_ukcp09_data(x[0], x[1])), axis=1)

        w_col_names = self.ukcp09_variable_names + ['Hottest_Heretofore']

        pip_weather_stats.columns = w_col_names

        pip_weather_stats['Temperature_Change_max'] = abs(
            pip_weather_stats['Maximum_Temperature_max'] - pip_weather_stats['Minimum_Temperature_min'])

        pip_weather_stats['Temperature_Change_min'] = abs(
            pip_weather_stats['Maximum_Temperature_min'] - pip_weather_stats['Minimum_Temperature_max'])

        return pip_weather_stats

    def integrate_nip_ukcp09_data(self, grids, period, stanox_section, pip_data, pickle_it=True):
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
        nip_weather = self.UKCP09.query_by_datetime_grid(period=period, grids=grids, pickle_it=pickle_it)

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
        obs_by_far = self.UKCP09.query_by_grid_datetime_heretofore(
            period=period, grids=grids, pickle_it=pickle_it)
        weather_stats.append(1 if weather_stats[0] > obs_by_far.Maximum_Temperature.max() else 0)

        return weather_stats

    def get_nip_ukcp09_stats(self, nip_data_, pip_data, weather_grid_col='Weather_Grid',
                             critical_period_col='Critical_Period', stanox_section_col='StanoxSection'):
        """
        Get prior-IP statistics of Weather variables for each incident.

        :param nip_data_: non-IP data
        :type nip_data_: pandas.DataFrame
        :param pip_data: prior-IP data
        :type pip_data: pandas.DataFrame
        :param weather_grid_col:
        :param critical_period_col:
        :param stanox_section_col:
        :return: stats of UKCP09 data for each incident record during the non-incident period
        :rtype: pandas.DataFrame

        **Test**::

            weather_grid_col = 'Weather_Grid'
            critical_period_col = 'Critical_Period'
            stanox_section_col = 'StanoxSection'
        """

        non_ip_weather_stats = \
            nip_data_[[weather_grid_col, critical_period_col, stanox_section_col]].apply(
                lambda x: pd.Series(self.integrate_nip_ukcp09_data(x[0], x[1], x[2], pip_data)), axis=1)

        non_ip_weather_stats.columns = self.ukcp09_variable_names + ['Hottest_Heretofore']

        non_ip_weather_stats['Temperature_Change_max'] = \
            non_ip_weather_stats.Maximum_Temperature_max - non_ip_weather_stats.Minimum_Temperature_min
        non_ip_weather_stats['Temperature_Change_min'] = \
            non_ip_weather_stats.Maximum_Temperature_min - non_ip_weather_stats.Minimum_Temperature_max

        return non_ip_weather_stats

    # == RADTOB ====================================================================================

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
            # stats_info = [np.nan] * (sum(Map(np.count_nonzero, self.RADTOBStatsCalc.values())))
            stats_info = np.nan

        else:
            # if 24 not in midas_radtob.OB_HOUR_COUNT:
            #     midas_radtob = midas_radtob.append(midas_radtob.iloc[-1, :])
            #     midas_radtob.VERSION_NUM.iloc[-1] = 0
            #     midas_radtob.OB_HOUR_COUNT.iloc[-1] = midas_radtob.OB_HOUR_COUNT.iloc[0:-1].sum()
            #     midas_radtob.GLBL_IRAD_AMT.iloc[-1] = midas_radtob.GLBL_IRAD_AMT.iloc[0:-1].sum()

            if 24 in midas_radtob.OB_HOUR_COUNT.values:
                temp = midas_radtob[midas_radtob.OB_HOUR_COUNT == 24]
                midas_radtob = pd.concat([temp, midas_radtob.loc[temp.last_valid_index() + 1:]])

            radtob_stats = midas_radtob.groupby('SRC_ID').aggregate(self.RADTOB_stats_calc)
            stats_info = radtob_stats.values.flatten()[0]

        return stats_info

    def integrate_pip_radtob(self, met_stn_id, period, route_name):
        """
        Gather solar radiation of the given period for each incident record.

        :param met_stn_id:
        :param period:
        :param route_name:
        :return:

        **Test**::

            met_stn_id = Incidents.Met_SRC_ID.iloc[4]
            period = Incidents.Critical_Period.iloc[4]
            route_name = Incidents.Route.iloc[4]
            use_suppl_dat = True
        """

        # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
        #
        # try:
        #     prior_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
        # except KeyError:
        #     prior_ip_radtob = pd.DataFrame()

        prior_ip_radtob = self.MIDAS.query_radtob_by_grid_datetime(met_stn_id, period, route_name)

        radtob_stats = self.calculate_radtob_stats(prior_ip_radtob)

        return radtob_stats

    def get_pip_radtob_stats(self, incidents, met_stn_id_col='Met_SRC_ID',
                             critical_period_col='Critical_Period', route_name_col='Route'):
        """
        Get prior-IP statistics of radiation data for each incident.

        :param incidents: data of Incidents
        :type incidents: pandas.DataFrame
        :param met_stn_id_col:
        :param critical_period_col:
        :param route_name_col:
        :return: statistics of radiation data for each incident record during the prior IP
        :rtype: pandas.DataFrame

        **Test**::

            Incidents
        """

        prior_ip_radtob_stats = incidents[[met_stn_id_col, critical_period_col, route_name_col]].apply(
            lambda x: pd.Series(self.integrate_pip_radtob(x[0], x[1], x[2])), axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        prior_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return prior_ip_radtob_stats

    def integrate_nip_radtob(self, met_stn_id, period, route_name, prior_ip_data, stanox_section):
        """
        Gather solar radiation of the corresponding non-incident period for each incident record.

        :param met_stn_id: e.g. met_stn_id = nip_data_.Met_SRC_ID.iloc[1]
        :param period: e.g. period = nip_data_.Critical_Period.iloc[1]
        :param route_name:
        :param stanox_section: e.g. location = nip_data_.StanoxSection.iloc[0]
        :param prior_ip_data:
        :return:
        """

        # irad_obs_ = irad_obs[irad_obs.SRC_ID.isin(met_stn_id)]
        #
        # try:
        #     non_ip_radtob = irad_obs_.set_index('OB_END_DATE').loc[period]
        # except KeyError:
        #     non_ip_radtob = pd.DataFrame()

        non_ip_radtob = self.MIDAS.query_radtob_by_grid_datetime(met_stn_id, period, route_name)

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

    def get_nip_radtob_stats(self, non_ip_data, prior_ip_data, met_stn_id_col='Met_SRC_ID',
                             critical_period_col='Critical_Period', route_name_col='Route',
                             stanox_section_col='StanoxSection'):
        """
        Get prior-IP statistics of radiation data for each incident.

        :param non_ip_data: non-IP data
        :type non_ip_data: pandas.DataFrame
        :param prior_ip_data: prior-IP data
        :type prior_ip_data: pandas.DataFrame
        :param met_stn_id_col:
        :param critical_period_col:
        :param route_name_col:
        :param stanox_section_col:
        :return: statistics of radiation data for each incident record during the non-incident period
        :rtype: pandas.DataFrame
        """

        cols = [met_stn_id_col, critical_period_col, route_name_col, stanox_section_col]
        non_ip_radtob_stats = non_ip_data[cols].apply(
            lambda x: pd.Series(self.integrate_nip_radtob(x[0], x[1], x[2], prior_ip_data, x[3])),
            axis=1)

        # r_col_names = specify_weather_variable_names(integrator.specify_radtob_stats_calculations())
        # r_col_names += ['GLBL_IRAD_AMT_total']
        non_ip_radtob_stats.columns = ['GLBL_IRAD_AMT_total']  # r_col_names

        return non_ip_radtob_stats

    # == Data of Weather conditions ================================================================

    def _prep_incident_data(self, reason_codes=None, seasons=None, start_datetime=None):
        """

        :param reason_codes:
        :param seasons:
        :param start_datetime:
        :return:
        """
        self.METEx.view_schedule8_cost_by_day_location_reason(
            route_name=self.route_names, weather_category=self.weather_category)
        dat = self.METEx.schedule8_cost_by_day_location_reason.copy()

        dat['StartDateTime'], dat['EndDateTime'] = map(
            pd.to_datetime, [dat['StartDateTime'], dat['EndDateTime']])

        if reason_codes is None:
            reason_codes = self.reason_codes
        dat = dat[dat['IncidentReasonCode'].isin(reason_codes) & (dat['WeatherCategory'] != 'Cold')]

        if seasons is None:
            seasons = self.seasons
        data_by_season = get_data_by_meteorological_seasons(
            data=dat, seasons=seasons, datetime_col='StartDateTime')

        if start_datetime is None:
            start_datetime = datetime.datetime(2006, 4, 1)
        data = data_by_season[(data_by_season['StartDateTime'] >= start_datetime)]

        if self.samples_only:  # For testing purpose only
            data = data.sample(n=self.sample_size, random_state=self.random_state)

        data['StartEasting'], data['StartNorthing'] = wgs84_to_osgb36(
            data['StartLongitude'], data['StartLatitude'])
        data['EndEasting'], data['EndNorthing'] = wgs84_to_osgb36(
            data['EndLongitude'], data['EndLatitude'])

        # Add 'Start_XY' column
        if 'Start_XY' not in data.columns:
            data['Start_XY'] = points_from_xy(data[['StartEasting', 'StartNorthing']])
        # Add 'End_XY' column
        if 'End_XY' not in data.columns:
            data['End_XY'] = points_from_xy(data[['EndEasting', 'EndNorthing']])
        # Add 'Midpoint_XY' column
        # Incidents['Midpoint_XY'] = Incidents[['Start_XY', 'End_XY']].apply(
        #     lambda xy: get_geometric_midpoint(xy[0], xy[1], as_geom=True), axis=1)
        data['Midpoint_XY'] = data[['Start_XY', 'End_XY']].apply(
            lambda xy: shapely.geometry.LineString((xy[0], xy[1])).centroid, axis=1)

        # Get radiation stations
        self.MIDAS.read_radiation_monitoring_stations()  # Met station locations
        met_stations = self.MIDAS.radiation_monitoring_stations.copy()
        met_stations_geom = shapely.geometry.MultiPoint(list(met_stations['XY']))

        # Find the closest radiation stations to each of the midpoints of incident location
        with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
            data['Met_SRC_ID'] = p.map(
                functools.partial(
                    find_closest_met_stn, met_stations=met_stations,
                    met_stations_geom=met_stations_geom),
                data['Midpoint_XY'])
            met_src_id_ = p.map(
                functools.partial(
                    find_closest_met_stn, met_stations=met_stations,
                    met_stations_geom=met_stations_geom),
                data['Start_XY'])

        data['Met_SRC_ID'] += pd.Series(met_src_id_)
        data['Met_SRC_ID'] = data['Met_SRC_ID'].map(lambda x: list(dict.fromkeys(x)))

        # Make a buffer zone for Weather data aggregation
        data['Buffer_Zone'] = data[['Start_XY', 'End_XY', 'Midpoint_XY']].apply(
            lambda x: create_weather_grid_buffer(x[0], x[1], x[2], min_radius=500, whisker=500),
            axis=1)

        # Get Weather observation grids
        self.UKCP09.read_observation_grids()  # Grids for observing Weather conditions
        obs_grids = self.UKCP09.observation_grids.copy()
        obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids['Grid_XY']))

        # Find UKCP09 grids that intersect with the buffer zones for each incident location
        with multiprocessing.Pool(processes=os.cpu_count() - 1) as p:
            data['Weather_Grid'] = p.map(
                functools.partial(
                    find_intersecting_weather_grid, obs_grids=obs_grids,
                    obs_grids_geom=obs_grids_geom),
                data['Buffer_Zone'])
            # data['Weather_Grid'] = data['Buffer_Zone'].Map(
            #     lambda x: find_intersecting_weather_grid(x, obs_grids, obs_grids_geom))

        return data

    def get_processed_incident_data(self, update=False, random_state=1):
        """

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> # -- Sample ---------------------------------------------------------------------------

            >>> # Regional; heat
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, sample_only=True)
            >>> incident_records = h_model_plus.get_processed_incident_data(update=True)

            >>> # Regional; heat and null Weather category
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, weather_category=None,
            ...                                            sample_only=True)
            >>> incident_records = h_model_plus.get_processed_incident_data(update=True)

            >>> # -- The whole data set ---------------------------------------------------------------

            >>> # Regional; heat
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)
            >>> incident_records = h_model_plus.get_processed_incident_data(update=True)

            >>> # Regional; heat and null Weather category
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, weather_category=None)
            >>> incident_records = h_model_plus.get_processed_incident_data(update=True)
        """

        if self.random_state != random_state:
            self.random_state = random_state

        pickle_filename = make_filename(
            "Incidents", self.route_names, self.weather_category,
            # "" if self.seasons is None else "_".join(self.seasons),
            "_".join(self.reason_codes),
            "sample" if self.samples_only else "",
            sep="_")

        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            data = load_pickle(path_to_pickle)

        else:
            data = self._prep_incident_data()

            save_pickle(data, path_to_pickle, verbose=True)

        return data

    def get_incident_location_weather(self, random_state=1, update=False, pickle_it=False,
                                      verbose=True):
        """
        Process data of Weather conditions for each incident location.

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
            if we consider the Weather conditions on the day of incident occurrence;
            ``'StartDateTime'`` otherwise.

        **Test**::

            pip_start_hrs = -24
            nip_start_hrs = -24
            random_state = 0
            update = False
            verbose = True

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> # -- Sample ---------------------------------------------------------------------------

            >>> # Regional; heat
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, sample_only=True)
            >>> dat = h_model_plus.get_incident_location_weather(update=True, pickle_it=True)

            >>> # Regional; heat and null Weather category
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, weather_category=None,
            ...                                            sample_only=True)
            >>> dat = h_model_plus.get_incident_location_weather(update=True, pickle_it=True)

            >>> # -- The whole data set ---------------------------------------------------------------

            >>> # Regional; heat
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)
            >>> dat = h_model_plus.get_incident_location_weather(update=True, pickle_it=True)

            >>> # Regional; heat and null Weather category
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2, weather_category=None)
            >>> dat = h_model_plus.get_incident_location_weather(update=True, pickle_it=True)
        """

        if self.random_state != random_state:
            self.random_state = random_state

        pickle_filename = make_filename(
            "Weather", self.route_names, self.weather_category,
            # "_".join([self.Seasons] if isinstance(self.Seasons, str) else self.Seasons),
            str(self.pip_start_hrs) + 'h', str(self.lp_days) + 'd' if self.lp_days else '-xd',
            str(self.nip_start_hrs) + 'h',
            "sample" if self.samples_only else "", sep="_")
        path_to_pickle = self.cdd_trial(pickle_filename)

        if os.path.isfile(path_to_pickle) and not update:
            incident_location_weather = load_pickle(path_to_pickle)

        else:
            try:
                # -- Incidents data -------------------------------------------------------------------

                incidents = self.get_processed_incident_data(update=False, random_state=random_state)

                # -- Data integration for the specified prior-IP --------------------------------------

                incidents = self.get_pip_records(incidents)

                # Get prior-IP statistics of Weather variables for each incident.
                pip_ukcp09_stats = self.get_pip_ukcp09_stats(
                    incidents, weather_grid_col='Weather_Grid', critical_period_col='Critical_Period')

                # Get prior-IP statistics of radiation data for each incident.
                pip_radtob_stats = self.get_pip_radtob_stats(
                    incidents, met_stn_id_col='Met_SRC_ID', critical_period_col='Critical_Period',
                    route_name_col='Route')

                pip_data = incidents.join(pip_ukcp09_stats).join(pip_radtob_stats)

                pip_data['Incident_Reported'] = 1

                # -- Data integration for the specified non-IP ----------------------------------------

                nip_data_ = self.get_nip_records(incidents, pip_data)

                nip_ukcp09_stats = self.get_nip_ukcp09_stats(
                    nip_data_, pip_data, weather_grid_col='Weather_Grid',
                    critical_period_col='Critical_Period', stanox_section_col='StanoxSection')

                nip_radtob_stats = self.get_nip_radtob_stats(
                    nip_data_, pip_data, met_stn_id_col='Met_SRC_ID',
                    critical_period_col='Critical_Period', route_name_col='Route',
                    stanox_section_col='StanoxSection')

                nip_data = nip_data_.join(nip_ukcp09_stats).join(nip_radtob_stats)

                nip_data['Incident_Reported'] = 0

                # -- Merge "pip_data" and "nip_data_" -------------------------------------------------
                incident_location_weather = pd.concat(
                    [pip_data, nip_data], axis=0, ignore_index=True, sort=False)

                # -- Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE) ------
                incident_location_weather = incident_location_weather.join(
                    categorise_track_orientations(incident_location_weather))

                # -- Categorise temperature: 25, 26, 27, 28, 29, 30 -----------------------------------
                incident_location_weather = incident_location_weather.join(
                    categorise_temperatures(
                        incident_location_weather, column_name='Maximum_Temperature_max'))

                if pickle_it:
                    save_pickle(incident_location_weather, path_to_pickle, verbose=verbose)

            except Exception as e:
                print("Failed to get Weather conditions for the incident locations. {}.".format(e))
                incident_location_weather = None

        return incident_location_weather

    def illustrate_weather_grid_buffer(self, single_point=True, save_as=".svg", dpi=600,
                                       verbose=True):
        """
        Plot a Weather-grid-buffer circle.

        :param single_point:
        :type single_point: bool
        :param save_as:
        :type save_as: str or None
        :param dpi:
        :type dpi: int or None
        :param verbose:
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> # -- An incident at a single point ----------------------------------------------------

            >>> h_model_plus.illustrate_weather_grid_buffer(save_as=None)
            >>> # Save the figure
            >>> h_model_plus.illustrate_weather_grid_buffer(save_as=".png", dpi=1200)

            >>> # -- An incident with start and end locations -----------------------------------------

            >>> h_model_plus.illustrate_weather_grid_buffer(single_point=False, save_as=None)
            >>> # Save the figure
            >>> h_model_plus.illustrate_weather_grid_buffer(single_point=False, save_as=".png", dpi=1200)
        """

        incidents = self.get_processed_incident_data()

        # Illustration of the buffer circle
        if single_point:
            idx = incidents[incidents.StartLocation == incidents.EndLocation].index[0]
        else:
            idx = incidents[incidents.StartLocation != incidents.EndLocation].index[0]

        start_point, end_point, midpoint = incidents.loc[idx, ['StartXY', 'EndXY', 'MidpointXY']]

        bf_circle = create_weather_grid_buffer(
            start_point, end_point, midpoint, min_radius=500, whisker=500)

        obs_grids = self.UKCP09.read_observation_grids()  # Grids for observing Weather conditions
        obs_grids_geom = shapely.geometry.MultiPolygon(list(obs_grids.Grid))

        i_obs_grids = find_intersecting_weather_grid(
            bf_circle, obs_grids, obs_grids_geom, as_grid_id=False)

        plt.figure(figsize=(7, 6))

        ax = plt.subplot2grid((1, 1), (0, 0))

        # -- Plot the incident location ---------------------------------------------------------------
        sx, sy, ex, ey = start_point.xy + end_point.xy

        if start_point == end_point:
            ax.plot(
                sx, sy, '#c64756', marker='o', markersize=9, linestyle='None',
                label='Incident location', zorder=4)
        else:
            ax.plot(
                sx, sy, '#c64756', marker='o', markersize=9, linestyle='None', label='Start location',
                zorder=4)
            ax.plot(
                ex, ey, '#16697a', marker='o', markersize=9, linestyle='None', label='End location',
                zorder=4)

        # -- Plot the Weather observation grid --------------------------------------------------------
        for g in i_obs_grids:
            x, y = g.exterior.xy
            ax.plot(x, y, color='#707070', linewidth=0.5, zorder=0)
            grid_patch = descartes.PolygonPatch(g, fc='#bce6eb', ec='none', alpha=0.5, zorder=0)
            ax.add_patch(grid_patch)
        plt.gca().set_aspect('equal')

        x_lines = [(min(line.get_xdata()), max(line.get_xdata())) for line in plt.gca().get_lines()]
        x_min, x_max = np.min(x_lines), np.max(x_lines)
        ax.xaxis.set_ticks(range(int(x_min), int(x_max) + 5000, 5000))

        y_lines = [(min(line.get_ydata()), max(line.get_ydata())) for line in plt.gca().get_lines()]
        y_min, y_max = np.min(y_lines), np.max(y_lines)
        ax.yaxis.set_ticks(range(int(y_min), int(y_max) + 5000, 5000))

        ax.plot(
            [], marker='s', label="Weather obs. grid", ms=16, color='none', markeredgecolor='#707070',
            markerfacecolor='#bce6eb', alpha=0.5)

        # -- Plot the buffer zone ---------------------------------------------------------------------
        bf_zone_patch = descartes.PolygonPatch(bf_circle, fc='#f0c929', ec='none', alpha=0.5, zorder=2)
        ax.add_patch(bf_zone_patch)
        ax.plot(
            [], marker='o', label='Buffer zone', ms=16, markeredgecolor='#f0c929', linestyle='None',
            markerfacecolor='#f0c929', alpha=0.5, zorder=2)  # fillstyle='none'

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
        legend = plt.legend(numpoints=1, loc='best', prop=font, fancybox=True, labelspacing=0.5)
        frame = legend.get_frame()
        frame.set_edgecolor('#e8e8e8')

        plt.tight_layout()

        if save_as:
            fig_filename = "weather_grid_buffer_circle"
            fig_filename = fig_filename + ("_single_point" if single_point else "_start_end")
            path_to_fig = self.cdd_trial(fig_filename + save_as)
            save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    def plot_temperature_deviation(self, lp_span=14, err_bar=True, save_as=".svg", dpi=600,
                                   update=False, pickle_it=False, verbose=True):
        """
        Plot temperature deviation.

        :param lp_span:
        :param err_bar:
        :param save_as:
        :type save_as: str or None
        :param dpi:
        :type dpi: int or None
        :param update:
        :param pickle_it:
        :param verbose:
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> # Regional; heat
            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> h_model_plus.plot_temperature_deviation(pickle_it=True, save_as=None)

            >>> # Save the figure
            >>> h_model_plus.plot_temperature_deviation(save_as=".png", dpi=1200)
        """

        default_lp = self.lp_days

        data_sets = []

        if verbose:
            print("Preparing datasets ... ")

        for d in range(1, lp_span + 1):
            self.__setattr__('LP', -d)

            if verbose:
                print("\t{} / {}".format(-d, -lp_span), end=" ... ")

            data_sets.append(
                self.get_incident_location_weather(update=update, pickle_it=pickle_it, verbose=False))

            if verbose:
                print("Done.")

        self.__setattr__('LP', default_lp)

        time_and_iloc = ['StartDateTime', 'EndDateTime', 'StanoxSection', 'IncidentDescription']
        selected_cols = time_and_iloc + ['Maximum_Temperature_max']

        base_data = data_sets[0]
        ip_temperature_max = base_data[base_data.Incident_Reported == 1][selected_cols]

        diff_means, diff_std = [], []
        for i in range(0, lp_span):
            data = data_sets[i]

            nip_temperature_max = data[data.Incident_Reported == 0][selected_cols]

            temp_diffs = pd.merge(
                ip_temperature_max, nip_temperature_max, on=time_and_iloc, suffixes=('_pip', '_nip'))
            temp_diff = temp_diffs.Maximum_Temperature_max_pip - temp_diffs.Maximum_Temperature_max_nip

            diff_means.append(temp_diff.abs().mean())
            diff_std.append(temp_diff.abs().std())

        plt.figure(figsize=(10, 5))

        if err_bar:
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
            fig_filename = "temperature_deviation"
            if self.weather_category:
                if isinstance(self.weather_category, str):
                    fig_filename += "_{}".format(self.weather_category.lower())
                elif isinstance(self.weather_category, list):
                    fig_filename += "_{}".format("_".join(self.weather_category).lower())
            path_to_fig = self.cdd_trial(fig_filename + save_as)

            save_fig(path_to_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    def prep_training_and_test_sets(self):
        """
        Further process the integrated data set and split it into a training set and a test set.

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> _, training_data, test_data = h_model_plus.prep_training_and_test_sets()

            >>> training_data.tail()
                  FinancialYear  ... const
            1357           2015  ...   1.0
            1358           2015  ...   1.0
            1359           2015  ...   1.0
            1360           2015  ...   1.0
            1361           2015  ...   1.0
            [5 rows x 88 columns]

            >>> test_data.tail()
                 FinancialYear  ... const
            300           2016  ...   1.0
            301           2016  ...   1.0
            302           2016  ...   1.0
            303           2016  ...   1.0
            304           2016  ...   1.0
            [5 rows x 88 columns]
        """

        # Get the mdata for modelling
        processed_dat = self.get_incident_location_weather(pickle_it=True)
        processed_data = processed_dat.dropna(subset=['Temperature_Change_max', 'GLBL_IRAD_AMT_total'])

        processed_data.GLBL_IRAD_AMT_total = processed_data.GLBL_IRAD_AMT_total / 1000

        # Remove outliers
        if 95 <= self.outlier_percentile <= 100:
            upper_limit = np.percentile(processed_data.DelayMinutes, self.outlier_percentile)
            processed_data = processed_data[processed_data.DelayMinutes <= upper_limit]
        # from pyhelpers.ops import get_extreme_outlier_bounds
        # l, u = get_extreme_outlier_bounds(processed_data.DelayMinutes, k=1.5)
        # processed_data = processed_data[
        #     processed_data.DelayMinutes.between(l, u, inclusive=True)]

        # Set the outcomes of non-incident records to 0
        outcome_columns = ['DelayMinutes', 'DelayCost', 'IncidentCount']
        processed_data.loc[processed_data.Incident_Reported == 0, outcome_columns] = 0

        # Select data before 2014 as the training data set, with the rest the test set
        training_set = processed_data[processed_data.StartDateTime < datetime.datetime(2016, 1, 1)]
        training_set.index = range(len(training_set))
        test_set = processed_data[processed_data.StartDateTime >= datetime.datetime(2016, 1, 1)]
        test_set.index = range(len(test_set))

        self.__setattr__('TrainingSet', training_set)
        self.__setattr__('TestSet', test_set)

        return processed_data, training_set, test_set

    def describe_training_set(self, save_as=".svg", dpi=600, verbose=True):
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

            >>> # Save the figure
            >>> h_model_plus.describe_training_set(save_as=".png", dpi=1200)
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
        training_set.Precipitation_mean.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
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
            fig_filename = "training_set_variables_description"
            if self.weather_category:
                if isinstance(self.weather_category, str):
                    fig_filename += "_{}".format(self.weather_category.lower())
                elif isinstance(self.weather_category, list):
                    fig_filename += "_{}".format("_".join(self.weather_category).lower())
            path_to_fig_file = self.cdd_trial(fig_filename + save_as)

            save_fig(path_to_fig_file, dpi, verbose=verbose, conv_svg_to_emf=True)

    @staticmethod
    def _p_value(lr, X_train):
        """
        Calculate z-scores for sklearn LogisticRegression.

        Source:
        https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
        """

        p = lr.predict_proba(X_train)
        n = len(p)
        m = len(lr.coef_[0]) + 1

        coefficients = np.concatenate([lr.intercept_, lr.coef_[0]])

        x_full = np.matrix(np.insert(np.array(X_train), 0, 1, axis=1))
        ans = np.zeros((m, m))

        for i in range(n):
            ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i, 1] * p[i, 0]

        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))
        t = coefficients / se

        # noinspection PyUnresolvedReferences
        p = (1 - scipy.stats.norm.cdf(abs(t))) * 2

        return p

    def logistic_regression(self, add_intercept=True, random_state=1, pickle_it=False, verbose=True):
        """
        Train/test a logistic regression model for predicting heat-related Incidents.

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> # Regional; heat: Anglia, Wessex, Wales and North & East
            >>> h_model_plus.route_names = ['Anglia', 'Wessex', 'Wales', 'North and East']
            >>> region_results = h_model_plus.logistic_regression(pickle_it=True)

            >>> # Regional; heat: Anglia
            >>> h_model_plus.route_names = ['Anglia']
            >>> anglia_results = h_model_plus.logistic_regression(pickle_it=True)

            >>> # Regional; heat: Wessex
            >>> h_model_plus.route_names = ['Wessex']
            >>> wessex_results = h_model_plus.logistic_regression(pickle_it=True)

            >>> # Regional; heat: Wales
            >>> h_model_plus.route_names = ['Wales']
            >>> wales_results = h_model_plus.logistic_regression(pickle_it=True)

            >>> # Regional; heat: North & East
            >>> h_model_plus.route_names = ['North and East']
            >>> ne_results = h_model_plus.logistic_regression(pickle_it=True)
        """

        # Get data for modelling
        _, training_set, test_set = self.prep_training_and_test_sets()

        X_train, X_test = training_set[self.variable_names], test_set[self.variable_names]
        y_train, y_test = training_set.Incident_Reported, test_set.Incident_Reported

        # import statsmodels.discrete.discrete_model as sm_dcm
        # import statsmodels.tools
        #
        # X_train_ = statsmodels.tools.add_constant(X_train, has_constant='add')
        # X_test_ = statsmodels.tools.add_constant(X_test, has_constant='add')
        #
        # if self.ModelType == 'LogisticRegression':
        #     lr = sm_dcm.Logit(y_train, X_train_)
        # else:
        #     lr = sm_dcm.Probit(y_train, X_test_)
        #
        # np.random.seed(random_state)
        #
        # try:
        #     lr_summary = lr.fit(
        #         maxiter=10000, method='newton', full_output=True, disp=True)
        # except np.linalg.LinAlgError:
        #     lr_summary = lr.fit(
        #         maxiter=10000, method='lbfgs', full_output=True, disp=True, pgtol=0.00001)
        #
        # print(lr_summary.summary2()) if verbose else print("")
        #
        # # Odds ratios
        # odds_ratios = pd.DataFrame(np.exp(lr_summary.params), columns=['OddsRatio'])
        # print("\n{}".format(odds_ratios)) if verbose else print("")
        #
        # # Prediction
        # incident_prob = lr_summary.predict(X_test_)

        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(
            C=1, tol=0.0001, max_iter=1000, solver='lbfgs', fit_intercept=True, intercept_scaling=1,
            random_state=random_state, multi_class='ovr', verbose=0)

        lr.fit(X_train, y_train)

        coefficients = lr.intercept_.tolist() + lr.coef_[0].tolist()
        p_values = [np.round(x, 4) for x in self._p_value(lr, X_train)]
        odds_ratios = np.exp(coefficients).tolist()

        lr_summary = pd.DataFrame(
            {'Coefficient': coefficients, 'P-value': p_values, 'OddsRatio': odds_ratios},
            index=['constant'] + self.variable_names)

        if verbose:
            print("\n{}".format(lr_summary))
            print("\nSize of training set: %d" % len(X_train))

        self.__setattr__('Model', lr)
        self.__setattr__('Summary', lr_summary)

        # Prediction (probabilities)
        incident_prob = lr.predict_proba(X_test)

        # ROC - False Positive Rate (fpr); True Positive Rate (tpr); Threshold
        fpr, tpr, thr = metrics.roc_curve(y_test.to_numpy(), incident_prob[:, 1], pos_label=1)
        # Area under the curve (AUC)
        auc = metrics.auc(fpr, tpr)

        threshold = np.min(thr[np.argmax(tpr + 1 - fpr)])

        self.__setattr__('FPR', fpr)
        self.__setattr__('TPR', tpr)
        self.__setattr__('AUC', auc)
        self.__setattr__('Threshold', threshold)

        # Mean accuracy on the test set
        mean_accuracy = lr.score(X_test, y_test)
        if verbose:
            print("\nMean accuracy: %.2f" % mean_accuracy)

        # Accuracy based on threshold
        incident_prediction = np.array([1 if x >= threshold else 0 for x in incident_prob[:, 1]])
        accuracy = np.divide(np.sum(y_test.to_numpy() == incident_prediction), len(y_test))
        if verbose:
            print("Prediction accuracy (given the threshold=%.2f): %.2f" % (threshold, accuracy))

        # Prediction of incident occurrences
        y_pred = lr.predict(X_test)
        f1_score = metrics.f1_score(y_test, y_pred)

        # Incident recall
        incidents_recall_score = metrics.recall_score(y_test, incident_prediction)
        # # Alternatively:
        # incident_only = test_set[y_test == 1].Incident_Reported
        # incident_acc = incident_only.eq(incident_prediction[incident_only.index])
        # incident_recall = np.divide(sum(incident_acc), len(incident_acc))
        if verbose:
            print("Incident recall score: %.2f\n" % incidents_recall_score)

        if pickle_it:
            repo = locals()
            names = ['training_set', 'test_set',
                     'lr', 'lr_summary', 'threshold', 'accuracy', 'incidents_recall_score']
            resources = {k: repo[k] for k in list(names)}
            result_pickle = make_filename(
                "result", self.route_names, self.weather_category,
                str(self.pip_start_hrs) + 'h', str(self.lp_days) + 'd' if self.lp_days else '-xd',
                str(self.nip_start_hrs) + 'h', sep="_")

            save_pickle(resources, self.cdd_trial(result_pickle), verbose=verbose)

        return training_set, test_set, lr, lr_summary, threshold, accuracy, incidents_recall_score

    def plot_roc(self, simplified=False, save_as=".svg", dpi=600, verbose=True):
        """
        Plot ROC.

        :param simplified:
        :param save_as: whether to save the figure or file extension
        :type save_as: str or bool or None
        :param dpi: DPI
        :type dpi: int or None
        :param verbose: whether to print relevant information in console, defaults to ``True``
        :type verbose: bool or int

        **Test**::

            >>> from modeller.prototype_ext import HeatAttributedIncidentsPlus

            >>> h_model_plus = HeatAttributedIncidentsPlus(trial_id=2)

            >>> results = h_model_plus.logistic_regression()

            >>> h_model_plus.plot_roc(save_as=None)

            >>> # Save the figure
            >>> h_model_plus.plot_roc(save_as=".png", dpi=1200)
        """

        if simplified:
            lr = self.__getattribute__('Model')
            test_set = self.__getattribute__('TestSet')

            X_test, y_test = test_set[self.variable_names], test_set.Incident_Reported

            metrics.plot_roc_curve(lr, X_test, y_test)

        else:
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

            plt.show()

        if save_as:
            fig_filename = "roc_curve"
            if simplified:
                fig_filename += "_simplified"
            if self.weather_category:
                if isinstance(self.weather_category, str):
                    fig_filename += "_{}".format(self.weather_category.lower())
                elif isinstance(self.weather_category, list):
                    fig_filename += "_{}".format("_".join(self.weather_category).lower())
            path_to_roc_fig = self.cdd_trial(fig_filename + save_as)

            save_fig(path_to_roc_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)

    def plot_pred_likelihood(self, save_as=".svg", dpi=600, verbose=True):
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

            >>> results = h_model_plus.logistic_regression()

            >>> h_model_plus.plot_pred_likelihood(save_as=None)

            >>> # Save the figure
            >>> h_model_plus.plot_pred_likelihood(save_as=".png", dpi=1200)
        """

        test_set = self.__getattribute__('TestSet')
        threshold = self.__getattribute__('Threshold')

        lr = self.__getattribute__('Model')
        incident_prob = lr.predict_proba(X=test_set[self.variable_names])[:, 1]

        incident_idx = test_set[test_set.Incident_Reported == 1].index.tolist()

        plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))

        ax.scatter(
            incident_prob[incident_idx], test_set.DelayMinutes.values[incident_idx],
            c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
            label="Heat-attributed incident (2016)")
        plt.axvline(
            x=threshold, label="Optimal threshold: %.2f" % threshold, color='#e5c100', linewidth=2)

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

        plt.show()

        if save_as:
            fig_filename = "pred_likelihood"
            if self.weather_category:
                if isinstance(self.weather_category, str):
                    fig_filename += "_{}".format(self.weather_category.lower())
                elif isinstance(self.weather_category, list):
                    fig_filename += "_{}".format("_".join(self.weather_category).lower())
            path_to_pred_fig = self.cdd_trial(fig_filename + save_as)

            save_fig(path_to_pred_fig, dpi=dpi, verbose=verbose, conv_svg_to_emf=True)


# if __name__ == '__main__':
#     from modeller import HeatAttributedIncidentsPlus
#
#     h_mod_plus = HeatAttributedIncidentsPlus(trial_id=2)
#
#     h_mod_plus_results = h_mod_plus.logistic_regression()
#
#     h_mod_plus.plot_roc(save_as=None)
#
#     h_mod_plus.plot_pred_likelihood(save_as=None)
