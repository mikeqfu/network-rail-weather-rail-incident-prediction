""" A prototype model in the context of heat-related Incidents """

import datetime
import itertools
import os

import datetime_truncate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import sklearn.metrics
import sklearn.utils.extmath
import statsmodels.discrete.discrete_model as sm
import statsmodels.tools.tools as sm_tools
from pyhelpers.dir import cdd
from pyhelpers.store import load_pickle, save_fig, save_pickle, save_svg_as_emf

import prototype.utils as proto_utils
import settings
from mssqlserver import metex
from prototype.model.wind import get_data_by_season, get_incident_location_vegetation

# Apply the preferences ==============================================================================================
settings.mpl_preferences(reset=False)
settings.pd_preferences(reset=False)
plt.rc('font', family='Times New Roman')

# ====================================================================================================================
""" Change directory """


# Change directory to "modelling\\prototype-Heat\\Trial_" and sub-directories
def cdd_mod_heat_proto(trial_id=0, *directories):
    path = cdd("modelling", "prototype-Heat", "Trial_{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Calculations for Weather data """


# Specify the statistics that need to be computed
def specify_weather_stats_calculations():
    weather_stats_computations = {'Temperature': (np.max, np.min, np.average),
                                  'RelativeHumidity': (np.max, np.min, np.average),
                                  'WindSpeed': np.max,
                                  'WindGust': np.max,
                                  'Snowfall': np.max,
                                  'TotalPrecipitation': (np.max, np.min, np.average)}
    return weather_stats_computations


# Get all Weather variable names
def weather_variable_names():
    # var_names = db.colnames_db_table('NR_METEX', table_name='Weather')[2:]
    agg_colnames = ['Temperature_max', 'Temperature_min', 'Temperature_avg',
                    'RelativeHumidity_max', 'RelativeHumidity_min', 'RelativeHumidity_avg',
                    'WindSpeed_max', 'WindGust_max', 'Snowfall_max',
                    'TotalPrecipitation_max', 'TotalPrecipitation_min', 'TotalPrecipitation_avg']
    wind_speed_variables = ['WindSpeed_avg', 'WindDirection_avg']
    return agg_colnames + wind_speed_variables


# Calculate average wind speed and direction
def calculate_wind_averages(wind_speeds, wind_directions):
    # component u, the zonal velocity
    u = - wind_speeds * np.sin(np.radians(wind_directions))
    # component v, the meridional velocity
    v = - wind_speeds * np.cos(np.radians(wind_directions))
    # sum up all u and v values and average it
    uav, vav = np.mean(u), np.mean(v)
    # Calculate average wind speed
    average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)
    # Calculate average wind direction
    if uav == 0:
        if vav == 0:
            average_wind_direction = 0
        else:
            average_wind_direction = 360 if vav > 0 else 180
    else:
        if uav > 0:
            average_wind_direction = 270 - 180 / np.pi * np.arctan(vav / uav)
        else:
            average_wind_direction = 90 - 180 / np.pi * np.arctan(vav / uav)
    return average_wind_speed, average_wind_direction


# Compute the statistics for all the Weather variables (except wind)
def calculate_weather_variables_stats(weather_data):
    """
    Note: to get the n-th percentile, use percentile(n)

    This function also returns the Weather dataframe indices. The corresponding Weather conditions in that Weather
    cell might cause wind-related Incidents.
    """
    # Calculate the statistics
    weather_stats_calculations = specify_weather_stats_calculations()
    weather_stats = weather_data.groupby('WeatherCell').aggregate(weather_stats_calculations)
    # Calculate average wind speeds and directions
    weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
        calculate_wind_averages(weather_data.WindSpeed, weather_data.WindDirection)
    if not weather_stats.empty:
        stats_info = weather_stats.values[0].tolist() + [weather_data.index.tolist()]
    else:
        stats_info = [np.nan] * len(weather_stats.columns) + [[None]]
    return stats_info


# Find Weather Cell ID
def find_weather_cell_id(longitude, latitude):
    weather_cell = metex.get_weather_cell()

    ll = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ll_Longitude, weather_cell.ll_Latitude)]
    ul = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ul_lon, weather_cell.ul_lat)]
    ur = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ur_Longitude, weather_cell.ur_Latitude)]
    lr = [shapely.geometry.Point(xy) for xy in zip(weather_cell.lr_lon, weather_cell.lr_lat)]

    poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]

    cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in range(len(weather_cell))]

    pt = shapely.geometry.Point(longitude, latitude)

    id_set = set(weather_cell.iloc[[i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
    if len(id_set) == 1:
        weather_cell_id = list(id_set)[0]
    else:
        weather_cell_id = list(id_set)
    return weather_cell_id


# Get TRUST and the relevant Weather data for each location
def get_incident_location_weather(route=None, weather=None, ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                  subset_weather_for_nip=False,
                                  update=False):
    """
    :param route: [str] Route name
    :param weather: [str] Weather category
    :param ip_start_hrs: [int/float]
    :param nip_ip_gap:
    :param nip_start_hrs: [int/float]
    :param subset_weather_for_nip: [bool]
    :param update: [bool]
    :return: [DataFrame]
    """

    filename = metex.make_filename("incident_location_weather", route, weather, ip_start_hrs, nip_ip_gap,
                                   nip_start_hrs)
    path_to_file = proto_utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        iw_data = load_pickle(path_to_file)
    else:
        try:
            # ----------------------------------------------------------------------------------------------------
            """ Get data """

            # Getting incident data for all incident locations
            sdata = metex.view_schedule8_cost_by_datetime_location_reason(route, weather, update)
            # Drop non-Weather-related incident records
            sdata = sdata[sdata.WeatherCategory != ''] if weather is None else sdata
            # Get data for the specified "Incident Periods"
            sdata['incident_duration'] = sdata.EndDate - sdata.StartDate
            sdata['critical_start'] = \
                sdata.StartDate.apply(datetime_truncate.truncate_hour) + datetime.timedelta(hours=ip_start_hrs)
            sdata['critical_end'] = sdata.StartDate
            sdata['critical_period'] = sdata.critical_end - sdata.critical_start

            if sdata.WeatherCell.dtype != 'int64':
                # Rectify the records for which Weather cell id is empty
                weather_cell = metex.get_weather_cell()
                ll = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ll_Longitude, weather_cell.ll_Latitude)]
                ul = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ul_lon, weather_cell.ul_lat)]
                ur = [shapely.geometry.Point(xy) for xy in zip(weather_cell.ur_Longitude, weather_cell.ur_Latitude)]
                lr = [shapely.geometry.Point(xy) for xy in zip(weather_cell.lr_lon, weather_cell.lr_lat)]
                poly_list = [[ll[i], ul[i], ur[i], lr[i]] for i in range(len(weather_cell))]
                cells = [shapely.geometry.Polygon([(p.x, p.y) for p in poly_list[i]]) for i in range(len(weather_cell))]

                for i in sdata[sdata.WeatherCell == ''].index:
                    pt = shapely.geometry.Point(sdata.StartLongitude.loc[i], sdata.StartLatitude.loc[i])
                    id_set = set(
                        weather_cell.iloc[[i for i, p in enumerate(cells) if pt.within(p)]].WeatherCellId.tolist())
                    if len(id_set) == 0:
                        pt_alt = shapely.geometry.Point(sdata.EndLongitude.loc[i], sdata.EndLatitude.loc[i])
                        id_set = set(
                            weather_cell.iloc[
                                [i for i, p in enumerate(cells) if pt_alt.within(p)]].WeatherCellId.tolist())
                    if len(id_set) == 0:
                        pass
                    else:
                        sdata.loc[i, 'WeatherCell'] = list(id_set)[0]

            # Get Weather data
            weather_data = metex.get_weather().reset_index()

            # ----------------------------------------------------------------------------------------------------
            """ Processing Weather data for IP - Get Weather conditions which led to Incidents for each record """

            def get_ip_weather_conditions(weather_cell_id, ip_start, ip_end):
                """
                :param weather_cell_id: [int] Weather Cell ID
                :param ip_start: [Timestamp] start of "incident period"
                :param ip_end: [Timestamp] end of "incident period"
                :return:
                """
                # Get Weather data about where and when the incident occurred
                ip_weather_data = weather_data[(weather_data.WeatherCell == weather_cell_id) &
                                               (weather_data.DateTime >= ip_start) &
                                               (weather_data.DateTime <= ip_end)]
                # Get the max/min/avg Weather parameters for those incident periods
                weather_stats_data = calculate_weather_variables_stats(ip_weather_data)
                return weather_stats_data

            # Get data for the specified IP
            ip_statistics = sdata.apply(
                lambda x: pd.Series(get_ip_weather_conditions(x.WeatherCell, x.critical_start, x.critical_end)), axis=1)

            ip_statistics.columns = weather_variable_names() + ['critical_weather_idx']
            ip_statistics['Temperature_diff'] = ip_statistics.Temperature_max - ip_statistics.Temperature_min

            ip_data = sdata.join(ip_statistics.dropna(), how='inner')
            ip_data['IncidentReported'] = 1

            # Get Weather data that did not ever cause Incidents according to records?
            if subset_weather_for_nip:
                weather_data = weather_data.loc[
                    weather_data.WeatherCell.isin(ip_data.WeatherCell) &
                    ~weather_data.index.isin(itertools.chain(*ip_data.critical_weather_idx))]

            # Processing Weather data for non-IP
            nip_data = sdata.copy(deep=True)
            nip_data.critical_end = nip_data.critical_start + datetime.timedelta(days=nip_ip_gap)
            nip_data.critical_start = nip_data.critical_end + datetime.timedelta(hours=nip_start_hrs)
            nip_data.critical_period = nip_data.critical_end - nip_data.critical_start

            # -----------------------------------------------------------------------
            """ Get data of Weather which did not cause Incidents for each record """

            def get_nip_weather_conditions(weather_cell_id, nip_start, nip_end, incident_location):
                # Get non-IP Weather data about where and when the incident occurred
                nip_weather_data = weather_data[
                    (weather_data.WeatherCell == weather_cell_id) &
                    (weather_data.DateTime >= nip_start) & (weather_data.DateTime <= nip_end)]
                # Get all incident period data on the same section
                ip_overlap = ip_data[
                    (ip_data.StanoxSection == incident_location) &
                    (((ip_data.critical_start < nip_start) & (ip_data.critical_end > nip_start)) |
                     ((ip_data.critical_start < nip_end) & (ip_data.critical_end > nip_end)))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not ip_overlap.empty:
                    nip_weather_data = nip_weather_data[
                        (nip_weather_data.DateTime < np.min(ip_overlap.critical_start)) |
                        (nip_weather_data.DateTime > np.max(ip_overlap.critical_end))]
                # Get the max/min/avg Weather parameters for those incident periods
                weather_stats_data = calculate_weather_variables_stats(nip_weather_data)
                return weather_stats_data

            # Get stats data for the specified "Non-Incident Periods"
            nip_statistics = nip_data.apply(
                lambda x: pd.Series(get_nip_weather_conditions(
                    x.WeatherCell, x.critical_start, x.critical_end, x.StanoxSection)), axis=1)
            nip_statistics.columns = weather_variable_names() + ['critical_weather_idx']
            nip_statistics['Temperature_diff'] = nip_statistics.Temperature_max - nip_statistics.Temperature_min
            nip_data = nip_data.join(nip_statistics.dropna(), how='inner')
            nip_data['IncidentReported'] = 0

            # Merge "ip_data" and "nip_data" into one DataFrame
            iw_data = pd.concat([nip_data, ip_data], axis=0, ignore_index=True)

            # --------------------------------
            """ Categorise wind directions """

            def categorise_wind_directions(direction_degree):
                if 0 <= direction_degree < 90:
                    return 1
                elif 90 <= direction_degree < 180:
                    return 2
                elif 180 <= direction_degree < 270:
                    return 3
                elif 270 <= direction_degree < 360:
                    return 4

            # Categorise average wind directions into 4 quadrants
            iw_data['wind_direction'] = iw_data.WindDirection_avg.apply(categorise_wind_directions)
            wind_direction = pd.get_dummies(iw_data.wind_direction, prefix='wind_direction')
            iw_data = iw_data.join(wind_direction)

            # ---------------------------------------------------------------------------------
            """ Categorise track orientations into four directions (N-S, E-W, NE-SW, NW-SE) """

            def categorise_track_orientations(data):
                data['track_orient'] = None
                # origin = (-0.565409, 51.23622)
                lon1, lat1, lon2, lat2 = data.StartLongitude, data.StartLatitude, data.EndLongitude, data.EndLatitude
                data['track_orient_angles'] = np.arctan2(lat2 - lat1, lon2 - lon1)  # Angles in radians, [-pi, pi]

                # N-S / S-N: [-np.pi*2/3, -np.pi/3] & [np.pi/3, np.pi*2/3]
                n_s = np.logical_or(
                    np.logical_and(data.track_orient_angles >= -np.pi * 2 / 3, data.track_orient_angles < -np.pi / 3),
                    np.logical_and(data.track_orient_angles >= np.pi / 3, data.track_orient_angles < np.pi * 2 / 3))
                data.track_orient[n_s] = 'N_S'

                # NE-SW / SW-NE: [np.pi/6, np.pi/3] & [-np.pi*5/6, -np.pi*2/3]
                ne_sw = np.logical_or(
                    np.logical_and(data.track_orient_angles >= np.pi / 6, data.track_orient_angles < np.pi / 3),
                    np.logical_and(data.track_orient_angles >= -np.pi * 5 / 6,
                                   data.track_orient_angles < -np.pi * 2 / 3))
                data.track_orient[ne_sw] = 'NE_SW'

                # NW-SE / SE-NW: [np.pi*2/3, np.pi*5/6], [-np.pi/3, -np.pi/6]
                nw_se = np.logical_or(
                    np.logical_and(data.track_orient_angles >= np.pi * 2 / 3, data.track_orient_angles < np.pi * 5 / 6),
                    np.logical_and(data.track_orient_angles >= -np.pi / 3, data.track_orient_angles < -np.pi / 6))
                data.track_orient[nw_se] = 'NW_SE'

                # E-W / W-E: [-np.pi, -np.pi*5/6], [-np.pi/6, np.pi/6], [np.pi*5/6, np.pi]
                data.track_orient.fillna('E_W', inplace=True)
                # e_w = np.logical_or(np.logical_or(
                #     np.logical_and(data.track_orient_angles >= -np.pi, data.track_orient_angles < -np.pi * 5 / 6),
                #     np.logical_and(data.track_orient_angles >= -np.pi/6, data.track_orient_angles < np.pi/6)),
                #     np.logical_and(data.track_orient_angles >= np.pi*5/6, data.track_orient_angles < np.pi))
                # data.track_orient[e_w] = 'E-W'

                return pd.get_dummies(data.track_orient, prefix='track_orientation')

            iw_data = iw_data.join(categorise_track_orientations(iw_data))

            # ----------------------------------------------------
            """ Categorise temperature: 25, 26, 27, 28, 29, 30 """

            # critical_temperature_dat = [
            #     iwdata.Temperature_max.map(lambda x: 1 if x >= t else -1) for t in range(25, 31)]
            # critical_temperature = ['Temperature_max ≥ {}°C'.format(t) for t in range(25, 31)]
            # for dat, col in zip(critical_temperature_dat, critical_temperature):
            #     iwdata[col] = dat

            def categorise_temperatures(data):
                data['temperature_category'] = None
                data.temperature_category[data.Temperature_max < 24] = 'Temperature_max < 24°C'
                data.temperature_category[data.Temperature_max == 24] = 'Temperature_max = 24°C'
                data.temperature_category[data.Temperature_max == 25] = 'Temperature_max = 25°C'
                data.temperature_category[data.Temperature_max == 26] = 'Temperature_max = 26°C'
                data.temperature_category[data.Temperature_max == 27] = 'Temperature_max = 27°C'
                data.temperature_category[data.Temperature_max == 28] = 'Temperature_max = 28°C'
                data.temperature_category[data.Temperature_max == 29] = 'Temperature_max = 29°C'
                data.temperature_category[data.Temperature_max >= 30] = 'Temperature_max ≥ 30°C'
                # data.temperature_category[data.Temperature_max > 30] = 'Temperature_max > 30°C'
                return data

            iw_data = categorise_temperatures(iw_data)
            iw_data = iw_data.join(pd.get_dummies(iw_data.temperature_category, prefix=''))

            save_pickle(iw_data, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            iw_data = None

    return iw_data


def temperature_deviation(nip_ip_gap=-14, add_err_bar=True, save_as=".svg", dpi=600):
    gap = np.abs(nip_ip_gap)
    data = [get_incident_location_weather(route='ANGLIA', weather='Heat', nip_ip_gap=-d) for d in range(1, gap + 1)]

    time_and_iloc = ['StartDate', 'EndDate', 'StanoxSection', 'IncidentDescription']
    ip_temperature_max = data[0][data[0].IncidentReported == 1][time_and_iloc + ['Temperature_max']]
    diff_means, diff_std = [], []
    for i in range(0, gap):
        nip_temperature_max = data[i][data[i].IncidentReported == 0][time_and_iloc + ['Temperature_max']]
        temp_diffs = pd.merge(ip_temperature_max, nip_temperature_max, on=time_and_iloc, suffixes=('_ip', '_nip'))
        temp_diff = temp_diffs.Temperature_max_ip - temp_diffs.Temperature_max_nip
        diff_means.append(temp_diff.abs().mean())
        diff_std.append(temp_diff.abs().std())

    plt.figure()
    if add_err_bar:
        container = plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', yerr=diff_std, capsize=4,
                            width=0.7, color='#9FAFBE')
        connector, cap_lines, (vertical_lines,) = container.errorbar.lines
        vertical_lines.set_color('#666666')
        for cap in cap_lines:
            cap.set_color('#da8067')
    else:
        plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', width=0.7, color='#9FAFBE')
        plt.grid(False)
    plt.xticks(np.arange(1, len(diff_means) + 1), fontsize=14)
    plt.xlabel('Latent period (Number of days)', fontsize=14)
    plt.ylabel('Temperature deviation (°C)', fontsize=14)
    plt.tight_layout()

    plt.savefig(cdd_mod_heat_proto(0, "Temp deviation" + save_as), dpi=dpi)


# ====================================================================================================================
""" Integrate both the Weather and Vegetation data """


# Integrate the Weather and Vegetation conditions for incident locations
def get_incident_data_with_weather_and_vegetation(route='ANGLIA', weather='Heat',
                                                  ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                                  shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                                                  update=False):
    filename = metex.make_filename("mod_data", route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                   shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_file = proto_utils.cd_prototype_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        m_data = load_pickle(path_to_file)
    else:
        try:
            # Get Schedule 8 incident and Weather data for locations
            iw_data = get_incident_location_weather(route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                    subset_weather_for_nip=False, update=update)
            # Get Vegetation conditions for the locations
            iv_data = get_incident_location_vegetation(route, shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)

            iv_features = [f for f in iv_data.columns if f not in ['IncidentCount', 'DelayCost', 'DelayMinutes']]
            iv_data = iv_data[iv_features]

            # Merge the above two data sets
            m_data = pd.merge(iw_data, iv_data, how='inner', on=list(set(iw_data.columns) & set(iv_features)))

            save_pickle(m_data, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            m_data = None

    return m_data


# ====================================================================================================================
""" modelling trials """


# Specify the explanatory variables considered in this prototype model
def specify_explanatory_variables_model_1():
    return [
        # 'Temperature_min',
        # 'Temperature_avg',
        # 'Temperature_max ≥ 25°C',
        # 'Temperature_max ≥ 26°C',
        # 'Temperature_max ≥ 27°C',
        # 'Temperature_max ≥ 28°C',
        # 'Temperature_max ≥ 29°C',
        # 'Temperature_max ≥ 30°C',
        # 'Temperature_max',
        'Temperature_diff',
        # '_Temperature_max < 24°C',
        '_Temperature_max = 24°C',
        '_Temperature_max = 25°C',
        '_Temperature_max = 26°C',
        '_Temperature_max = 27°C',
        '_Temperature_max = 28°C',
        '_Temperature_max = 29°C',
        '_Temperature_max ≥ 30°C',
        # 'track_orientation_E_W',
        'track_orientation_NE_SW',
        'track_orientation_NW_SE',
        'track_orientation_N_S',
        # 'WindGust_max',
        # 'WindSpeed_avg',
        # 'WindDirection_avg',
        # 'WindSpeed_max',
        # # 'wind_direction_1',  # [0°, 90°)
        # 'wind_direction_2',  # [90°, 180°)
        # 'wind_direction_3',  # [180°, 270°)
        # 'wind_direction_4',  # [270°, 360°)
        # 'RelativeHumidity_max',
        # 'RelativeHumidity_avg',
        # 'Snowfall_max',
        # 'TotalPrecipitation_max',
        # 'TotalPrecipitation_avg',
        # 'Electrified',
        # 'CoverPercentAlder',
        # 'CoverPercentAsh',
        # 'CoverPercentBeech',
        # 'CoverPercentBirch',
        # 'CoverPercentConifer',
        # 'CoverPercentElm',
        # 'CoverPercentHorseChestnut',
        # 'CoverPercentLime',
        # 'CoverPercentOak',
        # 'CoverPercentPoplar',
        # 'CoverPercentShrub',
        # 'CoverPercentSweetChestnut',
        # 'CoverPercentSycamore',
        # 'CoverPercentWillow',
        # 'CoverPercentOpenSpace',
        # 'CoverPercentOther',
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


def specify_explanatory_variables_model_2():
    return [
        # 'Temperature_min',
        # 'Temperature_avg',
        # 'Temperature_max ≥ 25°C',
        # 'Temperature_max ≥ 26°C',
        # 'Temperature_max ≥ 27°C',
        # 'Temperature_max ≥ 28°C',
        # 'Temperature_max ≥ 29°C',
        # 'Temperature_max ≥ 30°C',
        # 'Temperature_max',
        'Temperature_diff',
        # '_Temperature_max < 24°C',
        '_Temperature_max = 24°C',
        '_Temperature_max = 25°C',
        '_Temperature_max = 26°C',
        '_Temperature_max = 27°C',
        '_Temperature_max = 28°C',
        '_Temperature_max = 29°C',
        '_Temperature_max ≥ 30°C',
        # 'track_orientation_E_W',
        'track_orientation_NE_SW',
        'track_orientation_NW_SE',
        'track_orientation_N_S',
        # 'WindGust_max',
        'WindSpeed_avg',
        # 'WindDirection_avg',
        # 'WindSpeed_max',
        # # 'wind_direction_1',  # [0°, 90°)
        # 'wind_direction_2',  # [90°, 180°)
        # 'wind_direction_3',  # [180°, 270°)
        # 'wind_direction_4',  # [270°, 360°)
        # 'RelativeHumidity_max',
        'RelativeHumidity_avg',
        # 'Snowfall_max',
        # 'TotalPrecipitation_max',
        'TotalPrecipitation_avg',
        # 'Electrified',
        # 'CoverPercentAlder',
        # 'CoverPercentAsh',
        # 'CoverPercentBeech',
        # 'CoverPercentBirch',
        # 'CoverPercentConifer',
        # 'CoverPercentElm',
        # 'CoverPercentHorseChestnut',
        # 'CoverPercentLime',
        # 'CoverPercentOak',
        # 'CoverPercentPoplar',
        # 'CoverPercentShrub',
        # 'CoverPercentSweetChestnut',
        # 'CoverPercentSycamore',
        # 'CoverPercentWillow',
        'CoverPercentOpenSpace',
        # 'CoverPercentOther',
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


# Describe basic statistics about the main explanatory variables
def describe_explanatory_variables(train_set, save_as=".pdf", dpi=None):
    plt.figure(figsize=(13, 5))
    colour = dict(boxes='#4c76e1', whiskers='DarkOrange', medians='#ff5555', caps='Gray')

    ax1 = plt.subplot2grid((1, 8), (0, 0))
    train_set.Temperature_diff.plot.box(color=colour, ax=ax1, widths=0.5, fontsize=12)
    ax1.set_xticklabels('')
    plt.xlabel('Temp. Diff.', fontsize=13, labelpad=39)
    plt.ylabel('(°C)', fontsize=12, rotation=0)
    ax1.yaxis.set_label_coords(0.05, 1.01)

    ax2 = plt.subplot2grid((1, 8), (0, 1), colspan=2)
    train_set.temperature_category.value_counts().plot.bar(color='#537979', rot=-45, fontsize=12)
    plt.xticks(range(0, 8), ['< 24°C', '24°C', '25°C', '26°C', '27°C', '28°C', '29°C', '≥ 30°C'], fontsize=12)
    plt.xlabel('Max. Temp.', fontsize=13, labelpad=7)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax2.yaxis.set_label_coords(0.0, 1.01)

    ax3 = plt.subplot2grid((1, 8), (0, 3))
    track_orient = train_set.track_orient.value_counts()
    track_orient.index = [i.replace('_', '-') for i in track_orient.index]
    track_orient.plot.bar(color='#a72a3d', rot=-90, fontsize=12)
    plt.xlabel('Track orientation', fontsize=13)
    plt.ylabel('No.', fontsize=12, rotation=0)
    ax3.yaxis.set_label_coords(0.0, 1.01)

    ax4 = plt.subplot2grid((1, 8), (0, 4))
    train_set.WindSpeed_avg.plot.box(color=colour, ax=ax4, widths=0.5, fontsize=12)
    ax4.set_xticklabels('')
    plt.xlabel('Average\nWind speed', fontsize=13, labelpad=29)
    plt.ylabel('($\\times$10 mph)', fontsize=12, rotation=0)
    ax4.yaxis.set_label_coords(0.2, 1.01)

    ax5 = plt.subplot2grid((1, 8), (0, 5))
    train_set.RelativeHumidity_avg.plot.box(color=colour, ax=ax5, widths=0.5, fontsize=12)
    ax5.set_xticklabels('')
    plt.xlabel('Average\nR.H.', fontsize=13, labelpad=29)
    plt.ylabel('(%)', fontsize=12, rotation=0)
    # plt.ylabel('($\\times$10%)', fontsize=12, rotation=0)
    ax5.yaxis.set_label_coords(0.0, 1.01)

    ax6 = plt.subplot2grid((1, 8), (0, 6))
    train_set.TotalPrecipitation_avg.plot.box(color=colour, ax=ax6, widths=0.5, fontsize=12)
    ax6.set_xticklabels('')
    plt.xlabel('Average\nTotal Precip.', fontsize=13, labelpad=29)
    plt.ylabel('(mm)', fontsize=12, rotation=0)
    ax6.yaxis.set_label_coords(0.0, 1.01)

    ax7 = plt.subplot2grid((1, 8), (0, 7))
    train_set.CoverPercentOpenSpace.plot.box(color=colour, ax=ax7, widths=0.5, fontsize=12)
    ax7.set_xticklabels('')
    plt.xlabel('Open Space\nCoverage', fontsize=13, labelpad=29)
    plt.ylabel('(%)', fontsize=12, rotation=0)
    ax7.yaxis.set_label_coords(0.0, 1.01)

    plt.tight_layout()

    path_to_file_weather = cdd_mod_heat_proto(0, "Variables" + save_as)
    plt.savefig(path_to_file_weather, dpi=dpi)
    if save_as == ".svg":
        save_svg_as_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


# A prototype model in the context of wind-related Incidents
def logistic_regression_model(trial_id=0,
                              route='ANGLIA', weather='Heat',
                              ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                              shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                              season='summer',
                              describe_var=False,
                              outlier_pctl=100,
                              add_const=True, seed=0, model='logit',
                              plot_roc=False, plot_predicted_likelihood=False,
                              save_as=".svg", dpi=None,
                              verbose=True):
    """
    :param trial_id:
    :param route: [str]
    :param weather: [str]
    :param ip_start_hrs: [int] or [float]
    :param nip_ip_gap:
    :param nip_start_hrs: [int] or [float]
    :param shift_yards_same_elr:
    :param shift_yards_diff_elr:
    :param hazard_pctl:
    :param season:
    :param describe_var: [bool]
    :param outlier_pctl:
    :param add_const:
    :param seed:
    :param model:
    :param plot_roc:
    :param plot_predicted_likelihood:
    :param save_as:
    :param dpi:
    # :param dig_deeper:
    :param verbose:
    :return:

    IncidentReason  IncidentReasonName    IncidentReasonDescription

    IQ              TRACK SIGN            Trackside sign blown down/light out etc.
    IW              COLD                  Non severe - Snow/Ice/Frost affecting infrastructure equipment',
                                          'Takeback Pumps'
    OF              HEAT/WIND             Blanket speed restriction for extreme heat or high wind in accordance with
                                          the Group Standards
    Q1              TKB PUMPS             Takeback Pumps
    X4              BLNK REST             Blanket speed restriction for extreme heat or high wind
    XW              WEATHER               Severe Weather not snow affecting infrastructure the responsibility of
                                          Network Rail
    XX              MISC OBS              Msc items on line (incl trees) due to effects of Weather responsibility of RT

    """
    # Get the m_data for modelling
    m_data = get_incident_data_with_weather_and_vegetation(route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                           shift_yards_same_elr, shift_yards_diff_elr,
                                                           hazard_pctl)

    # Select season data: 'spring', 'summer', 'autumn', 'winter'
    m_data = get_data_by_season(m_data, season)

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        m_data = m_data[m_data.DelayMinutes <= np.percentile(m_data.DelayMinutes, outlier_pctl)]

    # Select features
    explanatory_variables = specify_explanatory_variables_model_2()

    # Add the intercept
    if add_const:
        m_data = sm_tools.add_constant(m_data)  # data['const'] = 1.0
        explanatory_variables = ['const'] + explanatory_variables

    #
    nip_idx = m_data.IncidentReported == 0
    m_data.loc[nip_idx, ['DelayMinutes', 'DelayCost', 'IncidentCount']] = 0

    # Select data before 2014 as training data set, with the rest being test set
    train_set = m_data[m_data.FinancialYear != 2014]
    test_set = m_data[m_data.FinancialYear == 2014]

    if describe_var:
        describe_explanatory_variables(train_set, save_as=save_as, dpi=dpi)

    np.random.seed(seed)
    try:
        if model == 'probit':
            mod = sm.Probit(train_set.IncidentReported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=False)
        else:
            mod = sm.Logit(train_set.IncidentReported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=False)
        print(result.summary()) if verbose else print("")

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['OddsRatio'])
        print("\n{}".format(odds_ratios)) if verbose else print("")

        # Prediction
        test_set['incident_prob'] = result.predict(test_set[explanatory_variables])

        # ROC  # False Positive Rate (FPR), True Positive Rate (TPR), Threshold
        fpr, tpr, thr = sklearn.metrics.roc_curve(test_set.IncidentReported, test_set.incident_prob)
        # Area under the curve (AUC)
        auc = sklearn.metrics.auc(fpr, tpr)
        ind = list(np.where((tpr + 1 - fpr) == np.max(tpr + np.ones(tpr.shape) - fpr))[0])
        threshold = np.min(thr[ind])

        # prediction accuracy
        test_set['incident_prediction'] = test_set.incident_prob.apply(lambda x: 1 if x >= threshold else 0)
        test = pd.Series(test_set.IncidentReported == test_set.incident_prediction)
        mod_acc = np.divide(test.sum(), len(test))
        print("\nAccuracy: %f" % mod_acc) if verbose else print("")

        # incident prediction accuracy
        incident_only = test_set[test_set.IncidentReported == 1]
        test_acc = pd.Series(incident_only.IncidentReported == incident_only.incident_prediction)
        incident_acc = np.divide(test_acc.sum(), len(test_acc))
        print("Incident accuracy: %f" % incident_acc) if verbose else print("")

        if plot_roc:
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
            save_fig(cdd_mod_heat_proto(trial_id, "ROC" + save_as), dpi=dpi)

        # Plot incident delay minutes against predicted probabilities
        if plot_predicted_likelihood:
            incident_ind = test_set.IncidentReported == 1
            plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            ax.scatter(test_set[incident_ind].incident_prob, test_set[incident_ind].DelayMinutes,
                       c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
                       label="Heat-related incident (2014/15)")
            plt.axvline(x=threshold, label="Threshold: %.2f" % threshold, color='#e5c100', linewidth=2)
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
            save_fig(cdd_mod_heat_proto(trial_id, "Predicted-likelihood" + save_as), dpi=dpi)

    except Exception as e:
        print(e)
        result = e
        mod_acc, incident_acc, threshold = np.nan, np.nan, np.nan

    return m_data, train_set, test_set, result, mod_acc, incident_acc, threshold
