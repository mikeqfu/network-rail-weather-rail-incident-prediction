""" A prototype model in the context of heat-related incidents """

import datetime
import itertools
import os
import re

import datetime_truncate
import matplotlib.pyplot as plt
import measurement.measures
import numpy as np
import pandas as pd
import shapely.geometry
import sklearn.metrics
import sklearn.utils.extmath
import statsmodels.discrete.discrete_model as smd
import statsmodels.tools.tools as smd_tools

import database_met as dbm
import database_veg as dbv
import railwaycodes_utils as rc_utils
import settings
from converters import str_to_num_mileage, mileage_to_str, mileage_to_yards, yards_to_mileage, svg_to_emf
from utils import cdd, load_pickle, save_pickle, find_match

# Apply the preferences ==============================================================================================
settings.mpl_preferences(reset=False)
settings.np_preferences(reset=False)
settings.pd_preferences(reset=False)
plt.rc('font', family='Times New Roman')

# ====================================================================================================================
""" Change directory """


# Change directory to "Data\\Model" and sub-directories
def cdd_mod_dat(*directories):
    path = cdd("Model", "dat")
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# Change directory to "Model\\Prototype_Heat\\Trial_" and sub-directories
def cdd_mod_heat(trial_id=0, *directories):
    path = cdd("Model", "Prototype_Heat", "Trial_{}".format(trial_id))
    os.makedirs(path, exist_ok=True)
    for directory in directories:
        path = os.path.join(path, directory)
    return path


# ====================================================================================================================
""" Calculations for weather data """


# Specify the statistics that need to be computed
def specify_weather_stats_calculations():
    weather_stats_computations = {'Temperature': (np.max, np.min, np.average),
                                  'RelativeHumidity': (np.max, np.min, np.average),
                                  'WindSpeed': np.max,
                                  'WindGust': np.max,
                                  'Snowfall': np.max,
                                  'TotalPrecipitation': (np.max, np.min, np.average)}
    return weather_stats_computations


# Get all weather variable names
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


# Compute the statistics for all the weather variables (except wind)
def calculate_weather_variables_stats(weather_data):
    """
    Note: to get the n-th percentitle, use percentile(n)

    This function also returns the weather dataframe indices. The corresponding weather conditions in that weather
    cell might cause wind-related incidents.
    """
    # Compute the statistics
    weather_stats_computations = specify_weather_stats_calculations()
    weather_stats = weather_data.groupby('WeatherCell').aggregate(weather_stats_computations)
    # Compute average wind speeds and directions
    weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
        calculate_wind_averages(weather_data.WindSpeed, weather_data.WindDirection)
    if not weather_stats.empty:
        stats_info = weather_stats.values[0].tolist() + [weather_data.index.tolist()]
    else:
        stats_info = [np.nan] * len(weather_stats.columns) + [[None]]
    return stats_info


# Find Weather Cell ID
def find_weather_cell_id(longitude, latitude):
    weather_cell = dbm.get_weather_cell()

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


# Get TRUST and the relevant weather data for each location
def get_incident_location_weather(route=None, weather=None, ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                  subset_weather_for_nip=False,
                                  update=False):
    """
    :param route: [str] Route name
    :param weather: [str] weather category
    :param ip_start_hrs: [int/float]
    :param nip_ip_gap:
    :param nip_start_hrs: [int/float]
    :param subset_weather_for_nip: [bool]
    :param update: [bool]
    :return: [DataFrame]
    """

    filename = dbm.make_filename("incident_location_weather", route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        iwdata = load_pickle(path_to_file)
    else:
        try:
            # ----------------------------------------------------------------------------------------------------
            """ Get data """

            # Getting incident data for all incident locations
            sdata = dbm.get_schedule8_cost_by_datetime_location_reason(route, weather, update)
            # Drop non-weather-related incident records
            sdata = sdata[sdata.WeatherCategory != ''] if weather is None else sdata
            # Get data for the specified "Incident Periods"
            sdata['incident_duration'] = sdata.EndDate - sdata.StartDate
            sdata['critical_start'] = \
                sdata.StartDate.apply(datetime_truncate.truncate_hour) + datetime.timedelta(hours=ip_start_hrs)
            sdata['critical_end'] = sdata.StartDate
            sdata['critical_period'] = sdata.critical_end - sdata.critical_start

            if sdata.WeatherCell.dtype != 'int64':
                # Rectify the records for which weather cell id is empty
                weather_cell = dbm.get_weather_cell()
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

            # Get weather data
            weather_data = dbm.get_weather().reset_index()

            # ----------------------------------------------------------------------------------------------------
            """ Processing weather data for IP - Get weather conditions which led to incidents for each record """

            def get_ip_weather_conditions(weather_cell_id, ip_start, ip_end):
                """
                :param weather_cell_id: [int] Weather Cell ID
                :param ip_start: [Timestamp] start of "incident period"
                :param ip_end: [Timestamp] end of "incident period"
                :return:
                """
                # Get weather data about where and when the incident occurred
                ip_weather_data = weather_data[(weather_data.WeatherCell == weather_cell_id) &
                                               (weather_data.DateTime >= ip_start) &
                                               (weather_data.DateTime <= ip_end)]
                # Get the max/min/avg weather parameters for those incident periods
                weather_stats_data = calculate_weather_variables_stats(ip_weather_data)
                return weather_stats_data

            # Get data for the specified IP
            ip_statistics = sdata.apply(
                lambda x: pd.Series(get_ip_weather_conditions(x.WeatherCell, x.critical_start, x.critical_end)), axis=1)

            ip_statistics.columns = weather_variable_names() + ['critical_weather_idx']
            ip_statistics['Temperature_diff'] = ip_statistics.Temperature_max - ip_statistics.Temperature_min

            ip_data = sdata.join(ip_statistics.dropna(), how='inner')
            ip_data['IncidentReported'] = 1

            # Get weather data that did not ever cause incidents according to records?
            if subset_weather_for_nip:
                weather_data = weather_data.loc[
                    weather_data.WeatherCell.isin(ip_data.WeatherCell) &
                    ~weather_data.index.isin(itertools.chain(*ip_data.critical_weather_idx))]

            # Processing weather data for non-IP
            nip_data = sdata.copy(deep=True)
            nip_data.critical_end = nip_data.critical_start + datetime.timedelta(days=nip_ip_gap)
            nip_data.critical_start = nip_data.critical_end + datetime.timedelta(hours=nip_start_hrs)
            nip_data.critical_period = nip_data.critical_end - nip_data.critical_start

            # -----------------------------------------------------------------------
            """ Get data of weather which did not cause incidents for each record """

            def get_nip_weather_conditions(weather_cell_id, nip_start, nip_end, incident_location):
                # Get non-IP weather data about where and when the incident occurred
                nip_weather_data = weather_data[
                    (weather_data.WeatherCell == weather_cell_id) &
                    (weather_data.DateTime >= nip_start) & (weather_data.DateTime <= nip_end)]
                # Get all incident period data on the same section
                ip_overlap = ip_data[
                    (ip_data.StanoxSection == incident_location) &
                    (((ip_data.critical_start < nip_start) & (ip_data.critical_end > nip_start)) |
                     ((ip_data.critical_start < nip_end) & (ip_data.critical_end > nip_end)))]
                # Skip data of weather causing incidents at around the same time; but
                if not ip_overlap.empty:
                    nip_weather_data = nip_weather_data[
                        (nip_weather_data.DateTime < np.min(ip_overlap.critical_start)) |
                        (nip_weather_data.DateTime > np.max(ip_overlap.critical_end))]
                # Get the max/min/avg weather parameters for those incident periods
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
            iwdata = pd.concat([nip_data, ip_data], axis=0, ignore_index=True)

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
            iwdata['wind_direction'] = iwdata.WindDirection_avg.apply(categorise_wind_directions)
            wind_direction = pd.get_dummies(iwdata.wind_direction, prefix='wind_direction')
            iwdata = iwdata.join(wind_direction)

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

            iwdata = iwdata.join(categorise_track_orientations(iwdata))

            # ----------------------------------------------------
            """ Categorise temperature: 25, 26, 27, 28, 29, 30 """

            # critical_temperature_dat = [
            #     iwdata.Temperature_max.map(lambda x: 1 if x >= t else -1) for t in range(25, 31)]
            # critical_temperature = ['Temperature_max ≥ {}°C'.format(t) for t in range(25, 31)]
            # for dat, col in zip(critical_temperature_dat, critical_temperature):
            #     iwdata[col] = dat

            def categorise_temperatures(data):
                data['temperatue_category'] = None
                data.temperatue_category[data.Temperature_max < 24] = 'Temperature_max < 24°C'
                data.temperatue_category[data.Temperature_max == 24] = 'Temperature_max = 24°C'
                data.temperatue_category[data.Temperature_max == 25] = 'Temperature_max = 25°C'
                data.temperatue_category[data.Temperature_max == 26] = 'Temperature_max = 26°C'
                data.temperatue_category[data.Temperature_max == 27] = 'Temperature_max = 27°C'
                data.temperatue_category[data.Temperature_max == 28] = 'Temperature_max = 28°C'
                data.temperatue_category[data.Temperature_max == 29] = 'Temperature_max = 29°C'
                data.temperatue_category[data.Temperature_max >= 30] = 'Temperature_max ≥ 30°C'
                # data.temperatue_category[data.Temperature_max > 30] = 'Temperature_max > 30°C'
                return data

            iwdata = categorise_temperatures(iwdata)
            iwdata = iwdata.join(pd.get_dummies(iwdata.temperatue_category, prefix=''))

            save_pickle(iwdata, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            iwdata = None

    return iwdata


def temperature_deviation(nip_ip_gap=-14, add_errbar=True, save_as=".svg", dpi=600):
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
    if add_errbar:
        container = plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', yerr=diff_std, capsize=4,
                            width=0.7, color='#9FAFBE')
        connector, caplines, (vertical_lines,) = container.errorbar.lines
        vertical_lines.set_color('#666666')
        for cap in caplines:
            cap.set_color('#da8067')
    else:
        plt.bar(np.arange(1, len(diff_means) + 1), diff_means, align='center', width=0.7, color='#9FAFBE')
        plt.grid(False)
    plt.xticks(np.arange(1, len(diff_means) + 1), fontsize=14)
    plt.xlabel('Latent period (Number of days)', fontsize=14)
    plt.ylabel('Temperature deviation (°C)', fontsize=14)
    plt.tight_layout()

    plt.savefig(cdd_mod_heat(0, "Temp deviation" + save_as), dpi=dpi)


# ====================================================================================================================
""" Calculations for vegetation data """


# Get Schedule 8 costs (minutes & cost) aggregated for each STANOX section
def get_incident_locations_from_metex_db(route=None, weather=None, same_elr=None):
    """
    :param route:
    :param weather:
    :param same_elr:
    :return:
    """
    # Load Schedule 8 costs data aggregated by financial year and STANOX section
    s8data = dbm.get_schedule8_cost_by_location(route, weather).loc[:, 'Route':]

    # Aggregate the data for each STANOX section
    incident_locations = s8data.groupby(list(s8data.columns)[:-3]).agg(np.sum)
    incident_locations.reset_index(inplace=True)

    # Create two additional columns about data of mileages (convert str to num)
    incident_locations[['start_mileage', 'end_mileage']] = \
        incident_locations[['StartMileage', 'EndMileage']].applymap(str_to_num_mileage)

    # Remove records for which ELR information was missing
    incident_locations = incident_locations[
        ~(incident_locations.StartELR.str.contains('^$')) & ~(incident_locations.EndELR.str.contains('^$'))]

    # # Remove records of 'WTS', as vegetation data is unavailable for this ELR
    # incident_locations = incident_locations[
    #     ~(incident_locations.StartELR.str.contains(re.compile('^$|WTS'))) &
    #     ~(incident_locations.EndELR.str.contains(re.compile('^$|WTS')))]

    # Get "ilocations_same_elr" / "ilocations_diff_elr", and "furlongs_veg_db"
    if same_elr is None:
        return incident_locations
    elif same_elr is True:
        # Subset the data for which the 'StartELR' and 'EndELR' are THE SAME
        same_elr_idx = incident_locations.StartELR == incident_locations.EndELR
        incident_locations_same_elr = incident_locations[same_elr_idx]
        return incident_locations_same_elr
    elif same_elr is False:
        # Subset the data for which the 'StartELR' and 'EndELR' are DIFFERENT
        diff_elr_idx = incident_locations.StartELR != incident_locations.EndELR
        incident_locations_diff_elr = incident_locations[diff_elr_idx]
        return incident_locations_diff_elr


# Get the ELR & mileage data of furlong locations
def get_furlongs_info_from_veg_db(location_data_only=False, update=False):
    """
    :param location_data_only:
    :param update:
    :return:
    """
    filename = "furlongs_veg_db"
    path_to_file = cdd_mod_dat(filename + ".pickle")

    if location_data_only:
        path_to_file = path_to_file.replace(filename, filename + "_loc_only")

    if os.path.isfile(path_to_file) and not update:
        furlongs = load_pickle(path_to_file)
    else:
        try:
            # Get the data of furlong location
            if location_data_only:  # using the original 'FurlongLocation'?
                furlong_location = dbv.get_furlong_location(useful_columns_only=True, update=update)
            else:  # using the merged data set 'furlong_vegetation_data'
                furlong_vegetation_data = dbv.get_furlong_vegetation_conditions(update=update)
                furlong_vegetation_data.set_index('FurlongID', inplace=True)
                furlong_location = furlong_vegetation_data.sort_index()

            # Column names of mileage data (as string)
            str_mileage_colnames = ['StartMileage', 'EndMileage']
            # Column names of ELR and mileage data (as string)
            elr_mileage_colnames = ['ELR'] + str_mileage_colnames

            furlongs = furlong_location.drop_duplicates(elr_mileage_colnames)

            # Create two new columns of mileage data (as float)
            num_mileage_colnames = ['start_mileage', 'end_mileage']
            furlongs[num_mileage_colnames] = furlongs[str_mileage_colnames].applymap(str_to_num_mileage)

            # Sort the furlong data by ELR and mileage
            furlongs.sort_values(['ELR'] + num_mileage_colnames, inplace=True)

            save_pickle(furlongs, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            furlongs = None

    return furlongs


# Get adjusted Start and End mileages
def adjust_start_end(furlongs, elr, start_mileage, end_mileage, shift_yards):
    """
    :param furlongs:
    :param elr:
    :param start_mileage:
    :param end_mileage:
    :param shift_yards:
    :return:
    """
    elr_furlongs = furlongs[furlongs.ELR == elr]

    def merge_start_and_end(start_array, end_array):
        end_array = np.array(end_array)
        assert len(start_array) == len(end_array)
        return pd.Series(np.append(start_array, end_array[-1]))

    try:
        elr_mileages = merge_start_and_end(elr_furlongs.start_mileage, elr_furlongs.end_mileage)
    except IndexError:
        return '', '', np.nan, np.nan, np.nan, []

    m_indices = pd.Index(elr_mileages)
    s_indices = pd.Index(elr_furlongs.StartMileage)
    e_indices = pd.Index(elr_furlongs.EndMileage)

    def num_mileage_shifting(mileage, y):
        yards = mileage_to_yards(mileage) + y
        str_mileage = yards_to_mileage(yards)
        return str_to_num_mileage(str_mileage)

    if start_mileage <= end_mileage:
        if start_mileage == end_mileage:
            start_mileage = num_mileage_shifting(start_mileage, -shift_yards)
            end_mileage = num_mileage_shifting(end_mileage, shift_yards)
        else:  # start_mileage < end_mileage
            pass
        # Get adjusted mileages of start and end locations ---------------
        try:
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'ffill')]
        except (ValueError, KeyError):
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'nearest')]
        try:
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'bfill')]
        except (ValueError, KeyError):
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'nearest')]
        # Get 'FurlongID's for extracting vegetation data ----------------
        try:
            s_idx = s_indices.get_loc(mileage_to_str(adj_start_mileage))
        except (ValueError, KeyError):
            s_idx = e_indices.get_loc(mileage_to_str(adj_start_mileage))
            adj_start_mileage = str_to_num_mileage(elr_furlongs.StartMileage.iloc[s_idx])
        try:
            e_idx = e_indices.get_loc(mileage_to_str(adj_end_mileage))
        except (ValueError, KeyError):
            e_idx = s_indices.get_loc(mileage_to_str(adj_end_mileage))
            adj_end_mileage = str_to_num_mileage(elr_furlongs.EndMileage.iloc[e_idx])
    else:  # start_mileage > end_mileage:
        # Get adjusted mileages of start and end locations ---------------
        try:
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'bfill')]
        except (ValueError, KeyError):
            adj_start_mileage = elr_mileages[m_indices.get_loc(start_mileage, 'nearest')]
        try:
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'ffill')]
        except (ValueError, KeyError):
            adj_end_mileage = elr_mileages[m_indices.get_loc(end_mileage, 'nearest')]
        # Get 'FurlongID's for extracting vegetation data ----------------
        try:
            s_idx = e_indices.get_loc(mileage_to_str(adj_start_mileage))
        except (ValueError, KeyError):
            s_idx = s_indices.get_loc(mileage_to_str(adj_start_mileage))
            adj_start_mileage = str_to_num_mileage(elr_furlongs.EndMileage.iloc[s_idx])
        try:
            e_idx = s_indices.get_loc(mileage_to_str(adj_end_mileage))
        except (ValueError, KeyError):
            e_idx = e_indices.get_loc(mileage_to_str(adj_end_mileage))
            adj_end_mileage = str_to_num_mileage(elr_furlongs.StartMileage.iloc[e_idx])

    if s_idx <= e_idx:
        e_idx = e_idx + 1 if e_idx < len(elr_mileages) else e_idx
        veg_furlongs = elr_furlongs.iloc[s_idx:e_idx]
    else:  # s_idx > e_idx
        s_idx = s_idx + 1 if s_idx < len(elr_mileages) else s_idx
        veg_furlongs = elr_furlongs.iloc[e_idx:s_idx]

    return \
        mileage_to_str(adj_start_mileage), \
        mileage_to_str(adj_end_mileage), \
        adj_start_mileage, adj_end_mileage, \
        measurement.measures.Distance(mile=np.abs(adj_end_mileage - adj_start_mileage)).yd, \
        veg_furlongs.index.tolist()


# Get furlongs data of incident locations each identified by the same ELRs
def get_incident_location_furlongs_same_elr(route=None, weather=None, shift_yards_same_elr=220, update=False):
    """
    :param route:
    :param weather:
    :param shift_yards_same_elr: yards
    :param update:
    :return:
    """
    filename = dbm.make_filename("incident_location_furlongs_same_ELRs", route, weather, shift_yards_same_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs_same_elr = load_pickle(path_to_file)
    else:
        try:
            # Get data about for which the 'StartELR' and 'EndELR' are THE SAME
            incident_locations_same_elr = get_incident_locations_from_metex_db(route, weather, same_elr=True)

            # Get furlong information
            furlongs = get_furlongs_info_from_veg_db(location_data_only=False, update=update)

            # Get data of each incident's furlong locations for extracting vegetation
            adjusted_mileages = incident_locations_same_elr.apply(
                lambda record: adjust_start_end(
                    furlongs, record.StartELR, record.start_mileage, record.end_mileage, shift_yards_same_elr), axis=1)
            # Column names
            colnames = ['StartMileage_adjusted',
                        'EndMileage_adjusted',
                        'start_mileage_adjusted',
                        'end_mileage_adjusted',
                        'total_yards_adjusted',  # yards
                        'critical_FurlongIDs']
            # Get adjusted mileage data
            adjusted_mileages_data = pd.DataFrame(list(adjusted_mileages), incident_locations_same_elr.index, colnames)

            save_pickle(adjusted_mileages_data, cdd_mod_dat("adjusted_mileages_same_ELRs_{}.pickle".format(route)))

            incident_locations_same_elr.drop(['start_mileage', 'end_mileage'], axis=1, inplace=True)
            incident_location_furlongs_same_elr = incident_locations_same_elr.join(
                adjusted_mileages_data[['total_yards_adjusted', 'critical_FurlongIDs']], how='inner').dropna()

            save_pickle(incident_location_furlongs_same_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs_same_elr = None

    return incident_location_furlongs_same_elr


# Get furlongs data by the same ELRs
def get_incident_furlongs_same_elr(route=None, weather=None, shift_yards_same_elr=220, update=False):
    filename = dbm.make_filename("incident_furlongs_same_elr", route, weather, shift_yards_same_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_furlongs_same_elr = load_pickle(path_to_file)
    else:
        try:
            incident_location_furlongs_same_elr = \
                get_incident_location_furlongs_same_elr(route, weather, shift_yards_same_elr, update)
            # Form a list containing all the furlong ID's
            veg_furlongs_idx = list(itertools.chain(*incident_location_furlongs_same_elr.critical_FurlongIDs))
            # Get furlong information
            furlongs = get_furlongs_info_from_veg_db(location_data_only=False, update=update)

            incident_furlongs_same_elr = furlongs.loc[veg_furlongs_idx]. \
                drop(['start_mileage', 'end_mileage'], axis=1).drop_duplicates(subset='AssetNumber')

            incident_furlongs_same_elr['IncidentReported'] = 1

            save_pickle(incident_furlongs_same_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_furlongs_same_elr = None

    return incident_furlongs_same_elr


# Get information of connecting points for different ELRs
def get_connecting_nodes(route=None, update=False):
    filename = dbm.make_filename("connecting_nodes_between_ELRs", route, weather=None)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        connecting_nodes = load_pickle(path_to_file)
    else:
        try:
            # Get data about where incidents occurred
            incident_locations_diff_elr = get_incident_locations_from_metex_db(route, same_elr=False)

            elr_mileage_cols = ['StartELR', 'StartMileage', 'EndELR', 'EndMileage', 'start_mileage', 'end_mileage']
            diff_elr_mileages = incident_locations_diff_elr[elr_mileage_cols].drop_duplicates()

            # Trying to get the connecting nodes ...
            def get_conn_mileages(start_elr, start_mileage, end_elr, end_mileage):
                s_end_mileage, e_start_mileage = rc_utils.get_conn_end_start_mileages(start_elr, end_elr)
                if s_end_mileage is None:
                    s_end_mileage = start_mileage
                if e_start_mileage is None:
                    e_start_mileage = end_mileage
                return s_end_mileage, e_start_mileage

            conn_mileages = diff_elr_mileages.apply(
                lambda x: pd.Series(get_conn_mileages(x.StartELR, x.start_mileage, x.EndELR, x.end_mileage)), axis=1)
            conn_mileages.columns = ['StartELR_EndMileage', 'EndELR_StartMileage']

            idx_columns = ['StartELR', 'StartMileage', 'EndELR', 'EndMileage']
            connecting_nodes = diff_elr_mileages[idx_columns].join(conn_mileages).set_index(idx_columns)

            save_pickle(connecting_nodes, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            connecting_nodes = None

    return connecting_nodes


# Get furlongs data of incident locations each identified by different ELRs
def get_incident_location_furlongs_diff_elr(route=None, weather=None, shift_yards_diff_elr=220, update=False):
    filename = dbm.make_filename("incident_location_furlongs_diff_ELRs", route, weather, shift_yards_diff_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs_diff_elr = load_pickle(path_to_file)
    else:
        try:
            # Get data for which the 'StartELR' and 'EndELR' are DIFFERENT
            incident_locations_diff_elr = get_incident_locations_from_metex_db(route, weather, same_elr=False)

            # Get furlong information
            furlongs = get_furlongs_info_from_veg_db(location_data_only=False, update=update)

            # Get connecting points for different (ELRs, mileages)
            connecting_nodes = get_connecting_nodes(route, update=update)

            incident_locations_diff_elr = incident_locations_diff_elr.join(
                connecting_nodes, on=connecting_nodes.index.names, how='inner')
            str_conn_colnames = ['StartELR_EndMileage', 'EndELR_StartMileage']
            num_conn_colnames = ['StartELR_end_mileage', 'EndELR_start_mileage']
            incident_locations_diff_elr[num_conn_colnames] = \
                incident_locations_diff_elr[str_conn_colnames].applymap(str_to_num_mileage)

            """ Get data of each incident's furlong locations for extracting vegetation """
            # Processing Start locations
            adjusted_start_elr_mileages = incident_locations_diff_elr.apply(
                lambda x: adjust_start_end(
                    furlongs, x.StartELR, x.start_mileage, x.StartELR_end_mileage, shift_yards_diff_elr),
                axis=1)

            # Column names for adjusted_start_elr_mileages_data
            start_elr_colnames = [
                'StartMileage_adjusted',
                'StartELR_EndMileage_adjusted',
                'start_mileage_adjusted',
                'StartELR_end_mileage_adjusted',
                'StartELR_total_yards_adjusted',  # yards
                'StartELR_FurlongIDs']

            # Form a dataframe for adjusted_start_elr_mileages_data
            adjusted_start_elr_mileages_data = pd.DataFrame(list(adjusted_start_elr_mileages),
                                                            index=incident_locations_diff_elr.index,
                                                            columns=start_elr_colnames)

            # Find the index for null values in adjusted_start_elr_mileages_data
            start_elr_null_idx = \
                adjusted_start_elr_mileages_data[adjusted_start_elr_mileages_data.isnull().any(axis=1)].index

            # Processing End locations
            adjusted_end_elr_mileages = incident_locations_diff_elr.apply(
                lambda record: adjust_start_end(
                    furlongs, record.EndELR, record.EndELR_start_mileage, record.end_mileage, shift_yards_diff_elr),
                axis=1)

            # Column names for adjusted_end_elr_mileages_data
            end_elr_colnames = [
                'EndELR_StartMileage_adjusted',
                'EndMileage_adjusted',
                'EndELR_start_mileage_adjusted',
                'end_mileage_adjusted',
                'EndELR_total_yards_adjusted',  # yards
                'EndELR_FurlongIDs']

            # Form a dataframe for adjusted_end_elr_mileages_data
            adjusted_end_elr_mileages_data = pd.DataFrame(list(adjusted_end_elr_mileages),
                                                          index=incident_locations_diff_elr.index,
                                                          columns=end_elr_colnames)

            # Find the index for null values in adjusted_end_elr_mileages_data
            end_elr_null_idx = adjusted_end_elr_mileages_data[adjusted_end_elr_mileages_data.isnull().any(axis=1)].index

            # --------------------------------------------------------------------------------------------
            adjusted_mileages_data = adjusted_start_elr_mileages_data.join(adjusted_end_elr_mileages_data)
            adjusted_mileages_data['total_yards_adjusted'] = list(zip(
                adjusted_mileages_data.StartELR_total_yards_adjusted.fillna(0),
                adjusted_mileages_data.EndELR_total_yards_adjusted.fillna(0)))
            adjusted_mileages_data['critical_FurlongIDs'] = \
                adjusted_mileages_data.StartELR_FurlongIDs + adjusted_mileages_data.EndELR_FurlongIDs

            # Save the adjusted_mileages_data
            save_pickle(adjusted_mileages_data, cdd_mod_dat("adjusted_mileages_diff_ELRs.pickle"))

            incident_locations_diff_elr.drop(
                str_conn_colnames + num_conn_colnames + ['start_mileage', 'end_mileage'], axis=1, inplace=True)

            colnames = incident_locations_diff_elr.columns
            start_loc_cols = [x for x in colnames if re.match('^Start(?!Location)', x)]
            end_loc_cols = [x for x in colnames if re.match('^End(?!Location)', x)]
            incident_locations_diff_elr.loc[start_elr_null_idx, start_loc_cols] = \
                incident_locations_diff_elr.loc[start_elr_null_idx, end_loc_cols].values
            incident_locations_diff_elr.loc[end_elr_null_idx, end_loc_cols] = \
                incident_locations_diff_elr.loc[end_elr_null_idx, start_loc_cols].values

            incident_location_furlongs_diff_elr = incident_locations_diff_elr.join(
                adjusted_mileages_data[['total_yards_adjusted', 'critical_FurlongIDs']], how='inner').dropna()

            save_pickle(incident_location_furlongs_diff_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs_diff_elr = None

    return incident_location_furlongs_diff_elr


# Get furlongs data by different ELRS
def get_incident_furlongs_diff_elr(route=None, weather=None, shift_yards_diff_elr=220, update=False):
    filename = dbm.make_filename("incident_furlongs_diff_ELRs", route, weather, shift_yards_diff_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incid_furlongs_diff_elr = load_pickle(path_to_file)
    else:
        try:
            incident_location_furlongs_diff_elr = get_incident_location_furlongs_diff_elr(route, weather,
                                                                                          shift_yards_diff_elr)

            # Form a list containing all the furlong ID's
            veg_furlongs_idx = list(itertools.chain(*incident_location_furlongs_diff_elr.critical_FurlongIDs))

            # Get furlong information
            furlongs = get_furlongs_info_from_veg_db(location_data_only=False)

            # Merge the data of the starts and ends
            incid_furlongs_diff_elr = furlongs.loc[veg_furlongs_idx]. \
                drop(['start_mileage', 'end_mileage'], axis=1).drop_duplicates(subset='AssetNumber')

            incid_furlongs_diff_elr['IncidentReported'] = 1

            save_pickle(incid_furlongs_diff_elr, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incid_furlongs_diff_elr = None

    return incid_furlongs_diff_elr


# Combine the incident furlong data of both of the above
def get_incident_location_furlongs(route=None, shift_yards_same_elr=220, shift_yards_diff_elr=220, update=False):
    filename = dbm.make_filename("incident_location_furlongs", route, None, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_location_furlongs = load_pickle(path_to_file)
    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            incident_location_furlongs_same_elr = \
                get_incident_location_furlongs_same_elr(route, None, shift_yards_same_elr)
            # Data of incident furlongs: start and end are identified by different ELRs
            incident_location_furlongs_diff_elr = \
                get_incident_location_furlongs_diff_elr(route, None, shift_yards_diff_elr)
            # Merge the above two data sets
            incident_location_furlongs = incident_location_furlongs_same_elr.append(
                incident_location_furlongs_diff_elr)
            incident_location_furlongs.sort_index(inplace=True)
            # incident_location_furlongs['IncidentReported'] = 1
            save_pickle(incident_location_furlongs, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_location_furlongs = None

    return incident_location_furlongs


# Combine the incident furlong data of both of the above
def get_incident_furlongs(route=None, shift_yards_same_elr=220, shift_yards_diff_elr=220, update=False):
    filename = dbm.make_filename("incident_furlongs", route, None, shift_yards_same_elr, shift_yards_diff_elr)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        incident_furlongs = load_pickle(path_to_file)
    else:
        try:
            # Data of incident furlongs: both start and end identified by the same ELR
            incid_furlongs_same_elr = get_incident_furlongs_same_elr(route, None, shift_yards_same_elr, update=update)
            # Data of incident furlongs: start and end are identified by different ELRs
            incid_furlongs_diff_elr = get_incident_furlongs_diff_elr(route, None, shift_yards_diff_elr, update=update)
            # Merge the above two data sets
            furlong_incidents = incid_furlongs_same_elr.append(incid_furlongs_diff_elr)
            furlong_incidents.drop_duplicates(subset='AssetNumber', inplace=True)
            # Data of furlong vegetation coverage and hazardous trees
            furlong_vegetation_data = dbv.get_furlong_vegetation_conditions(route)
            incident_furlongs = furlong_vegetation_data.join(
                furlong_incidents[['IncidentReported']], on='FurlongID', how='inner')
            incident_furlongs.sort_values(by='StructuredPlantNumber', inplace=True)
            # # incident_furlongs.index = range(len(incident_furlongs))
            save_pickle(incident_furlongs, path_to_file)
        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            incident_furlongs = None

    return incident_furlongs


# Specify the statistics that need to be computed
def specify_vegetation_stats_calculations(features):
    # "CoverPercent..."
    cover_percents = [x for x in features if re.match('^CoverPercent[A-Z]', x)]

    veg_stats_calc = dict(zip(cover_percents, [np.sum] * len(cover_percents)))

    veg_stats_calc.update({'AssetNumber': np.count_nonzero,
                           'TreeNumber': np.sum,
                           'TreeNumberUp': np.sum,
                           'TreeNumberDown': np.sum,
                           'Electrified': np.any,
                           'DateOfMeasure': lambda x: tuple(x),
                           # 'AssetDesc1': np.all,
                           # 'IncidentReported': np.any
                           'HazardTreeNumber': np.sum})

    # variables for hazardous trees
    hazard_min = [x for x in features if re.match('^HazardTree.*min$', x)]
    hazard_max = [x for x in features if re.match('^HazardTree.*max$', x)]
    hazard_rest = [x for x in features if re.match('^HazardTree[a-z]((?!_).)*$', x)]
    # Computations for hazardous trees variables
    hazard_calc = [dict(zip(hazard_rest, [lambda x: tuple(x)] * len(hazard_rest))),
                   dict(zip(hazard_min, [np.min] * len(hazard_min))),
                   dict(zip(hazard_max, [np.max] * len(hazard_max)))]

    # Update vegetation_stats_computations
    veg_stats_calc.update({k: v for d in hazard_calc for k, v in d.items()})

    return cover_percents, hazard_rest, veg_stats_calc


# Get vegetation conditions for incident locations
def get_incident_location_vegetation(route=None, shift_yards_same_elr=220, shift_yards_diff_elr=220,
                                     hazard_pctl=50, update=False):
    """
    Note that the "CoverPercent..." in furlong_vegetation_data has been
    amended when furlong_data was read. Check the function get_furlong_data().
    """
    filename = dbm.make_filename("incident_location_vegetation", route, None,
                                 shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        ivdata = load_pickle(path_to_file)
    else:
        try:
            # Get data of furlong vegetation coverage and hazardous trees
            furlong_vegetation_data = dbv.get_furlong_vegetation_conditions(update=update).set_index('FurlongID')

            # Get all column names as features
            features = furlong_vegetation_data.columns
            # Specify the statistics that need to be computed
            cover_percents, hazard_rest, veg_stats_calc = specify_vegetation_stats_calculations(features)

            # Get features which would be filled with "0" and "inf", respectively
            fill_0 = [x for x in features if re.match('.*height', x)] + ['HazardTreeNumber']
            fill_inf = [x for x in features if re.match('^.*prox|.*diam', x)]

            # Define a function that computes vegetation stats for each incident record
            def compute_vegetation_variables_stats(furlong_ids, start_elr, end_elr, total_yards_adjusted):
                """
                Note: to get the n-th percentitle may use percentile(n)
                """
                vegetation_data = furlong_vegetation_data.loc[furlong_ids]

                veg_stats = vegetation_data.groupby('ELR').aggregate(veg_stats_calc)
                veg_stats[cover_percents] = veg_stats[cover_percents].div(veg_stats.AssetNumber, axis=0).values

                if start_elr == end_elr:
                    if np.isnan(veg_stats.HazardTreeNumber.values):
                        veg_stats[fill_0] = 0.0
                        veg_stats[fill_inf] = 999999.0
                    else:
                        assert isinstance(hazard_pctl, int) and 0 <= hazard_pctl <= 100
                        veg_stats[hazard_rest] = veg_stats[hazard_rest].applymap(
                            lambda x: np.nanpercentile(tuple(itertools.chain(*pd.Series(x).dropna())), hazard_pctl))
                else:
                    if np.all(np.isnan(veg_stats.HazardTreeNumber.values)):
                        veg_stats[fill_0] = 0.0
                        veg_stats[fill_inf] = 999999.0
                        calc_further = {k: lambda y: np.average(y) for k in hazard_rest}
                    else:
                        veg_stats[hazard_rest] = veg_stats[hazard_rest].applymap(
                            lambda y: tuple(itertools.chain(*pd.Series(y).dropna())))
                        hazard_rest_func = [lambda y: np.nanpercentile(np.sum(y), hazard_pctl)]
                        calc_further = dict(zip(hazard_rest, hazard_rest_func * len(hazard_rest)))

                    # Specify further calculations
                    calc_further.update({'AssetNumber': np.sum})
                    calc_further.update(dict(DateOfMeasure=lambda y: tuple(itertools.chain(*y))))
                    calc_further.update({k: lambda y: tuple(y) for k in cover_percents})
                    veg_stats_calc_further = veg_stats_calc.copy()
                    veg_stats_calc_further.update(calc_further)

                    # Rename index (by which the dataframe can be grouped)
                    veg_stats.index = pd.Index(data=['-'.join(set(veg_stats.index))] * len(veg_stats.index), name='ELR')
                    veg_stats = veg_stats.groupby(veg_stats.index).aggregate(veg_stats_calc_further)

                    # Calculate the cover percents across two neighbouring ELRs
                    def overall_cover_percent(cp_start_and_end):
                        # (start * end) / (start + end)
                        multiplier = np.prod(total_yards_adjusted) / np.sum(total_yards_adjusted)
                        # 1/start, 1/end
                        cp_start, cp_end = cp_start_and_end
                        s_, e_ = np.divide(1, total_yards_adjusted)
                        # a numerator
                        n = e_ * cp_start + s_ * cp_end
                        # a denominator
                        d = np.sum(cp_start_and_end) if np.all(cp_start_and_end) else 1
                        factor = multiplier * np.divide(n, d)
                        return factor * d

                    veg_stats[cover_percents] = veg_stats[cover_percents].applymap(overall_cover_percent)

                # Calculate tree densities (number of trees per furlong)
                veg_stats['TreeDensity'] = veg_stats.TreeNumber.div(np.divide(
                    np.sum(total_yards_adjusted), 220.0))
                veg_stats['HazardTreeDensity'] = veg_stats.HazardTreeNumber.div(
                    np.divide(np.sum(total_yards_adjusted), 220.0))

                # Rearrange the order of features
                veg_stats = veg_stats[sorted(veg_stats.columns)]

                return veg_stats.values[0].tolist()

            # Get incident_location_furlongs
            incident_location_furlongs = \
                get_incident_location_furlongs(route, shift_yards_same_elr, shift_yards_diff_elr, update)

            # Compute vegetation stats for each incident record
            vegetation_statistics = incident_location_furlongs.apply(
                lambda x: pd.Series(compute_vegetation_variables_stats(
                    x.critical_FurlongIDs, x.StartELR, x.EndELR, x.total_yards_adjusted)), axis=1)

            vegetation_statistics.columns = sorted(list(veg_stats_calc.keys()) + ['TreeDensity', 'HazardTreeDensity'])
            veg_percent = [x for x in cover_percents if re.match('^CoverPercent*.[^Open|thr]', x)]
            vegetation_statistics['CoverPercentVegetation'] = vegetation_statistics[veg_percent].apply(np.sum, axis=1)

            hazard_rest_pctl = [''.join([x, '_%s' % hazard_pctl]) for x in hazard_rest]
            rename_features = dict(zip(hazard_rest, hazard_rest_pctl))
            rename_features.update({'AssetNumber': 'AssetCount'})
            vegetation_statistics.rename(columns=rename_features, inplace=True)

            ivdata = incident_location_furlongs.join(vegetation_statistics)

            # Electrified
            ivdata.Electrified = ivdata.Electrified.apply(int)

            save_pickle(ivdata, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            ivdata = None

    return ivdata


# ====================================================================================================================
""" Integrate both the weather and vegetation data """


# Integrate the weather and vegetation conditions for incident locations
def get_incident_data_with_weather_and_vegetation(route='ANGLIA', weather='Heat',
                                                  ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                                                  shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                                                  update=False):
    filename = dbm.make_filename("mod_data", route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                 shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    path_to_file = cdd_mod_dat(filename)

    if os.path.isfile(path_to_file) and not update:
        mdata = load_pickle(path_to_file)
    else:
        try:
            # Get Schedule 8 incident and weather data for locations
            iwdata = get_incident_location_weather(route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                   subset_weather_for_nip=False, update=update)
            # Get vegetation conditions for the locations
            ivdata = get_incident_location_vegetation(route, shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)

            iv_features = [f for f in ivdata.columns if f not in ['IncidentCount', 'DelayCost', 'DelayMinutes']]
            ivdata = ivdata[iv_features]

            # Merge the above two data sets
            mdata = pd.merge(iwdata, ivdata, how='inner', on=list(set(iwdata.columns) & set(iv_features)))

            save_pickle(mdata, path_to_file)

        except Exception as e:
            print("Getting '{}' ... failed due to {}.".format(filename, e))
            mdata = None

    return mdata


# ====================================================================================================================
""" Model trials """


# Get data for specified season(s)
def get_data_by_season(mdata, season):
    """
    :param mdata:
    :param season: [str] 'spring', 'summer', 'autumn', 'winter'; if None, returns data of all seasons
    :return:
    """
    if season is None:
        return mdata
    else:
        spring_data, summer_data, autumn_data, winter_data = [pd.DataFrame()] * 4
        for y in pd.unique(mdata.FinancialYear):
            data = mdata[mdata.FinancialYear == y]
            # Get data for spring -----------------------------------
            spring_start1 = datetime.datetime(year=y, month=4, day=1)
            spring_end1 = spring_start1 + pd.DateOffset(months=2)
            spring_start2 = datetime.datetime(year=y + 1, month=3, day=1)
            spring_end2 = spring_start2 + pd.DateOffset(months=1)
            spring = ((data.StartDate >= spring_start1) & (data.StartDate < spring_end1)) | \
                     ((data.StartDate >= spring_start2) & (data.StartDate < spring_end2))
            spring_data = pd.concat([spring_data, data.loc[spring]])
            # Get data for summer ----------------------------------
            summer_start = datetime.datetime(year=y, month=6, day=1)
            summer_end = summer_start + pd.DateOffset(months=3)
            summer = (data.StartDate >= summer_start) & (data.StartDate < summer_end)
            summer_data = pd.concat([summer_data, data.loc[summer]])
            # Get data for autumn ----------------------------------
            autumn_start = datetime.datetime(year=y, month=9, day=1)
            autumn_end = autumn_start + pd.DateOffset(months=3)
            autumn = (data.StartDate >= autumn_start) & (data.StartDate < autumn_end)
            autumn_data = pd.concat([autumn_data, data.loc[autumn]])
            # Get data for winter -----------------------------------
            winter_start = datetime.datetime(year=y, month=12, day=1)
            winter_end = winter_start + pd.DateOffset(months=3)
            winter = (data.StartDate >= winter_start) & (data.StartDate < winter_end)
            winter_data = pd.concat([winter_data, data.loc[winter]])

        seasons = ['spring', 'summer', 'autumn', 'winter']
        season = [season] if isinstance(season, str) else season
        season = [find_match(s, seasons) for s in season]
        seasonal_data = eval("pd.concat([%s], ignore_index=True)" % ', '.join(['{}_data'.format(s) for s in season]))

        return seasonal_data


# Specify the explanatory variables considered in this prototype model
def specify_explanatory_variables():
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
    train_set.temperatue_category.value_counts().plot.bar(color='#537979', rot=-45, fontsize=12)
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

    path_to_file_weather = cdd_mod_heat(0, "Variables" + save_as)
    plt.savefig(path_to_file_weather, dpi=dpi)
    if save_as == ".svg":
        svg_to_emf(path_to_file_weather, path_to_file_weather.replace(save_as, ".emf"))


# A prototype model in the context of wind-related incidents
def logistic_regression_model(trial_id=0,
                              route='ANGLIA', weather='Heat',
                              ip_start_hrs=-24, nip_ip_gap=-5, nip_start_hrs=-24,
                              shift_yards_same_elr=220, shift_yards_diff_elr=220, hazard_pctl=50,
                              season='summer',
                              describe_var=False,
                              outlier_pctl=100,
                              add_const=True, seed=0, model='logit',
                              plot_roc=False, plot_pred_likelihood=False,
                              save_as=".tiff", dpi=600,
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
    :param plot_pred_likelihood:
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
    XW              WEATHER               Severe weather not snow affecting infrastructure the responsibility of
                                          Network Rail
    XX              MISC OBS              Msc items on line (incl trees) due to effects of weather responsibility of RT

    """
    # Get the mdata for modelling
    mdata = get_incident_data_with_weather_and_vegetation(route, weather, ip_start_hrs, nip_ip_gap, nip_start_hrs,
                                                          shift_yards_same_elr, shift_yards_diff_elr,
                                                          hazard_pctl)

    # Select season data: 'spring', 'summer', 'autumn', 'winter'
    mdata = get_data_by_season(mdata, season)

    # Remove outliers
    if 95 <= outlier_pctl <= 100:
        mdata = mdata[mdata.DelayMinutes <= np.percentile(mdata.DelayMinutes, outlier_pctl)]

    # TreeDensity (number of trees per 10 yards)
    mdata['TreeDensity'] = mdata['TreeDensity'] / 22.0

    # Select features
    explanatory_variables = specify_explanatory_variables()

    # Add the intercept
    if add_const:
        mdata = smd_tools.add_constant(mdata)  # data['const'] = 1.0
        explanatory_variables = ['const'] + explanatory_variables

    #
    nip_idx = mdata.IncidentReported == 0
    mdata.loc[nip_idx, ['DelayMinutes', 'DelayCost', 'IncidentCount']] = 0

    # Select data before 2014 as training data set, with the rest being test set
    train_set = mdata[mdata.FinancialYear != 2014]
    test_set = mdata[mdata.FinancialYear == 2014]

    if describe_var:
        describe_explanatory_variables(train_set, save_as=save_as, dpi=dpi)

    try:
        np.random.seed(seed)
        if model == 'probit':
            mod = smd.Probit(train_set.IncidentReported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=False)
        else:
            mod = smd.Logit(train_set.IncidentReported, train_set[explanatory_variables])
            result = mod.fit(method='newton', maxiter=10000, full_output=True, disp=False)

        if verbose:
            print(result.summary())

        # Odds ratios
        odds_ratios = pd.DataFrame(np.exp(result.params), columns=['OddsRatio'])
        if verbose:
            print("\nOdds ratio:")
            print(odds_ratios)

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
        if verbose:
            print("\nAccuracy: %f" % mod_acc)

        # incident prediction accuracy
        incid_only = test_set[test_set.IncidentReported == 1]
        test_acc = pd.Series(incid_only.IncidentReported == incid_only.incident_prediction)
        incid_acc = np.divide(test_acc.sum(), len(test_acc))
        if verbose:
            print("Incident accuracy: %f" % incid_acc)

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
            # plt.subplots_adjust(left=0.10, bottom=0.1, right=0.96, top=0.96)
            plt.tight_layout()
            path_to_file_roc = cdd_mod_heat(trial_id, "ROC" + save_as)
            plt.savefig(cdd_mod_heat(trial_id, "ROC" + save_as), dpi=dpi)
            plt.savefig(path_to_file_roc, dpi=dpi)
            if save_as == ".svg":
                svg_to_emf(path_to_file_roc, path_to_file_roc.replace(save_as, ".emf"))  # Fig. 6.

        # Plot incident delay minutes against predicted probabilities
        if plot_pred_likelihood:
            incid_ind = test_set.IncidentReported == 1
            plt.figure()
            ax = plt.subplot2grid((1, 1), (0, 0))
            ax.scatter(test_set[incid_ind].incident_prob, test_set[incid_ind].DelayMinutes,
                       c='#D87272', edgecolors='k', marker='o', linewidths=1.5, s=80,  # alpha=.5,
                       label="Heat-related incident (2014/15)")
            plt.axvline(x=threshold, label="Threshold: %.2f" % threshold, color='#e5c100', linewidth=2)
            legend = plt.legend(scatterpoints=1, loc='best', fontsize=14, fancybox=True, labelspacing=0.6)
            frame = legend.get_frame()
            frame.set_edgecolor('k')
            plt.xlim(xmin=0, xmax=1.03)
            plt.ylim(ymin=-15)
            ax.set_xlabel("Likelihood of heat-related incident occurrence", fontsize=14,
                          fontweight='bold')
            ax.set_ylabel("Delay minutes", fontsize=14, fontweight='bold')
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.tight_layout()
            path_to_file_pred = cdd_mod_heat(trial_id, "Predicted-likelihood" + save_as)
            plt.savefig(path_to_file_pred, dpi=dpi)
            if save_as == ".svg":
                svg_to_emf(path_to_file_pred, path_to_file_pred.replace(save_as, ".emf"))

    except Exception as e:
        print(e)
        result = e
        mod_acc, incid_acc, threshold = np.nan, np.nan, np.nan

    # from utils import get_variable_names
    # repo = locals()
    # resources = {k: repo[k]
    #              for k in get_variable_names(mdata, train_set, test_set, result, mod_acc, incid_acc, threshold)}
    # filename = dbm.make_filename("data", route, weather,
    #                              ip_start_hrs, ip_end_hrs, nip_start_hrs,
    #                              shift_yards_same_elr, shift_yards_diff_elr, hazard_pctl)
    # save_pickle(resources, cdd_mod_heat(trial_id, filename))

    return mdata, train_set, test_set, result, mod_acc, incid_acc, threshold
