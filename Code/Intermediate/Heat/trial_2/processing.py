""" Integrate data of Incidents and Weather """

import functools
import os
import random

import datetime_truncate
import gc
import geopandas as gpd
import geopy.distance
import matplotlib.font_manager
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhelpers.settings
import pyhelpers.store
import pyproj
import shapely.geometry
import shapely.ops

import Intermediate.utils as itm_utils
import mssql_metex

pyhelpers.settings.np_preferences()
pyhelpers.settings.pd_preferences()


# Create shapely.points for StartLocations and EndLocations
def create_start_end_shapely_points(incidents_data):
    print("Creating shapely.geometry.Points for each incident location ... ", end="")
    data = incidents_data.copy()
    # Make shapely.geometry.points in longitude and latitude
    data.insert(data.columns.get_loc('StartLatitude') + 1, 'StartLonLat',
                gpd.points_from_xy(data.StartLongitude, data.StartLatitude))
    data.insert(data.columns.get_loc('EndLatitude') + 1, 'EndLonLat',
                gpd.points_from_xy(data.EndLongitude, data.EndLatitude))
    data.insert(data.columns.get_loc('EndLonLat') + 1, 'MidLonLat',
                data[['StartLonLat', 'EndLonLat']].apply(
                    lambda x: shapely.geometry.LineString([x.StartLonLat, x.EndLonLat]).centroid, axis=1))
    # # Add Easting and Northing points
    # import converters
    # start_en_pt = [converters.wgs84_to_osgb36(data.StartLongitude[i], data.StartLatitude[i]) for i in data.index]
    # data = pd.concat([data, pd.DataFrame(start_en_pt, columns=['StartEasting', 'StartNorthing'])], axis=1)
    # data['StartEN'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    # end_en_pt = [converters.wgs84_to_osgb36(data.EndLongitude[i], data.EndLatitude[i]) for i in data.index]
    # data = pd.concat([data, pd.DataFrame(end_en_pt, columns=['EndEasting', 'EndNorthing'])], axis=1)
    # data['EndEN'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)
    # # data[['StartEasting', 'StartNorthing']] = data[['StartLongitude', 'StartLatitude']].apply(
    # #     lambda x: pd.Series(converters.wgs84_to_osgb36(x.StartLongitude, x.StartLatitude)), axis=1)
    # # data['StartEN'] = gpd.points_from_xy(data.StartEasting, data.StartNorthing)
    # # data[['EndEasting', 'EndNorthing']] = data[['EndLongitude', 'EndLatitude']].apply(
    # #     lambda x: pd.Series(converters.wgs84_to_osgb36(x.EndLongitude, x.EndLatitude)), axis=1)
    # # data['EndEN'] = gpd.points_from_xy(data.EndEasting, data.EndNorthing)
    print("Done.")
    return data


# Create a circle buffer for an incident location
def create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end, whisker_km=0.008, as_geom=True):
    """
    Ref: https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python

    :param midpoint: [shapely.geometry.point.Point] e.g. midpoint = incidents.MidLonLat.iloc[0]
    :param incident_start: [shapely.geometry.point.Point] e.g. incident_start = incidents.StartLonLat.iloc[0]
    :param incident_end: [shapely.geometry.point.Point] e.g. incident_end = incidents.EndLonLat.iloc[0]
    :param whisker_km: [num] an extended length to the diameter (i.e. on both sides of the start and end locations)
    :param as_geom: [bool]
    :return: [shapely.geometry.Polygon; list of tuples]
    """
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lon_0={lon} +lat_0={lat} +x_0=0 +y_0=0'
    project = functools.partial(
        pyproj.transform, pyproj.Proj(aeqd_proj.format(lon=midpoint.x, lat=midpoint.y)), pyproj.Proj(init='epsg:4326'))
    if incident_start != incident_end:
        radius_km = geopy.distance.distance(incident_start.coords, incident_end.coords).km / 2 + whisker_km
    else:
        radius_km = 2
    buffer = shapely.ops.transform(project, shapely.geometry.Point(0, 0).buffer(radius_km * 1000))
    buffer_circle = buffer if as_geom else buffer.exterior.coords[:]
    return buffer_circle


# Find all intersecting weather cells
def find_intersecting_weather_cells(x, as_geom=False):
    """
    :param x: [shapely.geometry.point.Point] e.g. x = incidents.Buffer_Zone.iloc[0]
    :param as_geom: [bool] whether to return shapely.geometry.Polygon of intersecting weather cells
    :return:
    """
    weather_cell_geoms = mssql_metex.get_weather_cell().Polygon_WGS84
    intxn_weather_cells = tuple(cell for cell in weather_cell_geoms if x.intersects(cell))
    if as_geom:
        return intxn_weather_cells
    else:
        intxn_weather_cell_ids = tuple(weather_cell_geoms[weather_cell_geoms == cell].index[0]
                                       for cell in intxn_weather_cells)
        if len(intxn_weather_cell_ids) == 1:
            intxn_weather_cell_ids = intxn_weather_cell_ids[0]
        return intxn_weather_cell_ids


# Illustration of the buffer circle
def illustrate_buffer_circle(midpoint, incident_start, incident_end, whisker_km=0.008, legend_loc='best'):
    """
    :param midpoint: e.g. midpoint = incidents.MidLonLat.iloc[2]
    :param incident_start: e.g. incident_start = incidents.StartLonLat.iloc[2]
    :param incident_end: e.g. incident_end = incidents.EndLonLat.iloc[2]
    :param whisker_km:
    :param legend_loc:
    """
    buffer_circle = create_circle_buffer_upon_weather_cell(midpoint, incident_start, incident_end, whisker_km)
    i_weather_cells = find_intersecting_weather_cells(buffer_circle, as_geom=True)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot2grid((1, 1), (0, 0))
    for g in i_weather_cells:
        x, y = g.exterior.xy
        ax.plot(x, y, color='#433f3f')
        polygons = matplotlib.patches.Polygon(g.exterior.coords[:], fc='#D5EAFF', ec='#4b4747', alpha=0.5)
        plt.gca().add_patch(polygons)
    ax.plot([], 's', label="Weather cell", ms=16, color='#D5EAFF', markeredgecolor='#4b4747')
    x, y = buffer_circle.exterior.xy
    ax.plot(x, y)
    sx, sy, ex, ey = incident_start.xy + incident_end.xy
    if incident_start == incident_end:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Incident location')
    else:
        ax.plot(sx, sy, 'b', marker='o', markersize=10, linestyle='None', label='Start location')
        ax.plot(ex, ey, 'g', marker='o', markersize=10, linestyle='None', label='End location')
    ax.set_xlabel('Longitude')  # ax.set_xlabel('Easting')
    ax.set_ylabel('Latitude')  # ax.set_ylabel('Northing')
    font = matplotlib.font_manager.FontProperties(family='Times New Roman', weight='normal', size=14)
    legend = plt.legend(numpoints=1, loc=legend_loc, prop=font, fancybox=True, labelspacing=0.5)
    frame = legend.get_frame()
    frame.set_edgecolor('k')
    plt.tight_layout()


# Integrate incidents and weather data
def get_incidents_with_weather(trial_id=0, route_name=None, weather_category='Heat',
                               regional=True, reason=None, season='summer',
                               prep_test=False, random_select=True, use_buffer_zone=False, illustrate_buf_cir=False,
                               lp=5 * 24, non_ip=24,
                               update=False):

    pickle_filename = mssql_metex.make_filename(
        "incidents-and-weather", route_name, weather_category, "regional" if regional else "",
        "-".join([season] if isinstance(season, str) else season), "trial" if prep_test else "", sep="-")
    path_to_pickle = itm_utils.cd_intermediate("Heat", "{}".format(trial_id), pickle_filename)

    if os.path.isfile(path_to_pickle) and not update:
        incidents_and_weather = pyhelpers.store.load_pickle(path_to_pickle)
    else:
        try:
            # Incidents data
            incidents_data = mssql_metex.view_schedule8_cost_by_datetime_location_reason(route_name)
            incidents = incidents_data.copy()
            incidents = incidents[incidents.WeatherCategory.isin(['', weather_category])]

            # Investigate only the following incident reasons
            if reason is not None:
                reason_codes = mssql_metex.get_incident_reason_metadata().index.tolist()
                assert all(x for x in reason if x in reason_codes) and isinstance(reason, list)
            else:
                reason = ['IR', 'XH', 'IB', 'JH']
            incidents = incidents[incidents.IncidentReasonCode.isin(reason)]
            # Select season data
            incidents = itm_utils.get_data_by_season(incidents, season)

            if regional:
                incidents = incidents[incidents.Route.isin(['South East', 'Anglia', 'Wessex'])]

            if prep_test:  # For initial testing ...
                if random_select:
                    incidents = incidents.iloc[random.sample(range(len(incidents)), prep_test), :]
                else:
                    incidents = incidents.iloc[-prep_test - 1:-1, :]

            # --------------------------------------------------------------------------------------------------------
            """ In the spatial context """

            incidents = create_start_end_shapely_points(incidents)

            if use_buffer_zone:
                # Make a buffer zone for gathering data of weather observations
                print("Creating a buffer zone for each incident location ... ", end="")
                incidents['Buffer_Zone'] = incidents.apply(
                    lambda x: create_circle_buffer_upon_weather_cell(
                        x.MidLonLat, x.StartLonLat, x.EndLonLat, whisker_km=0.0), axis=1)
                print("Done.")

                # Find all weather observation grids intersecting with the buffer zone for each incident location
                print("Delimiting zone for calculating weather statistics  ... ", end="")
                incidents['WeatherCell_Obs'] = incidents.Buffer_Zone.map(lambda x: find_intersecting_weather_cells(x))
                print("Done.")

                if illustrate_buf_cir:
                    # Example 1
                    x_incident_start, x_incident_end = incidents.StartLonLat.iloc[12], incidents.EndLonLat.iloc[12]
                    x_midpoint = incidents.MidLonLat.iloc[12]
                    illustrate_buffer_circle(x_midpoint, x_incident_start, x_incident_end, whisker_km=0.0,
                                             legend_loc='upper left')
                    pyhelpers.store.save_fig(itm_utils.cd_intermediate("Heat", "Buffer_circle_example_1.png"), dpi=1200)
                    # Example 2
                    x_incident_start, x_incident_end = incidents.StartLonLat.iloc[16], incidents.EndLonLat.iloc[16]
                    x_midpoint = incidents.MidLonLat.iloc[16]
                    illustrate_buffer_circle(x_midpoint, x_incident_start, x_incident_end, whisker_km=0.0,
                                             legend_loc='lower right')
                    pyhelpers.store.save_fig(itm_utils.cd_intermediate("Heat", "Buffer_circle_example_2.png"), dpi=1200)

            # --------------------------------------------------------------------------------------------------------
            """ In the temporal context """

            incidents['Incident_Duration'] = incidents.EndDateTime - incidents.StartDateTime
            # Incident period (IP)
            incidents['Critical_StartDateTime'] = incidents.StartDateTime.map(
                lambda x: x.replace(hour=0, minute=0, second=0)) - pd.DateOffset(hours=24)
            incidents['Critical_EndDateTime'] = incidents.StartDateTime.map(
                lambda x: x.replace(minute=0) + pd.Timedelta(hours=1) if x.minute > 45 else x.replace(minute=0))
            incidents['Critical_Period'] = incidents.Critical_EndDateTime - incidents.Critical_StartDateTime

            # Specify the statistics needed for Weather observations (except radiation)
            def specify_weather_stats_calculations():
                """
                :rtype: dict
                """
                weather_stats_calculations = {'Temperature': (max, min, np.average),
                                              'RelativeHumidity': max,
                                              'WindSpeed': max,
                                              'WindGust': max,
                                              'Snowfall': sum,
                                              'TotalPrecipitation': sum}
                return weather_stats_calculations

            # Calculate average wind speed and direction
            def calculate_wind_averages(wind_speeds, wind_directions):
                u = - wind_speeds * np.sin(np.radians(wind_directions))  # component u, the zonal velocity
                v = - wind_speeds * np.cos(np.radians(wind_directions))  # component v, the meridional velocity
                uav, vav = np.mean(u), np.mean(v)  # sum up all u and v values and average it
                average_wind_speed = np.sqrt(uav ** 2 + vav ** 2)  # Calculate average wind speed
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

            # Get all Weather variable names
            def specify_weather_variable_names():
                # var_names = db.colnames_db_table('NR_METEX', table_name='Weather')[2:]
                weather_stats_calculations = specify_weather_stats_calculations()
                stats_names = [x + '_max' for x in weather_stats_calculations.keys()]
                stats_names[stats_names.index('TotalPrecipitation_max')] = 'TotalPrecipitation_sum'
                stats_names[stats_names.index('Snowfall_max')] = 'Snowfall_sum'
                stats_names.insert(stats_names.index('Temperature_max') + 1, 'Temperature_min')
                stats_names.insert(stats_names.index('Temperature_min') + 1, 'Temperature_avg')
                stats_names.insert(stats_names.index('Temperature_avg') + 1, 'Temperature_dif')
                wind_speed_variables = ['WindSpeed_avg', 'WindDirection_avg']
                return stats_names + wind_speed_variables + ['Hottest_Heretofore']

            # Get the highest temperature of year by far
            def get_highest_temperature_of_year_by_far(weather_cell_id, period_start_dt):
                # Whether "max_temp = weather_stats[0]" is the hottest of year so far
                yr_start_dt = datetime_truncate.truncate_year(period_start_dt)
                # Specify a directory to pickle slices of weather observation data
                weather_dat_dir = itm_utils.cdd_intermediate("weather-slices")
                # Get weather observations
                weather_obs = mssql_metex.view_weather_by_id_datetime(
                    weather_cell_id, yr_start_dt, period_start_dt, pickle_it=False, dat_dir=weather_dat_dir)
                weather_obs_by_far = weather_obs[
                    (weather_obs.DateTime < period_start_dt) & (weather_obs.DateTime > yr_start_dt)]
                highest_temp = weather_obs_by_far.Temperature.max()
                return highest_temp

            # Calculate the statistics for the weather-related variables (except radiation)
            def calculate_weather_stats(weather_obs, weather_stats_calculations, values_only=True):
                if weather_obs.empty:
                    weather_stats = [np.nan] * (sum(map(np.count_nonzero, weather_stats_calculations.values())) + 4)
                    if not values_only:
                        weather_stats = pd.DataFrame(weather_stats, columns=specify_weather_variable_names())
                else:
                    # Create a pseudo id for groupby() & aggregate()
                    weather_obs['Pseudo_ID'] = 0
                    # Calculate basic statistics
                    weather_stats = weather_obs.groupby('Pseudo_ID').aggregate(weather_stats_calculations)
                    # Calculate average wind speeds and directions
                    weather_stats['WindSpeed_avg'], weather_stats['WindDirection_avg'] = \
                        calculate_wind_averages(weather_obs.WindSpeed, weather_obs.WindDirection)
                    # Lowest temperature between the time of the highest temperature and 00:00
                    highest_temp_dt = weather_obs[
                        weather_obs.Temperature == weather_stats.Temperature['max'][0]].DateTime.min()
                    weather_stats.Temperature['min'] = weather_obs[
                        weather_obs.DateTime < highest_temp_dt].Temperature.min()
                    # Temperature change between the the highest and lowest temperatures
                    weather_stats.insert(3, 'Temperature_dif',
                                         weather_stats.Temperature['max'] - weather_stats.Temperature['min'])
                    # Find out weather cell ids
                    weather_cell_obs = weather_obs.WeatherCell.unique()
                    weather_cell_id = weather_cell_obs[0] if len(weather_cell_obs) == 1 else tuple(weather_cell_obs)
                    obs_start_dt = weather_obs.DateTime.min()  # Observation start datetime
                    # Whether it is the hottest of the year by far
                    highest_temp = get_highest_temperature_of_year_by_far(weather_cell_id, obs_start_dt)
                    highest_temp_obs = weather_stats.Temperature['max'][0]
                    weather_stats['Hottest_Heretofore'] = 1 if highest_temp_obs >= highest_temp else 0
                    weather_stats.columns = specify_weather_variable_names()
                    # Scale up variable
                    scale_up_vars = ['WindSpeed_max', 'WindGust_max', 'WindSpeed_avg', 'RelativeHumidity_max',
                                     'Snowfall_sum']
                    weather_stats[scale_up_vars] = weather_stats[scale_up_vars] / 10.0
                    weather_stats.index.name = None
                    if values_only:
                        weather_stats = weather_stats.values[0].tolist()
                return weather_stats

            # Calculate weather statistics based on the retrieved weather observation data
            def get_ip_weather_stats(weather_cell_id, start_dt, end_dt):
                """
                :param weather_cell_id: e.g. weather_cell_id = incidents.WeatherCell.iloc[0]
                :param start_dt: e.g. start_dt = incidents.Critical_StartDateTime.iloc[0]
                :param end_dt: e.g. end_dt = incidents.Critical_EndDateTime.iloc[0]
                :return:
                """
                # Specify a directory to pickle slices of weather observation data
                weather_dat_dir = itm_utils.cdd_intermediate("weather-slices")
                # Query weather observations
                ip_weather = mssql_metex.view_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                                     pickle_it=False, dat_dir=weather_dat_dir)
                # Calculate basic statistics of the weather observations
                weather_stats_calculations = specify_weather_stats_calculations()
                weather_stats = calculate_weather_stats(ip_weather, weather_stats_calculations, values_only=True)
                return weather_stats

            # Prior-IP ---------------------------------------------------
            print("Calculating weather statistics for IPs ... ", end="")
            incidents[specify_weather_variable_names()] = incidents.apply(
                lambda x: pd.Series(get_ip_weather_stats(
                    x.WeatherCell_Obs if use_buffer_zone else x.WeatherCell,
                    x.Critical_StartDateTime, x.Critical_EndDateTime)), axis=1)
            print("Done.")

            gc.collect()

            incidents.Hottest_Heretofore = incidents.Hottest_Heretofore.astype(int)
            incidents['Incident_Reported'] = 1

            # Non-IP ---------------------------------------------------
            non_ip_data = incidents.copy(deep=True)

            non_ip_data.Critical_StartDateTime = incidents.Critical_StartDateTime - pd.DateOffset(hours=non_ip + lp)
            # non_ip_data.Critical_EndDateTime = non_ip_data.Critical_StartDateTime + pd.DateOffset(hours=non_ip)
            non_ip_data.Critical_EndDateTime = non_ip_data.Critical_StartDateTime + incidents.Critical_Period

            # Gather gridded Weather observations of the corresponding non-incident period for each incident record
            def get_non_ip_weather_stats(weather_cell_id, start_dt, end_dt, stanox_section):
                """
                :param weather_cell_id: weather_cell_id = non_ip_data.WeatherCell.iloc[0]
                :param start_dt: e.g. start_dt = non_ip_data.Critical_StartDateTime.iloc[0]
                :param end_dt: e.g. end_dt = non_ip_data.Critical_EndDateTime.iloc[0]
                :param stanox_section: e.g. stanox_section = non_ip_data.StanoxSection.iloc[0]
                :return:
                """
                # Specify a directory to pickle slices of weather observation data
                weather_dat_dir = itm_utils.cdd_intermediate("weather-slices")
                # Query weather observations
                non_ip_weather = mssql_metex.view_weather_by_id_datetime(weather_cell_id, start_dt, end_dt,
                                                                         pickle_it=False, dat_dir=weather_dat_dir)
                # Get all incident period data on the same section
                ip_overlap = incidents[
                    (incidents.StanoxSection == stanox_section) &
                    (((incidents.Critical_StartDateTime <= start_dt) & (incidents.Critical_EndDateTime >= start_dt)) |
                     ((incidents.Critical_StartDateTime <= end_dt) & (incidents.Critical_EndDateTime >= end_dt)))]
                # Skip data of Weather causing Incidents at around the same time; but
                if not ip_overlap.empty:
                    non_ip_weather = non_ip_weather[
                        (non_ip_weather.DateTime < min(ip_overlap.Critical_StartDateTime)) |
                        (non_ip_weather.DateTime > max(ip_overlap.Critical_EndDateTime))]
                # Calculate weather statistics
                weather_stats_calculations = specify_weather_stats_calculations()
                weather_stats = calculate_weather_stats(non_ip_weather, weather_stats_calculations, values_only=True)
                return weather_stats

            print("Calculating weather statistics for Non-IPs ... ", end="")
            non_ip_data[specify_weather_variable_names()] = non_ip_data.apply(
                lambda x: pd.Series(get_non_ip_weather_stats(
                    x.WeatherCell_Obs if use_buffer_zone else x.WeatherCell,
                    x.Critical_StartDateTime, x.Critical_EndDateTime, x.StanoxSection)), axis=1)
            print("Done.")

            gc.collect()

            non_ip_data['Incident_Reported'] = 0
            # non_ip_data.DelayMinutes = 0.0
            non_ip_data.DelayCost = 0.0

            # Combine IP data and Non-IP data ----------------------------------------------------
            incidents_and_weather = pd.concat([incidents, non_ip_data], axis=0, ignore_index=True)

            # Get track orientation
            incidents_and_weather = itm_utils.categorise_track_orientations(incidents_and_weather)

            # Create temperature categories
            incidents_and_weather = itm_utils.categorise_temperatures(incidents_and_weather, 'Temperature_max')

            pyhelpers.store.save_pickle(incidents_and_weather, path_to_pickle)

        except Exception as e:
            print(e)
            incidents_and_weather = None

    return incidents_and_weather
