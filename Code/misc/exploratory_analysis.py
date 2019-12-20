""" Exploratory analysis """

import pandas as pd
from pyhelpers.geom import find_closest_points_between, get_midpoint, wgs84_to_osgb36
from pyhelpers.settings import pd_preferences
from pyhelpers.store import save_excel

from models.prototype.plot_hotspots import get_shp_coordinates
from mssqlserver import metex
from utils import cdd

pd_preferences()

# Get data of Schedule 8 incidents
incident_locations = metex.view_schedule8_costs_by_location(route_name=None, weather_category=None, update=False)
# Find a pseudo-midpoint location for each incident location
pseudo_midpoints = get_midpoint(incident_locations.StartLongitude.values, incident_locations.StartLatitude.values,
                                incident_locations.EndLongitude.values, incident_locations.EndLatitude.values,
                                as_geom=False)

# Get a collection of coordinates of the railways in GB
railway_coordinates = get_shp_coordinates('Great Britain', osm_layer='railways', osm_feature='rail')

# Find the "midpoint" of each incident location
midpoints = find_closest_points_between(pseudo_midpoints, railway_coordinates)
incident_locations[['MidLongitude', 'MidLatitude']] = pd.DataFrame(midpoints)
incident_locations['MidEasting'], incident_locations['MidNorthing'] = wgs84_to_osgb36(
    incident_locations.MidLongitude.values, incident_locations.MidLatitude.values)

# Split the data by "region"
for region in incident_locations.Region.unique():
    region_data = incident_locations[incident_locations.Region == region]
    # Sort data by (frequency of incident occurrences, delay minutes, delay cost)
    region_data.sort_values(['WeatherCategory', 'IncidentCount', 'DelayMinutes', 'DelayCost'], ascending=False, inplace=True)
    region_data.index = range(len(region_data))
    save_excel(region_data, cdd("Incidents\\Exploratory analysis", region.replace(" ", "-").lower() + ".csv"),
               verbose=True)
