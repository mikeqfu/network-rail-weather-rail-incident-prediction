import numpy as np


# Calculate average wind speed and direction
def calculate_wind_averages(wind_speeds, wind_directions):
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
