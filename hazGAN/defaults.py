"""Default settings for modelling."""
import numpy as np

SEED = 42

# Bay of Bengal bounds (EPSG:4326)
xmin = 80.0
xmax = 95.0
ymin = 10.0
ymax = 25.0

bay_of_bengal_crs = 24346 # https://epsg.io/24346

channel_labels = {0: r'wind speed [ms$^{-1}$]', 1: r'mean sea level pressure [Pa]'}
channel_labels = {0: r'wind speed [ms$^{-1}$]', 1: r'total precipitation [m]'}
longitude = np.linspace(xmin, xmax, 3)
latitude = np.linspace(ymin, ymax, 4)