"""Default settings for modelling."""
import numpy as np

SEED = 42

# Bay of Bengal bounds (EPSG:4326)
xmin = 80.0
xmax = 95.0
ymin = 10.0
ymax = 25.0

bay_of_bengal_crs = 24346 # https://epsg.io/24346

channel_labels = {0: r'wind speed [ms$^{-1}$]', 1: r'total precipitation [m]', 2: r'mean sea level pressure [Pa]'}

longitude = np.linspace(xmin, xmax, 3)
latitude = np.linspace(ymin, ymax, 4)

TEST_YEAR = 2022

def PADDINGS():
    """Wrap in function to avoid early initialization."""
    return np.array([[1, 1], [1, 1], [0, 0]])


KNOWN_OUTLIERS = np.array([
    '1992-04-15T00:00:00.000000000',
    '1952-05-09T00:00:00.000000000',
    '1995-05-02T00:00:00.000000000'
    ], dtype='datetime64[ns]')


SAMPLE_CONFIG = {
    'generator_width': 64,
    'nconditions': 2,
    'embedding_depth': 64,
    'latent_dim': 64,
    'lrelu': 0.2,
    'critic_width': 64,
    'lambda_gp': 10,
    'lambda_var': 1,
    'input_policy': 'concat',
    'latent_space_distn': 'gumbel',
    'augment_policy': '',
    'gumbel': True,
    'seed': 42,
    'optimizer': 'Adam',
    'learning_rate': 1e-4,
    'beta_1': 0.9,
    'beta_2': 0.99,
    'training_balance': 5,
    'latent_dims': 128,
    'fields': ['u10', 'tp']
}


OBSERVATION_POINTS = {
    # ----bangladesh----
    'chittagong': (91.8466, 22.3569),
    'dhaka': (90.4125, 23.8103),
    # 'khulna': (89.5911, 22.8456),
    # ----myanmar----
    # 'sittwe': (92.9000, 20.1500),
    # 'rangoon': (96.1561, 16.8409),
    # ----india----
    # 'kolkata': (88.3639, 22.5726),
    # 'madras': (80.2707, 13.0827),
    # 'vishakapatham': (83.3165, 17.6868),
    # 'haldia': (87.9634, 22.0253),
    # ----noaa buoys----
    'buoy_23223': (89.483, 17.333),
    # 'buoy_23219': (88.998, 13.472),
    'buoy_23218': (88.5, 10.165),
    # 'buoy_23009': (90.0, 15.0),
    'buoy_23008': (90.0, 12.0),
    'buoy_23007': (90.0, 8.0)
}