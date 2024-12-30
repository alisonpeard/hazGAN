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