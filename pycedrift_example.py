import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pyproj import Transformer

import pycedrift as pyce

# prepare gps station data as list of pd dataframes
paths_gps = ['GPS/2', 'GPS/3', 'GPS/5']
stations = pyce.read_stations_txt(paths_gps, to_epsg='EPSG:3413')

# read and prepare snow probe measurements
path_snow = 'snowdata/20240923/Katrin_20240923.dat'
snow = pd.read_csv(path_snow, skiprows=[2,3], header=1)
snow['time'] = pd.to_datetime(snow.TIMESTAMP, format='mixed')

# add x and y in EPSG:3413
snow['longitude'], snow['latitude'] = snow.Longitude_a + snow.Longitude_b/60, snow.latitude_a + snow.latitude_b/60 # thats problematic, watch out for the signs!
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3413')
snow['x'], snow['y'] = transformer.transform(snow.latitude, snow.longitude)

# drift correction
snow['x_corr'], snow['y_corr'] = pyce.drift_corr(snow.x, snow.y, snow.time, stations, ref_time=ref_time, translation_only=True, plot=False)

# save as csv
snow.to_csv(path_snow[:-4]+'_snow_drift-corr.csv')