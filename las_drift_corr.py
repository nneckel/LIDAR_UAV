import laspy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer

import pycedrift as pyce

# import pygmt
# import rasterio


def las_to_time(gps_time):
    gps_epoch = pd.to_datetime('1980-01-06')
    time = gps_epoch + pd.to_timedelta(gps_time+1e9, 's')
    return time


# load GPS ground stations
fnames = ['GPS/1/240901_1.pos', 'GPS/2/240901_1.pos', 'GPS/5/240901_1.pos']
stations = pyce.read_stations(fnames, to_epsg=metric_crs)

ref_time = pd.to_datetime('2024-09-02 22:20:00')


path = 'terra_las/cloudf05b72f3790fec08_Block_3.las'
src_crs = 'EPSG:32652'
metric_crs = 'EPSG:3413'

las = laspy.read(path)

transformer = Transformer.from_crs(src_crs, metric_crs)
x, y = transformer.transform(las.x, las.y)

time = las_to_time(las.gps_time)

# drift correction
x_corr, y_corr = pyce.drift_corr(x, y, time, stations, ref_time=ref_time, translation_only=True)

transformer = Transformer.from_crs(metric_crs, src_crs)
las.x , las.y = transformer.transform(x_corr, y_corr)

las.write(path[:-4]+'_drift_corr.las')