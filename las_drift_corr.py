import laspy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer, CRS
import pygmt
import rasterio

import pycedrift as pyce



def las_to_time(gps_time):
    '''
    converts gps_time from .las file to pandas datetime
    :param gps_time: time from laspy object e.g. las.gps_time
    :return time: pandas datetime Series
    '''
    gps_epoch = pd.to_datetime('1980-01-06')
    time = gps_epoch + pd.to_timedelta(gps_time + 1e9, 's') # las gps time is gievn in seconds since 1980-01-06 minus 1e9
    return time

def las_drift_correction(las, stations, ref_time=None, metric_crs='EPSG:3413', convert_back_to_crs=False):
    # corrects drift of a las object in place.
    # las files should be in 'metric_crs', otherwise problems while writing to the las object might occur.
    # stations must be a list of pandas DataFrames with columns x, y and time that contains coordinates of base stations

    # transform to a metric CRS if needed
    src_crs = las.header.parse_crs()
    if src_crs != metric_crs:
        transformer = Transformer.from_crs(src_crs, metric_crs)
        x, y = transformer.transform(las.x, las.y)
    else:
        x, y = las.x, las.y

    time = las_to_time(las.gps_time)

    if ref_time is None:
        ref_time = time[0]

    # drift correction
    x_corr, y_corr = pyce.drift_corr(x, y, time, stations, ref_time=ref_time)

    if src_crs != metric_crs:
        if convert_back_to_crs:
            x_corr, y_corr = transformer.transform(x_corr, y_corr)
        else:
            las.header.add_crs(CRS.from_string(metric_crs))
    
    las.x , las.y = x_corr, y_corr
    las.header.update(las._points)


def plot_las(las, skip=50, vmin=None, vmax=None):
    plt.scatter(las[::skip].x, las[::skip].y, c=las[::skip].z, s=.1, vmin=vmin, vmax=vmax)


def rasterdef_from_las(las, resolution):
    # defines a raster in terms of bounding box and resolution
    # las can be a las object or a list of las objects
    if type(las) == list:
        xmin = np.min([la.x.min() for la in las])
        xmax = np.max([la.x.max() for la in las])
        ymin = np.min([la.y.min() for la in las])
        ymax = np.max([la.y.max() for la in las])
    else:
        xmin, xmax, ymin, ymax = las.x.min(), las.x.max(), las.y.min(), las.y.max()

    xmin = xmin
    ymin = ymin
    xmax = xmin + ((xmax-xmin)//resolution + 1) * resolution
    ymax = ymin + ((ymax-ymin)//resolution + 1) * resolution

    return {'resolution':resolution, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}

def las_to_grid(las, rasterdef, lower_origin=False):
    # rasterizes a las object to a grid given by a rasterdef dictionary
    xyz_gridded = pygmt.xyz2grd(x=las.x, y=las.y, z=las.z, spacing=rasterdef['resolution'], region=[rasterdef['xmin'], rasterdef['xmax'], rasterdef['ymin'], rasterdef['ymax']])
    if not lower_origin:
        xyz_gridded = np.flipud(xyz_gridded)
    return xyz_gridded

def save_to_tif(grid, path, rasterdef, crs='EPSG:3413'):
    height, width = grid.shape
    res = rasterdef['resolution']
    transform = rasterio.transform.from_bounds(west=rasterdef['xmin']-.5*res, 
                                               south=rasterdef['ymin']-.5*res, 
                                               east=rasterdef['xmax']+.5*res, 
                                               north=rasterdef['ymax']+.5*res, 
                                               width=width, height=height)
    
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(grid, 1)

# def las_to_csv(las, path):
#     pd.DataFrame(data={'x':np.array(las.x), 'y':np.array(las.y), 'z':np.array(las.z)}).to_csv(path)




las_files = ['cloud6cfdf1aac201eb67.las', 'cloud672d836ce2c2d474.las']
metric_crs = 'EPSG:3413'
ref_time = pd.to_datetime('2024-09-02 22:20:00')
resolution = 50

# load GPS ground stations
fnames = ['GPS/GPS_1_8320_20240908.zip', 'GPS/GPS_2_8806_20240908.zip', 'GPS/GPS_3_8802_20240908.zip', 'GPS/GPS_4_8312_20240908.zip', 'GPS/GPS_5_8315_20240908.zip', 'GPS/GPS_6_8319_20240908.zip']
stations = pyce.read_stations_zip(fnames, to_epsg=metric_crs)


# do drift correction of las files and save
las = []
for las_file in las_files:
    la = laspy.read(las_file)
    las_drift_correction(la, stations, ref_time=ref_time, metric_crs=metric_crs)
    la.write(las_file[:-4]+'_drift_corr.las')
    las += [la]

# calculate raster bounds
rasterdef = rasterdef_from_las(las, resolution)

# rasterize las
grids = [las_to_grid(la, rasterdef) for la in las]

# average rasters
mean_grid = np.nanmean(np.array(grids), axis=0)

# save
save_to_tif(mean_grid, 'mean_las.tif', rasterdef)