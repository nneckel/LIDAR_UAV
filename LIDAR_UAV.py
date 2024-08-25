# import open3d as o3d
import laspy
import numpy as np
import sys
# from datetime import datetime, timedelta
import fnmatch
import pandas as pd
from pyproj import Transformer
import pygmt
import rasterio


def GetFileList(path,wildcard):
    filelist = []
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, wildcard):
            filelist = np.append(filelist,file)
    return np.sort(filelist)


def gridding(x, y, z, resolution=None):
    xmin = x.min()-resolution
    ymin = y.min()-resolution
    xmax = xmin + ((x.max()-xmin)//resolution + 1) * resolution
    ymax = ymin + ((y.max()-ymin)//resolution + 1) * resolution
    raster = {'resolution':resolution, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
    print(raster)
    grid = pygmt.xyz2grd(x=x, y=y, z=z, spacing=raster['resolution'], region=[raster['xmin'], raster['xmax'], raster['ymin'], raster['ymax']])
    return grid


def save_to_tif(raster, path, crs='EPSG:3413'):
    height, width = len(raster.y), len(raster.x)
    res_x = abs(raster.x[1]-raster.x[0])
    res_y = abs(raster.y[1]-raster.y[0])
    transform=rasterio.transform.from_bounds(west=raster.x[0]-.5*res_x, south=raster.y[-1]+.5*res_y, east=raster.x[-1]+.5*res_x, north=raster.y[0]-.5*res_y, 
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
        dst.write(raster, 1)


def read_rtk(path, date):
    '''
    Reads files from RTK base station into pandas DataFrame. Date is converted to datetime and rows with invalid data are droped.
    
    :param path: path to RTKxxx.DAT file
    :param data: date of acquisition, needed to compute reference date (last Sunday)

    :return df: pandas DataFrame with columns 'time', 'lat', 'lon', 'hgt'
    '''
    df = pd.read_csv(path, sep=',', names=['bestpos', 'time', 'type', 'lat', 'lon', 'hgt', 'std', '1', '2'], index_col=False)

    df = df[(df.time!='0ms')*(df.lat!='lat:0.000000000')].copy()

    date = pd.to_datetime(date)
    last_sunday = pd.to_datetime((date - pd.to_timedelta((date.weekday()+1) % 7, 'd')).date())
    df.time = last_sunday + pd.to_timedelta(df.time)

    df.lat = (df.lat.str[4:]).astype(np.float64)
    df.lon = (df.lon.str[4:]).astype(np.float64)
    df.hgt = (df.hgt.str[4:]).astype(np.float64)

    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3413')
    df['x'], df['y'] = transformer.transform(df.lat, df.lon)
    
    return df[['time', 'lat', 'lon', 'x', 'y', 'hgt']].copy()


def read_las(path, only_with_rgb=False):
    '''
    Reads .las file and returns pandas DataFrame with columns time, lat, lon (EPSG:4326), x, y (EPSG:3413), z, r,g,b
    '''

    if path[-4:]=='.las':
        las_files = [laspy.read(path)]
    else:
        # reading a whole dictionary could be implemented
        # GetFileList(path, '*.las')
        print('Pass path to .las file')

    las_array = []
    for las_file in las_files:

        if only_with_rgb:
            mask = ((las_file.red + las_file.green + las_file.blue) != 0)
            las_file = las_file[mask]

        lat, lon = las_file.y, las_file.x
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3413')
        x, y = transformer.transform(lat, lon)

        gps_epoch = pd.to_datetime('1980-01-06')
        print(las_file.gps_time)
        time = gps_epoch + pd.to_timedelta(las_file.gps_time+1e9, 's')

        las = pd.DataFrame(data={'time':np.array(time), 'lat':np.array(lat), 'lon':np.array(lon), 'x':np.array(x), 'y':np.array(y), 
            'z':np.array(las_file.z), 'r':np.array(las_file.red), 'g':np.array(las_file.green), 'b':np.array(las_file.blue)})
        las_array += [las]
    
    las = pd.concat(las_array)
    return las


def drift_corr_from_gps(las, station, ref_date=None):
    '''
    Adds drift-corrected coordinates to the gem dataframe. 
    The correction is based on a GPS ground station coordinate list, that must span the time of gem observations.
    The GPS positions are interpolated to times of gem observations.
    '''
    las = las.copy()
    if ref_date==None:
        ref_date = las.time.mean()
    
    ref_date = pd.Series(pd.to_datetime(ref_date))

    x_station_ref = np.interp(ref_date, station.time, station.x)[0]
    y_station_ref = np.interp(ref_date, station.time, station.y)[0]

    x_station = np.interp(las.time, station.time, station.x)
    y_station = np.interp(las.time, station.time, station.y)
    
    las['x_corr'] = las.x + x_station_ref - x_station
    las['y_corr'] = las.y + y_station_ref - y_station

    transformer = Transformer.from_crs('EPSG:3413', 'EPSG:4326')
    las['lat_corr'], las['lon_corr'] = transformer.transform(las['x_corr'], las['y_corr'])

    return las



inputLASFILE = sys.argv[1]
#LASFILE = laspy.read(inputLASFILE)

las = read_las(path=inputLASFILE, only_with_rgb=True)

station = read_RTK('terra_las/RTK021.DAT', date='2024-08-20')

las = drift_corr_from_gps(las, station)
