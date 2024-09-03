import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def read_stations(fnames, to_epsg=None):
    '''
    Reads .pos files from GPS stations and returns array of pandas DataFrames with columns time, lat, lon. 
    If to_epsg is given columns x and y are added in respectove CRS.
    :param fnames: list of paths.
    :param to_epsg: EPSG code, e.g., 'EPSG:3413'

    :return stations: list of pandas DataFrames
    '''
    stations = [pd.read_csv(
        fname, sep='  ', header=None, comment='%', 
        usecols=[0,1,2,4], names=['time', 'lat', 'lon', 'height'], parse_dates=['time'])
        for fname in fnames]
    
    if to_epsg is not None:
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', to_epsg)
        for gps in stations:
            gps.loc[:,'x'], gps.loc[:,'y'] = transformer.transform(gps.lat, gps.lon)
    
    return stations


def calc_transform(stations, time, ref_time=None, translation_only=False):
    '''
    Computes the translation and rotation that transforms the coordinates of the gps stations to their position at ref_time.
    :param stations: list of pandas DataFrames (one per station) that each contain the columns 'time', 'x', 'y'. 
        x and y must be in metric coordinates.
    :param time: (n), pandas Series of datetimes for which to calculate the transform. Should be within the period covered by the stations
    :param ref_time: datetime. Should be within the period covered by the stations. If None, mean of time is used.

    :return T: (n,2) numpy array of translation vectors
    :return R: (n,2,2) numpy array of rotation matrices, not returned if translation_only=True
    :return center_rotation: point around rotation is calculated, not returned if translation_only=True
    '''

    if ref_time is None:
        ref_time = pd.Series(time.mean())
        print('ref_time is set to', ref_time[0])
    else:
        ref_time = pd.Series(pd.to_datetime(ref_time))

    time = pd.to_datetime(time)

    gps_ref = np.array([[np.interp(ref_time, gps.time, gps.x)[0] for gps in stations], [np.interp(ref_time, gps.time, gps.y)[0] for gps in stations]]).T
    gps_ref_mean = gps_ref.mean(axis=0) # centeroid, mean over all stations

    # shape: (stations, points, x/y)
    gps = np.array([[np.interp(time, gps.time, gps.x) for gps in stations], [np.interp(time, gps.time, gps.y) for gps in stations]]).T.swapaxes(0,1)
    gps_mean = gps.mean(axis=0) # centeroid, mean over all stations

    # Translation
    T = gps_ref_mean - gps_mean

    if translation_only:
        return T, None, None

    # Rotation
    # see https://medium.com/@hirok4/estimate-rotation-matrix-from-corresponding-point-cloud-9f7e7b155370
    if len(gps)<2:
        print('Warning: Rotation can not be computed from less than two stations.')

    gps_rel = gps - gps_mean
    gps_ref_rel = gps_ref - gps_ref_mean

    cov = np.einsum('ij,ibk->bkj', gps_ref_rel, gps_rel)

    U, S, Vh = np.linalg.svd(cov, compute_uv=True)

    R = np.matmul(U,Vh)

    # avoid reflection
    det_is_neg = (np.linalg.det(R) < 0)
    Vh[det_is_neg] = Vh[det_is_neg] * np.array([[1,1],[-1,-1]])
    R = np.matmul(U,Vh)

    center_rotation = gps_ref_mean

    return T, R, center_rotation


def transform(points, T, R=None, center=None):
    '''
    Applies translation and rotation to points.
    :param points: (n, 2), x and y coordinate
    :param T: (n, 2), translation vectors
    :param R: (n, 2, 2), rotation matrices
    :param center: (2), center of rotation

    :return points_trans: (n,2), transformed points
    '''
    if R is None:
        points_trans = points + T
    else:
        points_trans = np.einsum('bij,bi->bj', R, points + T - center) + center # maybe correction is in wrong direction, swap i and j
    
    return points_trans



def epsg(lat, lon):
    if lat > 60:
        return 'EPSG:3413'
    if lat < -60: 
        return 'EPSG:3031'
    else:
        print('No CRS determined!')
        return None


def drift_corr(x, y, time, stations, ref_time=None, translation_only=False, plot=True):
    '''
    Takes point coordinates in metric coordinate system and times capured on a drifting platform 
    and shifts the points to the position on the platform at ref_time. 
    The motion of the platform is derived from gps and time information in stations
    :param x, y:, (n), x and y coordinate
    :param time: (n), list of datetime at which the coordinates where captured
    :param ref_time: datetime to which the points are shifted
    :param plot: if True the calculated corrections are plotted

    :return x_corr, y_corr: drift corrected x and y coordinates. 
    '''

    time = pd.to_datetime(time)
    T, R, center = trans_param = calc_transform(stations, time, ref_time=ref_time, translation_only=translation_only)
    
    points = np.array([x,y]).T
    points_corr = transform(points, *trans_param)

    if plot:
        plt.subplot(2,1,1)
        plt.plot(time, T[:,0], label='x')
        plt.plot(time, T[:,1], label='y')
        plt.ylabel('Translation')
        plt.legend()
        
        plt.subplot(2,1,2)
        if R is not None:
            plt.plot(time, np.arccos(R[:,0,0]) / np.pi * 180)
        plt.xlabel('Time')
        plt.ylabel('Rotation (°)')

        plt.tight_layout()
        plt.show()

    x_corr, y_corr = points_corr[:,0], points_corr[:,1]

    return x_corr, y_corr

if __name__ == '__main__':
    import sys

    test = sys.argv[1]


    ...


    # add check for gps gaps
    # add just translation to apply transfrom