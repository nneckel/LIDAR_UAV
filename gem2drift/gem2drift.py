import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pyproj import Transformer
import yaml
import os
import sys

#import pycedrift as pyce

def gap_test(time_series, time_test):
    '''
    Calculates the time between neighbouring entries in time_series that contain the times in time_test
    :param time_series: (n), time series to test
    :param time_test:  (m), times at which to test

    :return timedelta_around_test: (m) length of interval( pandas timedelta) of time_series around times in time_test
    '''
    time_series = pd.to_datetime(time_series).to_numpy()
    time_test = pd.to_datetime(time_test).to_numpy()
    
    indices = np.searchsorted(time_series, time_test)
    time_before_test = time_series[indices-1]
    time_after_test = time_series[indices]
    timedelta_around_test = pd.to_timedelta(time_after_test - time_before_test)

    return timedelta_around_test

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


def make_equidistant(gem, dist=2.0, xcol='x_corr', ycol='y_corr'):
    '''
    Uses corrected coordinates to group points by distance along track and average these groups.
    This results in aproximately equidistant points.
    Distance along the new track is added as column.
    '''
    gem = gem.copy()
    x = gem[xcol].to_numpy()
    y = gem[ycol].to_numpy()
    gem['groups'] = np.concatenate([[0.0],(((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)**.5).cumsum()]) // dist
    gem = gem.groupby(gem['groups']).mean()
    x = gem[xcol].to_numpy()
    y = gem[ycol].to_numpy()
    gem['distance'] = np.concatenate([[0.0],(((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)**.5).cumsum()])

    return gem

def drift_corr(x, y, time, stations, ref_time=None, translation_only=False, plot=False):
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

def drift_corr_latlon(lat, lon, time, stations, ref_time=None, translation_only=False, plot=True):
    '''
    Like drift_corr, but takes and returns coordinates in lat lon (EGSG:4326).
    :param lat, lon:
    :param time:
    :param ref_time:
    :param plot:

    :return lat_corr, lon_corr: 
    '''
    from pyproj import Transformer

    metric_crs = epsg(np.median(lat), np.median(lon))

    transformer = Transformer.from_crs('EPSG:4326', metric_crs)
    x, y = transformer.transform(lat, lon)

    x_corr, y_corr = drift_corr(x, y, time, stations, ref_time=ref_time, translation_only=translation_only, plot=plot)

    transformer = Transformer.from_crs(metric_crs, 'EPSG:4326')
    lat_corr, lon_corr = transformer.transform(x_corr, y_corr)

    return lat_corr, lon_corr




def calc_drift_2point(x, y, time, start_index=None, end_index=None, ref_time=None):
    if start_index==None:
        start_index = 0
    if end_index==None:
        end_index = -1
    if ref_time==None:
        ref_time = time.mean()

    x, y, time = pd.Series(x), pd.Series(y), pd.Series(time)
    
    dt = (time.iloc[end_index] - time.iloc[start_index]).total_seconds()
    vx = (x.iloc[end_index] - x.iloc[start_index]) / dt
    vy = (y.iloc[end_index] - y.iloc[start_index]) / dt
    print('Drift velocity: vx =', vx, 'vy =', vy)

    drift = {'vx':vx, 'vy':vy, 'ref_time':ref_time}

    return drift

def apply_drift_2point(x, y, time, vx, vy, ref_time):
    x, y, time = pd.Series(x), pd.Series(y), pd.Series(time)

    x_corr = x - vx * (time-ref_time).dt.total_seconds()
    y_corr = y - vy * (time-ref_time).dt.total_seconds()

    return x_corr, y_corr


def find_closest_points(x_ice, y_ice, x_snow, y_snow):
    '''
    Returns the indices of the points x_snow, y_snow that are closest to the points x_ice, y_ice.
    :param x_ice, y_ice: (n), coordinates of ice thickness measurements in metric coordinate system
    :param x_snow, y_snow: (m) coordinates of snow depth measurements in metric coordinate system
    :return closest_indices: (n) indices of x_snow, y_snow found to be closest
    :return closest_distance: (n) distance to clostest point
    '''
    ice_points = np.stack((x_ice, y_ice), axis=-1)  # Shape (N_ice, 2)
    snow_points = np.stack((x_snow, y_snow), axis=-1)  # Shape (N_snow, 2)
    
    # Calculate the squared differences between each ice point and each snow point
    diff = ice_points[:, np.newaxis, :] - snow_points[np.newaxis, :, :]  # Shape (N_ice, N_snow, 2)
    
    # Compute the Euclidean distances squared
    distances_squared = np.sum(diff**2, axis=-1)  # Shape (N_ice, N_snow)
    
    # Find the indices of the closest snow points
    closest_indices = np.argmin(distances_squared, axis=1)
    closest_distance = np.min(distances_squared, axis=1)**.5
    
    return closest_indices, closest_distance

def gps_to_df(path, gps_read_info, to_epsg=None):
    if path[-4:] == '.zip':
        # df = gps_to_df_zip(path, **gps_read_info['csv_read_attributes'])
        import zipfile
        with zipfile.ZipFile(path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.txt'):
                    zip_ref.read(file_name)
                    df = pd.read_csv(zip_ref.open(file_name,'r'), **gps_read_info['csv_read_attributes'])
    else:
        df = pd.read_csv(path, **gps_read_info['csv_read_attributes'])

    if gps_read_info['datetime_column'] != None:
        time = pd.to_datetime(df[gps_read_info['datetime_column']])
    else:
        time = pd.to_datetime(df[gps_read_info['date_column']] + ' ' + df[gps_read_info['time_column']])
    gps = pd.DataFrame(data={
        'time': time, 
        'lon':df[gps_read_info['lon_column']], 
        'lat':df[gps_read_info['lat_column']]})
    
    if to_epsg is not None:
        transformer = Transformer.from_crs('EPSG:4326', to_epsg)
        gps['x'], gps['y'] = transformer.transform(gps.lat, gps.lon)
            
    return gps


def fix_midnight_bug(time, cut_time='12:00:00'):
    '''
    Should be used if the time recorded over midnight did not switch the date.
    '''
    faulty = pd.to_datetime(time) < pd.to_datetime(time).iloc[0]
    time_corr = time.where(~faulty, other=time + pd.to_timedelta(24,'h'))
    return time_corr


def main(yaml_path):

    '''
    add test and plots
    '''
    dir = os.path.dirname(yaml_path)
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.safe_load(f)
    yaml_file

    path_gem = os.path.join(dir, yaml_file['thickness_file'])
    gem = pd.read_csv(path_gem, skipinitialspace=True)
    gem.time = pd.to_datetime(gem.time, format='mixed')

    if yaml_file['fix_midnight_bug']:
        print('Fixing the midnight bug. This should only be done if data was recorded during midnight.')
        gem.time = fix_midnight_bug(gem.time)

    # remove points without coordinates
    percantage_valid_coordinates = len(gem[gem.latitude!=0]) / len(gem) *100
    print('percantage_valid_coordinates =', percantage_valid_coordinates)
    gem = gem[np.abs(gem.latitude)!=0].copy()

    # add metric coordinates to gem
    metric_crs = yaml_file['metric_crs']
    transformer = Transformer.from_crs('EPSG:4326', metric_crs)
    gem['x'], gem['y'] = transformer.transform(gem.latitude, gem.longitude)

    # correct drift
    if yaml_file['ref_time'] == 'mean':
        ref_time = gem.time.mean()
    else:
        ref_time = pd.to_datetime(yaml_file['ref_time'])

    if yaml_file['method'] == 'gps':
        # load gps stations
        print('load gps stations')
        stations = [gps_to_df(os.path.join(dir, gps_file), yaml_file['gps_read_info'], to_epsg=metric_crs) for gps_file in yaml_file['gps_files']]
        for gps in stations:
            print(gps)
        
        print('drift correct gem data with gps data')
        gem['x_corr'], gem['y_corr'] = drift_corr(gem.x, gem.y, gem.time, stations, ref_time=ref_time, translation_only=yaml_file['translation_only'], plot=False)
        
    elif yaml_file['method'] == 'end_to_end':
        print('drift correct gem data by matching first and last point')
        drift = calc_drift_2point(gem.x, gem.y, gem.time, ref_time=ref_time)
        gem['x_corr'], gem['y_corr'] = apply_drift_2point(gem.x, gem.y, gem.time, **drift)
    else:
        raise Exception('method must be gps or end_to_end')

    # add correncted lat lon
    transformer = Transformer.from_crs(metric_crs, 'EPSG:4326')
    gem['lat_corr'], gem['lon_corr'] = transformer.transform(gem.x_corr, gem.y_corr)

    # snow
    if yaml_file['add_snow_depth']:
        print('load snow data')
        paths_snow = [os.path.join(dir, snow_file) for snow_file in yaml_file['snow_files']]
        snow = pd.concat([pd.read_csv(path_snow, skiprows=[2,3], header=1) for path_snow in paths_snow])
        snow['time'] = pd.to_datetime(snow.TIMESTAMP, format='mixed')
        snow['snow_depth'] = snow.DepthCm/100
        snow['lon'], snow['lat'] = snow.Longitude_a + snow.Longitude_b/60, snow.latitude_a + snow.latitude_b/60

        # add metric coordinates to snow
        transformer = Transformer.from_crs('EPSG:4326', metric_crs)
        snow['x'], snow['y'] = transformer.transform(snow.lat, snow.lon)
        
        # drift correction
        if yaml_file['method'] == 'gps':
            print('drift correct snow data with gps data')
            snow['x_corr'], snow['y_corr'] = drift_corr(snow.x, snow.y, snow.time, stations, ref_time=ref_time, translation_only=False, plot=False)
            
        elif yaml_file['method'] == 'end_to_end':
            print('drift correct snow data with drift computed from first and last gem measurement')
            snow['x_corr'], snow['y_corr'] = apply_drift_2point(snow.x, snow.y, snow.time, **drift)
        
        # add correncted lat lon
        transformer = Transformer.from_crs(metric_crs, 'EPSG:4326')
        snow['lat_corr'], snow['lon_corr'] = transformer.transform(snow.x_corr, snow.y_corr)

        # save snow
        snow.to_csv(paths_snow[0][:-4]+'_drift-corr.csv')
        print(paths_snow[0][:-4]+'_drift-corr.csv' + ' (drift corrected snow) saved')

        # add closest snow measurement to gem
        print('find snow measurements closest to gem measurements')
        closest_indices, closest_distance = find_closest_points(gem.x_corr, gem.y_corr, snow.x_corr, snow.y_corr)
        gem['snow_depth'] = snow.snow_depth.iloc[closest_indices].to_numpy()
        gem['distance_to_snow_meas'] = closest_distance

    gem.to_csv(path_gem[:-4]+'_drift-corr.csv')
    print(path_gem[:-4]+'_drift-corr.csv' + ' (drift corrected gem) saved')

    # make equidistant
    if yaml_file['equidist']:
        gem_equi = make_equidistant(gem, dist=yaml_file['equidist'])
        gem_equi.to_csv(path_gem[:-4]+'_drift-corr_equidist.csv')
        print(path_gem[:-4]+'_drift-corr_equidist.csv' + ' (drift corrected equdistant gem) saved')

    # plotting
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    plt.figure(figsize=(9,9))
    plt.plot(gem.x, gem.y, '.', label='gem raw', markersize=2)
    plt.plot(gem.x_corr, gem.y_corr, '.', label='gem drift-corr', markersize=2)
    if yaml_file['add_snow_depth']:
        plt.plot(snow.x, snow.y, '.', label='snow raw', markersize=1)
        plt.plot(snow.x_corr, snow.y_corr, '.', label='snow drift-corr', markersize=1)
    plt.gca().set_aspect('equal')
    scalebar = AnchoredSizeBar(plt.gca().transData, 100, '100m', 'lower left', pad=1, frameon=False, size_vertical=0)
    plt.gca().add_artist(scalebar)
    plt.yticks(None, None, rotation=90, va='center')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(2))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(2))
    plt.xlabel(metric_crs)
    plt.legend()
    plt.savefig(path_gem[:-4]+'_drift_corr_map.png', dpi=200)

    plt.figure(figsize=(9,9))
    if yaml_file['method'] == 'gps':
        for gps in stations:
            plt.plot(gps.time, gps.y, '.', label='GPS')
    plt.plot(gem.time, gem.y, '.', label='gem raw')
    plt.axvline(pd.to_datetime(ref_time), label='ref_time')
    plt.legend()
    plt.ylabel('y')
    plt.savefig(path_gem[:-4]+'_timeseries.png', dpi=200)
    


if __name__ == '__main__':
    yaml_path = sys.argv[1]
    main(yaml_path)
