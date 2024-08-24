#import open3d as o3d
import laspy
import numpy as np
import sys
from datetime import datetime, timedelta

def GetFileList(path,wildcard):
	filelist = []
	for file in os.listdir(path):
		if fnmatch.fnmatch(file, wildcard):
			filelist = np.append(filelist,file)
	return np.sort(filelist)

def DriftCorr(REFcoords,cameraGPSdate,psx,psy,REFpsX_init,REFpsY_init):
	idx = np.searchsorted(REFcoords[:,0],cameraGPSdate)
	REFpsX,REFpsY = REFcoords[idx,3],REFcoords[idx,4]
	dx = REFpsX-REFpsX_init
	dy = REFpsY-REFpsY_init
	psx_corr = psx-dx
	psy_corr = psy-dy
	lon_corr,lat_corr,z_corr = TransCoordsPSToLatLon(EPSG).TransformPoint(psx_corr,psy_corr)
	return lat_corr,lon_corr,psx_corr,psy_corr

inputLASFILE = sys.argv[1]
LASFILE = laspy.read(inputLASFILE)

# GPS-Zeitursprung: 6. Januar 1980, 00:00:00 UTC
gps_epoch = datetime(1980, 1, 6)

#for dimension in las.point_format.dimensions:
#	print(dimension.name)

#confidence = las.confidence
lon = LASFILE.x
lat = LASFILE.y
z = LASFILE.z
i = LASFILE.intensity
time = LASFILE.gps_time + 1e9

# Umwandlung der ersten GPS-Zeit in ein datetime-Objekt
datetime_first_gps = gps_epoch + timedelta(seconds=time[0])

# Anzeigen der umgewandelten Zeit
print(datetime_first_gps)



print(X,Y,Z)

#np.savetxt('z.txt', np.c_[X,Y,Z])
#np.savetxt('confidence.txt', np.c_[X,Y,confidence])
