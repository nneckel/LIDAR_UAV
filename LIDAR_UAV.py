#import open3d as o3d
import laspy
import numpy as np
import sys
from datetime import datetime, timedelta

#print("Load a ply point cloud, print it, and render it")
##pcd = o3d.io.read_point_cloud("table_scene_lms400.pcd")
##pcd = o3d.io.read_point_cloud("20200318_01.pcd")
#pcd = o3d.io.read_point_cloud("20200722_01.pcd")
#cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=1.0)
#    #display_inlier_outlier(voxel_down_pcd, ind)

##o3d.visualization.draw_geometries([pcd])
##o3d.visualization.draw_geometries([cl])
#o3d.io.write_point_cloud("tmp.pcd", cl)

inputLASFILE = sys.argv[1]
LASFILE = laspy.read(inputLASFILE)

# GPS-Zeitursprung: 6. Januar 1980, 00:00:00 UTC
gps_epoch = datetime(1980, 1, 6)


#for dimension in las.point_format.dimensions:
#	print(dimension.name)

#confidence = las.confidence
X = LASFILE.X
Y = LASFILE.Y
Z = LASFILE.Z
I = LASFILE.intensity
TIME = LASFILE.gps_time

# Umwandlung der ersten GPS-Zeit in ein datetime-Objekt
datetime_first_gps = gps_epoch + timedelta(seconds=TIME[0])

<<<<<<< HEAD
# Anzeigen der umgewandelten Zeit
print(datetime_first_gps)



#X = X/100+200000
#Y = Y/100-200000
#Z = Z/100-1000

print(X,Y,Z)

##np.savetxt('z.txt', np.c_[X,Y,Z])
#np.savetxt('confidence.txt', np.c_[X,Y,confidence])
=======
#np.savetxt('z.txt', np.c_[X,Y,Z])
np.savetxt('confidence.txt', np.c_[X,Y,confidence])

# this is a test added by jonathan
>>>>>>> b13705da8d7514336b225cecbc4c10b1e817f753
