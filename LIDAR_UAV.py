import open3d as o3d
import laspy
import numpy as np

#print("Load a ply point cloud, print it, and render it")
##pcd = o3d.io.read_point_cloud("table_scene_lms400.pcd")
##pcd = o3d.io.read_point_cloud("20200318_01.pcd")
#pcd = o3d.io.read_point_cloud("20200722_01.pcd")
#cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=1.0)
#    #display_inlier_outlier(voxel_down_pcd, ind)

##o3d.visualization.draw_geometries([pcd])
##o3d.visualization.draw_geometries([cl])
#o3d.io.write_point_cloud("tmp.pcd", cl)

las = laspy.read("20200318_01.las")

#for dimension in las.point_format.dimensions:
#	print(dimension.name)

confidence = las.confidence
X = las.X
Y = las.Y
Z = las.Z

X = X/100+200000
Y = Y/100-200000
Z = Z/100-1000

#np.savetxt('z.txt', np.c_[X,Y,Z])
np.savetxt('confidence.txt', np.c_[X,Y,confidence])
