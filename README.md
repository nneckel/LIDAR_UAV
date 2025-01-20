# LIDAR_UAV
pycedrift.py: library for general drift correction of gps data with multiple gps base stations.

las_drift_corr.py: postprocessing of .las point cloud files files. Includes drift correction, rasterization and combination of multiple las files.

gem2drift: Drift correction for the use with GEM (ice thickness) and magnaprobe (snow depth) data. Intendet to be used in comination with the gem2-seaice-toolbox (https://gitlab.awi.de/sitem/gem2-seaice-toolbox). See tutorial: https://gitlab.awi.de/sitem/gem2-processing-tutorial 
gem2drift.py is ment to be applied on the ice thickness .csv files produced by gem2hstrns.py (gem2-seaice-toolbox). It is capable of shifting gps coordinates to their position at a given reference time. The ice motion can be calculated either from gps stations fixed to the drifting ice or by assuming start and end point of the survey are equal. Further, a combination with snow depth data is possible by finding the closest snowdepth point for every ice thickness measurement.
