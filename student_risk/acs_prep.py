#%%
import saspy
import subprocess
import os
from osgeo import ogr

#%%
driver = ogr.GetDriverByName("OpenFileGDB")

#%%
for year in range(2012, 2013):

	gdb = driver.Open(f"C:\\Users\\nathan.lindstedt\\Desktop\\shapefiles\\zcta\\ACS_{year}_5YR_ZCTA.gdb.zip")

	try:
		features = []

		for featsClass_idx in range(gdb.GetLayerCount()):
			featsClass = gdb.GetLayerByIndex(featsClass_idx)
			features.append(featsClass.GetName())

		my_indices = [0, 1, 2, 14, 16, 18, 23, 30]

		filtered_features = [features[i] for i in my_indices]

		for feature in filtered_features:
			cmd_lst = ["C:\\Program Files\\QGIS 3.16\\bin\\ogr2ogr.exe", "-skipfailures", "-f", "CSV", f"C:\\Users\\nathan.lindstedt\\Desktop\\acs_raw\\acs_{year}_5yr_zcta_{feature}.csv", f"C:\\Users\\nathan.lindstedt\\Desktop\\shapefiles\\zcta\\ACS_{year}_5YR_ZCTA.gdb.zip", f"{feature}"]
			subprocess.check_call(cmd_lst)
	finally:
		del gdb

# %%
