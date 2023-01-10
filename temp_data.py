import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# netCDF4 needs to be installed in your environment for this to work
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import geopandas as gpd
import earthpy as et

file = 'C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981/NLDAS_198101_all_stats_DS.nc'

mean_temp_xr = xr.open_dataset(file)

climate_crs = mean_temp_xr.rio.crs
climate_crs

# View first 5 latitude values
mean_temp_xr["TMP_mean"]["lat"].values[:5]

temp_2000_2005 = (mean_temp_xr["TMP_mean"], mean_temp_xr["lat"], mean_temp_xr["lon"])

# Read lat and long grids from netCDF
lats = nc_data.variables['lat'][:]
lons = nc_data.variables['lon'][:]

# Read the elevation data
elev = nc_data.variables['TMP_mean'][:]

# Plot
fig = plt.figure(figsize=(14, 7))

# Robinson projection
m = Basemap(projection='robin', lon_0=0, resolution='c')

# Create grid of lon-lat coordinates
xx, yy = np.meshgrid(lons, lats)

# Plot elev grid
im = m.pcolormesh(xx, yy, elev[0,:,:], latlon=True, cmap=plt.cm.jet)

air = nc_data.TMP_mean

# Plot using xarray
plt.figure(figsize=(14, 7))

# Define the desired output projection using Cartopy. Options are listed here:
# http://scitools.org.uk/cartopy/docs/latest/crs/projections.html#cartopy-projection-list
ax = plt.axes(projection=ccrs.Robinson())

# Plot the data
nc_data.TMP_mean[0].plot.pcolormesh(ax=ax,
                           transform=ccrs.PlateCarree(), # Define original data projection
                                                         # PlateCarree is just a Cartesian
                                                         # grid based on lat/lon values,
                                                         # which is what we have in the
                                                         # original file
                           x='lon', y='lat',
                           add_colorbar=True)

# Add coastlines
ax.coastlines()

plt.tight_layout()