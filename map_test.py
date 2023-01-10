from netCDF4 import Dataset
import numpy as np

my_example_nc_file = 'C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981/NLDAS_198101_all_stats_DS.nc'
fh = Dataset(my_example_nc_file, mode='r')

lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
tmean = fh.variables['TMP_mean'][:]

tmean_units = fh.variables['TMP_mean'].units

fh.close()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Get some parameters for the Stereographic Projection
lon_0 = lons.mean()
lat_0 = lats.mean()

m = Basemap(width=5000000, height=3500000,
            resolution='l', projection='stere',
            lat_ts=40, lat_0=lat_0, lon_0=lon_0)

# Because our lon and lat variables are 1D,
# use meshgrid to create 2D arrays
# Not necessary if coordinates are already in 2D arrays.
lon, lat = np.meshgrid(lons, lats)
xi, yi = m(lon, lat)

# Plot Data
cs = m.pcolor(xi, yi, np.squeeze(tmean))

# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label(tmean_units)

# Add Title
plt.title('NLDAS Mean Temperature')

plt.show()
