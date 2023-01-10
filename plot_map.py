from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

nc = NetCDFFile('C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981/NLDAS_198101_all_stats_DS.nc') # note this file is 2.5 degree, so low resolution data

lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
t2 = nc.variables['TMP_mean'][:] # 2 meter temperature


map = Basemap(projection='merc',llcrnrlon=-93.,llcrnrlat=35.,urcrnrlon=-73.,urcrnrlat=45.,resolution='i') # projection, lat/lon extents and resolution of polygons to draw
# resolutions: c - crude, l - low, i - intermediate, h - high, f - full

map.drawcoastlines()
map.drawstates()
map.drawcountries()
map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
map.drawcounties() # you can even add counties (and other shapefiles!)

parallels = np.arange(30,50,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-95,-70,5.) # make longitude lines every 5 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

lons,lats= np.meshgrid(lon-180,lat) # for this dataset, longitude is 0 through 360, so you need to subtract 180 to properly display on map
x,y = map(lons,lats)

temp = map.contourf(x,y,t2[1,:,:])
cb = map.colorbar(temp,"bottom", size="5%", pad="2%")
plt.title('2m Temperature')
cb.set_label('Temperature (K)')
plt.show()