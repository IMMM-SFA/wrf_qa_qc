import datetime
from suntime import Sun
import netCDF4
import os
import pandas as pd

#UTC time not local time
# set directory
data_input_dir = 'C:/Users/mcgr323/projects/wrf'
#data_input_dir = '/global/cscratch1/sd/mcgrathc'

nc_file = os.path.join(data_input_dir, 'wrfout_d01_1986-01-01_01-00-00.nc')
nc = netCDF4.Dataset(nc_file, mode='r')
nc.variables.keys()

nc.T2.to_dataframe().to_csv('glw.csv')
lat = nc.variables['XLAT'][:]
lon = nc.variables['XLONG'][:]
time_var = nc.variables['Times']
dtime = netCDF4.num2date(time_var[:], time_var.units)
glw = nc.variables['GLW'][:]

# a pandas.Series designed for time series of a 2D lat,lon grid
glw_ts = pd.Series(glw, index=dtime)

# latitude and longitude fetch
latitude = lat
longitude = lon
sun = Sun(lat, lon)

#apply pandas column

# date in your machine's local time zone
time_zone = datetime.date(2020, 9, 13)
sun_rise = sun.get_local_sunrise_time(time_zone)
sun_dusk = sun.get_local_sunset_time(time_zone)