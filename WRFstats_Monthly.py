import numpy as np
import pandas as pd
import xarray as xr
import salem as sl
import dask
import os
from glob import glob


#%% function to find the previous month containing parts of the given month
def previous_month(year_month):
    
    year = year_month[0 : year_month.find("-")]
    month = year_month[year_month.rfind("-")+1 :]
    
    months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    idx = months_list.index(month)

    month_minus = months_list[idx - 1]
    
    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year
    
    previousmonth = year_minus + "-" + month_minus
    
    return previousmonth


#%% function to combine and deaccumulate precipitation variables into new variable
def deacc_precip(ds, ds_variables):
    
    # check if the variable PRECIP was included in the variables list, if so then create variable from rain variables and deaccumulate
    if "PRECIP" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis = 0, prepend = np.array([ds["PRECIP"][0].values]))
    
    # deaccumulate rain variables
    ds["RAINC"].values = np.diff(ds["RAINC"].values, axis = 0, prepend = np.array([ds["RAINC"][0].values]))
    ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis = 0, prepend = np.array([ds["RAINSH"][0].values]))
    ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis = 0, prepend = np.array([ds["RAINNC"][0].values]))


#%% function for calculating magnitude of velocity vectors
def magnitude(ds):
    
    U = ds["U10"]
    V = ds["V10"]
    ds["WINDSPEED"] = np.sqrt( U**2 + V**2 )
    

#%% function for calculating descriptive stats on monthly netCDF data
def WRFstats(input_path, output_path, start, stop, 
             ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","WINDSPEED","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                           "ZOL","RAINC","RAINSH","RAINNC","PRECIP","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
             ):
    
    """
    Function for calculating descriptive statistics on monthly netCDF data between a given range of months.
    
    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    output_path : Str Path for the output netCDF files to be stored.
    start : Str "YYYY-MM" Start date of files to open.
    stop : Str "YYYY-MM" End date of files to open (inclusive).
    ds_variables : List Variables to run stats on.
        
    Returns
    -------
    stats : DataSet netCDF file for storage of descriptive statistics.
    
    """
    
    n = 0 # counter
    stats_list = []
    
    # create list of range of months to open
    months = pd.date_range(start, stop, freq="MS").strftime("%Y-%m").tolist()
    
    # iterate through each month and create dataset
    for month in months:
        
        # create list of files in the given month in the range of months specified
        nc_files = sorted(glob(os.path.join(input_path, f"wrfout_d01_*{month}*")))
        
        # find the previous month and take the last file of that month to extract any overlapping dates
        previousmonth = previous_month(month)
        previousmonth_lastfile = sorted(glob(os.path.join(input_path, f"wrfout_d01_*{previousmonth}*")))[-1]
        nc_files.insert(0, previousmonth_lastfile)
        
        ds = sl.open_mf_wrf_dataset(nc_files) # open all netCDF files in month and create xarray dataset using salem
        ds = ds.sel(time = slice(f"{month}")) # slice by the current month
        
        # combine and deaccumulate precipitation variables into new variable
        deacc_precip(ds, ds_variables)
        
        # create new variable for wind speed from magnitudes of velocity vectors
        magnitude(ds)
        
        # calculate descriptive stats on file using xarray
        mean_ds = ds[ds_variables].mean(dim = "time", skipna = True)
        median_ds = ds[ds_variables].median(dim = "time", skipna = True)
        stddev_ds = ds[ds_variables].std(dim = "time", skipna = True)
        max_ds = ds[ds_variables].max(dim = "time", skipna = True)
        min_ds = ds[ds_variables].min(dim = "time", skipna = True)
        
        # concatenate stats
        #stats = xr.merge([mean_ds, median_ds, stddev_ds, max_ds, min_ds])
        #stats_list.append(stats) # store in list?
        
        # specify the location for the output of the program
        output_filename = os.path.join(output_path + f"WRF_{month}_")
        
        # save each output stats as a netCDF file
        #stats.to_netcdf(path = output_filename + "Stats.nc")
        mean_ds.to_netcdf(path = output_filename + "Mean_DS.nc")
        median_ds.to_netcdf(path = output_filename + "Median_DS.nc")
        stddev_ds.to_netcdf(path = output_filename + "StdDev_DS.nc")
        max_ds.to_netcdf(path = output_filename + "Max_DS.nc")
        min_ds.to_netcdf(path = output_filename + "Min_DS.nc")
        
        n += 1 # iterate counter
        
    
    return stats_list


#%% run code

# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
input_path = "/project/projectdirs/m2702/gsharing/CONUS_TGW_WRF_Historical/"
output_path = "/project/projectdirs/m2702/gsharing/QAQC/"
start = "1999-01"
stop = "1999-12"

# run the WRFstats program
stats = WRFstats(input_path, output_path, start, stop)
