import numpy as np
import xarray as xr
import salem as sl
import os
from glob import glob
from datetime import datetime, timedelta
from WRFstats_Functions import temp_conv, deacc_precip, windspeed, descriptive_stats


#%% function for opening a variable range of data

def variable_range_data(input_path, start, stop):
    
    """
    Function for opening a variable range of files between speicifed dates (inclusive).
    
    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    start : Str "YYYY-MM-DD" Start date of files to open.
    stop : Str "YYYY-MM-DD" End date of files to open.
    
    Returns
    -------
    rangefiles : List of netCDF files in specified date range.
    
    """
    
    # find the starting and ending year
    startyear = start[0 : start.find("-")]
    stopyear = stop[0 : stop.find("-")]
    
    # collect all files for the years in the given range
    years = [str(int(startyear) + i) for i in range(int(stopyear) - int(startyear) + 1)]    
    yearfiles_list = [sorted(glob(os.path.join(input_path, f"wrfout_*{year}*"))) for year in years]
    yearfiles = [file for list in yearfiles_list for file in list]
    
    # calculate the time difference in the given date range
    startdate = datetime.strptime(start, "%Y-%m-%d")
    stopdate = datetime.strptime(stop, "%Y-%m-%d")
    delta = stopdate - startdate
    
    # create list of all dates between the given range of dates
    filedates = [str((startdate + timedelta(days = d)).strftime("%Y-%m-%d")) for d in range(delta.days + 1)]
    
    # collect just the files between the date range that are present in the folder
    rangefiles = [glob(os.path.join(input_path, f"wrfout_d01_*{date}*"))[0] for date in filedates if date in str(yearfiles)]
    
    return rangefiles


#%% function for calculating rolling max and min

def WRFminmax(max, min, max_roll, min_roll, n):
    
    """
    Function for calculating rolling maximum and minimum values for each variable.
    
    Input
    ----------
    max : DataSet Maximum values for current file.
    min : DataSet Minimum values for current file.
    max_roll : DataSet Rolling maximum values.
    min_roll : DataSet Rolling minimum values.
    n : Int Current value of iteration counter.
    
    Returns
    -------
    max_roll, min_roll : DataSets for storage of maximum and minimum values.
    
    """
    
    ## check if this is the first file in the run, if so then assign values from current dataset
    if n == 0:
        max_roll = max
        min_roll = min
        
    else:
        # check if new max is greater than old max, if so then replace it, if none are greater then skip loop
        if (max > max_roll).any():
            max_roll = xr.where(max > max_roll, max, max_roll)
        
        # check if new min is less than old min, if so then replace it, if none are less then skip loop
        if (min < max_roll).any():
            min_roll = xr.where(min < min_roll, min, min_roll)
    
    return max_roll, min_roll


#%% function for calculating cumulative standard deviation and sample size

def WRFstddev(stddev, sample_size, stddev_roll, sample_size_roll, n):
    
    """
    Function for calculating pooled standard deviation and rolling sample size values for each variable.
    
    Input
    ----------
    stddev : DataSet Standard deviation values for current file.
    stddev_roll : DataSet Rolling standard deviation values.
    sample_size : DataSet Sample sizes of each variable for current file.
    sample_size_roll : DataSet Rolling sample sizes of each variable.
    n : Int Current value of iteration counter.
    
    Returns
    -------
    stddev_roll, sample_size_roll : DataSets for storage of rolling standard deviation and sample size values.
    
    """
    
    # check if this is the first file in the run, if so then assign values from current dataset
    if n == 0:
        sample_size_roll = sample_size
        stddev_roll = stddev
    else:
        sample_size_roll = sample_size_roll + sample_size

        # Cohen pooled method of combining weighted std devs, assuming similar variances in samples
        sd1 = stddev_roll
        sd2 = stddev
        n1 = sample_size_roll
        n2 = sample_size
        stddev_roll = np.sqrt( ( ((n1 - 1) * sd1**2) + ((n2 - 1) * sd2**2 ) ) / (n1 + n2 - 2) )

    return stddev_roll, sample_size_roll
    

#%% function for aggregating rolling stats on netCDF data

def WRFstats(input_path, output_path, start, stop, 
             ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","WINDSPEED","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                           "ZOL","RAINC","RAINSH","RAINNC","PRECIP","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
             ):
    
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files between a given range of dates.
    
    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    output_path : Str Path for the output netCDF files to be stored.
    year : Str Year of data for files to open.
    month : Str Month of data for files to open.
    ds_variables : List Variables to run stats on.
        
    Returns
    -------
    mean_roll : DataSet netCDF file for storage of rolling mean.
    avg_median_roll : DataSet netCDF file for storage of rolling average median.
    stddev_roll : DataSet netCDF file for storage of rolling standard deviation.
    max_roll : DataSet netCDF file for storage of rolling maximums.
    min_roll : DataSet netCDF file for storage of rolling minimums.
    
    """
    
    # create list of netCDF files at path
    nc_files = variable_range_data(input_path, start, stop)
    
    # create rolling stats variables, set intial values
    n = 0 # counter
    mean_roll = 0
    avg_median_roll = 0
    stddev_roll = None
    sample_size_roll = None
    max_roll = None
    min_roll = None
    
    # iterate through each nc file and create dataset
    for file in nc_files:
        
        ds = sl.open_wrf_dataset(file) # open netCDF data path and create xarray dataset using salem
        
        # check if last file in run, if so then slice data by stop date
        if file == nc_files[-1]:
            ds = ds.sel(time = slice(f"{stop}"))
        
        # convert T2 variable from K to F or C
        temp_conv(ds, ds_variables)
        
        # combine and deaccumulate precipitation variables into PRECIP variable
        deacc_precip(ds, ds_variables)
        
        # create new variable WINDSPEED from magnitudes of velocity vectors
        windspeed(ds, ds_variables)
        
        # calculate descriptive stats on file using xarray
        mean, avg_median, stddev, max, min = descriptive_stats(ds, ds_variables)
        sample_size = ds[ds_variables].count(dim = "time")
        
        # aggregate means using cumulative moving average method
        mean_roll = (mean + (n * mean_roll)) / (n + 1)
        
        # aggregate average medians using cumulative moving average method
        avg_median_roll = (avg_median + (n * avg_median_roll)) / (n + 1)
        
        # function for calculating rolling std dev and cumulative sample size
        stddev_roll, sample_size_roll = WRFstddev(stddev, sample_size, stddev_roll, sample_size_roll, n)
        
        # function for calculating rolling max and min
        max_roll, min_roll = WRFminmax(max, min, max_roll, min_roll, n)
        
        n += 1 # iterate counter
        
        
    # specify the location for the output of the program
    output_filename = os.path.join(output_path + f"{start}_{stop}_")

    # save each output stat as a netCDF file
    mean_roll.to_netcdf(path = output_filename + "Mean_DS.nc")
    avg_median_roll.to_netcdf(path = output_filename + "Avg_Median_DS.nc")
    stddev_roll.to_netcdf(path = output_filename + "StdDev_DS.nc")
    max_roll.to_netcdf(path = output_filename + "Max_DS.nc")
    min_roll.to_netcdf(path = output_filename + "Min_DS.nc")
    
    return mean_roll, avg_median_roll, stddev_roll, max_roll, min_roll


#%% run code

# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
input_path = "/project/projectdirs/m2702/gsharing/CONUS_TGW_WRF_Historical/"
output_path = "/project/projectdirs/m2702/gsharing/QAQC/"
start = "1999-01-01"
stop = "1999-12-31"

# run the WRFstats program
mean_ds, avg_median_ds, stddev_ds, max_ds, min_ds = WRFstats(input_path, output_path, start, stop)
