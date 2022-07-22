import numpy as np
import xarray as xr
import timeit
import os
from glob import glob
import salem as sl

# start timer
start_time = timeit.default_timer()


#%% function for opening one month of data
 
def one_month_data(path, year, month):
    
    """
    Function for opening one month of data including overlapping files.
    
    Input
    ----------
    path : Str Path to netCDF files for analysis.
    year : Str Year of data for files to open.
    month : Str Month of data for files to open.
    
    Returns
    -------
    onemonthdata : List of netCDF files in specified month and year.
    
    """
    
    # collect all files of a given month and year
    monthdata = sorted(glob(os.path.join(path, f"wrfout_*{year}-{month}*.nc")))
    
    # find the following and preceeding months and year
    month_minus, year_minus = previous_file(year, month)
    
    # take the file that preceeds the specified month and add to collected files, return sorted list
    last_month = sorted(glob(os.path.join(path, f"wrfout_*{year_minus}-{month_minus}*.nc")))[-1]
    monthdata.append(last_month)
    onemonthdata = sorted(monthdata)
    
    return onemonthdata


# function to find the previous file containing parts of the given month
def previous_file(year, month):
    
    months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    idx = months_list.index(month)
    
    month_minus = months_list[idx - 1] # get previous mnonth
    
    # if month is January, return previous year, else current year
    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year
    
    return month_minus, year_minus


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
    

#%% function for aggregating rolling stats on netCDF data

def WRFstats(path, year, month, 
             ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","WINDSPEED","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                           "ZOL","RAINC","RAINSH","RAINNC","PRECIP","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
             ):
    
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.
    
    Input
    ----------
    path: Str Path to netCDF files for analysis.
    year: Str Year of data for files to open.
    month: Str Month of data for files to open.
    ds_variables: List Variables to run stats on.
        
    Returns
    -------
    mean_roll: DataSet netCDF file for storage of rolling mean.
    stddev_roll: DataSet netCDF file for storage of rolling standard deviation.
    max_roll: DataSet netCDF file for storage of rolling maximums.
    min_roll: DataSet netCDF file for storage of rolling minimums.
    
    """
    
    # create list of netCDF files at path
    #nc_files = [file for file in os.listdir(path) if file.endswith(".nc")]
    nc_files = one_month_data(path, year, month)
    
    # create rolling stats variables, set intial values
    n = 0 # counter
    sample_size_roll = None
    mean_roll = 0
    stddev_roll = None
    median_roll = 0
    max_roll = None
    min_roll = None
    
    # iterate through each nc file and create dataset
    for file in nc_files:
        
        ds_sl = sl.open_wrf_dataset(file) # open netCDF data path and create xarray dataset using salem
        ds = ds_sl.sel(time = f"{year}-{month}") # slice data by year and month
        
        # combine and deaccumulate precipitation variables into new variable
        deacc_precip(ds, ds_variables)
        
        # create new variable for wind speed from magnitudes of velocity vectors
        magnitude(ds)
        
        # calculate descriptive stats on file using xarray
        mean = ds[ds_variables].mean(dim = "time", skipna = True)
        stddev = ds[ds_variables].std(dim = "time", skipna = True)
        median = ds[ds_variables].median(dim = "time", skipna = True)
        max = ds[ds_variables].max(dim = "time", skipna = True)
        min = ds[ds_variables].min(dim = "time", skipna = True)
        sample_size = ds[ds_variables].count(dim = "time")
        
        # aggregate means using cumulative moving average method
        mean_roll = (mean + (n * mean_roll)) / (n + 1)
        
        # function for calculating rolling std dev and cumulative sample size
        stddev_roll, sample_size_roll = WRFstddev(stddev, sample_size, stddev_roll, sample_size_roll, n)
        
        # function for calculating rolling max and min
        max_roll, min_roll = WRFminmax(max, min, max_roll, min_roll, n)
        
        #TODO cumulative methods for median
        median_roll = (median + (n * median_roll)) / (n + 1)
        
        n += 1 # iterate counter
        
        print("Run: ", n) # just to make sure it's still working...
        print("Current Runtime: ", timeit.default_timer() - start_time, "\n")

    
    return mean_roll, stddev_roll, max_roll, min_roll


#%% run code

path = r"C:\Users\mart229\OneDrive - PNNL\Desktop\Stuff\IM3\WRF\Month_Data\\"
year = "2007"
month = "01"

#WRFstats = WRFstats(path, year, month)
mean_ds, stddev_ds, max_ds, min_ds = WRFstats(path, year, month)

#print(WRFstats)
print("\n", "Total Runtime: ", timeit.default_timer() - start_time) # end timer and print
