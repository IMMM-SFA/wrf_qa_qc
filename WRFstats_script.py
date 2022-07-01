import pandas as pd
import numpy as np
import xarray as xr
import timeit
import os
from glob import glob
import salem as sl

# start timer
start_time = timeit.default_timer()

# set pandas display options
pd.set_option("display.expand_frame_repr", False)


#%% function for opening one month of data

def one_month_data(path, year, month):
    
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

    month_minus = months_list[idx - 1]
    
    # if month is January, return previous year, else current year
    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year
    
    return month_minus, year_minus


#%% function for calculating rolling max and min

def WRFminmax(max_df, min_df, max_roll, min_roll, n):
    
    """
    Function for calculating rolling maximum and minimum values for each variable.
    
    Parameters
    ----------
    max_df : DataFrame
        Pandas DataFrame with maximum values for current file.
    min_df : DataFrame
        Pandas DataFrame with minimum values for current file.
    max_roll : DataFrame
        Pandas DataFrame with rolling maximum values.
    min_roll : DataFrame
        Pandas DataFrame with rolling minimum values.
    n : Int
        Current value of iteration counter.
    Returns
    -------
    max_roll, min_roll : Series
        Pandas DataFrame for storage of maximum and minimum values.
    
    """
    ## check if this is the first file in the run, if so then assign values from current dataset
    if n == 0:
        max_roll = max_df.reset_index(drop = True)
        min_roll = min_df.reset_index(drop = True)
        
    else:
        # check if new max is greater than old max, if so then replace it, if none are greater then skip loop
        if (max_df.reset_index(drop = True) > max_roll).any():
            for i in range(len(max_df)):
                if max_df.reset_index(drop = True)[i] > max_roll[i]:
                    max_roll[i] = max_df.reset_index(drop = True)[i]
        
        # check if new min is less than old min, if so then replace it, if none are less then skip loop
        if (min_df.reset_index(drop = True) < min_roll).any():
            for i in range(len(min_df)):
                if min_df.reset_index(drop = True)[i] < min_roll[i]:
                    min_roll[i] = min_df.reset_index(drop = True)[i]
    
    return max_roll, min_roll


#%% function for calculating cumulative standard deviation and sample size

def WRFstddev(stddev_df, stddev_roll, sample_size_series, sample_size_roll, n):
    
    """
    Function for calculating pooled standard deviation and rolling sample size values for each variable.
    
    Parameters
    ----------
    stddev_df : DataFrame
        Pandas DataFrame with standard deviation values for current file.
    stddev_roll : DataFrame
        Pandas DataFrame with rolling standard deviation values.
    sample_size_series : Series
        Pandas series with sample sizes of each variable for current file.
    sample_size_roll : Series
        Pandas series with rolling sample sizes of each variable.
    n : Int
        Current value of iteration counter.
    Returns
    -------
    stddev_roll, sample_size_roll : Series
        Pandas series for storage of standard deviation and sample size values.
    
    """
    
    # check if this is the first file in the run, if so then assign values from current dataset
    if n == 0:
        sample_size_roll = sample_size_series
        stddev_roll = stddev_df
    else:
        sample_size_roll = sample_size_roll + sample_size_series

        # Cohen pooled method of combining weighted std devs, assuming similar variances in samples
        sd1 = stddev_roll.reset_index(drop = True)
        sd2 = stddev_df.reset_index(drop = True)
        n1 = sample_size_roll.reset_index(drop = True)
        n2 = sample_size_series.reset_index(drop = True)
        stddev_roll = np.sqrt( ( ((n1 - 1) * sd1**2) + ((n2 - 1) * sd2**2 ) ) / (n1 + n2 - 2) )

    return stddev_roll, sample_size_roll


#%% function to combine and deaccumulate precipitation variables into new variable

def deacc_precip(ds, ds_variables):
    
    if "PRECIP" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis = 0, prepend = np.array([ds["PRECIP"][0].values]))
        
    ds["RAINC"].values = np.diff(ds["RAINC"].values, axis = 0, prepend = np.array([ds["RAINC"][0].values]))
    ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis = 0, prepend = np.array([ds["RAINSH"][0].values]))
    ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis = 0, prepend = np.array([ds["RAINNC"][0].values]))    


#%% function for aggregating rolling stats on netCDF data

def WRFstats(path, year, month, 
             ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                           "ZOL","RAINC","RAINSH","RAINNC","PRECIP","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
             ):
    
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.
    
    Parameters
    ----------
    path : Str
        Path to netCDF files for analysis.
    Returns
    -------
    stats_df : DataFrame
        Pandas DataFrame for storage of stats.
    
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
        
        # calculate descriptive stats on file using xarray
        mean = ds[ds_variables].mean()
        stddev = ds[ds_variables].std()
        median = ds[ds_variables].median()
        max = ds[ds_variables].max()
        min = ds[ds_variables].min()
        
        # convert stats to pandas for storage
        mean_df = mean.to_pandas().reset_index(drop = True)
        stddev_df = stddev.to_pandas()
        median_df = median.to_pandas().reset_index(drop = True)
        max_df = max.to_pandas()
        min_df = min.to_pandas()
        
        #var_names = [key for key in ds.variables.keys()][3:] # extract variable names for index matching
        sample_size = [eval(f"ds.{var}.size", {"ds": ds}) for var in ds_variables] # find size of each variable
        sample_size_series = pd.Series(data = sample_size, dtype = float) # convert to pandas series for calculation in formula
        
        # aggregate means using cumulative moving average method
        mean_roll = (mean_df + (n * mean_roll)) / (n + 1)
        
        # function for calculating rolling std dev and cumulative sample size
        stddev_roll, sample_size_roll = WRFstddev(stddev_df, stddev_roll, sample_size_series, sample_size_roll, n)
        
        # function for calculating rolling max and min
        max_roll, min_roll = WRFminmax(max_df, min_df, max_roll, min_roll, n)
        
        #TODO cumulative methods for median
        median_roll = (median_df + (n * median_roll)) / (n + 1)
        
        n += 1 # iterate counter
        
        print("Run: ", n) # just to make sure it's still working...
        print("Current Runtime: ", timeit.default_timer() - start_time, "\n")
        
    # create dictionary of stats and convert to DataFrame
    stats = {
            "Variable": ds_variables,
            "Mean": mean_roll,
            "Standard Dev.": stddev_roll,
            "Median": median_roll,
            "Maximum": max_roll,
            "Minimum": min_roll
            }
    
    stats_df = pd.DataFrame(data = stats, index = None)
    
    return stats_df


#%% run code

path = r"C:\Users\mart229\OneDrive - PNNL\Desktop\Stuff\IM3\WRF\Month_Data\\"
year = "2007"
month = "01"

WRFstats = WRFstats(path, year, month)

print(WRFstats)
print("\n", "Total Runtime: ", timeit.default_timer() - start_time) # end timer and print
