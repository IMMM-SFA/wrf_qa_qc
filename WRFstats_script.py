import pandas as pd
import xarray as xr
import timeit
import os
#import salem

# start timer
start_time = timeit.default_timer()

# set pandas display options
pd.set_option("display.expand_frame_repr", False)


#%% function for rolling stats on netCDF data

def WRFstats(path):
    
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.
    
    Parameters
    ----------
    path : STR
        Path to netCDF files for analysis.
    Returns
    -------
    stats_df : DataFrame
        Pandas DataFrame for storage of variables.
    
    """
    
    # create list of files at path
    nc_files = [file for file in os.listdir(path) if file.endswith(".nc")]
    
    # create rolling stats variables
    n = 0
    mean_roll = 0
    stddev_roll = 0
    median_roll = 0
    max_roll = 0
    min_roll = 0
    
    # iterate through each nc file and create dataset
    for file in nc_files:
        
        ds = xr.open_dataset(path + file) # open netCDF data path and create xarray dataset
        
        # calculate descriptive stats on file using xarray
        mean = ds.mean()
        stddev = ds.std()
        median = ds.median()
        max = ds.max()
        min = ds.min()
        
        # convert stats to pandas for storage
        mean_df = mean.to_pandas()
        stddev_df = stddev.to_pandas()
        median_df = median.to_pandas()
        max_df = max.to_pandas()
        min_df = min.to_pandas()
        
        var_names = [key for key in mean.variables.keys()] # extract variable names for index matching
        
        # aggregate stats using cumulative moving average method
        #TODO cumulative methods for std dev, median, max, min?
        mean_roll = (mean_df + (n * mean_roll)) / (n + 1)
        stddev_roll = (stddev_df + (n * stddev_roll)) / (n + 1)
        median_roll = (median_df + (n * median_roll)) / (n + 1)
        max_roll = (max_df.iloc[1:] + (n * max_roll)) / (n + 1)
        min_roll = (min_df.iloc[1:] + (n * min_roll)) / (n + 1)
        
        n += 1 # iterate counter
        
    # create DataFrame of stats
    stats = {
            "Variable": var_names,
            "Mean": mean_roll,
            "Standard Dev.": stddev_roll,
            "Median": median_roll,
            "Maximum": max_roll,
            "Minimum": min_roll
            }
    
    stats_df = pd.DataFrame(data = stats, index = None)
    
    return stats_df


#%% run code

path = r"C:\Users\mart229\OneDrive - PNNL\Desktop\Stuff\IM3\WRF\netCDF_Data\\"
WRFstats = WRFstats(path)

print("Total Runtime: ", timeit.default_timer() - start_time) # end timer and print

