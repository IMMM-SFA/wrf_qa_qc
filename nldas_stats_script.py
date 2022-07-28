import pandas as pd
import numpy as np
import timeit
import os
from glob import glob
import xarray as xr
from dateutil.rrule import rrule, MONTHLY
from datetime import datetime

# start timer
start_time = timeit.default_timer()

# set pandas display options
pd.set_option("display.expand_frame_repr", False)


# %% function for opening one month of data

def one_month_data(path, year, month):
    # collect all files of a given month and year
    onemonthdata = sorted(glob(os.path.join(path, f"NLDAS_FORA0125_H.A*{year}{month}*.002.grb.SUB.nc4")))

    return onemonthdata


# %% function for calculating rolling max and min

def NLDASminmax(max_df, min_df, max_roll, min_roll, n):
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
        max_roll = max_df.reset_index(drop=True)
        min_roll = min_df.reset_index(drop=True)

    else:
        # check if new max is greater than old max, if so then replace it, if none are greater then skip loop
        if (max_df.reset_index(drop=True) > max_roll).any():
            for i in range(len(max_df)):
                if max_df.reset_index(drop=True)[i] > max_roll[i]:
                    max_roll[i] = max_df.reset_index(drop=True)[i]

        # check if new min is less than old min, if so then replace it, if none are less then skip loop
        if (min_df.reset_index(drop=True) < min_roll).any():
            for i in range(len(min_df)):
                if min_df.reset_index(drop=True)[i] < min_roll[i]:
                    min_roll[i] = min_df.reset_index(drop=True)[i]

    return max_roll, min_roll


# %% function for calculating cumulative standard deviation and sample size

def NLDASstddev(stddev_df, stddev_roll, sample_size_series, sample_size_roll, n):
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
        sd1 = stddev_roll.reset_index(drop=True)
        sd2 = stddev_df.reset_index(drop=True)
        n1 = sample_size_roll.reset_index(drop=True)
        n2 = sample_size_series.reset_index(drop=True)
        stddev_roll = np.sqrt((((n1 - 1) * sd1 ** 2) + ((n2 - 1) * sd2 ** 2)) / (n1 + n2 - 2))

    return stddev_roll, sample_size_roll


# %% function for aggregating rolling stats on netCDF data

def NLDASstats(path, year, month,
               ds_variables=["TMP", "SPFH", "PRES", "UGRD", "VGRD", "DLWRF",
                             "PEVAP", "APCP", "DSWRF"]
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
    # nc_files = [file for file in os.listdir(path) if file.endswith(".nc")]
    nc_files = one_month_data(path, year, month)

    # create rolling stats variables, set intial values
    n = 0  # counter
    sample_size_roll = None
    mean_roll = 0
    stddev_roll = None
    max_roll = None
    min_roll = None

    # iterate through each nc file and create dataset
    for file in nc_files:
        ds_xr = xr.open_dataset(file)  # open netCDF data path and create xarray dataset using salem
        ds = ds_xr.sel(time=f"{year}-{month}")  # slice data by year and month

        # calculate descriptive stats on file using xarray
        mean = ds[ds_variables].mean()
        stddev = ds[ds_variables].std()
        max = ds[ds_variables].max()
        min = ds[ds_variables].min()

        # convert stats to pandas for storage
        mean_df = mean.to_pandas().reset_index(drop=True)
        stddev_df = stddev.to_pandas()
        max_df = max.to_pandas()
        min_df = min.to_pandas()

        # var_names = [key for key in ds.variables.keys()][3:] # extract variable names for index matching
        sample_size = [eval(f"ds.{var}.size", {"ds": ds}) for var in ds_variables]  # find size of each variable
        sample_size_series = pd.Series(data=sample_size,
                                       dtype=float)  # convert to pandas series for calculation in formula

        # aggregate means using cumulative moving average method
        mean_roll = (mean_df + (n * mean_roll)) / (n + 1)

        # function for calculating rolling std dev and cumulative sample size
        stddev_roll, sample_size_roll = NLDASstddev(stddev_df, stddev_roll, sample_size_series, sample_size_roll, n)

        # function for calculating rolling max and min
        max_roll, min_roll = NLDASminmax(max_df, min_df, max_roll, min_roll, n)

        n += 1  # iterate counter

        print("Run: ", n)  # just to make sure it's still working...
        print("Current Runtime: ", timeit.default_timer() - start_time, "\n")

    # create dictionary of stats and convert to DataFrame
    stats = {
        "Time": f"{year}-{month}",
        "Variable": ds_variables,
        "Mean": mean_roll,
        "Standard Dev.": stddev_roll,
        "Maximum": max_roll,
        "Minimum": min_roll
    }

    stats_df = pd.DataFrame(data=stats, index=None)

    return stats_df


# Month selection function commented out for now becasue really want year functionality
# def months(start_m, start_yr, end_m, end_yr):
#     """
#       Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.
#
#       Parameters
#       ----------
#       start_m : int
#           Month to start analysis
#       start_yr : int
#           Year to start analysis.
#       end_m : int
#           Month to end analysis
#       end_yr : int
#           Year to end analysis.
#       Returns
#       -------
#       stats_df : DataFrame
#           Vector of year-month iterations
#
#       """
#
#     start = datetime(start_yr, start_m, 1)
#     end = datetime(end_yr, end_m, 1)
#     return [(d.month, d.year) for d in rrule(MONTHLY, dtstart=start, until=end)]


# def stats_by_month(path, start_m, start_yr, end_m, end_yr):
#     """
#     Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.
#
#     Parameters
#     ----------
#     path : Str
#         Path to netCDF files for analysis.
#      start_m : int
#           Month to start analysis
#      start_yr : int
#           Year to start analysis.
#      end_m : int
#           Month to end analysis
#      end_yr : int
#           Year to end analysis.
#     Returns
#     -------
#     stats_df : DataFrame
#         Pandas DataFrame for storage of stats.
#
#
#     """
#     month_itr = months(start_m, start_yr, end_m, end_yr)
#
#     appended_data = []
#     for i in range(1, len(month_itr)):
#         year = str(month_itr[i][1])
#         month = str(month_itr[i][0]).zfill(2)
#         NLDAS_stats = NLDASstats(path, year, month)
#         appended_data.append(NLDAS_stats)
#
#     # see pd.concat documentation for more info
#     appended_data = pd.concat(appended_data)
#     # write DataFrame to an excel sheet
#     csv_monthly_filename = os.path.join(path, 'NLDAS_Monthly_Min_Max_Values_' + str(start_m).zfill(2) + "_" + str(start_yr)
#                                         + '_UTC_to_' + str(end_m).zfill(2) + "_" + str(end_yr) + '_UTC.csv')
#     appended_data.to_csv(csv_monthly_filename, sep=',', index=False)


def stats_by_year(path, year):
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files at a given location.

    Parameters
    ----------
    path : Str
        Path to netCDF files for analysis.
     year : int
          Year to run analysis
    Returns
    -------
    stats_df : DataFrame
        Pandas DataFrame for storage of stats.


    """
    month_list = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    appended_data = []
    for i in range(1, len(month_list)):
        year = str(year)
        month = str(month_list[i]).zfill(2)
        NLDAS_stats = NLDASstats(path, year, month)
        appended_data.append(NLDAS_stats)

    # see pd.concat documentation for more info
    appended_data = pd.concat(appended_data)
    # write DataFrame to an excel sheet
    csv_monthly_filename = os.path.join(path,
                                        'NLDAS_Monthly_Min_Max_Values_' + year + '_UTC.csv')
    appended_data.to_csv(csv_monthly_filename, sep=',', index=False)


path = 'C:\\Users\\mcgr323\\projects\\wrf'
year = 2007

stats_by_year(path, year)

print("\n", "Total Runtime: ", timeit.default_timer() - start_time)  # end timer and print
