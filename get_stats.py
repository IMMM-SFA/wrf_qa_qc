import numpy as np
import xarray as xr
import os
from glob import glob
import salem as sl
from datetime import datetime, timedelta


def variable_range_data(input_path, start, stop, filename="wrfout_d01_"):
    """
    Function for opening a variable range of files between speicifed dates (inclusive).

    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    start : Str "YYYY-MM-DD" Start date of files to open.
    stop : Str "YYYY-MM-DD" End date of files to open.
    filename : Str basic beginning file structure of the netcdf file

    Returns
    -------
    rangefiles : List of netCDF files in specified date range.

    """

    # find the starting year, month, and day
    startyear = start[0: start.find("-")]
    startmonth = start[start.find("-") + 1: start.rfind("-")]
    startday = start[start.rfind("-") + 1:]

    # find the ending year, month, and day
    stopyear = stop[0: stop.find("-")]
    stopmonth = stop[stop.find("-") + 1: stop.rfind("-")]
    stopday = stop[stop.rfind("-") + 1:]

    # collect all files for the years in the given range
    years = [str(int(startyear) + i) for i in range(int(stopyear) - int(startyear) + 1)]
    yearfiles_list = [sorted(glob(os.path.join(input_path, f"{filename}*{year}*"))) for year in years]
    yearfiles = [file for list in yearfiles_list for file in list]

    # calculate the time difference in the given date range
    startdate = datetime.strptime(start, "%Y-%m-%d")
    stopdate = datetime.strptime(stop, "%Y-%m-%d")
    delta = stopdate - startdate

    # create list of all dates between the given range of dates
    filedates = [str((startdate + timedelta(days=d)).strftime("%Y-%m-%d")) for d in range(delta.days + 1)]

    # collect just the files between the date range that are present in the folder
    rangefiles = [glob(os.path.join(input_path, f"{filename}*{date}*"))[0] for date in filedates if
                  date in str(yearfiles)]

    return rangefiles


def get_min_max(max, min, max_roll, min_roll, n):
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

    # check if this is the first file in the run, if so then assign values from current dataset
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


def get_stddev(stddev, sample_size, stddev_roll, sample_size_roll, n):
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
        stddev_roll = np.sqrt((((n1 - 1) * sd1 ** 2) + ((n2 - 1) * sd2 ** 2)) / (n1 + n2 - 2))

    return stddev_roll, sample_size_roll


def deacc_precip(ds, ds_variables):
    """
        Function for calculating magnitude of velocity vectors

        Input
        ----------
        ds: DataSet from netcdf file
        ds_variables : List Variables to run stats on.

        """
    # check if the variable PRECIP was included in the variables list, if so then create variable from rain variables
    # and deaccumulate
    if "PRECIP" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis=0, prepend=np.array([ds["PRECIP"][0].values]))

    # deaccumulate rain variables
    ds["RAINC"].values = np.diff(ds["RAINC"].values, axis=0, prepend=np.array([ds["RAINC"][0].values]))
    ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis=0, prepend=np.array([ds["RAINSH"][0].values]))
    ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis=0, prepend=np.array([ds["RAINNC"][0].values]))


def magnitude(ds, wind_u="U10", wind_v="V10"):
    """
        Function for calculating magnitude of velocity vectors

        Input
        ----------
        ds: DataSet from netcdf file
        wind_u: Str variable name for U wind vector (varies for each dataset)
        wind_v: Str variable name for V wind vector (varies for each dataset)

        """

    U = ds["U10"]
    V = ds["V10"]
    ds["WINDSPEED"] = np.sqrt(U ** 2 + V ** 2)



def all_stats(input_path, output_path, start, stop, package_type=[0],
             ds_variables=["LU_INDEX", "Q2", "T2", "PSFC", "U10", "V10", "WINDSPEED", "SFROFF", "UDROFF", "ACSNOM",
                           "SNOW", "SNOWH", "WSPD", "BR","ZOL", "RAINC", "RAINSH", "RAINNC", "PRECIP", "SNOWNC",
                           "GRAUPELNC", "HAILNC", "SWDOWN","GLW", "UST", "SNOWC", "SR"]
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
    package_type: Int of whether to use 0=salem (WRF) or 1=xarray (other datasets)

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
    n = 0  # counter
    mean_roll = 0
    avg_median_roll = 0
    stddev_roll = None
    sample_size_roll = None
    max_roll = None
    min_roll = None

    # method for WRF (salem package)
    if package_type == 0:
        # iterate through each nc file and create dataset
        for file in nc_files:

            ds = sl.open_wrf_dataset(file)  # open netCDF data path and create xarray dataset using salem

            # check if last file in run, if so then slice data by stop date
            if file == nc_files[-1]:
                ds = ds.sel(time=slice(f"{stop}"))

            # combine and de-accumulate precipitation variables into new variable
            deacc_precip(ds, ds_variables)

            # create new variable for wind speed from magnitudes of velocity vectors
            magnitude(ds)

            # calculate descriptive stats on file using xarray
            mean = ds[ds_variables].mean(dim="time", skipna=True)
            stddev = ds[ds_variables].std(dim="time", skipna=True)
            avg_median = ds[ds_variables].median(dim="time", skipna=True)
            max = ds[ds_variables].max(dim="time", skipna=True)
            min = ds[ds_variables].min(dim="time", skipna=True)
            sample_size = ds[ds_variables].count(dim="time")

            # aggregate means using cumulative moving average method
            mean_roll = (mean + (n * mean_roll)) / (n + 1)

            # aggregate average medians using cumulative moving average method
            avg_median_roll = (avg_median + (n * avg_median_roll)) / (n + 1)

            # function for calculating rolling std dev and cumulative sample size
            stddev_roll, sample_size_roll = get_stddev(stddev, sample_size, stddev_roll, sample_size_roll, n)

            # function for calculating rolling max and min
            max_roll, min_roll = get_min_max(max, min, max_roll, min_roll, n)

            n += 1  # iterate counter

    # method for all other datasets (xarray)
    else:
        # iterate through each nc file and create dataset
        for file in nc_files:

            ds = xr.open_dataset(file) # open netCDF data path and create xarray dataset using salem

            # check if last file in run, if so then slice data by stop date
            if file == nc_files[-1]:
                ds = ds.sel(time=slice(f"{stop}"))

            # create new variable for wind speed from magnitudes of velocity vectors
            magnitude(ds)

            # calculate descriptive stats on file using xarray
            mean = ds[ds_variables].mean(dim="time", skipna=True)
            stddev = ds[ds_variables].std(dim="time", skipna=True)
            avg_median = ds[ds_variables].median(dim="time", skipna=True)
            max = ds[ds_variables].max(dim="time", skipna=True)
            min = ds[ds_variables].min(dim="time", skipna=True)
            sample_size = ds[ds_variables].count(dim="time")

            # aggregate means using cumulative moving average method
            mean_roll = (mean + (n * mean_roll)) / (n + 1)

            # aggregate average medians using cumulative moving average method
            avg_median_roll = (avg_median + (n * avg_median_roll)) / (n + 1)

            # function for calculating rolling std dev and cumulative sample size
            stddev_roll, sample_size_roll = get_stddev(stddev, sample_size, stddev_roll, sample_size_roll, n)

            # function for calculating rolling max and min
            max_roll, min_roll = get_min_max(max, min, max_roll, min_roll, n)

            n += 1  # iterate counter

    # specify the location for the output of the program
    output_filename = os.path.join(output_path + f"{start}_{stop}_")

    # save each output stat as a netCDF file
    mean_roll.to_netcdf(path=output_filename + "Mean_DS.nc")
    avg_median_roll.to_netcdf(path=output_filename + "Avg_Median_DS.nc")
    stddev_roll.to_netcdf(path=output_filename + "StdDev_DS.nc")
    max_roll.to_netcdf(path=output_filename + "Max_DS.nc")
    min_roll.to_netcdf(path=output_filename + "Min_DS.nc")

    return mean_roll, avg_median_roll, stddev_roll, max_roll, min_roll
