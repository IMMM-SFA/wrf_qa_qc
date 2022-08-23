import datetime
from suntime import Sun, SunTimeException
import os
from glob import glob
import salem as sl
from datetime import datetime, timedelta


# %% function for opening a variable range of data

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
    yearfiles_list = [sorted(glob(os.path.join(input_path, f"wrfout_*{year}*"))) for year in years]
    yearfiles = [file for list in yearfiles_list for file in list]

    # calculate the time difference in the given date range
    startdate = datetime.strptime(start, "%Y-%m-%d")
    stopdate = datetime.strptime(stop, "%Y-%m-%d")
    delta = stopdate - startdate

    # create list of all dates between the given range of dates
    filedates = [str((startdate + timedelta(days=d)).strftime("%Y-%m-%d")) for d in range(delta.days + 1)]

    # collect just the files between the date range that are present in the folder
    rangefiles = [glob(os.path.join(input_path, f"wrfout_d01_*{date}*"))[0] for date in filedates if
                  date in str(yearfiles)]

    return rangefiles


def WRFsun(input_path, output_path, start, stop):
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
    # create list of netCDF files at path
    nc_files = variable_range_data(input_path, start, stop)

    for file in nc_files:

        ds = sl.open_wrf_dataset(file)  # open netCDF data path and create xarray dataset using salem

        # check if last file in run, if so then slice data by stop date
        if file == nc_files[-1]:
            ds = ds.sel(time=slice(f"{stop}"))

    sun = Sun(latitude, longitude)


input_path = "C:/Users/mcgr323/projects/wrf/"
output_path = "C:/Users/mcgr323/projects/wrf/wrf_output/"
start = "1986-01-01"
stop = "1986-01-02"