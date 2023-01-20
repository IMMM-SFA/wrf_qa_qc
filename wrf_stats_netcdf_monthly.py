import pandas as pd
import salem as sl
import os
import xarray as xr
from glob import glob
from wrf_stats_netcdf_monthly import temp_conv, deacc_precip, windspeed
from wrf_stats_netcdf_monthly import descriptive_stats, sw_test, skew_kurtosis_test


# %% function to find the previous month containing parts of the given month
def previous_month(year_month):
    year = year_month[0: year_month.find("-")]
    month = year_month[year_month.rfind("-") + 1:]

    months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    idx = months_list.index(month)

    month_minus = months_list[idx - 1]

    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year

    previousmonth = year_minus + "-" + month_minus

    return previousmonth


# %% function for calculating stats on monthly netCDF data
def WRFstats(input_path, output_path, start, stop, descriptive=True, distribution=True, outliers=True,
             ds_variables=["LU_INDEX", "Q2", "T2", "PSFC", "U10", "V10", "SFROFF", "UDROFF", "ACSNOM", "SNOW", "SNOWH",
                           "WSPD", "BR",
                           "ZOL", "RAINC", "RAINSH", "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SWDOWN", "GLW", "UST",
                           "SNOWC", "SR"]
             ):
    """
    Function for calculating descriptive statistics and statistical outliers on monthly WRF netCDF data between a given range of months.

    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    output_path : Str Path for the output netCDF files to be stored.
    start : Str "YYYY-MM" Start date of files to open.
    stop : Str "YYYY-MM" End date of files to open (inclusive).
    ds_variables : List Variables to run stats on.

    Returns
    -------
    stats_list : List of datasets for storage of statistics output.

    """

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

        ds = sl.open_mf_wrf_dataset(nc_files)  # open all netCDF files in month and create xarray dataset using salem
        ds = ds.sel(time=slice(f"{month}"))  # slice by the current month
        ds.load()  # load into memory for computations

        # convert T2 variable from K to F or C
        temp_conv(ds, ds_variables)

        # combine and deaccumulate precipitation variables into PRECIP variable
        deacc_precip(ds, ds_variables)

        # create new variable WINDSPEED from magnitudes of velocity vectors
        windspeed(ds, ds_variables)

        # calculate descriptive stats on files using xarray
        if descriptive == True:
            all_stats = descriptive_stats(ds, ds_variables)

        else:
            all_stats = (None,) * 5

        # calculate distribution stats on files using xarray
        if distribution == True:
            # Shapiro-Wilks test function for normality, gives percent of distributions that are normal
            sw_ds, normality = sw_test(ds, ds_variables)

            # skew and kurtosis tests
            skew_ds, kurtosis_ds = skew_kurtosis_test(ds, ds_variables)

        else:
            sw_ds, normality, skew_ds, kurtosis_ds = (None,) * 4


        # concatenate stats into dictionary and save as numpy dict
        stats_combined = xr.merge([all_stats, sw_ds, skew_ds, normality, kurtosis_ds])

        # get string for year
        year_dir = month[0:4]

        # create path for year
        year_path = os.path.join(output_path, year_dir)

        # checking if the directory demo_folder exist or not.
        if not os.path.exists(year_path):
            # if the demo_folder directory is not present create it
            os.makedirs(year_path)

        # specify the location for the output of the program
        output_filename = os.path.join(year_path + "/" + f"tgw_wrf_hourly_{month}_all_stats.nc")

        # save each output stat as a netCDF file
        stats_combined.to_netcdf(path=output_filename)

    return


# %% run code

# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
# input_path = "/project/projectdirs/m2702/gsharing/CONUS_TGW_WRF_Historical/"
# output_path = "/project/projectdirs/m2702/gsharing/QAQC/"

input_path = "C:/Users/mcgr323/projects/wrf/wrf_input/"
output_path = "C:/Users/mcgr323/projects/wrf/wrf_output/"
start = "2006-12"
stop = "2007-12"

# run the WRFstats program
stats = WRFstats(input_path, output_path, start, stop)

