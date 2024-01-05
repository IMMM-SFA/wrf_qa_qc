import pandas as pd
import salem as sl
import os
import xarray as xr
import dask
from glob import glob

from create_climate_variables import temp_conv, deacc_precip, windspeed, rel_humidity
from find_stats_and_dist import descriptive_stats, sw_test, skew_kurtosis_test
from find_outliers import iqr_test, zscore_test, rh_anomaly, find_lan_nlad, lan_nlad_storage, find_zeros, find_negatives


def previous_month(year_month: str) -> str:
    """
    Returns the year-month string for the previous month given a year-month string.

    Parameters:
    -----------
    year_month : str
        A string in the format "YYYY-MM" representing the year and month.

    Returns:
    --------
    str
        A string in the format "YYYY-MM" representing the year and month of the previous month.
    """

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


def wrf_stats(input_path: str, output_path: str, start: str, stop: str,
              descriptive: bool = True, distribution: bool = True, outliers: bool = True, anomalies: bool = True,
              ds_variables: list[str] = None) -> list[xr.Dataset]:
    """
    Calculates descriptive statistics and statistical outliers on monthly WRF netCDF data between a given range of months.

    Parameters
    ----------
    input_path : str
        The path to the netCDF files for analysis.
    output_path : str
        The path for the output netCDF files to be stored.
    start : str
        The start date of the files to open, in the format "YYYY-MM".
    stop : str
        The end date of the files to open (inclusive), in the format "YYYY-MM".
    descriptive : bool, optional
        Whether to calculate descriptive statistics. Default is True.
    distribution : bool, optional
        Whether to calculate distribution statistics. Default is True.
    outliers : bool, optional
        Whether to calculate statistical outliers. Default is True.
    anomalies : bool, optional
        Whether to find specific anomalies in the dataset. Default is True.
    ds_variables : list of str, optional
        List of variable names to calculate statistics on. If not provided, defaults to a list of commonly used variables.

    Returns
    -------
    list of xarray.Dataset
        A list of datasets for storage of statistics output.
    """

    if ds_variables is None:
        ds_variables = ["LU_INDEX", "Q2", "T2", "PSFC", "U10", "V10", "SFROFF", "UDROFF", "ACSNOM", "SNOW", "SNOWH",
                        "WSPD", "BR", "ZOL", "RAINC", "RAINSH", "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SWDOWN",
                        "GLW", "UST", "SNOWC", "SR", 'T2F', 'T2C', 'PRECIP', 'WINDSPEED', 'RH']

    # create list of range of months to open
    months = pd.date_range(start, stop, freq="MS").strftime("%Y-%m").tolist()

    # iterate through each month and create dataset
    for month in months:

        # create list of files in the given month in the range of months specified
        nc_files = sorted(glob(os.path.join(input_path, f"tgw_wrf_historical_hourly_*{month}*")))

        # find the previous month and take the last file of that month to extract any overlapping dates
        previousmonth = previous_month(month)
        previousmonth_lastfile = sorted(glob(os.path.join(input_path, f"tgw_wrf_historical_hourly_*{previousmonth}*")))[
            -1]
        nc_files.insert(0, previousmonth_lastfile)

        ds = sl.open_mf_wrf_dataset(nc_files)  # open all netCDF files in month and create xarray dataset using salem
        ds = ds.sel(time=slice(f"{month}"))  # slice by the current month
        ds.load()

        # convert T2 variable from K to F or C
        temp_conv(ds, ds_variables)

        # combine and deaccumulate precipitation variables into PRECIP variable
        deacc_precip(ds, ds_variables)

        # create new variable WINDSPEED from magnitudes of velocity vectors
        windspeed(ds, ds_variables)

        # calculate relative humidity and create new variable RH
        rel_humidity(ds, ds_variables)

        # calculate descriptive stats on files using xarray
        if descriptive:
            all_stats = descriptive_stats(ds, ds_variables)

        else:
            all_stats = (None,) * 5

        # calculate distribution stats on files using xarray
        if distribution:
            # Shapiro-Wilks test function for normality, gives percent of distributions that are normal
            sw_ds, normality_ds = sw_test(ds, ds_variables)

            # skew and kurtosis tests
            skew_ds, kurtosis_ds = skew_kurtosis_test(ds, ds_variables)

        else:
            sw_ds, normality, skew_ds, kurtosis_ds = (None,) * 4

        if outliers:
            # run IQR test
            all_iqr_stats, all_iqr_outliers = iqr_test(ds, ds_variables, iqr_threshold=3)

            # run z-score test
            z_outlier_ds = zscore_test(ds, ds_variables, z_threshold=4)

        # find specific anomalies in the dataset
        if anomalies:
            # check for RH anomalies
            rh_anomaly_ds = rh_anomaly(ds, ds_variables)

            # calculate dusk/dawn times and find occurrences of light at night (LAN) and no light at day (NLAD)
            light_anomaly_ds = find_lan_nlad(ds, ds_variables)

            # find occurrences of zero where they should not occur
            zeros_ds = find_zeros(ds, ds_variables)

            # find occurrences of negatives where they should not occur
            negatives_ds = find_negatives(ds, ds_variables)

            # concatenate stats into dictionary and save as numpy dict
        stats_combined = xr.merge([all_stats, sw_ds, skew_ds, kurtosis_ds, all_iqr_stats])
        outliers_combined = xr.merge([all_iqr_outliers, z_outlier_ds, rh_anomaly_ds, light_anomaly_ds,
                                      zeros_ds, negatives_ds])

        # get string for year
        # get string for year
        year_dir = month[0:4]

        # create path for year
        year_path = os.path.join(output_path, year_dir)

        # checking if the directory demo_folder exist or not.
        if not os.path.exists(year_path):
            # if the demo_folder directory is not present create it
            os.makedirs(year_path)

        # specify the location for the output of the program
        stats_output_filename = os.path.join(year_path + f"tgw_wrf_hourly_{month}_all_stats.nc")
        outlier_output_filename = os.path.join(year_path + f"tgw_wrf_hourly_{month}_all_outliers.nc")

        # save each output stat as a netCDF file
        stats_combined.to_netcdf(path=stats_output_filename)
        outliers_combined.to_netcdf(path=outlier_output_filename)

    return


