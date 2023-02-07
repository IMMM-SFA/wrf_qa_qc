import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import zscore


# %% function to convert T2 variable from K to F or C
def temp_conv(ds, ds_variables, F=True, C=True):
    """
    Function for converting Kelvin to Fahrenheit or Celsius

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on
    F: boolean Fahrenheit
    C: boolean Celsius

    Returns
    -------
    None

    """
    if "T2" in ds_variables:

        K = ds["T2"]

        # convert to F
        if F == True:
            ds["T2F"] = 1.8 * (K - 273.15) + 32
            # ds_variables.append("T2F")

        # convert to C
        if C == True:
            ds["T2C"] = K - 273.15
            # ds_variables.append("T2C")


# %% function to combine and deaccumulate precipitation variables into new variable
def deacc_precip(ds, ds_variables):
    """
    Function for deaccumlating precipitation (only for TGW dataset)

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on

    Returns
    -------
    None

    """
    # check if rain variables included in the variables list, if so then create PRECIP variable and deaccumulate
    if "RAINC" in ds_variables and "RAINSH" in ds_variables and "RAINNC" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis=0, prepend=np.array([ds["PRECIP"][0].values]))
        # ds_variables.append("PRECIP")

    # deaccumulate rain variables
    if "RAINC" in ds_variables:
        ds["RAINC"].values = np.diff(ds["RAINC"].values, axis=0, prepend=np.array([ds["RAINC"][0].values]))

    if "RAINSH" in ds_variables:
        ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis=0, prepend=np.array([ds["RAINSH"][0].values]))

    if "RAINNC" in ds_variables:
        ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis=0, prepend=np.array([ds["RAINNC"][0].values]))


# %% function for calculating magnitude of wind velocity vectors
def windspeed(ds, ds_variables):
    """
    Function for calculating windspeed from U and V

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on

    Returns
    -------
    None

    """
    if "U10" in ds_variables and "V10" in ds_variables:
        U = ds["U10"]
        V = ds["V10"]
        ds["WINDSPEED"] = np.sqrt(U ** 2 + V ** 2)
        # ds_variables.append("WINDSPEED")


# %% function to rename stats
def rename_stats(mean_ds, median_ds, stddev_ds, max_ds, min_ds,
                 ds_variables):
    """
        Function for renaming the xarray variables for mean, med, min, max and std dev

        Input
        ----------
        mean_ds : xarray of monthly mean ds_variables
        median_ds : xarray of monthly median ds_variables
        stddev_ds : xarray of monthly std dev ds_variables
        max_ds : xarray of monthly max ds_variables
        min_ds : xarray of monthly min ds_variables
        ds_variables : List Variables to rename

        Returns
        -------
        mean_df : xarray with string "_mean" added to each df variable
        med_df : xarray with string "_med" added to each df variable
        stddev_df : xarray with string "_std" added to each df variable
        max_df : xarray with string "_max" added to each df variable
        min_df : xarray with string "_min" added to each df variable

        """

    length = len(ds_variables)

    for i in range(length):
        mean_ds = mean_ds.rename({ds_variables[i]: f"{ds_variables[i]}_mean"})
        median_ds = median_ds.rename({ds_variables[i]: f"{ds_variables[i]}_med"})
        stddev_ds = stddev_ds.rename({ds_variables[i]: f"{ds_variables[i]}_std"})
        max_ds = max_ds.rename({ds_variables[i]: f"{ds_variables[i]}_max"})
        min_ds = min_ds.rename({ds_variables[i]: f"{ds_variables[i]}_min"})

    return mean_ds, median_ds, stddev_ds, max_ds, min_ds


# %% calculate descriptive stats on file using xarray

def descriptive_stats(ds, ds_variables):
    """
    Function for calculating mean, median, min, max, and std dev from raw data

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on

    Returns
    -------
    stats_all : xarray of combined varaibles with variables strings

    """
    mean_ds = ds[ds_variables].mean(dim="time", skipna=True)
    median_ds = ds[ds_variables].median(dim="time", skipna=True)
    stddev_ds = ds[ds_variables].std(dim="time", skipna=True)
    max_ds = ds[ds_variables].max(dim="time", skipna=True)
    min_ds = ds[ds_variables].min(dim="time", skipna=True)

    mean_df, med_df, stddev_df, max_df, min_df = rename_stats(mean_ds, median_ds, stddev_ds, max_ds, min_ds,
                                                              ds_variables)

    all_stats = xr.merge([mean_df, med_df, max_df, min_df, stddev_df])

    return all_stats


# %% skew and kurtosis tests


def rename_skew(skew_ds, kurtosis_ds, ds_variables):
    """
    Function for renaming skew and kurtosis variables within the xarray

    Input
    ----------
    skew_ds : xarray of monthly skew ds_variables
    kurtosis_ds : xarray of monthly kurtosis ds_variables

    Returns
    -------
    skew_ds: xarray with string "_skew" added to each df variable
    kurtosis_ds: xarray with string "_kurt" added to each df variable
    """

    length = len(ds_variables)

    for i in range(length):
        skew_ds = skew_ds.rename({ds_variables[i]: f"{ds_variables[i]}_skew"})
        kurtosis_ds = kurtosis_ds.rename({ds_variables[i]: f"{ds_variables[i]}_kurt"})

    return skew_ds, kurtosis_ds


def skew_func(ds_var):
    skewness = skew(ds_var, axis=0, nan_policy="omit")

    return np.array([skewness])


def kurtosis_func(ds_var):
    kurtosisness = kurtosis(ds_var, axis=0, nan_policy="omit")

    return np.array([kurtosisness])


def skew_kurtosis_test(ds, ds_variables):
    """
    Function for calculating skew and kurtosis

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on

    Returns
    -------
    skew_ds : xarray of skew for all given ds_variables
    kurtosis_ds : xarray of kurtosis for all given ds_variables

    """

    skew_list = []
    kurt_list = []

    for ds_var in ds_variables:
        skew_test = xr.apply_ufunc(skew_func, ds[ds_var], input_core_dims=[["time"]], output_core_dims=[["output"]],
                                   vectorize=True, output_dtypes=[np.dtype("float32")])

        kurtosis_test = xr.apply_ufunc(kurtosis_func, ds[ds_var], input_core_dims=[["time"]],
                                       output_core_dims=[["output"]],
                                       vectorize=True, output_dtypes=[np.dtype("float32")])

        skewness = skew_test.isel(output=0)
        kurtosisness = kurtosis_test.isel(output=0)

        skew_list.append(skewness)
        kurt_list.append(kurtosisness)

    skew_ds = xr.merge(skew_list)
    kurtosis_ds = xr.merge(kurt_list)

    skew_ds, kurtosis_ds = rename_skew(skew_ds, kurtosis_ds, ds_variables)

    return skew_ds, kurtosis_ds


# %% Shapiro-Wilks test function for normality

def rename_sw(sw_ds, ds_variables, normality):
    """
    Function for renaming skew and kurtosis variables within the xarray

    Input
    ----------
    sw_ds: xarray of Sharpio-Wilks test results for given ds_variables
    normality_ds : xarray of normality vlaue for given ds_variables

    Returns
    -------
    sw_ds: xarray with string "_sw" added to each df variable
    normality_ds: xarray with string "_norm" added to each df variable
    """

    length = len(ds_variables)

    for i in range(length):
        sw_ds = sw_ds.rename({ds_variables[i]: f"{ds_variables[i]}_sw"})
        # normality_ds = normality_ds.rename({ds_variables[i]: f"{ds_variables[i]}_norm"})
        normality = {f"{key}_norm": value for key, value in normality.items()}

    return sw_ds, normality


def sw_func(ds_var):
    teststat, p = shapiro(ds_var)

    return np.array([[teststat, p]])


def sw_test(ds, ds_variables):
    """
    Function for calculating Sharpio-Wilks test

    Input
    ----------
    ds : xarray of raw data
    ds_variables: list of variables within the df to perform stats on

    Returns
    -------
    sw_ds : xarray of Sharpio-Wilks test for all given ds_variables
    normality_ds : xarray of normality for all given ds_variables

    """

    pval_list = []
    normality = {}

    for ds_var in ds_variables:
        shapiro_test = xr.apply_ufunc(sw_func, ds[ds_var], input_core_dims=[["time"]], output_core_dims=[["output"]],
                                      vectorize=True, output_dtypes=[np.dtype("float32")])

        p = shapiro_test.isel(output=1)

        pval_list.append(p)

        percent_normal = (p.values > 0.05).sum() / (p.values >= 0).sum()
        normality[ds_var] = percent_normal

    sw_ds = xr.merge(pval_list)
    sw_ds, normality = rename_sw(sw_ds, ds_variables, normality)

    return sw_ds, normality


# %% function to rename stats
def rename_iqr_stats(iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper, outlier_lower,
                     outlier_upper_inv,
                     outlier_lower_inv, ds_variables):
    """
        Function for renaming the xarray variables for mean, med, min, max and std dev

        Input
        ----------
        mean_ds : xarray of monthly mean ds_variables
        median_ds : xarray of monthly median ds_variables
        stddev_ds : xarray of monthly std dev ds_variables
        max_ds : xarray of monthly max ds_variables
        min_ds : xarray of monthly min ds_variables
        ds_variables : List Variables to rename

        Returns
        -------
        mean_df : xarray with string "_mean" added to each df variable
        med_df : xarray with string "_med" added to each df variable
        stddev_df : xarray with string "_std" added to each df variable
        max_df : xarray with string "_max" added to each df variable
        min_df : xarray with string "_min" added to each df variable

        """

    length = len(ds_variables)

    for i in range(length):
        iqr_ds = iqr_ds.rename({ds_variables[i]: f"{ds_variables[i]}_iqr"})
        q75_ds = q75_ds.rename({ds_variables[i]: f"{ds_variables[i]}_q75"})
        upper_threshold = upper_threshold.rename({ds_variables[i]: f"{ds_variables[i]}_upper"})
        lower_threshold = lower_threshold.rename({ds_variables[i]: f"{ds_variables[i]}_lower"})
        outlier_upper = outlier_upper.rename({ds_variables[i]: f"{ds_variables[i]}_out_up"})
        outlier_lower = outlier_lower.rename({ds_variables[i]: f"{ds_variables[i]}_out_low"})
        outlier_upper_inv = outlier_upper_inv.rename({ds_variables[i]: f"{ds_variables[i]}_out_up_inv"})
        outlier_lower_inv = outlier_lower_inv.rename({ds_variables[i]: f"{ds_variables[i]}_out_low_inv"})

    return iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper, outlier_lower, outlier_upper_inv, \
           outlier_lower_inv


# %% outlier detection with IQR test
def IQR_Test(ds, ds_variables, iqr_threshold=1.5):
    q75_ds = ds[ds_variables].quantile(q=0.75, dim="time", skipna="True").astype("float32")
    q25_ds = ds[ds_variables].quantile(q=0.25, dim="time", skipna="True").astype("float32")

    iqr_ds = q75_ds - q25_ds
    IQR_val = iqr_threshold * iqr_ds
    upper_threshold = q75_ds + IQR_val
    lower_threshold = q25_ds - IQR_val

    outlier_upper = ds[ds_variables].where(ds[ds_variables] < (upper_threshold))
    outlier_lower = ds[ds_variables].where(ds[ds_variables] > (lower_threshold))

    outlier_upper_inv = ds[ds_variables].where(ds[ds_variables] > (upper_threshold))
    outlier_lower_inv = ds[ds_variables].where(ds[ds_variables] < (lower_threshold))

    iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper, outlier_lower, outlier_upper_inv, \
    outlier_lower_inv = rename_iqr_stats(iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper,
                                         outlier_lower, outlier_upper_inv, outlier_lower_inv, ds_variables)

    iqr_outliers = xr.merge(
        [iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper, outlier_lower, outlier_upper_inv, \
         outlier_lower_inv], compat='override')

    return iqr_outliers


def rename_z_stats(zscore_ds, z_outlier_upper, z_outlier_lower, z_outlier_upper_inv, z_outlier_lower_inv, ds_variables):
    """
        Function for renaming the xarray variables for mean, med, min, max and std dev

        Input
        ----------
        mean_ds : xarray of monthly mean ds_variables
        median_ds : xarray of monthly median ds_variables
        stddev_ds : xarray of monthly std dev ds_variables
        max_ds : xarray of monthly max ds_variables
        min_ds : xarray of monthly min ds_variables
        ds_variables : List Variables to rename

        Returns
        -------
        mean_df : xarray with string "_mean" added to each df variable
        med_df : xarray with string "_med" added to each df variable
        stddev_df : xarray with string "_std" added to each df variable
        max_df : xarray with string "_max" added to each df variable
        min_df : xarray with string "_min" added to each df variable

        """

    length = len(ds_variables)

    for i in range(length):
        zscore_ds = zscore_ds.rename({ds_variables[i]: f"{ds_variables[i]}_zscore"})
        z_outlier_upper = z_outlier_upper.rename({ds_variables[i]: f"{ds_variables[i]}_z_out_upper"})
        z_outlier_lower = z_outlier_lower.rename({ds_variables[i]: f"{ds_variables[i]}_z_out_lower"})
        z_outlier_upper_inv = z_outlier_upper_inv.rename({ds_variables[i]: f"{ds_variables[i]}_z_out_upper_inv"})
        z_outlier_lower_inv = z_outlier_lower_inv.rename({ds_variables[i]: f"{ds_variables[i]}_z_out_lower_inv"})

    return zscore_ds, z_outlier_upper, z_outlier_lower, z_outlier_upper_inv, z_outlier_lower_inv


# %% z-score outlier test
def ZScore_Test(ds, ds_variables, z_threshold=3):
    def ZS_func(ds_var):
        z = zscore(ds_var, axis=0, nan_policy="omit")

        return np.array([z])

    z_list = []

    for ds_var in ds_variables:
        zscore_test = xr.apply_ufunc(ZS_func, ds[ds_var], input_core_dims=[["time"]], output_core_dims=[["time"]],
                                     vectorize=True, output_dtypes=[np.dtype("float32")])

        z_list.append(zscore_test)

    zscore_ds = xr.merge(z_list)

    z_outlier_upper = ds[ds_variables].where(zscore_ds[ds_variables] < z_threshold)
    z_outlier_lower = ds[ds_variables].where(zscore_ds[ds_variables] > -z_threshold)

    z_outlier_upper_inv = ds[ds_variables].where(zscore_ds[ds_variables] > z_threshold)
    z_outlier_lower_inv = ds[ds_variables].where(zscore_ds[ds_variables] < -z_threshold)

    z_outliers = xr.merge(
        [zscore_ds, z_outlier_upper, z_outlier_lower, z_outlier_upper_inv, z_outlier_lower_inv, z_threshold], compat='override')

    return z_outliers

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
             ds_variables=None):
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
    if ds_variables is None:
        ds_variables = ["LU_INDEX", "Q2", "T2", "PSFC", "U10", "V10", "SFROFF", "UDROFF", "ACSNOM", "SNOW", "SNOWH",
                        "WSPD", "BR", "ZOL", "RAINC", "RAINSH", "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SWDOWN",
                        "GLW", "UST",
                        "SNOWC", "SR", 'T2F', 'T2C', 'PRECIP', 'WINDSPEED']

    stats_list = []

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

        if outliers == True:
            # run IQR test
            iqr_ds = IQR_Test(ds, ds_variables, iqr_threshold=1.5)

            # run z-score test
            z_ds = ZScore_Test(ds, ds_variables, z_threshold=3)

        # concatenate stats into dictionary and save as numpy dict
        stats_combined = xr.merge([all_stats, sw_ds, skew_ds, normality, kurtosis_ds, iqr_ds, z_ds])

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


input_path = "/global/cfs/cdirs/m2702/gsharing/tgw-wrf-conus/historical_1980_2019/hourly/"
output_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/"

start = "2006-12"
stop = "2007-12"

# run the WRFstats program
WRFstats(input_path, output_path, start, stop)

