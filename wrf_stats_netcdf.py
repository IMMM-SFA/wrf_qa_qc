import numpy as np
import xarray as xr
from scipy.stats import shapiro, kurtosis, skew


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
            ds_variables.append("T2F")

        # convert to C
        if C == True:
            ds["T2C"] = K - 273.15
            ds_variables.append("T2C")


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
        ds_variables.append("PRECIP")

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
        ds_variables.append("WINDSPEED")


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

def rename_sw(sw_ds, normality_ds, ds_variables):
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
        normality_ds = normality_ds.rename({ds_variables[i]: f"{ds_variables[i]}_norm"})

    return sw_ds, normality_ds


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
    #sw_ds = rename_sw(sw_ds, ds_variables)

    return sw_ds, normality
