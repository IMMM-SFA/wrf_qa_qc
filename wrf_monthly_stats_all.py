import pandas as pd
import salem as sl
import os
import xarray as xr
import dask
import numpy as np
from glob import glob
from scipy.stats import shapiro, kurtosis, skew, zscore

import sys


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


# %% calculate relative humidity and return where rh over 100% or negative
# expect rh to be between 0 and 1, but small deviations above 1 are allowed
def rel_humidity(ds, ds_variables):
    if "T2" in ds_variables and "PSFC" in ds_variables and "Q2" in ds_variables:
        es = 6.112 * np.exp(17.67 * (ds["T2"] - 273.15) / (ds["T2"] - 29.65))
        qvs = 0.622 * es / (0.01 * ds["PSFC"] - (1.0 - 0.622) * es)
        rh = ds["Q2"] / qvs

        ds["RH"] = rh
        # ds_variables.append("RH")


def rename_stats(mean_ds, median_ds, stddev_ds, max_ds, min_ds, ds_variables):
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
    all_stats : xarray with string added to each df variable for each statistic

    """

    # List of new variable names for each statistic
    stat_suffixes = ['_mean', '_med', '_std', '_max', '_min']

    # Rename variables for each statistic using a loop
    stats = [mean_ds, median_ds, stddev_ds, max_ds, min_ds]
    all_stats = []
    for i, stat in enumerate(stats):
        renamed = stat.rename({var: f"{var}{stat_suffixes[i]}" for var in ds_variables})
        all_stats.append(renamed)

    # Merge all statistics into one xarray
    all_stats = xr.merge(all_stats)

    return all_stats


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

    all_stats = rename_stats(mean_ds, median_ds, stddev_ds, max_ds, min_ds, ds_variables)

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

    rename_dict = {var: f"{var}_skew" for var in ds_variables}
    skew_ds = skew_ds.rename(rename_dict)

    rename_dict = {var: f"{var}_kurt" for var in ds_variables}
    kurtosis_ds = kurtosis_ds.rename(rename_dict)

    return skew_ds, kurtosis_ds


def skew_func(ds_var):
    skewness = skew(ds_var, axis=0, nan_policy="omit", keepdims=True)

    return skewness


def kurtosis_func(ds_var):
    kurtosisness = kurtosis(ds_var, axis=0, nan_policy="omit", keepdims=True)

    return kurtosisness


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
    normality_ds : xarray of normality value for given ds_variables

    Returns
    -------
    sw_ds: xarray with string "_sw" added to each df variable
    normality_ds: xarray with string "_norm" added to each df variable
    """

    sw_ds = sw_ds.rename({var: f"{var}_sw" for var in ds_variables})
    normality = {f"{var}_norm": value for var, value in normality.items()}

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


# %% outlier detection with IQR test
def IQR_Test(ds, ds_variables, iqr_threshold=3):
    q75_ds = ds[ds_variables].quantile(q=0.75, dim="time", skipna="True").astype("float32")
    q25_ds = ds[ds_variables].quantile(q=0.25, dim="time", skipna="True").astype("float32")

    iqr_ds = q75_ds - q25_ds
    IQR_val = iqr_threshold * iqr_ds
    iqr_upper_threshold = q75_ds + IQR_val
    iqr_lower_threshold = q25_ds - IQR_val

    iqr_outlier_upper = ds[ds_variables].where(ds[ds_variables] > (iqr_upper_threshold))
    iqr_outlier_lower = ds[ds_variables].where(ds[ds_variables] < (iqr_lower_threshold))

    return iqr_ds, q75_ds, q25_ds, iqr_upper_threshold, iqr_lower_threshold, iqr_outlier_upper, iqr_outlier_lower


# %% pandas iqr outlier storage function
def iqr_outlier_storage(ds, ds_variables, iqr_outlier_upper, iqr_outlier_lower, iqr_upper_threshold,
                        iqr_lower_threshold):
    outlier_upper_dict = {i: np.where(iqr_outlier_upper[i].notnull()) for i in ds_variables}
    outlier_lower_dict = {i: np.where(iqr_outlier_lower[i].notnull()) for i in ds_variables}

    iqr_outlier_df_list = []

    for var in ds_variables:

        val_upper_outliers = ds[var].values[tuple(outlier_upper_dict[var])]
        time_upper_outliers = ds[var].time[tuple(outlier_upper_dict[var])[0]]
        lat_upper_outliers = ds[var].south_north[tuple(outlier_upper_dict[var])[1]]
        lon_upper_outliers = ds[var].west_east[tuple(outlier_upper_dict[var])[2]]
        q75_upper = iqr_upper_threshold[var].values[tuple(outlier_upper_dict[var])[1:]]
        diffq75_upper_outliers = val_upper_outliers - q75_upper

        val_lower_outliers = ds[var].values[tuple(outlier_lower_dict[var])]
        time_lower_outliers = ds[var].time[tuple(outlier_lower_dict[var])[0]]
        lat_lower_outliers = ds[var].south_north[tuple(outlier_lower_dict[var])[1]]
        lon_lower_outliers = ds[var].west_east[tuple(outlier_lower_dict[var])[2]]
        q25_lower = iqr_lower_threshold[var].values[tuple(outlier_lower_dict[var])[1:]]
        diffq25_lower_outliers = q25_lower - val_lower_outliers

        time_upper_list = list(time_upper_outliers.values)
        lat_upper_list = list(lat_upper_outliers.values)
        lon_upper_list = list(lon_upper_outliers.values)
        value_upper_list = list(val_upper_outliers)
        q75_upper_list = list(q75_upper)
        diffq75_upper_list = list(diffq75_upper_outliers)

        time_lower_list = list(time_lower_outliers.values)
        lat_lower_list = list(lat_lower_outliers.values)
        lon_lower_list = list(lon_lower_outliers.values)
        value_lower_list = list(val_lower_outliers)
        q25_lower_list = list(q25_lower)
        diffq25_lower_list = list(diffq25_lower_outliers)

        dict_upper = {"Time": [], "Lat": [], "Lon": [], "Value": [], "Q75": [], "QDiff": []}
        dict_lower = {"Time": [], "Lat": [], "Lon": [], "Value": [], "Q25": [], "QDiff": []}

        for i in range(len(value_upper_list)):
            dict_upper["Time"].append(time_upper_list[i])
            dict_upper["Lat"].append(lat_upper_list[i])
            dict_upper["Lon"].append(lon_upper_list[i])
            dict_upper["Value"].append(value_upper_list[i])
            dict_upper["Q75"].append(q75_upper_list[i])
            dict_upper["QDiff"].append(diffq75_upper_list[i])

        for i in range(len(value_lower_list)):
            dict_lower["Time"].append(time_lower_list[i])
            dict_lower["Lat"].append(lat_lower_list[i])
            dict_lower["Lon"].append(lon_lower_list[i])
            dict_lower["Value"].append(value_lower_list[i])
            dict_lower["Q25"].append(q25_lower_list[i])
            dict_lower["QDiff"].append(diffq25_lower_list[i])

        iqr_outlier_upper_df = pd.DataFrame(dict_upper)
        iqr_outlier_lower_df = pd.DataFrame(dict_lower)

        iqr_outliers_df = pd.concat([iqr_outlier_upper_df, iqr_outlier_lower_df], ignore_index=True)

        iqr_outlier_df_list.append(iqr_outliers_df)

    iqr_outlier_df_dict = {ds_variables[i]: iqr_outlier_df_list[i] for i in range(len(ds_variables))}

    return iqr_outlier_df_dict


def iqr_to_xarray(ds, ds_variables, iqr_threshold=3):
    iqr_ds, q75_ds, q25_ds, iqr_upper_threshold, iqr_lower_threshold, iqr_outlier_upper, iqr_outlier_lower = IQR_Test(
        ds, ds_variables, iqr_threshold)

    iqr_outlier_df_dict = iqr_outlier_storage(ds, ds_variables, iqr_outlier_upper, iqr_outlier_lower,
                                              iqr_upper_threshold, iqr_lower_threshold)

    iqr_outlier_ds = xr.Dataset.from_dict(iqr_outlier_df_dict)

    return iqr_outlier_ds


def ZScore_Test(ds, ds_variables, z_threshold=4):
    def ZS_func(ds_var):
        z = zscore(ds_var, axis=0, nan_policy="omit")
        return np.array([z])

    z_list = []

    for ds_var in ds_variables:
        zscore_test = xr.apply_ufunc(
            ZS_func,
            ds[ds_var],
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[np.dtype("float32")],
        )

        z_list.append(zscore_test)

    zscore_ds = xr.merge(z_list)

    z_outlier_upper = ds[ds_variables].where(zscore_ds[ds_variables] > z_threshold)
    z_outlier_lower = ds[ds_variables].where(zscore_ds[ds_variables] < -z_threshold)

    # Combine the outputs into an xarray dataset
    z_outlier_dataset = xr.Dataset(
        {
            "zscore": zscore_ds,
            "z_outlier_upper": z_outlier_upper,
            "z_outlier_lower": z_outlier_lower,
        }
    )
    z_outlier_dataset.attrs["z_threshold"] = z_threshold

    return z_outlier_dataset


def rh_anomaly(ds, ds_variables, RH_threshold=1.15, RH_threshold_neg=-0.015):
    if "RH" in ds_variables:
        rh_over100_ds = ds["RH"].where(ds["RH"] > RH_threshold)
        rh_negative_ds = ds["RH"].where(ds["RH"] < RH_threshold_neg)

        return rh_over100_ds, rh_negative_ds


def RH_dict_creation(ds, ds_variables, rh_over100_ds, rh_negative_ds):
    if "RH" in ds_variables:

        RH_over100_dict = {"RH_Over100": np.where(rh_over100_ds.notnull())}
        RH_negative_dict = {"RH_Negative": np.where(rh_negative_ds.notnull())}

        val_RH_Over100 = ds["RH"].values[tuple(RH_over100_dict["RH_Over100"])]
        time_RH_Over100 = ds["RH"].time[tuple(RH_over100_dict["RH_Over100"])[0]]
        lat_RH_Over100 = ds["RH"].south_north[tuple(RH_over100_dict["RH_Over100"])[1]]
        lon_RH_Over100 = ds["RH"].west_east[tuple(RH_over100_dict["RH_Over100"])[2]]

        val_RH_Negative = ds["RH"].values[tuple(RH_negative_dict["RH_Negative"])]
        time_RH_Negative = ds["RH"].time[tuple(RH_negative_dict["RH_Negative"])[0]]
        lat_RH_Negative = ds["RH"].south_north[tuple(RH_negative_dict["RH_Negative"])[1]]
        lon_RH_Negative = ds["RH"].west_east[tuple(RH_negative_dict["RH_Negative"])[2]]

        RH_Over100_time_list = list(time_RH_Over100.values)
        RH_Over100_lat_list = list(lat_RH_Over100.values)
        RH_Over100_lon_list = list(lon_RH_Over100.values)
        RH_Over100_val_list = list(val_RH_Over100)

        RH_Negative_time_list = list(time_RH_Negative.values)
        RH_Negative_lat_list = list(lat_RH_Negative.values)
        RH_Negative_lon_list = list(lon_RH_Negative.values)
        RH_Negative_val_list = list(val_RH_Negative)

        RH_Over100_dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}
        for i in range(len(RH_Over100_val_list)):
            RH_Over100_dict["Time"].append(RH_Over100_time_list[i])
            RH_Over100_dict["Lat"].append(RH_Over100_lat_list[i])
            RH_Over100_dict["Lon"].append(RH_Over100_lon_list[i])
            RH_Over100_dict["Value"].append(RH_Over100_val_list[i])

        RH_Negative_dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}
        for i in range(len(RH_Negative_val_list)):
            RH_Negative_dict["Time"].append(RH_Negative_time_list[i])
            RH_Negative_dict["Lat"].append(RH_Negative_lat_list[i])
            RH_Negative_dict["Lon"].append(RH_Negative_lon_list[i])
            RH_Negative_dict["Value"].append(RH_Negative_val_list[i])

        RH_Over100_df = pd.DataFrame(RH_Over100_dict)
        RH_Negative_df = pd.DataFrame(RH_Negative_dict)

        RH_anomaly_df_dict = {"RH_Over100": RH_Over100_df, "RH_Negative": RH_Negative_df}

        return RH_anomaly_df_dict


def RH_xarray_storage(ds, ds_variables):
    rh_over100_ds, rh_negative_ds = rh_anomaly(ds, ds_variables)
    RH_anomaly_df_dict = RH_dict_creation(ds, ds_variables, rh_over100_ds, rh_negative_ds)

    RH_Over100_df = RH_anomaly_df_dict["RH_Over100"].drop_duplicates(subset=["Time", "Lat", "Lon"])
    RH_Negative_df = RH_anomaly_df_dict["RH_Negative"].drop_duplicates(subset=["Time", "Lat", "Lon"])

    RH_Over100_ds = xr.Dataset(RH_Over100_df, coords={"time": RH_Over100_df["Time"], "lat": RH_Over100_df["Lat"],
                                                      "lon": RH_Over100_df["Lon"]})
    RH_Negative_ds = xr.Dataset(RH_Negative_df, coords={"time": RH_Negative_df["Time"], "lat": RH_Negative_df["Lat"],
                                                        "lon": RH_Negative_df["Lon"]})

    RH_Over100_ds["Value"].attrs["long_name"] = "Relative Humidity Anomaly > 115%"
    RH_Negative_ds["Value"].attrs["long_name"] = "Relative Humidity Anomaly < -1.5%"

    RH_Negative_ds = RH_Negative_ds.rename_vars({"Value": "RH_neg"})
    RH_Over100_ds = RH_Over100_ds.rename_vars({"Value": "RH_over100"})

    RH_Over100_ds = RH_Over100_ds.sel(time=~RH_Over100_ds.indexes['time'].duplicated())
    RH_Negative_ds = RH_Negative_ds.sel(time=~RH_Negative_ds.indexes['time'].duplicated())

    RH_Over100_ds = RH_Over100_ds.sel(lat=~RH_Over100_ds.indexes['lat'].duplicated())
    RH_Negative_ds = RH_Negative_ds.sel(lat=~RH_Negative_ds.indexes['lat'].duplicated())

    RH_Over100_ds = RH_Over100_ds.sel(lon=~RH_Over100_ds.indexes['lon'].duplicated())
    RH_Negative_ds = RH_Negative_ds.sel(lon=~RH_Negative_ds.indexes['lon'].duplicated())

    RH_anomaly_ds = xr.combine_by_coords([RH_Over100_ds, RH_Negative_ds], compat='override')

    return RH_anomaly_ds


# %% find light at night (LAN) and no light at day (NLAD) v2.0
def has_sunlight(latitude: xr.DataArray, longitude: xr.DataArray, time: List[datetime]):
    timestamps = [pd.Timestamp(t) for t in time]
    datetimes = [ts.to_pydatetime() for ts in timestamps]
    datetimes = [time.replace(tzinfo=timezone.utc) for time in datetimes if time.tzinfo is None]

    dates = sorted(list({t.date() for t in datetimes}))
    frames = []

    def get_dawn(lat, lon, d):

        try:
            return dawn(LocationInfo(name="", region="", timezone="UTC", latitude=lat, longitude=lon,
                                     ).observer, date=d, depression=-2)

        except ValueError as e:
            # hack to fix precision errors for CONUS
            return get_dawn(lat, lon + 0.5, d)

    def get_dusk(lat, lon, d):

        try:
            return dusk(LocationInfo(name="", region="", timezone="UTC", latitude=lat, longitude=lon,
                                     ).observer, date=d, depression=-2)

        except ValueError as e:
            # hack to fix precision errors for CONUS
            return get_dusk(lat, lon + 0.5, d)

    for date in dates:

        today_dawn = xr.apply_ufunc(get_dawn, latitude, longitude, date, vectorize=True)
        today_dusk = xr.apply_ufunc(get_dusk, latitude, longitude, date, vectorize=True)

        for t in [t for t in datetimes if t.date() == date]:

            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)

            frames.append(xr.apply_ufunc(
                lambda tdawn, tdusk, t: ((t <= tdusk) or (t >= tdawn)) if tdawn > tdusk else (tdawn <= t <= tdusk),
                today_dawn, today_dusk, t, vectorize=True).values)

    return xr.DataArray(np.stack(frames, axis=2).transpose((2, 0, 1)), dims=["time", "lat", "lon"])


def find_LAN_NLAD(ds, ds_variables):
    if "SWDOWN" in ds_variables:

        has_light = has_sunlight(ds.lat, ds.lon, ds.time.values)

        LAN = ds["SWDOWN"].where(has_light.values == False).where(ds["SWDOWN"] >= 50)
        NLAD = ds["SWDOWN"].where(has_light.values == True).where(ds["SWDOWN"] == 0)

    else:
        LAN, NLAD, has_light = (None,) * 3

    return LAN, NLAD, has_light


def LAN_NLAD_storage(ds, ds_variables, LAN, NLAD, has_light):
    if "SWDOWN" in ds_variables:

        LAN_dict = {"LAN": np.where(LAN.notnull())}
        NLAD_dict = {"NLAD": np.where(NLAD.notnull())}

        val_LAN = ds["SWDOWN"].values[tuple(LAN_dict["LAN"])]
        time_LAN = ds["SWDOWN"].time[tuple(LAN_dict["LAN"])[0]]
        lat_LAN = ds["SWDOWN"].south_north[tuple(LAN_dict["LAN"])[1]]
        lon_LAN = ds["SWDOWN"].west_east[tuple(LAN_dict["LAN"])[2]]
        haslight_LAN = has_light.values[tuple(LAN_dict["LAN"])]

        val_NLAD = ds["SWDOWN"].values[tuple(NLAD_dict["NLAD"])]
        time_NLAD = ds["SWDOWN"].time[tuple(NLAD_dict["NLAD"])[0]]
        lat_NLAD = ds["SWDOWN"].south_north[tuple(NLAD_dict["NLAD"])[1]]
        lon_NLAD = ds["SWDOWN"].west_east[tuple(NLAD_dict["NLAD"])[2]]
        haslight_NLAD = has_light.values[tuple(NLAD_dict["NLAD"])]

        LAN_time_list = list(time_LAN.values)
        LAN_lat_list = list(lat_LAN.values)
        LAN_lon_list = list(lon_LAN.values)
        LAN_val_list = list(val_LAN)
        haslight_LAN_list = list(haslight_LAN)

        NLAD_time_list = list(time_NLAD.values)
        NLAD_lat_list = list(lat_NLAD.values)
        NLAD_lon_list = list(lon_NLAD.values)
        NLAD_val_list = list(val_NLAD)
        haslight_NLAD_list = list(haslight_NLAD)

        LAN_dict_i = {"Time": [], "Lat": [], "Lon": [], "has_light": [], "Value": []}
        for i in range(len(LAN_val_list)):
            LAN_dict_i["Time"].append(LAN_time_list[i])
            LAN_dict_i["Lat"].append(LAN_lat_list[i])
            LAN_dict_i["Lon"].append(LAN_lon_list[i])
            LAN_dict_i["has_light"].append(haslight_LAN_list[i])
            LAN_dict_i["Value"].append(LAN_val_list[i])

        NLAD_dict_i = {"Time": [], "Lat": [], "Lon": [], "has_light": [], "Value": []}
        for i in range(len(NLAD_val_list)):
            NLAD_dict_i["Time"].append(NLAD_time_list[i])
            NLAD_dict_i["Lat"].append(NLAD_lat_list[i])
            NLAD_dict_i["Lon"].append(NLAD_lon_list[i])
            NLAD_dict_i["has_light"].append(haslight_NLAD_list[i])
            NLAD_dict_i["Value"].append(NLAD_val_list[i])

        LAN_df = pd.DataFrame(LAN_dict_i)
        NLAD_df = pd.DataFrame(NLAD_dict_i)

        return LAN_df, NLAD_df


# %% find zeros where 0 should not occur
def find_zeros(ds, ds_variables,
               checklist=["T2", "PSFC", "RH"]):
    vars = [checklist[i] for i in range(len(checklist)) if checklist[i] in ds_variables]

    zeros_ds = ds[vars].where(ds[vars] == 0)

    return zeros_ds, vars


def zeros_storage(ds, zeros_ds, vars):
    zeros_dict = {i: np.where(zeros_ds[i].notnull()) for i in vars}

    has_zeros_vars = [var for var in vars if len(zeros_dict[var][0] > 0)]

    zeros_df_list = []

    for var in has_zeros_vars:

        val_zero = ds[var].values[tuple(zeros_dict[var])]
        time_zero = ds[var].time[tuple(zeros_dict[var])[0]]
        lat_zero = ds[var].south_north[tuple(zeros_dict[var])[1]]
        lon_zero = ds[var].west_east[tuple(zeros_dict[var])[2]]

        time_list = list(time_zero.values)
        lat_list = list(lat_zero.values)
        lon_list = list(lon_zero.values)
        val_list = list(val_zero)

        dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}

        for i in range(len(val_list)):
            dict["Time"].append(time_list[i])
            dict["Lat"].append(lat_list[i])
            dict["Lon"].append(lon_list[i])
            dict["Value"].append(val_list[i])

        zeros_df = pd.DataFrame(dict)

        zeros_df_list.append(zeros_df)

    zeros_df_dict = {has_zeros_vars[i]: zeros_df_list[i] for i in range(len(has_zeros_vars))}

    return zeros_df_dict


# %% find negatives where negative values should not occur
def find_negatives(ds, ds_variables,
                   checklist=["LU_INDEX", "T2", "PSFC", "SFROFF", "UDROFF", "ACSNOM", "SNOW", "SNOWH",
                              "RAINC", "RAINSH", "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SWDOWN"]):
    vars = [checklist[i] for i in range(len(checklist)) if checklist[i] in ds_variables]

    negatives_ds = ds[vars].where(ds[vars] < 0)

    return negatives_ds, vars


def negatives_storage(ds, negatives_ds, vars):
    negatives_dict = {i: np.where(negatives_ds[i].notnull()) for i in vars}

    has_negatives_vars = [var for var in vars if len(negatives_dict[var][0] > 0)]

    negatives_df_list = []

    for var in has_negatives_vars:

        val_negative = ds[var].values[tuple(negatives_dict[var])]
        time_negative = ds[var].time[tuple(negatives_dict[var])[0]]
        lat_negative = ds[var].south_north[tuple(negatives_dict[var])[1]]
        lon_negative = ds[var].west_east[tuple(negatives_dict[var])[2]]

        time_list = list(time_negative.values)
        lat_list = list(lat_negative.values)
        lon_list = list(lon_negative.values)
        val_list = list(val_negative)

        dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}

        for i in range(len(val_list)):
            dict["Time"].append(time_list[i])
            dict["Lat"].append(lat_list[i])
            dict["Lon"].append(lon_list[i])
            dict["Value"].append(val_list[i])

        negatives_df = pd.DataFrame(dict)

        negatives_df_list.append(negatives_df)

    negatives_df_dict = {has_negatives_vars[i]: negatives_df_list[i] for i in range(len(has_negatives_vars))}

    return negatives_df_dict


def combine_dataframes_to_xarray(LAN_df, NLAD_df, zeros_df_dict, negatives_df_dict):
    # Create a dictionary to hold the data arrays
    data_dict = {}

    # Convert the dataframes to xarray data arrays
    for key, value in zeros_df_dict.items():
        data_dict[f"zeros_{key}"] = xr.DataArray(value["Value"], coords=[value["Time"], value["Lat"], value["Lon"]],
                                                 dims=["time", "lat", "lon"])

    for key, value in negatives_df_dict.items():
        data_dict[f"negatives_{key}"] = xr.DataArray(value["Value"], coords=[value["Time"], value["Lat"], value["Lon"]],
                                                     dims=["time", "lat", "lon"])

    data_dict["LAN"] = xr.DataArray(LAN_df["Value"], coords=[LAN_df["Time"], LAN_df["Lat"], LAN_df["Lon"]],
                                    dims=["time", "lat", "lon"])
    data_dict["NLAD"] = xr.DataArray(NLAD_df["Value"], coords=[NLAD_df["Time"], NLAD_df["Lat"], NLAD_df["Lon"]],
                                     dims=["time", "lat", "lon"])

    # Combine the data arrays into a single xarray dataset
    anomaly_ds = xr.Dataset(data_dict)

    return anomaly_ds


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
def WRFstats(input_path, output_path, start, stop, descriptive=True, distribution=True, outliers=True, anomalies=True,
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
                        "GLW", "UST", "SNOWC", "SR", 'T2F', 'T2C', 'PRECIP', 'WINDSPEED', 'RH']

    stats_list = []

    # create list of range of months to open
    months = pd.date_range(start, stop, freq="MS").strftime("%Y-%m").tolist()
    dt64 = [np.datetime64(m, "ns") for m in months]  # convert months to datetime64[ns]

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
            iqr_ds = IQR_Test(ds, ds_variables, iqr_threshold=3)

            # run z-score test
            z_outlier_ds = ZScore_Test(ds, ds_variables, z_threshold=4)

        # find specific anomalies in the dataset
        if anomalies == True:
            # check for RH anomalies
            RH_anomaly_ds = RH_xarray_storage(ds, ds_variables)

            # calculate dusk/dawn times and find occurrences of light at night (LAN) and no light at day (NLAD)
            LAN, NLAD, has_light = find_LAN_NLAD(ds, ds_variables)
            LAN_df, NLAD_df = LAN_NLAD_storage(ds, ds_variables, LAN, NLAD, has_light)

            # find occurrences of zero where they should not occur
            zeros_ds, vars = find_zeros(ds, ds_variables)
            zeros_df_dict = zeros_storage(ds, zeros_ds, vars)

            # find occurrences of negatives where they should not occur
            negatives_ds, vars = find_negatives(ds, ds_variables)
            negatives_df_dict = negatives_storage(ds, negatives_ds, vars)

            anomaly_ds = combine_dataframes_to_xarray(LAN_df, NLAD_df, zeros_df_dict, negatives_df_dict)

        # concatenate stats into dictionary and save as numpy dict
        stats_combined = xr.merge(
            [all_stats, sw_ds, skew_ds, normality, kurtosis_ds, iqr_ds, z_outlier_ds, RH_anomaly_ds, anomaly_ds])

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


if __name__ == "__main__":

    input_path = "/global/cfs/cdirs/m2702/gsharing/tgw-wrf-conus/historical_1980_2019/hourly/"
    output_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/"

    if len(sys.argv) > 1:
        year = sys.argv[1]
        start = f'{year}-01'
        stop = f'{year}-12'

    else:
        start = "2007-01"
        stop = "2007-12"

    # run the WRFstats program
    WRFstats(input_path, output_path, start, stop)
