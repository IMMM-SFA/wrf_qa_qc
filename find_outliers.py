import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import zscore
from astral import LocationInfo
from astral.sun import dusk, dawn
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict


def iqr_test(ds: xr.Dataset, ds_variables: List[str], iqr_threshold: int = 3) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate outliers using the interquartile range (IQR) test for each variable in the input dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to calculate outliers for.
    ds_variables : List of str
        List of variable names to calculate outliers for.
    iqr_threshold : int, optional
        The IQR threshold used to define outliers. Default is 3.

    Returns
    -------
    Tuple of xr.Dataset
        Tuple containing the IQR statistics dataset and the outliers dataset for each input variable.

    Notes
    -----
    This function calculates the interquartile range (IQR) for each variable along the time dimension.

    Outliers are defined as values that are above the upper threshold or below the lower threshold.

    The returned datasets contain the following variables for each input variable:
    - '<variable_name>_iqr': The interquartile range (IQR)
    - '<variable_name>_q75': The 75th percentile
    - '<variable_name>_q25': The 25th percentile
    - '<variable_name>_upper_thresh': The upper threshold
    - '<variable_name>_lower_thresh': The lower threshold
    - '<variable_name>_upper_outliers': The upper outliers (values above the upper threshold)
    - '<variable_name>_lower_outliers': The lower outliers (values below the lower threshold)

    Missing or NaN values are skipped when calculating percentiles and thresholds.
    """
    q75_ds = ds[ds_variables].quantile(q=0.75, dim="time", skipna="True").astype("float32")
    q25_ds = ds[ds_variables].quantile(q=0.25, dim="time", skipna="True").astype("float32")

    iqr_ds = q75_ds - q25_ds
    IQR_val = iqr_threshold * iqr_ds
    iqr_upper_threshold = q75_ds + IQR_val
    iqr_lower_threshold = q25_ds - IQR_val

    iqr_outlier_upper = ds[ds_variables].where(ds[ds_variables] > iqr_upper_threshold)
    iqr_outlier_lower = ds[ds_variables].where(ds[ds_variables] < iqr_lower_threshold)

    iqr_ds = iqr_ds.rename({var: f"{var}_iqr" for var in ds_variables})
    q75_ds = q75_ds.rename({var: f"{var}_q75" for var in ds_variables})
    q25_ds = q25_ds.rename({var: f"{var}_q25" for var in ds_variables})
    iqr_upper_threshold = iqr_upper_threshold.rename({var: f"{var}_upper_thresh" for var in ds_variables})
    iqr_lower_threshold = iqr_lower_threshold.rename({var: f"{var}_lower_thresh" for var in ds_variables})
    iqr_outlier_upper = iqr_outlier_upper.rename({var: f"{var}_upper_outliers" for var in ds_variables})
    iqr_outlier_lower = iqr_outlier_lower.rename({var: f"{var}_lower_outliers" for var in ds_variables})

    q75_ds = q75_ds.reset_coords("quantile")
    q25_ds = q25_ds.reset_coords("quantile")
    iqr_lower_threshold = iqr_lower_threshold.reset_coords("quantile")
    iqr_upper_threshold = iqr_upper_threshold.reset_coords("quantile")
    iqr_outlier_lower = iqr_outlier_lower.reset_coords("quantile")
    iqr_outlier_upper = iqr_outlier_upper.reset_coords("quantile")

    iqr_stats_ds = xr.merge([iqr_ds, q75_ds, q25_ds, iqr_upper_threshold, iqr_lower_threshold], compat='override')
    iqr_outliers_ds = xr.merge([iqr_outlier_upper, iqr_outlier_lower], compat='override')

    iqr_stats_ds = iqr_stats_ds.drop_vars("quantile")
    iqr_outliers_ds = iqr_outliers_ds.drop_vars("quantile")

    return iqr_stats_ds, iqr_outliers_ds


def zscore_test(ds: xr.Dataset, ds_variables: list, z_threshold: int = 4) -> xr.Dataset:
    """
    Calculate outliers using the Z-score test for each variable in the input dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to calculate outliers for.
    ds_variables : list of str
        List of variable names to calculate outliers for.
    z_threshold : int, optional
        The Z-score threshold used to define outliers. Default is 4.

    Returns
    -------
    xr.Dataset
        Dataset containing the Z-score and the upper and lower outliers for each variable.

    Notes
    -----
    This function calculates the Z-score for each variable along the time dimension.

    Outliers are defined as values that are above the upper threshold or below the lower threshold.

    The returned dataset contains the following variables for each input variable:
    - 'zscore': The Z-score
    - 'z_outlier_upper': The upper outliers (values above the upper threshold)
    - 'z_outlier_lower': The lower outliers (values below the lower threshold)

    Missing or NaN values are skipped when calculating the Z-score and outliers.
    """
    zscore_ds = xr.Dataset()

    for ds_var in ds_variables:
        # Compute the Z-score
        z = xr.apply_ufunc(
            zscore,
            ds[ds_var],
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[np.dtype("float32")],
            kwargs={"nan_policy": "omit"},
        )

        # Compute the upper and lower outliers
        z_outliers = ds[ds_var].where(abs(z) > z_threshold)

        # Add the Z-score and outliers to the output dataset
        zscore_ds[f"{ds_var}_zscore"] = z
        zscore_ds[f"{ds_var}_z_outlier_upper"] = z_outliers.where(z > z_threshold)
        zscore_ds[f"{ds_var}_z_outlier_lower"] = z_outliers.where(z < -z_threshold)

    zscore_ds.attrs["z_threshold"] = z_threshold
    return zscore_ds


def rh_anomaly(ds, ds_variables, rh_threshold=1.15, rh_threshold_neg=-0.015) -> xr.Dataset:
    """
    Calculate the anomalies in relative humidity from the given dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to calculate anomalies for.
    ds_variables : list of str
        List of variable names to calculate anomalies for.
    rh_threshold : float, optional
        The threshold used to define RH values greater than 100%. Default is 1.15.
    rh_threshold_neg : float, optional
        The threshold used to define RH values less than 0%. Default is -0.015.

    Returns
    -------
    xr.Dataset
        Dataset containing the relative humidity anomalies.

    Notes
    -----
    This function calculates the anomalies in relative humidity (RH) greater than 100% and less than 0%.

    Outliers are defined as values that are above the upper threshold or below the lower threshold.

    The returned dataset contains the following variables:
    - 'RH_over100': The RH anomalies greater than 100%
    - 'RH_neg': The RH anomalies less than 0%

    The dataset is indexed by time, latitude, and longitude.

    Missing or NaN values are skipped when calculating RH anomalies.
    """

    if "RH" in ds_variables:
        rh_over100_ds = ds["RH"].where(ds["RH"] > rh_threshold).rename("RH_over100")
        rh_negative_ds = ds["RH"].where(ds["RH"] < rh_threshold_neg).rename("RH_neg")
        rh_anomaly_ds = xr.merge([rh_over100_ds, rh_negative_ds], join="outer")

    return rh_anomaly_ds


def has_sunlight(latitude: xr.DataArray, longitude: xr.DataArray, time: List[datetime]) -> xr.DataArray:
    """
    Determine whether a given location has sunlight at a specific time.

    Parameters
    ----------
    latitude : xr.DataArray
        Array of latitudes for each location.
    longitude : xr.DataArray
        Array of longitudes for each location.
    time : List[datetime]
        List of datetime objects representing the times to check.

    Returns
    -------
    xr.DataArray
        DataArray of boolean values indicating whether each location has sunlight at each time.
        The shape is (time, lat, lon).
    """
    timestamps = [pd.Timestamp(t) for t in time]
    datetimes = [ts.to_pydatetime() for ts in timestamps]
    datetimes = [time.replace(tzinfo=timezone.utc) for time in datetimes if time.tzinfo is None]

    dates = sorted(list({t.date() for t in datetimes}))
    frames = []

    def get_dawn(lat, lon, d):
        try:
            return dawn(LocationInfo(name="", region="", timezone="UTC", latitude=lat, longitude=lon).observer,
                        date=d, depression=-2)
        except ValueError as e:
            # hack to fix precision errors for CONUS
            return get_dawn(lat, lon + 0.5, d)

    def get_dusk(lat, lon, d):
        try:
            return dusk(LocationInfo(name="", region="", timezone="UTC", latitude=lat, longitude=lon).observer,
                        date=d, depression=-2)
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


def find_lan_nlad(ds: xr.Dataset, ds_variables: list) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Find locations with light at night (LAN) and no light at day (NLAD) in a given dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variables to check for LAN and NLAD.
    ds_variables : list of str
        List of variable names to check for LAN and NLAD.

    Returns
    -------
    Tuple of xr.DataArray
        Tuple containing the LAN DataArray, the NLAD DataArray, and the DataArray indicating whether each location
        has sunlight. The shape of each DataArray is (time, lat, lon). If the 'SWDOWN' variable is not present in the
        input dataset, returns None for all three DataArrays.
    """

    if "SWDOWN" in ds_variables:

        has_light = has_sunlight(ds.lat, ds.lon, ds.time.values)

        lan = ds["SWDOWN"].where(has_light.values == False).where(ds["SWDOWN"] >= 50)
        nlad = ds["SWDOWN"].where(has_light.values == True).where(ds["SWDOWN"] == 0)
        lan_xr = xr.Dataset().assign(SWDOWN_LAN=lan)
        nlad_xr = xr.Dataset().assign(SWDOWN_NLAD=nlad)
        light_anomaly_df = xr.merge([lan_xr, nlad_xr])


    else:
        None

    return light_anomaly_df


def find_zeros(ds: xr.Dataset, ds_variables: List[str], checklist: Optional[List[str]] = None) -> xr.Dataset:
    """
    Replaces non-zero values in the given variables in the dataset with NaN, and returns only the variables in the checklist.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset.
    ds_variables : List[str]
        A list of variables to search for zeros in.
    checklist : Optional[List[str]], default=None
        A list of variables to check for zeros in. If None, default value ["T2", "PSFC", "RH"] is used.

    Returns:
    --------
    xr.Dataset
        A copy of the input dataset with non-zero values in the specified variables replaced with NaN, and only the variables
        in the checklist.
    """

    if checklist is None:
        checklist = ["T2", "PSFC", "RH"]

    vars = [var_name for var_name in ds_variables if var_name in checklist]

    zeros_mask = ds[vars] == 0
    ds_with_nans = ds.copy()
    ds_with_nans[vars] = ds_with_nans[vars].where(zeros_mask, other=np.nan)
    zero_ds = ds_with_nans[vars]
    zero_ds = zero_ds.rename({var: f"{var}_has_zero" for var in vars})

    return zero_ds


def find_negatives(ds: xr.Dataset, ds_variables: List[str], checklist: Optional[List[str]] = None) -> xr.Dataset:
    """
    Replaces non-zero values in the given variables in the dataset with NaN, and returns only the variables in the checklist.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset.
    ds_variables : List[str]
        A list of variables to search for zeros in.
    checklist : Optional[List[str]], default=None
        A list of variables to check for zeros in. If None, default value ["T2", "PSFC", "RH"] is used.

    Returns:
    --------
    xr.Dataset
        A copy of the input dataset with non-zero values in the specified variables replaced with NaN, and only the variables
        in the checklist.
    """

    if checklist is None:
        checklist = ["LU_INDEX", "T2", "PSFC", "SFROFF", "UDROFF", "ACSNOM", "SNOW", "SNOWH", "RAINC", "RAINSH",
                     "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SWDOWN"]

    vars = [var_name for var_name in ds_variables if var_name in checklist]

    negative_mask = ds[vars] == 0
    ds_with_nans = ds.copy()
    ds_with_nans[vars] = ds_with_nans[vars].where(negative_mask, other=np.nan)
    negative_ds = ds_with_nans[vars]
    negative_ds = negative_ds.rename({var: f"{var}_has_negative" for var in vars})

    return negative_ds
