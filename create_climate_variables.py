import xarray as xr
import numpy as np
from typing import List


def temp_conv(ds: xr.Dataset, ds_variables: List[str], f: bool = True, c: bool = True) -> None:
    """
    Convert temperature from Kelvin to Fahrenheit or Celsius.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the temperature data to be converted.
    ds_variables : list of str
        List of temperature variable names in the dataset.
    f : bool, optional
        Whether to convert temperature to Fahrenheit. Default is True.
    c : bool, optional
        Whether to convert temperature to Celsius. Default is True.

    Returns
    -------
    None
    """
    if "T2" in ds_variables:
        K = ds["T2"]

        # convert to F
        if f:
            ds["T2F"] = 1.8 * (K - 273.15) + 32

        # convert to C
        if c:
            ds["T2C"] = K - 273.15


def deacc_precip(ds: xr.Dataset, ds_variables: List[str]) -> None:
    """
    Deaccumulate precipitation variables into a new variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the precipitation data to be deaccumulated.
    ds_variables : list of str
        List of precipitation variable names in the dataset.

    Returns
    -------
    None
    """
    # check if rain variables included in the variables list, if so then create PRECIP variable and deaccumulate
    if "RAINC" in ds_variables and "RAINSH" in ds_variables and "RAINNC" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis=0, prepend=np.array([ds["PRECIP"][0].values]))

    # deaccumulate rain variables
    if "RAINC" in ds_variables:
        ds["RAINC"].values = np.diff(ds["RAINC"].values, axis=0, prepend=np.array([ds["RAINC"][0].values]))

    if "RAINSH" in ds_variables:
        ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis=0, prepend=np.array([ds["RAINSH"][0].values]))

    if "RAINNC" in ds_variables:
        ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis=0, prepend=np.array([ds["RAINNC"][0].values]))


def windspeed(ds: xr.Dataset, ds_variables: List[str]) -> None:
    """
    Calculate the magnitude of wind velocity vectors from U and V components.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind data to calculate windspeed from.
    ds_variables : list of str
        List of wind variable names in the dataset.

    Returns
    -------
    None
    """
    if "U10" in ds_variables and "V10" in ds_variables:
        U = ds["U10"]
        V = ds["V10"]
        ds["WINDSPEED"] = np.sqrt(U ** 2 + V ** 2)


def rel_humidity(ds: xr.Dataset, ds_variables: List[str]) -> None:
    """
    Calculate relative humidity from temperature, pressure and specific humidity.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the atmospheric data to calculate relative humidity from.
    ds_variables : list of str
        List of variable names in the dataset, including "T2", "PSFC" and "Q2".

    Returns
    -------
    None

    Notes
    -----
    Relative humidity (RH) is calculated as the ratio of the mixing ratio (Q) to the saturation mixing ratio (Qs) at
    the same temperature and pressure. The saturation mixing ratio is calculated using the Clausius-Clapeyron equation.
    The RH values that exceed 100% or are negative are considered to be invalid.

    The expected range for RH values is between 0 and 1, but small deviations above 1 are allowed due to rounding
    errors or other factors.
    """
    if "T2" in ds_variables and "PSFC" in ds_variables and "Q2" in ds_variables:
        es = 6.112 * np.exp(17.67 * (ds["T2"] - 273.15) / (ds["T2"] - 29.65))
        qvs = 0.622 * es / (0.01 * ds["PSFC"] - (1.0 - 0.622) * es)
        rh = ds["Q2"] / qvs

        ds["RH"] = rh