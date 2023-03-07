import xarray as xr
import numpy as np
from scipy.stats import shapiro, kurtosis, skew
from typing import List, Tuple, Dict, Union, Iterable


def rename_stats(mean_ds: xr.Dataset, median_ds: xr.Dataset, stddev_ds: xr.Dataset, max_ds: xr.Dataset,
                 min_ds: xr.Dataset, ds_variables: List[str]) -> xr.Dataset:
    """
    Rename the xarray variables for mean, median, min, max and standard deviation.

    Parameters
    ----------
    mean_ds : xarray.Dataset
        Dataset containing the mean values of the input variables.
    median_ds : xarray.Dataset
        Dataset containing the median values of the input variables.
    stddev_ds : xarray.Dataset
        Dataset containing the standard deviation values of the input variables.
    max_ds : xarray.Dataset
        Dataset containing the maximum values of the input variables.
    min_ds : xarray.Dataset
        Dataset containing the minimum values of the input variables.
    ds_variables : list of str
        List of variable names to be renamed.

    Returns
    -------
    all_stats : xarray.Dataset
        Dataset containing all the statistical values with the variable names modified to include suffixes indicating
        the statistic type.

    Notes
    -----
    The suffixes used to indicate the statistical values are '_mean', '_med', '_std', '_max', and '_min'.
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


def descriptive_stats(ds: xr.Dataset, ds_variables: List[str]) -> xr.Dataset:
    """
    Calculate mean, median, minimum, maximum and standard deviation of the input variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the input variables to calculate statistics for.
    ds_variables : list of str
        List of variable names to calculate statistics for.

    Returns
    -------
    stats_all : xarray.Dataset
        Dataset containing all the statistical values with the variable names modified to include suffixes indicating
        the statistic type.

    Notes
    -----
    Missing or NaN values are ignored when calculating statistics.

    The returned dataset contains the following variables for each input variable: '_mean', '_med', '_std', '_max',
    and '_min'.
    """
    mean_ds = ds[ds_variables].mean(dim="time", skipna=True)
    median_ds = ds[ds_variables].median(dim="time", skipna=True)
    stddev_ds = ds[ds_variables].std(dim="time", skipna=True)
    max_ds = ds[ds_variables].max(dim="time", skipna=True)
    min_ds = ds[ds_variables].min(dim="time", skipna=True)

    all_stats = rename_stats(mean_ds, median_ds, stddev_ds, max_ds, min_ds, ds_variables)

    return all_stats


def rename_skew(skew_ds: xr.Dataset, kurtosis_ds: xr.Dataset, ds_variables: List[str]) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Rename the skewness and kurtosis variables within the xarray.

    Parameters
    ----------
    skew_ds : xarray.Dataset
        Dataset containing the skewness values of the input variables.
    kurtosis_ds : xarray.Dataset
        Dataset containing the kurtosis values of the input variables.
    ds_variables : list of str
        List of variable names to be renamed.

    Returns
    -------
    Tuple of xr.Dataset
        Tuple containing the skewness and kurtosis datasets with the variable names modified to include suffixes
        indicating the statistic type.

    Notes
    -----
    The suffixes used to indicate the skewness and kurtosis are '_skew' and '_kurt', respectively.
    """
    rename_dict = {var: f"{var}_skew" for var in ds_variables}
    skew_ds = skew_ds.rename(rename_dict)

    rename_dict = {var: f"{var}_kurt" for var in ds_variables}
    kurtosis_ds = kurtosis_ds.rename(rename_dict)

    return skew_ds, kurtosis_ds


def skew_func(ds_var: xr.DataArray) -> np.ndarray:
    """
    Calculate the skewness of an xarray DataArray along the time dimension.

    Parameters
    ----------
    ds_var : xarray.DataArray
        DataArray containing the input variable.

    Returns
    -------
    np.ndarray
        ndarray containing the skewness values of the input variable.
    """
    skewness = skew(ds_var, axis=0, nan_policy="omit", keepdims=True)

    return skewness


def kurtosis_func(ds_var: xr.DataArray) -> Union[np.ndarray, Iterable, int, float]:
    """
    Calculate the kurtosis of an xarray DataArray along the time dimension.

    Parameters
    ----------
    ds_var : xarray.DataArray
        DataArray containing the input variable.

    Returns
    -------
    xarray.DataArray
        DataArray containing the kurtosis values of the input variable.
    """
    kurtosisness = kurtosis(ds_var, axis=0, nan_policy="omit", keepdims=True)

    return kurtosisness


def skew_kurtosis_test(ds: xr.Dataset, ds_variables: List[str]) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate skewness and kurtosis values for each variable in the input dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to calculate statistics for.
    ds_variables : list of str
        List of variable names to calculate statistics for.

    Returns
    -------
    Tuple of xr.Dataset
        Tuple containing the skewness and kurtosis datasets with the variable names modified to include suffixes
        indicating the statistic type.

    Notes
    -----
    This function calculates the skewness and kurtosis for each variable along the time dimension.

    The returned dataset contains the following variables for each input variable: '_skew', and '_kurt'.

    Missing or NaN values are ignored when calculating statistics.
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


def sw_func(ds_var: np.ndarray) -> np.ndarray:
    """
    Calculate the Shapiro-Wilks test for a single variable.

    Parameters
    ----------
    ds_var : numpy.ndarray
        Array containing the data for the variable to test.

    Returns
    -------
    numpy.ndarray
        Array containing the test statistic and p-value for the Shapiro-Wilks test.

    Notes
    -----
    This function calculates the Shapiro-Wilks test for a single variable along the time dimension.
    """
    teststat, p = shapiro(ds_var)

    return np.array([[teststat, p]])


def sw_test(ds: xr.Dataset, ds_variables: list) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate the Shapiro-Wilks test for each variable in the input dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variables to calculate statistics for.
    ds_variables : list of str
        List of variable names to calculate statistics for.

    Returns
    -------
    Tuple of xr.Dataset and xr.Dataset
        Tuple containing the results of the Shapiro-Wilks test and the normality values for each variable with
        modified variable names.

    Notes
    -----
    This function calculates the Shapiro-Wilks test for each variable along the time dimension.

    The returned dataset contains the Shapiro-Wilks test results for each input variable with the variable names
    modified to include the suffix "_sw".

    The returned dataset contains the percent of normality for each variable with modified variable names.

    Missing or NaN values are ignored when calculating statistics.
    """
    pval_list = []
    normality_list = []

    for ds_var in ds_variables:
        shapiro_test = xr.apply_ufunc(sw_func, ds[ds_var], input_core_dims=[["time"]], output_core_dims=[["output"]],
                                      vectorize=True, output_dtypes=[np.dtype("float32")])

        p = shapiro_test.isel(output=1)

        pval_list.append(p)

        percent_normal = (p.values > 0.05).sum() / (p.values >= 0).sum()
        normality_list.append(percent_normal)

    sw_ds = xr.merge(pval_list)
    sw_ds, normality_ds = rename_sw(sw_ds, ds_variables, normality_list)

    return sw_ds, normality_ds


def rename_sw(sw_ds, ds_variables, normality_list):
    """
    Rename variables in the input dataset to include "_sw" suffix and add percent normality as a new variable.

    Parameters
    ----------
    sw_ds : xr.Dataset
        Dataset containing the Shapiro-Wilks test results.
    ds_variables : list of str
        List of variable names.
    normality_list : list of float
        List of percent normality for each variable.

    Returns
    -------
    Tuple of xr.Dataset and xr.Dataset
        Tuple containing the modified dataset with variable names and the normality values for each variable with
        modified variable names.
    """
    new_var_names = [var + "_sw" for var in ds_variables]
    sw_ds = sw_ds.rename({var: new_var_name for var, new_var_name in zip(ds_variables, new_var_names)})

    normality_ds = xr.Dataset({"percent_normality": (["variable"], normality_list)}, coords={"variable": new_var_names})

    return sw_ds, normality_ds