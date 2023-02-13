import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import zscore


#%% outlier detection with IQR test
def IQR_Test(ds, ds_variables, iqr_threshold=1.5):
    
    q75_ds = ds[ds_variables].quantile(q = 0.75, dim = "time", skipna = "True").astype("float32")
    q25_ds = ds[ds_variables].quantile(q = 0.25, dim = "time", skipna = "True").astype("float32")
    
    iqr_ds = q75_ds - q25_ds
    IQR_val = iqr_threshold * iqr_ds
    iqr_upper_threshold = q75_ds + IQR_val
    iqr_lower_threshold = q25_ds - IQR_val
    
    iqr_outlier_upper = ds[ds_variables].where(ds[ds_variables] > (iqr_upper_threshold))
    iqr_outlier_lower = ds[ds_variables].where(ds[ds_variables] < (iqr_lower_threshold))
    
    return iqr_ds, q75_ds, q25_ds, iqr_upper_threshold, iqr_lower_threshold, iqr_outlier_upper, iqr_outlier_lower


#%% z-score outlier test
def ZScore_Test(ds, ds_variables, z_threshold=3):
    
    def ZS_func(ds_var):
        
        z = zscore(ds_var, axis = 0, nan_policy = "omit")
        
        return np.array([z])
    
    
    z_list = []
    
    for ds_var in ds_variables:
        
        zscore_test = xr.apply_ufunc(ZS_func, ds[ds_var], input_core_dims = [["time"]], output_core_dims = [["time"]],
                                     vectorize = True, output_dtypes = [np.dtype("float32")])

        z_list.append(zscore_test)

    zscore_ds = xr.merge(z_list)
    
    z_outlier_upper = ds[ds_variables].where(zscore_ds[ds_variables] > z_threshold)
    z_outlier_lower = ds[ds_variables].where(zscore_ds[ds_variables] < -z_threshold)
        
    return zscore_ds, z_outlier_upper, z_outlier_lower, z_threshold


#%% pandas iqr outlier storage function
def iqr_outlier_storage(ds, ds_variables, iqr_outlier_upper, iqr_outlier_lower, iqr_upper_threshold, iqr_lower_threshold):
    
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
        
        dict_upper = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "Q75": [], "QDiff": []}
        dict_lower = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "Q25": [], "QDiff": []}
        
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
        
        iqr_outliers_df = pd.concat([iqr_outlier_upper_df, iqr_outlier_lower_df], ignore_index = True)
        
        iqr_outlier_df_list.append(iqr_outliers_df)
    
    iqr_outlier_df_dict = {ds_variables[i]: iqr_outlier_df_list[i] for i in range(len(ds_variables))}
    
    return iqr_outlier_df_dict


#%% pandas z outlier storage function
def z_outlier_storage(ds, ds_variables, zscore_ds, z_outlier_upper, z_outlier_lower, z_threshold):
    
    z_outlier_upper_dict = {i: np.where(z_outlier_upper[i].notnull()) for i in ds_variables}
    z_outlier_lower_dict = {i: np.where(z_outlier_lower[i].notnull()) for i in ds_variables}
    
    z_outlier_df_list = []
    
    for var in ds_variables:
    
        # reshape dict tuple to zscore_ds shape
        arr0 = tuple(z_outlier_upper_dict[var])[0],
        arr1 = tuple(z_outlier_upper_dict[var])[1],
        arr2 = tuple(z_outlier_upper_dict[var])[2],
        z_outlier_upper_dict_reshaped = arr1 + arr2 + arr0
    
        z_val_upper_outliers = ds[var].values[tuple(z_outlier_upper_dict[var])]
        z_time_upper_outliers = ds[var].time[tuple(z_outlier_upper_dict[var])[0]]
        z_lat_upper_outliers = ds[var].south_north[tuple(z_outlier_upper_dict[var])[1]]
        z_lon_upper_outliers = ds[var].west_east[tuple(z_outlier_upper_dict[var])[2]]
        zscore_upper_outliers = zscore_ds[var].values[(z_outlier_upper_dict_reshaped)]
        z_diff_upper_outliers = zscore_upper_outliers - z_threshold
    
        # reshape dict tuple to zscore_ds shape
        arr0 = tuple(z_outlier_lower_dict[var])[0],
        arr1 = tuple(z_outlier_lower_dict[var])[1],
        arr2 = tuple(z_outlier_lower_dict[var])[2],
        z_outlier_lower_dict_reshaped = arr1 + arr2 + arr0
    
        z_val_lower_outliers = ds[var].values[tuple(z_outlier_lower_dict[var])]
        z_time_lower_outliers = ds[var].time[tuple(z_outlier_lower_dict[var])[0]]
        z_lat_lower_outliers = ds[var].south_north[tuple(z_outlier_lower_dict[var])[1]]
        z_lon_lower_outliers = ds[var].west_east[tuple(z_outlier_lower_dict[var])[2]]
        zscore_lower_outliers = zscore_ds[var].values[(z_outlier_lower_dict_reshaped)]
        z_diff_lower_outliers = (-z_threshold) - zscore_lower_outliers
    
        z_time_upper_list = list(z_time_upper_outliers.values)
        z_lat_upper_list = list(z_lat_upper_outliers.values)
        z_lon_upper_list = list(z_lon_upper_outliers.values)
        z_value_upper_list = list(z_val_upper_outliers)
        zscore_upper_list = list(zscore_upper_outliers)
        z_diff_upper_list = list(z_diff_upper_outliers)
    
        z_time_lower_list = list(z_time_lower_outliers.values)
        z_lat_lower_list = list(z_lat_lower_outliers.values)
        z_lon_lower_list = list(z_lon_lower_outliers.values)
        z_value_lower_list = list(z_val_lower_outliers)
        zscore_lower_list = list(zscore_lower_outliers)
        z_diff_lower_list = list(z_diff_lower_outliers)
    
        z_dict_upper = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "Z-Score": [], "Z-Threshold": [], "Z-Diff": []}
        z_dict_lower = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "Z-Score": [], "Z-Threshold": [], "Z-Diff": []}
    
        for i in range(len(z_value_upper_list)):
            z_dict_upper["Time"].append(z_time_upper_list[i])
            z_dict_upper["Lat"].append(z_lat_upper_list[i])
            z_dict_upper["Lon"].append(z_lon_upper_list[i])
            z_dict_upper["Value"].append(z_value_upper_list[i])
            z_dict_upper["Z-Score"].append(zscore_upper_list[i])
            z_dict_upper["Z-Threshold"].append(z_threshold)
            z_dict_upper["Z-Diff"].append(z_diff_upper_list[i])
    
        for i in range(len(z_value_lower_list)):
            z_dict_lower["Time"].append(z_time_lower_list[i])
            z_dict_lower["Lat"].append(z_lat_lower_list[i])
            z_dict_lower["Lon"].append(z_lon_lower_list[i])
            z_dict_lower["Value"].append(z_value_lower_list[i])
            z_dict_lower["Z-Score"].append(zscore_lower_list[i])
            z_dict_lower["Z-Threshold"].append(-z_threshold)
            z_dict_lower["Z-Diff"].append(z_diff_lower_list[i])
    
        z_outlier_upper_df = pd.DataFrame(z_dict_upper)
        z_outlier_lower_df = pd.DataFrame(z_dict_lower)
    
        z_outliers_df = pd.concat([z_outlier_upper_df, z_outlier_lower_df], ignore_index = True)
        
        z_outlier_df_list.append(z_outliers_df)
        
    z_outlier_df_dict = {ds_variables[i]: z_outlier_df_list[i] for i in range(len(ds_variables))}
        
    return z_outlier_df_dict


