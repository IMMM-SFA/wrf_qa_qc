import pandas as pd
import numpy as np
import xarray as xr
import salem as sl
from glob import glob
import os
from datetime import datetime, timedelta


#%% find previous month to current
def previous_month(year_month):
    
    year = year_month[0 : year_month.find("-")]
    month = year_month[year_month.rfind("-")+1 :]
    
    months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    idx = months_list.index(month)

    month_minus = months_list[idx - 1]
    
    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year
    
    previousmonth = year_minus + "-" + month_minus
    
    return previousmonth


#%% find next month to current
def next_month(year_month):

    currentdate = datetime.strptime(year_month, "%Y-%m")
    nextmonth = currentdate + timedelta(days=31)
    nextmonth_str = nextmonth.strftime("%Y-%m")

    return nextmonth_str


#%% function to convert T2 variable from K to F or C
def temp_conv(ds, ds_variables, F=True, C=True):
    
    if "T2" in ds_variables:
        
        K = ds["T2"]
        
        #convert to F
        if F == True:
            ds["T2F"] = 1.8 * (K - 273.15) + 32
            ds_variables.append("T2F")
        
        #convert to C
        if C == True:
            ds["T2C"] = K - 273.15
            ds_variables.append("T2C")


#%% function to combine and deaccumulate precipitation variables into new variable
def deacc_precip(ds, ds_variables):
    
    # check if rain variables included in the variables list, if so then create PRECIP variable and deaccumulate
    if "RAINC" in ds_variables and "RAINSH" in ds_variables and "RAINNC" in ds_variables:
        ds["PRECIP"] = ds["RAINC"] + ds["RAINSH"] + ds["RAINNC"]
        ds["PRECIP"].values = np.diff(ds["PRECIP"].values, axis = 0, prepend = np.array([ds["PRECIP"][0].values]))
        ds_variables.append("PRECIP")
        
    # deaccumulate rain variables
    if "RAINC" in ds_variables:
        ds["RAINC"].values = np.diff(ds["RAINC"].values, axis = 0, prepend = np.array([ds["RAINC"][0].values]))
    
    if "RAINSH" in ds_variables:
        ds["RAINSH"].values = np.diff(ds["RAINSH"].values, axis = 0, prepend = np.array([ds["RAINSH"][0].values]))
    
    if "RAINNC" in ds_variables:
        ds["RAINNC"].values = np.diff(ds["RAINNC"].values, axis = 0, prepend = np.array([ds["RAINNC"][0].values]))


#%% function for calculating magnitude of wind velocity vectors
def windspeed(ds, ds_variables):
    
    if "U10" in ds_variables and "V10" in ds_variables: 
    
        U = ds["U10"]
        V = ds["V10"]
        ds["WINDSPEED"] = np.sqrt( U**2 + V**2 )
        ds_variables.append("WINDSPEED")


#%% calculate relative humidity
def rel_humidity(ds, ds_variables):
    
    if "T2" in ds_variables and "PSFC" in ds_variables and "Q2" in ds_variables: 
    
        es = 6.112 * np.exp(17.67 * (ds["T2"] - 273.15) / (ds["T2"] - 29.65))
        qvs = 0.622 * es / (0.01 * ds["PSFC"] - (1.0 - 0.622) * es)
        rh = ds["Q2"] / qvs
        
        ds["RH"] = rh
        ds_variables.append("RH")


#%% storage function
def outlier_dict_storage(ds, ds_variables, dt64, i, month, outliers, 
                         outlier_upper_type, outlier_lower_type=None, 
                         upper_threshold=None, lower_threshold=None, 
                         reshape=False):
    
    outliers["time"] = np.sort(outliers["time"].values)
    outlier_upper = outliers.sel(time = slice(dt64[i], dt64[i+1]))
    outlier_upper_dict = {var: np.where(outlier_upper[f"{var}_{outlier_upper_type}"].notnull()) for var in ds_variables}
    
    if outlier_lower_type is not None:
        outlier_lower = outliers.sel(time = slice(dt64[i], dt64[i+1]))
        outlier_lower_dict = {var: np.where(outlier_lower[f"{var}_{outlier_lower_type}"].notnull()) for var in ds_variables}
    
    outlier_df_list = []
    
    for var in ds_variables:
        
        if reshape == True:
            # reshape dict tuple to zscore_ds shape
            arr0 = tuple(outlier_upper_dict[var])[0],
            arr1 = tuple(outlier_upper_dict[var])[1],
            arr2 = tuple(outlier_upper_dict[var])[2],
            outlier_upper_dict_reshaped = arr1 + arr2 + arr0
        
        val_upper_outliers = ds[var].values[tuple(outlier_upper_dict[var])]
        time_upper_outliers = ds[var].time[tuple(outlier_upper_dict[var])[0]]
        lat_upper_outliers = ds[var].south_north[tuple(outlier_upper_dict[var])[1]]
        lon_upper_outliers = ds[var].west_east[tuple(outlier_upper_dict[var])[2]]
        if upper_threshold is not None:
            # check if zscore
            if type(upper_threshold) is int:
                upper_threshold_outliers = outlier_upper[f"{var}_zscore"].values[(outlier_upper_dict_reshaped)]
                diff_upper_outliers = upper_threshold_outliers - upper_threshold
            else:
                upper_threshold_outliers = upper_threshold[f"{var}_upper_thresh"].values[tuple(outlier_upper_dict[var])[1:]]
                diff_upper_outliers = val_upper_outliers - upper_threshold_outliers
        
        if outlier_lower_type is not None:
            
            if reshape == True:
                # reshape dict tuple to zscore_ds shape
                arr0 = tuple(outlier_lower_dict[var])[0],
                arr1 = tuple(outlier_lower_dict[var])[1],
                arr2 = tuple(outlier_lower_dict[var])[2],
                outlier_lower_dict_reshaped = arr1 + arr2 + arr0
            
            val_lower_outliers = ds[var].values[tuple(outlier_lower_dict[var])]
            time_lower_outliers = ds[var].time[tuple(outlier_lower_dict[var])[0]]
            lat_lower_outliers = ds[var].south_north[tuple(outlier_lower_dict[var])[1]]
            lon_lower_outliers = ds[var].west_east[tuple(outlier_lower_dict[var])[2]]
            if lower_threshold is not None:
                # check if zscore
                if type(lower_threshold) is int:
                    lower_threshold_outliers = outlier_lower[f"{var}_zscore"].values[(outlier_lower_dict_reshaped)]
                    diff_lower_outliers = lower_threshold - lower_threshold_outliers
                else:
                    lower_threshold_outliers = lower_threshold[f"{var}_lower_thresh"].values[tuple(outlier_lower_dict[var])[1:]]
                    diff_lower_outliers = lower_threshold_outliers - val_lower_outliers
        
        time_upper_list = list(time_upper_outliers.values)
        lat_upper_list = list(lat_upper_outliers.values)
        lon_upper_list = list(lon_upper_outliers.values)
        value_upper_list = list(val_upper_outliers)
        if upper_threshold is not None:
            upper_threshold_list = list(upper_threshold_outliers)
            diff_upper_list = list(diff_upper_outliers)
        
        if outlier_lower_type is not None:
            time_lower_list = list(time_lower_outliers.values)
            lat_lower_list = list(lat_lower_outliers.values)
            lon_lower_list = list(lon_lower_outliers.values)
            value_lower_list = list(val_lower_outliers)
            if lower_threshold is not None:
                lower_threshold_list = list(lower_threshold_outliers)
                diff_lower_list = list(diff_lower_outliers)
        
        dict_upper = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "UpperThreshold": [], "UpperDiff": []}
        if outlier_lower_type is not None:
            dict_lower = {"Time": [], "Lat": [],  "Lon": [],  "Value": [], "LowerThreshold": [], "LowerDiff": []}
        
        for i in range(len(value_upper_list)):
            dict_upper["Time"].append(time_upper_list[i])
            dict_upper["Lat"].append(lat_upper_list[i])
            dict_upper["Lon"].append(lon_upper_list[i])
            dict_upper["Value"].append(value_upper_list[i])
            if upper_threshold is not None:
                dict_upper["UpperThreshold"].append(upper_threshold_list[i])
                dict_upper["UpperDiff"].append(diff_upper_list[i])
            else:
                dict_upper["UpperThreshold"].append("N/A")
                dict_upper["UpperDiff"].append("N/A")
        
        if outlier_lower_type is not None:
            for i in range(len(value_lower_list)):
                dict_lower["Time"].append(time_lower_list[i])
                dict_lower["Lat"].append(lat_lower_list[i])
                dict_lower["Lon"].append(lon_lower_list[i])
                dict_lower["Value"].append(value_lower_list[i])
                if lower_threshold is not None:
                    dict_lower["LowerThreshold"].append(lower_threshold_list[i])
                    dict_lower["LowerDiff"].append(diff_lower_list[i])
                else:
                    dict_lower["LowerThreshold"].append("N/A")
                    dict_lower["LowerDiff"].append("N/A")
        
        outlier_upper_df = pd.DataFrame(dict_upper)
        if outlier_lower_type is not None:
            outlier_lower_df = pd.DataFrame(dict_lower)
        
        if outlier_lower_type is not None:
            outliers_df = pd.concat([outlier_upper_df, outlier_lower_df], ignore_index = True)
        else:
            outliers_df = outlier_upper_df
        
        outlier_df_list.append(outliers_df)
    
    outlier_df_dict = {ds_variables[i]: outlier_df_list[i] for i in range(len(ds_variables))}
    
    return outlier_df_dict


#%% month loop
def WRFstats_Analysis(wrf_path, outlier_path, stats_path, output_path, start, stop, 
                      ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                                    "ZOL","RAINC","RAINSH","RAINNC","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
                      ):
    
    """
    Function for storing statistical outliers and anomalies as tables on monthly WRF netCDF data between a given range of months.
    
    Input
    ----------
    wrf_path : Str Path to netCDF files for analysis.
    outlier_path : Str Path to netCDF outliers file
    stats_path : Str Path to netCDF statistics file
    output_path : Str Path for the output netCDF files to be stored.
    start : Str "YYYY-MM" Start date of files to open. (inclusive)
    stop : Str "YYYY-MM" End date of files to open (NOT inclusive).
    ds_variables : List Variables to run stats on.
        
    Returns
    -------
    stats_list : List of datasets for storage of analysis output.
    
    """
    
    # create list of range of months to open
    months = pd.date_range(start, stop, freq="MS").strftime("%Y-%m").tolist()
    years = np.unique([month[:month.find("-")] for month in months[:-1]])
    dt64 = [np.datetime64(m, "ns") for m in months] # convert months to datetime64[ns]
    
    for year in years:
        
        outliers_list = []
        
        year_months = [ month for month in months[:-1] if month[:month.find("-")] == str(year) ]
        nextmonth = next_month(year_months[-1])
        year_months.append(nextmonth)
        dt64_year = [np.datetime64(y_m, "ns") for y_m in year_months]
        
        # iterate through each month and create dataset
        for i, month in enumerate(year_months):
            if i < 12:
                
                #open files
                #historical datasets
                nc_files = sorted(glob(os.path.join(wrf_path, f"tgw_wrf_historical_hourly_{month}*")))

                try:
                    previousmonth = previous_month(f"{month}")
                    previousmonth_lastfile = sorted(glob(os.path.join(wrf_path, f"tgw_wrf_historical_hourly_{previousmonth}*")))[-1]
                    nc_files.insert(0, previousmonth_lastfile)
                except:
                    pass

                ds = sl.open_mf_wrf_dataset(nc_files)
                ds["time"] = np.sort(ds["time"].values)
                ds = ds.sel(time = slice(dt64_year[i], dt64_year[i+1]))

                #outliers
                outlier_path = f"/global/cfs/projectdirs/m2702/gsharing/QAQC/historical/{year}/tgw_wrf_hourly_{month}_historical_all_outliers.nc"
                outliers = xr.open_dataset(outlier_path)

                #stats/thresholds
                stats_path = f"/global/cfs/projectdirs/m2702/gsharing/QAQC/historical/{year}/tgw_wrf_hourly_{month}_all_stats.nc"
                stats = xr.open_dataset(stats_path)


                # convert T2 variable from K to F or C
                temp_conv(ds, ds_variables)

                # combine and deaccumulate precipitation variables into PRECIP variable
                deacc_precip(ds, ds_variables)

                # create new variable WINDSPEED from magnitudes of velocity vectors
                windspeed(ds, ds_variables)

                # calculate relative humidity and create new variable RH
                rel_humidity(ds, ds_variables)

                #iqr outliers
                iqr_outliers_df = outlier_dict_storage(ds, ds_variables, dt64, i, month, outliers,
                                                       outlier_upper_type="upper_outliers", outlier_lower_type="lower_outliers",
                                                       upper_threshold=stats, lower_threshold=stats,
                                                       reshape=False
                                                       )

                #z outliers
                z_outliers_df = outlier_dict_storage(ds, ds_variables, dt64, i, month, outliers,
                                                     outlier_upper_type="z_outlier_upper", outlier_lower_type="z_outlier_lower",
                                                     upper_threshold=4, lower_threshold=-4,
                                                     reshape=True
                                                     )

                #has zero
                zero_outliers_df = outlier_dict_storage(ds, ["T2", "PSFC", "RH"], dt64, i, month, outliers,
                                                        outlier_upper_type="has_zero", outlier_lower_type=None,
                                                        upper_threshold=None, lower_threshold=None,
                                                        reshape=False
                                                        )

                #has negative
                ds_vars = ["LU_INDEX","T2","PSFC","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","RAINC","RAINSH","RAINNC","SNOWNC","GRAUPELNC","HAILNC","SWDOWN"]
                negative_outliers_df = outlier_dict_storage(ds, ds_vars, dt64, i, month, outliers,
                                                            outlier_upper_type="has_negative", outlier_lower_type=None,
                                                            upper_threshold=None, lower_threshold=None,
                                                            reshape=False
                                                            )

                #LAN
                LAN_outliers_df = outlier_dict_storage(ds, ["SWDOWN"], dt64, i, month, outliers,
                                                       outlier_upper_type="LAN", outlier_lower_type=None,
                                                       upper_threshold=None, lower_threshold=None,
                                                       reshape=False
                                                       )

                #NLAD
                NLAD_outliers_df = outlier_dict_storage(ds, ["SWDOWN"], dt64, i, month, outliers,
                                                        outlier_upper_type="NLAD", outlier_lower_type=None,
                                                        upper_threshold=None, lower_threshold=None,
                                                        reshape=False
                                                        )

                #RH_over100 & RH_neg
                RH_outliers_df = outlier_dict_storage(ds, ["RH"], dt64, i, month, outliers,
                                                      outlier_upper_type="over100", outlier_lower_type="neg",
                                                      upper_threshold=None, lower_threshold=None,
                                                      reshape=False
                                                      )

                outliers_dict = {
                                  "IQR": iqr_outliers_df,
                                  "ZScore": z_outliers_df,
                                  "Zeros": zero_outliers_df,
                                  "Negatives": negative_outliers_df,
                                  "LAN": LAN_outliers_df,
                                  "NLAD": NLAD_outliers_df,
                                  "Rel. Humidity": RH_outliers_df
                                  }

                outliers_list.append(outliers_dict)
            

        output_filename = os.path.join(output_path + f"WRFstats_Analysis_{year}.npy")
        #np.savez_compressed(output_filename, outliers=outliers_list)
        np.save(output_filename, outliers_list)
        
    
    return outliers_list


#%% run code

wrf_path = "/global/cfs/cdirs/m2702/gsharing/tgw-wrf-conus/historical_1980_2019/hourly/"
outlier_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/historical/"
stats_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/historical/"

output_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/analysis/"

start = "1980-02"
stop = "2020-01"

outliers = WRFstats_Analysis(wrf_path, outlier_path, stats_path, output_path, start, stop)
