import numpy as np
import pandas as pd
import salem as sl
import os
from glob import glob
from WRFstats_Functions import temp_conv, deacc_precip, windspeed
from WRFstats_Functions import descriptive_stats, SW_Test, skew_kurtosis_test
from WRF_QAQC_OutlierAnalysis_Functions import IQR_Test, ZScore_Test, iqr_outlier_storage, z_outlier_storage


#%% function to find the previous month containing parts of the given month
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


#%% function for calculating stats on monthly netCDF data
def WRFstats(input_path, output_path, start, stop, descriptive=True, distribution=True, outliers=True,
             ds_variables=["LU_INDEX","Q2","T2","PSFC","U10","V10","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH","WSPD","BR",
                           "ZOL","RAINC","RAINSH","RAINNC","SNOWNC","GRAUPELNC","HAILNC","SWDOWN","GLW","UST","SNOWC","SR"]
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
        
        ds = sl.open_mf_wrf_dataset(nc_files) # open all netCDF files in month and create xarray dataset using salem
        ds = ds.sel(time = slice(f"{month}")) # slice by the current month
        ds.load() # load into memory for computations
        
        # convert T2 variable from K to F or C
        temp_conv(ds, ds_variables)
        
        # combine and deaccumulate precipitation variables into PRECIP variable
        deacc_precip(ds, ds_variables)
        
        # create new variable WINDSPEED from magnitudes of velocity vectors
        windspeed(ds, ds_variables)
        
        # calculate descriptive stats on files using xarray
        if descriptive == True:
            mean_ds, median_ds, stddev_ds, max_ds, min_ds = descriptive_stats(ds, ds_variables)
        
        else:    
            mean_ds, median_ds, stddev_ds, max_ds, min_ds = (None,)*5
        
        # calculate distribution stats on files using xarray
        if distribution == True:
            # Shapiro-Wilks test function for normality, gives percent of distributions that are normal
            SW_ds, normality = SW_Test(ds, ds_variables)
            
            # skew and kurtosis tests
            skew_ds, kurtosis_ds = skew_kurtosis_test(ds, ds_variables)
            
        else:
            SW_ds, normality, skew_ds, kurtosis_ds = (None,)*4
        
        # calculate statistical outliers
        if outliers == True:
            # outlier detection with IQR test
            iqr_ds, q75_ds, q25_ds, upper_threshold, lower_threshold, outlier_upper, outlier_lower, outlier_upper_inv, outlier_lower_inv = IQR_Test(ds, ds_variables, iqr_threshold=3)
            iqr_outlier_df_dict = iqr_outlier_storage(ds, ds_variables, outlier_upper, outlier_lower, upper_threshold, lower_threshold)
            
            # outlier detection with Z-score test
            zscore_ds, z_outlier_upper, z_outlier_lower, z_outlier_upper_inv, z_outlier_lower_inv, z_threshold = ZScore_Test(ds, ds_variables, z_threshold=4)
            z_outlier_df_dict = z_outlier_storage(ds, ds_variables, zscore_ds, z_outlier_upper, z_outlier_lower, z_threshold)
            
        else:
            iqr_ds, q75_ds, q25_ds, outlier_upper, outlier_lower, iqr_outlier_df_dict = (None,)*6
            zscore_ds, z_outlier_upper, z_outlier_lower, z_threshold, z_outlier_df_dict = (None,)*5
        
        # specify the location for the output of the program
        output_filename = os.path.join(output_path + f"WRFstats_{month}.npy")
        
        # concatenate stats into dictionary and save as numpy dict
        stats_dict = {
                      "Means": mean_ds,
                      "Medians": median_ds,
                      "Standard Deviation": stddev_ds,
                      "Max": max_ds,
                      "Min": min_ds,
                      "Shapiro-Wilks": SW_ds,
                      "% Normal": normality,
                      "Skew": skew_ds,
                      "Kurtosis": kurtosis_ds,
                      "Q75": q75_ds,
                      "Q25": q25_ds,
                      "IQR": iqr_ds,
                      "IQR Outliers": iqr_outlier_df_dict,
                      "Z-Scores": zscore_ds,
                      "Z-Score Outliers": z_outlier_df_dict
                      }
        
        np.save(os.path.join(output_path, output_filename), stats_dict)
        stats_list.append(stats_dict)
    
    
    return stats_list


#%% run code

# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
input_path = "/project/projectdirs/m2702/gsharing/CONUS_TGW_WRF_Historical/"
output_path = "/project/projectdirs/m2702/gsharing/QAQC/"
start = "1989-01"
stop = "1989-12"

# run the WRFstats program
stats = WRFstats(input_path, output_path, start, stop)

