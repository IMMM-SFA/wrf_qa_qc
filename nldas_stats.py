import numpy as np
import xarray as xr
import os
import pandas as pd
from glob import glob


# function for calculating magnitude of velocity vectors

def magnitude(ds):
    U = ds["UGRD"]
    V = ds["VGRD"]
    ds["WINDSPEED"] = np.sqrt(U ** 2 + V ** 2)


# function for aggregating rolling stats on netCDF data

def NLDASstats(input_path, output_path, start, stop,
               ds_variables=["TMP", "SPFH", "PRES", "UGRD", "VGRD", "DLWRF",
                             "PEVAP", "APCP", "DSWRF"]
               ):
    """
    Function for running moving (rolling) descriptive statistics on all netCDF files between a given range of dates.

    Input
    ----------
    input_path : Str Path to netCDF files for analysis.
    output_path : Str Path for the output netCDF files to be stored.
    start : Str start date for files
    stop : Str stop date for files
    ds_variables : List Variables to run stats on.

    Returns
    -------
    mean_roll : DataSet netCDF file for storage of rolling mean.
    avg_median_roll : DataSet netCDF file for storage of rolling average median.
    stddev_roll : DataSet netCDF file for storage of rolling standard deviation.
    max_roll : DataSet netCDF file for storage of rolling maximums.
    min_roll : DataSet netCDF file for storage of rolling minimums.

    """
    n = 0  # counter

    # create list of range of months to open
    months = pd.date_range(start, stop, freq="MS").strftime("%Y%m").tolist()

    # iterate through each month and create dataset
    for month in months:

        # create list of files in the given month in the range of months specified
        nc_files = sorted(glob(os.path.join(input_path, f"NLDAS_FORA0125_H.A*{month}*.002.grb.SUB.nc4")))

        ds = xr.open_mfdataset(nc_files)  # open all netCDF files in month and create xarray dataset using salem
        # ds = ds.sel(time=slice(f"{month}"))  # slice by the current month

        # create new variable for wind speed from magnitudes of velocity vectors
        magnitude(ds)

        # calculate descriptive stats on file using xarray
        mean_ds = ds[ds_variables].mean(dim="time", skipna=True)
        median_ds = ds[ds_variables].median(dim="time", skipna=True)
        stddev_ds = ds[ds_variables].std(dim="time", skipna=True)
        max_ds = ds[ds_variables].max(dim="time", skipna=True)
        min_ds = ds[ds_variables].min(dim="time", skipna=True)

        mean_df = mean_ds.rename(
            {ds_variables[0]: f"{ds_variables[0]}_mean", ds_variables[1]: f"{ds_variables[1]}_mean",
             ds_variables[2]: f"{ds_variables[2]}_mean", ds_variables[3]: f"{ds_variables[3]}_mean",
             ds_variables[4]: f"{ds_variables[4]}_mean", ds_variables[5]: f"{ds_variables[5]}_mean",
             ds_variables[6]: f"{ds_variables[6]}_mean", ds_variables[7]: f"{ds_variables[7]}_mean",
             ds_variables[8]: f"{ds_variables[8]}_mean"})

        max_df = max_ds.rename({ds_variables[0]: f"{ds_variables[0]}_max", ds_variables[1]: f"{ds_variables[1]}_max",
                                ds_variables[2]: f"{ds_variables[2]}_max", ds_variables[3]: f"{ds_variables[3]}_max",
                                ds_variables[4]: f"{ds_variables[4]}_max", ds_variables[5]: f"{ds_variables[5]}_max",
                                ds_variables[6]: f"{ds_variables[6]}_max", ds_variables[7]: f"{ds_variables[7]}_max",
                                ds_variables[8]: f"{ds_variables[8]}_max"})

        min_df = min_ds.rename({ds_variables[0]: f"{ds_variables[0]}_min", ds_variables[1]: f"{ds_variables[1]}_min",
                                ds_variables[2]: f"{ds_variables[2]}_min", ds_variables[3]: f"{ds_variables[3]}_min",
                                ds_variables[4]: f"{ds_variables[4]}_min", ds_variables[5]: f"{ds_variables[5]}_min",
                                ds_variables[6]: f"{ds_variables[6]}_min", ds_variables[7]: f"{ds_variables[7]}_min",
                                ds_variables[8]: f"{ds_variables[8]}_min"})

        med_df = median_ds.rename({ds_variables[0]: f"{ds_variables[0]}_med",
                                   ds_variables[1]: f"{ds_variables[1]}_med",
                                   ds_variables[2]: f"{ds_variables[2]}_med",
                                   ds_variables[3]: f"{ds_variables[3]}_med",
                                   ds_variables[4]: f"{ds_variables[4]}_med",
                                   ds_variables[5]: f"{ds_variables[5]}_med",
                                   ds_variables[6]: f"{ds_variables[6]}_med",
                                   ds_variables[7]: f"{ds_variables[7]}_med",
                                   ds_variables[8]: f"{ds_variables[8]}_med"})

        std_dev_df = stddev_ds.rename({ds_variables[0]: f"{ds_variables[0]}_std",
                                       ds_variables[1]: f"{ds_variables[1]}_std",
                                       ds_variables[2]: f"{ds_variables[2]}_std",
                                       ds_variables[3]: f"{ds_variables[3]}_std",
                                       ds_variables[4]: f"{ds_variables[4]}_std",
                                       ds_variables[5]: f"{ds_variables[5]}_std",
                                       ds_variables[6]: f"{ds_variables[6]}_std",
                                       ds_variables[7]: f"{ds_variables[7]}_std",
                                       ds_variables[8]: f"{ds_variables[8]}_std"})

        # concatenate stats
        all_stats = xr.merge([mean_df, med_df, max_df, min_df, std_dev_df])

        # get string for year
        year_dir = month[0:4]

        # create path for year
        year_path = os.path.join(output_path, year_dir)

        # checking if the directory demo_folder exist or not.
        if not os.path.exists(year_path):
            # if the demo_folder directory is not present create it
            os.makedirs(year_path)

        # specify the location for the output of the program
        output_filename = os.path.join(year_path + "/" + f"NLDAS_{month}_all_stats_DS.nc")

        # save each output stat as a netCDF file
        all_stats.to_netcdf(path=output_filename)

        n += 1  # iterate counter

    return


# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
input_path = "/global/cfs/projectdirs/m2702/QAQC/NLDAS/NLDAS_input"
output_path = "/global/cfs/projectdirs/m2702/QAQC/NLDAS/NLDAS_output"

start = "1980-01-01"
stop = "1990-01-01"

# run the NLDASstats program
all_stats = NLDASstats(input_path, output_path, start, stop)
