# %% run code

# specify the path to the location of the files to be analyzed,
# the path for the output to be stored, and the start and stop dates
input_path = "/project/projectdirs/m2702/gsharing/CONUS_TGW_WRF_Historical/"
output_path = "/project/projectdirs/m2702/gsharing/QAQC/"
start = "1999-01-01"
stop = "1999-12-31"

# run the WRFstats program
mean_ds, avg_median_ds, stddev_ds, max_ds, min_ds = WRFstats(input_path, output_path, start, stop)