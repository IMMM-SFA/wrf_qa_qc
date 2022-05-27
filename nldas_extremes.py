# Import all of the required libraries and packages:
import os
import glob
import pandas as pd
import xarray as xr
import numpy as np

# Set the data input and output directories:
data_input_dir = '/global/cscratch1/sd/mcgrathc'
data_output_dir = '/global/cscratch1/sd/mcgrathc'


# Define a function to read in a single NLDAS file and extract the minimum and maximum value of several
# meteorological variables:
def process_single_hour_min_max_values(filename):
    # Read in the NLDAS file using xarray:
    nldas = xr.open_dataset(filename)

    # Create a new dataframe to output the results:
    output_df = pd.DataFrame()

    # Store the time variable as a pandas datetime:
    output_df.at[0, 'Time'] = pd.to_datetime(nldas.time).item()

    # Compute and store the maximum and minimum wind speeds
    output_df.at[0, 'WSPD_Min'] = (
        np.sqrt(np.square(nldas.UGRD) + np.square(nldas.VGRD))).min().item()  # Minimum 10-m wind speed in m/s
    output_df.at[0, 'WSPD_Max'] = (
        np.sqrt(np.square(nldas.UGRD) + np.square(nldas.VGRD))).max().item()  # Maximum 10-m wind speed in m/s

    # Store the maximum and minimum temperature, surface pressure, specific humidity, and downwelling and long- and
    # shortwave radiation:
    output_df.at[0, 'T2_Min'] = nldas.TMP.min().item()  # Minimum 2-m temperature in K
    output_df.at[0, 'T2_Max'] = nldas.TMP.max().item()  # Maximum 2-m temperature in K
    output_df.at[0, 'PS_Min'] = nldas.PRES.min().item()  # Minimum surface pressure in Pa
    output_df.at[0, 'PS_Max'] = nldas.PRES.max().item()  # Maximum surface pressure in Pa
    output_df.at[0, 'Q2_Min'] = nldas.SPFH.min().item()  # Minimum 2-m specific humidity in kg/kg
    output_df.at[0, 'Q2_Max'] = nldas.SPFH.max().item()  # Maximum 2-m specific humidity in kg/kg
    output_df.at[
        0, 'GLW_Min'] = nldas.DLWRF.min().item()  # Minimum downwelling longwave radiation at the surface in W/m^2
    output_df.at[
        0, 'GLW_Max'] = nldas.DLWRF.max().item()  # Maximum downwelling longwave radiation at the surface in W/m^2
    output_df.at[
        0, 'SWDOWN_Min'] = nldas.DSWRF.min().item()  # Minimum downwelling shortwave radiation at the surface in W/m^2
    output_df.at[
        0, 'SWDOWN_Max'] = nldas.DSWRF.max().item()  # Maximum downwelling shortwave radiation at the surface in W/m^2

    # Round off the decimal places for all variables:
    output_df['WSPD_Min'] = output_df['WSPD_Min'].round(2)
    output_df['WSPD_Max'] = output_df['WSPD_Max'].round(2)
    output_df['T2_Min'] = output_df['T2_Min'].round(2)
    output_df['T2_Max'] = output_df['T2_Max'].round(2)
    output_df['PS_Min'] = output_df['PS_Min'].round(2)
    output_df['PS_Max'] = output_df['PS_Max'].round(2)
    output_df['Q2_Min'] = output_df['Q2_Min'].round(6)
    output_df['Q2_Max'] = output_df['Q2_Max'].round(6)
    output_df['GLW_Min'] = output_df['GLW_Min'].round(2)
    output_df['GLW_Max'] = output_df['GLW_Max'].round(2)
    output_df['SWDOWN_Min'] = output_df['SWDOWN_Min'].round(2)
    output_df['SWDOWN_Max'] = output_df['SWDOWN_Max'].round(2)

    return output_df


list_of_files = sorted(glob.glob(os.path.join(data_input_dir + 'NLDAS_FORA0125_H.A*.002.grb.SUB.nc4')))

# Loop over the states and interpolate their loads to an annual time step:
for file in range(len(list_of_files)):
    # If this is the first step in the loop then create a new output dataframe to store the results else just append
    # the results to the existing output dataframe:
    if file == 0:
        hourly_min_max_df = process_single_hour_min_max_values(list_of_files[file])
    else:
        hourly_min_max_df = hourly_min_max_df.append(process_single_hour_min_max_values(list_of_files[file]))

# Create strings of the starting and ending times and generate the .csv output file name:
start_time = str(hourly_min_max_df['Time'].min()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(
    "_00_00", "")
end_time = str(hourly_min_max_df['Time'].max()).replace(" ", "_").replace("-", "_").replace(":", "_").replace("_00_00",
                                                                                                              "")
hourly_csv_output_filename = os.path.join(data_output_dir,
                                          'NLDAS_Hourly_Min_Max_Values_' + start_time + '_UTC_to_' + end_time + '_UTC.csv')

# Save the hourly output dataframe to a .csv file:
hourly_min_max_df.to_csv(hourly_csv_output_filename, sep=',', index=False)

# Display the head of the hourly output dataframe:
hourly_min_max_df.head(10)

# Create a new dataframe to output the global max/min results:
global_min_max_df = pd.DataFrame()

# Store the minimum and maximum values over the whole time period processed:
global_min_max_df.at[0, 'Start_Time'] = start_time
global_min_max_df.at[0, 'End_Time'] = end_time
global_min_max_df.at[0, 'WSPD_Min'] = hourly_min_max_df['WSPD_Min'].min()
global_min_max_df.at[0, 'WSPD_Max'] = hourly_min_max_df['WSPD_Max'].max()
global_min_max_df.at[0, 'T2_Min'] = hourly_min_max_df['T2_Min'].min()
global_min_max_df.at[0, 'T2_Max'] = hourly_min_max_df['T2_Max'].max()
global_min_max_df.at[0, 'PS_Min'] = hourly_min_max_df['PS_Min'].min()
global_min_max_df.at[0, 'PS_Max'] = hourly_min_max_df['PS_Max'].max()
global_min_max_df.at[0, 'Q2_Min'] = hourly_min_max_df['Q2_Min'].min()
global_min_max_df.at[0, 'Q2_Max'] = hourly_min_max_df['Q2_Max'].max()
global_min_max_df.at[0, 'GLW_Min'] = hourly_min_max_df['GLW_Min'].min()
global_min_max_df.at[0, 'GLW_Max'] = hourly_min_max_df['GLW_Max'].max()
global_min_max_df.at[0, 'SWDOWN_Min'] = hourly_min_max_df['SWDOWN_Min'].min()
global_min_max_df.at[0, 'SWDOWN_Max'] = hourly_min_max_df['SWDOWN_Max'].max()

# Generate the .csv output file name:
global_csv_output_filename = os.path.join(data_output_dir, 'NLDAS_Global_Min_Max_Values_' + start_time + '_UTC_to_' +
                                          end_time + '_UTC.csv')

# Save the hourly output dataframe to a .csv file:
global_min_max_df.to_csv(global_csv_output_filename, sep=',', index=False)

