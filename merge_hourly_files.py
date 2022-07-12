import pandas as pd
import os
import glob

# set directory
data_input_dir = '/global/cscratch1/sd/mcgrathc'

# read in csv files
nldas_extm_1980 = pd.read_csv(glob.glob(os.path.join(data_input_dir + 'NLDAS_Hourly_Min_Max_Values_1980*')))
nldas_extm_1990 = pd.read_csv(glob.glob(os.path.join(data_input_dir + 'NLDAS_Hourly_Min_Max_Values_1990*')))
nldas_extm_2000 = pd.read_csv(glob.glob(os.path.join(data_input_dir + 'NLDAS_Hourly_Min_Max_Values_2000*')))
nldas_extm_2010 = pd.read_csv(glob.glob(os.path.join(data_input_dir + 'NLDAS_Hourly_Min_Max_Values_2010*')))
nldas_extm_2020 = pd.read_csv(glob.glob(os.path.join(data_input_dir + 'NLDAS_Hourly_Min_Max_Values_2020*')))

# merge df from 1980 to 2020 together
frames = [nldas_extm_1980, nldas_extm_1990, nldas_extm_2000, nldas_extm_2010, nldas_extm_2020]
result = pd.concat(frames)

# Generate the .csv output file name:
filename = os.path.join(data_input_dir, 'NLDAS_Extreme_Hourly_Max_Min_1980_to_2020.csv')

# Save the hourly output dataframe to a .csv file:
result.to_csv(filename, sep=',', index=False)
