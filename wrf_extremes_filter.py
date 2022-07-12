import pandas as pd
import os

# set directory
data_input_dir = '/global/cscratch1/sd/mcgrathc'

# read in the NLDAS extremes
nldas_df = os.path.join(data_input_dir, 'NLDAS_Extreme_Hourly_Max_Min_1980_to_2020.csv')

nldas_extm = pd.read_csv(nldas_df)

# read in the wrf extremes
wrf_df = os.path.join(data_input_dir, 'WRF_Extreme_Hourly_Max_Min_1980_to_2020.csv')

wrf_extm = pd.read_csv(wrf_df)

# merge nldas and wrf
extm_merge = nldas_df.merge(wrf_df, on='Time', how='left')

WSPD_Min	
WSPD_Max
T2_Min
T2_Max
PS_Min
PS_Max
Q2_Min
Q2_Max
GLW_Min
GLW_Max
SWDOWN_Min
SWDOWN_Max
