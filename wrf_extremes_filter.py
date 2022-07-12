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


# selecting rows based on condition
rslt_df = extm_merge.loc[(extm_merge['WSPD_Min'] < ) &
                         (extm_merge['WSPD_Max'] > ) &
                        (extm_merge['T2_Min'] < ) &
                        (extm_merge['T2_Max'] > ) &
                        (extm_merge['PS_Min'] < ) &
                        (extm_merge['PS_Max'] > ) &
                        (extm_merge['Q2_Min'] < ) &
                        (extm_merge['Q2_Max'] > ) &
                        (extm_merge['GLW_Min'] < ) &
                        (extm_merge['GLW_Max'] > ) &
                        (extm_merge['SWDOWN_Min'] < ) &
                        (extm_merge['SWDOWN_Max'] > ) &
                        ]

