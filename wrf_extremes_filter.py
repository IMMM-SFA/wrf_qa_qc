import pandas as pd

data_input_dir = '/global/cscratch1/sd/mcgrathc'

start_time = "_00_00"
end_time =  "_00_00"

nldas_extm_ df = pd. read_csv(os.path.join(data_input_dir, 'NLDAS_Global_Min_Max_Values_' + start_time +
                          '_UTC_to_' + end_time + '_UTC.csv'))

nldas_extm_ df