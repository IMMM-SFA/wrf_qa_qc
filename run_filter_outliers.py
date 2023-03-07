import sys
from filter_outliers_nan import remove_NaNs

if __name__ == "__main__":

    input_path = "/global/cfs/projectdirs/m2702/gsharing/tgw-wrf-conus/historical_1980_2019/hourly/"
    output_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/"

    if len(sys.argv) > 1:
        year = sys.argv[1]
        start = f'{year}-01'
        stop = f'{year}-12'

    else:
        print('Please specify start and stop years for analysis')

    # run the filter outliers program
    remove_NaNs(input_path, output_path, start, stop)