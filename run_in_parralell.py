import sys
from get_monthly_stats import wrf_stats


if __name__ == "__main__":

    input_path = "/global/cfs/cdirs/m2702/gsharing/tgw-wrf-conus/historical_1980_2019/hourly/"
    output_path = "/global/cfs/projectdirs/m2702/gsharing/QAQC/"

    if len(sys.argv) > 1:
        year = sys.argv[1]
        start = f'{year}-01'
        stop = f'{year}-12'

    else:
        print('Please specify start and stop years for analysis')

    # run the WRFstats program
    wrf_stats(input_path, output_path, start, stop)
