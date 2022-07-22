import os
from glob import glob
import xarray as xr

path = r"C:\Users\mcgr323\projects\wrf"
year = "2007"
month = "01"
monthdata = sorted(glob(os.path.join(path, f"NLDAS_FORA0125_H.A*{year}{month}*.002.grb.SUB.nc4")))
# find the following and preceeding months and year

def previous_file(year, month):
    months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    idx = months_list.index(month)

    month_minus = months_list[idx - 1]

    # if month is January, return previous year, else current year
    if month == "01":
        year_minus = str(int(year) - 1)
    else:
        year_minus = year

    return month_minus, year_minus


month_minus, year_minus = previous_file(year, month)

# take the file that preceeds the specified month and add to collected files, return sorted list
last_month = sorted(glob(os.path.join(path, f"NLDAS_FORA0125_H.A*{year_minus}{month_minus}*.002.grb.SUB.nc4")))[-1]
monthdata.append(last_month)
onemonthdata = sorted(monthdata)

for file in nc_files:
    ds_sl = sl.open_wrf_dataset(file)

