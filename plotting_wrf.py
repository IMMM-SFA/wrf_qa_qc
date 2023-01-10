import hvplot.xarray  # noqa
import xarray as xr
import os
from glob import glob
from hvplot import show
import cartopy.crs as ccrs

input_path = "C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981"

months = ['01', '02', '03']
for month in months:
    nc_files = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
    ds = xr.open_mfdataset(nc_files)

show(ds['TMP_mean'].hvplot.quadmesh(width=400, cmap='viridis'))

ds.hvplot.quadmesh(
    'lon', 'lat', 'TMP_mean', projection=ccrs.Orthographic(-90, 30),
    global_extent=True, frame_height=540, cmap='viridis',
    coastline=True
)

show(ds.hvplot.quadmesh(
    'lon', 'lat', 'TMP_mean', projection='LambertConformal'
))

ds.hvplot.quadmesh(x='lon', y='lat', z=['TMP_mean', 'SPFH_mean', 'UGRD_mean', 'DLWRF_mean'],
                   width=350, height=300, subplots=True, shared_axes=False).cols(2)
