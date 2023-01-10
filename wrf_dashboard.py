import xarray as xr
from glob import glob
import os
import panel as pn
from netCDF4 import Dataset
import hvplot.xarray
from hvplot import show
from bokeh.embed import components


input_path = "C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981"

month = '01'
jan_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_jan = xr.open_mfdataset(jan_nc)

month = '02'
feb_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_feb = xr.open_mfdataset(feb_nc)

month = '03'
mar_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_mar = xr.open_mfdataset(mar_nc)

month = '04'
apr_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_apr = xr.open_mfdataset(apr_nc)

month = '05'
may_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_may = xr.open_mfdataset(may_nc)

month = '06'
jun_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_jun = xr.open_mfdataset(jun_nc)

month = '07'
jul_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_jul = xr.open_mfdataset(jul_nc)

month = '08'
aug_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_aug = xr.open_mfdataset(aug_nc)

month = '09'
sep_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_sep = xr.open_mfdataset(sep_nc)

month = '10'
oct_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_oct = xr.open_mfdataset(oct_nc)

month = '11'
nov_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_nov = xr.open_mfdataset(nov_nc)

month = '12'
dec_nc = sorted(glob(os.path.join(input_path, f"NLDAS_1981{month}_all_stats_DS.nc")))
ds_dec = xr.open_mfdataset(dec_nc)

init_var = 'TMP_mean'
var_select = pn.widgets.Select(name='NLDAS Statistics:', options=list(ds_jan.data_vars),
                               value=init_var)


@pn.depends(var_select)
def plot_jan(var):
    mesh = ds_jan[var].hvplot.quadmesh(x='lon', y='lat', title="January",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_feb(var):
    mesh = ds_feb[var].hvplot.quadmesh(x='lon', y='lat', title="February",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_mar(var):
    mesh = ds_mar[var].hvplot.quadmesh(x='lon', y='lat', title="March",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_apr(var):
    mesh = ds_apr[var].hvplot.quadmesh(x='lon', y='lat', title="April",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_may(var):
    mesh = ds_may[var].hvplot.quadmesh(x='lon', y='lat', title="May",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_jun(var):
    mesh = ds_jun[var].hvplot.quadmesh(x='lon', y='lat', title="June",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_jul(var):
    mesh = ds_jul[var].hvplot.quadmesh(x='lon', y='lat', title="July",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_aug(var):
    mesh = ds_aug[var].hvplot.quadmesh(x='lon', y='lat', title="August",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_sep(var):
    mesh = ds_sep[var].hvplot.quadmesh(x='lon', y='lat', title="September",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_oct(var):
    mesh = ds_oct[var].hvplot.quadmesh(x='lon', y='lat', title="October",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_nov(var):
    mesh = ds_nov[var].hvplot.quadmesh(x='lon', y='lat', title="November",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_dec(var):
    mesh = ds_dec[var].hvplot.quadmesh(x='lon', y='lat', title="December",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=400, height=200).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


plots_box = pn.WidgetBox(pn.Column(pn.Row(var_select)),
                         pn.Row(pn.bind(plot_jan, var_select),
                                pn.bind(plot_feb, var_select),
                                pn.bind(plot_mar, var_select)),
                         pn.Row(pn.bind(plot_apr, var_select),
                                pn.bind(plot_may, var_select),
                                pn.bind(plot_jun, var_select)),
                         pn.Row(pn.bind(plot_jul, var_select),
                                pn.bind(plot_aug, var_select),
                                pn.bind(plot_sep, var_select)),
                         pn.Row(pn.bind(plot_oct, var_select),
                                pn.bind(plot_nov, var_select),
                                pn.bind(plot_dec, var_select)),
                         align="start",
                         sizing_mode="stretch_both")

dashboard = pn.Row(plots_box, sizing_mode="stretch_both")
dashboard.servable('NLDAS Dashboard')
dashboard.show()
