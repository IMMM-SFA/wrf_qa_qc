import xarray as xr
from glob import glob
import os
import panel as pn

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

init_var = 'TMP_mean'
var_select = pn.widgets.Select(name='NLDAS Statistics:', options=list(ds_jan.data_vars),
                               value=init_var)


@pn.depends(var_select)
def plot_jan(var):
    mesh = ds_jan[var].hvplot.quadmesh(x='lon', y='lat', title="January",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_feb(var):
    mesh = ds_feb[var].hvplot.quadmesh(x='lon', y='lat', title="February",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_mar(var):
    mesh = ds_mar[var].hvplot.quadmesh(x='lon', y='lat', title="March",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_apr(var):
    mesh = ds_apr[var].hvplot.quadmesh(x='lon', y='lat', title="April",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_may(var):
    mesh = ds_may[var].hvplot.quadmesh(x='lon', y='lat', title="May",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


@pn.depends(var_select)
def plot_jun(var):
    mesh = ds_jun[var].hvplot.quadmesh(x='lon', y='lat', title="June",
                                       attr_labels=False,
                                       cmap='rainbow',
                                       width=600, height=400).opts(alpha=0.7,
                                                                   active_tools=['wheel_zoom', 'pan'])
    return pn.panel(mesh)


plots_box = pn.WidgetBox(pn.Column(pn.Row(var_select)),
                         pn.Row(pn.bind(plot_jan, var_select),
                                pn.bind(plot_feb, var_select),
                                pn.bind(plot_mar, var_select)),
                         pn.Row(pn.bind(plot_apr, var_select),
                                pn.bind(plot_may, var_select),
                                pn.bind(plot_jun, var_select)),
                         align="start",
                         sizing_mode="stretch_width")

dashboard = pn.Row(plots_box, sizing_mode="stretch_width")
dashboard.servable('NLDAS Dashboard')
dashboard.show()
