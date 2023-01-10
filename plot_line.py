import hvplot
import hvplot.xarray
import xarray as xr
import holoviews as hv
import panel as pn
from panel.interact import interact
import numpy as np

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.layouts import column, grid
from bokeh.models import ColumnDataSource, CustomJS, Slider, HoverTool

hv.extension('bokeh')


def data_visualization(request):
    ds = xr.open_dataset('C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981/NLDAS_198101_all_stats_DS.nc')
    df = ds.to_dataframe()


    def plot(variable):
        return ds[variable].hvplot.line()

    model = interact(plot, variable=list(ds.data_vars))

    script, div = components(model.get_root())

    return render(request, 'visualization.html', {'script': script, 'div': div})

import nctoolkit as nc
file = 'C:/Users/mcgr323/projects/wrf/nldas_stats_91/1981/NLDAS_198101_all_stats_DS.nc'
import nctoolkit as nc
ds = nc.open_data(file)
ds.TMP_mean()