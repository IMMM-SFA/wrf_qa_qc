# wrf_qa_qc
Repository for the Weather Research and Forecasting Model (WRF) QA/QC process within IM3 

* `create_climate_variables.py` creates the new climate varaibles (like RH), `find_stats_and_dist.py` is the functions for analysis of stats and distributions, `find_outliers.py` is the functions for analysis of outers, `get_monthly_stats.py` calls the functions and stores them as netcdf files in nested yearly directories by month for "*all_stats.nc", "*all_outliers.nc" and "*normality.nc" (normality stored separately from "*all_stats.nc, since it is a single value across time, and space). Finally `run_in_parralell.py` runs the analysis in parallel by year

* `filter_outliers_nan.py`, and `run_filter_outliers.py` removes the padded nans from the "*all_outliers.nc" files

# MetaRepo
[Martell et al. (2024) Geoscientific Model Development](https://mxjmartell.github.io/Martell-etal_2024_GeoscientificModelDevelopment/ "Martell et al. (2024)")
