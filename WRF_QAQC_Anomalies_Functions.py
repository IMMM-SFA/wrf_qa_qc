import pandas as pd
import numpy as np
import xarray as xr
import salem as sl
from astral import LocationInfo
from astral.sun import dusk, dawn
from datetime import datetime, timezone
from typing import List


#%% find light at night (LAN) and no light at day (NLAD) v2.0
def has_sunlight(latitude: xr.DataArray, longitude: xr.DataArray, time: List[datetime]):
    
    timestamps = [pd.Timestamp(t) for t in time]
    datetimes = [ts.to_pydatetime() for ts in timestamps]
    datetimes = [time.replace(tzinfo = timezone.utc) for time in datetimes if time.tzinfo is None]
    
    dates = list({t.date() for t in datetimes})
    frames = []
    
    def get_dawn(lat, lon, d):
        
        try:
            return dawn(LocationInfo(name = "", region = "", timezone = "UTC", latitude = lat, longitude = lon,).observer, date = d)
        
        except ValueError as e:
            # hack to fix precision errors for CONUS
            return get_dawn(lat, lon+0.5, d)
        
    def get_dusk(lat, lon, d):
        
        try:
            return dusk(LocationInfo(name = "", region = "", timezone = "UTC", latitude = lat, longitude = lon,).observer, date = d)
        
        except ValueError as e:
            # hack to fix precision errors for CONUS
            return get_dusk(lat, lon+0.5, d)
        
    for date in dates:
        
        today_dawn = xr.apply_ufunc(get_dawn, latitude, longitude, date, vectorize = True)
        today_dusk = xr.apply_ufunc(get_dusk, latitude, longitude, date, vectorize = True)
        
        for t in [t for t in datetimes if t.date() == date]:
            
            if t.tzinfo is None:
                t = t.replace(tzinfo = timezone.utc)
                
            frames.append(xr.apply_ufunc(
                    lambda tdawn, tdusk, t: ((t <= tdusk) or (t >= tdawn)) if tdawn > tdusk else (tdawn <= t <= tdusk),
                    today_dawn, today_dusk, t, vectorize = True).values)
            
    #return xr.DataArray(np.stack(frames, axis = 2), dims = ["lat", "lon", "time"])
    return xr.DataArray(np.stack(frames, axis=2).transpose((2, 0, 1)), dims=["time", "lat", "lon"])


def find_LAN_NLAD(ds, ds_variables):
    
    if "SWDOWN" in ds_variables:
        
        has_light = has_sunlight(ds.lat, ds.lon, ds.time.values)
        
        LAN = ds["SWDOWN"].where(has_light.values == False).where(ds["SWDOWN"] >= 50)
        NLAD = ds["SWDOWN"].where(has_light.values == True).where(ds["SWDOWN"] == 0)
        
    else:
        LAN, NLAD, has_light = (None,)*3
    
    return LAN, NLAD, has_light


def LAN_NLAD_storage(ds, ds_variables, LAN, NLAD, has_light):
    
    if "SWDOWN" in ds_variables: 
    
        LAN_dict = {"LAN": np.where(LAN.notnull())}
        NLAD_dict = {"NLAD": np.where(NLAD.notnull())}
        
        val_LAN = ds["SWDOWN"].values[tuple(LAN_dict["LAN"])]
        time_LAN = ds["SWDOWN"].time[tuple(LAN_dict["LAN"])[0]]
        lat_LAN = ds["SWDOWN"].south_north[tuple(LAN_dict["LAN"])[1]]
        lon_LAN = ds["SWDOWN"].west_east[tuple(LAN_dict["LAN"])[2]]
        haslight_LAN = has_light.values[tuple(LAN_dict["LAN"])]
        
        val_NLAD = ds["SWDOWN"].values[tuple(NLAD_dict["NLAD"])]
        time_NLAD = ds["SWDOWN"].time[tuple(NLAD_dict["NLAD"])[0]]
        lat_NLAD = ds["SWDOWN"].south_north[tuple(NLAD_dict["NLAD"])[1]]
        lon_NLAD = ds["SWDOWN"].west_east[tuple(NLAD_dict["NLAD"])[2]]
        haslight_NLAD = has_light.values[tuple(NLAD_dict["NLAD"])]
        
        LAN_time_list = list(time_LAN.values)
        LAN_lat_list = list(lat_LAN.values)
        LAN_lon_list = list(lon_LAN.values)
        LAN_val_list = list(val_LAN)
        haslight_LAN_list = list(haslight_LAN)
        
        NLAD_time_list = list(time_NLAD.values)
        NLAD_lat_list = list(lat_NLAD.values)
        NLAD_lon_list = list(lon_NLAD.values)
        NLAD_val_list = list(val_NLAD)
        haslight_NLAD_list = list(haslight_NLAD)
        
        LAN_dict_i = {"Time": [], "Lat": [], "Lon": [], "has_light": [], "Value": []}
        for i in range(len(LAN_val_list)):
            LAN_dict_i["Time"].append(LAN_time_list[i])
            LAN_dict_i["Lat"].append(LAN_lat_list[i])
            LAN_dict_i["Lon"].append(LAN_lon_list[i])
            LAN_dict_i["has_light"].append(haslight_LAN_list[i])
            LAN_dict_i["Value"].append(LAN_val_list[i])
        
        NLAD_dict_i = {"Time": [], "Lat": [], "Lon": [], "has_light": [], "Value": []}
        for i in range(len(NLAD_val_list)):
            NLAD_dict_i["Time"].append(NLAD_time_list[i])
            NLAD_dict_i["Lat"].append(NLAD_lat_list[i])
            NLAD_dict_i["Lon"].append(NLAD_lon_list[i])
            NLAD_dict_i["has_light"].append(haslight_NLAD_list[i])
            NLAD_dict_i["Value"].append(NLAD_val_list[i])
        
        LAN_df = pd.DataFrame(LAN_dict_i)
        NLAD_df = pd.DataFrame(NLAD_dict_i)
        
        LAN_NLAD_df_dict = {"LAN": LAN_df, "NLAD": NLAD_df}
        
        return LAN_NLAD_df_dict


#%% find zeros 
def find_zeros(ds, ds_variables,
               checklist = ["T2", "PSFC", "RH"]):
    
    vars = [checklist[i] for i in range(len(checklist)) if checklist[i] in ds_variables]
    
    zeros_ds = ds[vars].where(ds[vars] == 0)
    
    return zeros_ds, vars


def zeros_storage(ds, zeros_ds, vars):
    
    zeros_dict = {i: np.where(zeros_ds[i].notnull()) for i in vars}
    
    has_zeros_vars = [var for var in vars if len(zeros_dict[var][0] > 0)]
    
    zeros_df_list = []
    
    for var in has_zeros_vars:
        
        val_zero = ds[var].values[tuple(zeros_dict[var])]
        time_zero = ds[var].time[tuple(zeros_dict[var])[0]]
        lat_zero = ds[var].south_north[tuple(zeros_dict[var])[1]]
        lon_zero = ds[var].west_east[tuple(zeros_dict[var])[2]]
        
        time_list = list(time_zero.values)
        lat_list = list(lat_zero.values)
        lon_list = list(lon_zero.values)
        val_list = list(val_zero)
        
        dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}
        
        for i in range(len(val_list)):
            dict["Time"].append(time_list[i])
            dict["Lat"].append(lat_list[i])
            dict["Lon"].append(lon_list[i])
            dict["Value"].append(val_list[i])
        
        zeros_df = pd.DataFrame(dict)
        
        zeros_df_list.append(zeros_df)
        
    zeros_df_dict = {has_zeros_vars[i]: zeros_df_list[i] for i in range(len(has_zeros_vars))}
    
    return zeros_df_dict


#%% find negatives
def find_negatives(ds, ds_variables,
               checklist = ["LU_INDEX","T2","PSFC","SFROFF","UDROFF","ACSNOM","SNOW","SNOWH",
                             "RAINC","RAINSH","RAINNC","SNOWNC","GRAUPELNC","HAILNC","SWDOWN", "RH"]):
    
    vars = [checklist[i] for i in range(len(checklist)) if checklist[i] in ds_variables]
    
    negatives_ds = ds[vars].where(ds[vars] < 0)
    
    return negatives_ds, vars


def negatives_storage(ds, negatives_ds, vars):
    
    negatives_dict = {i: np.where(negatives_ds[i].notnull()) for i in vars}
    
    has_negatives_vars = [var for var in vars if len(negatives_dict[var][0] > 0)]
    
    negatives_df_list = []
    
    for var in has_negatives_vars:
        
        val_negative = ds[var].values[tuple(negatives_dict[var])]
        time_negative = ds[var].time[tuple(negatives_dict[var])[0]]
        lat_negative = ds[var].south_north[tuple(negatives_dict[var])[1]]
        lon_negative = ds[var].west_east[tuple(negatives_dict[var])[2]]
        
        time_list = list(time_negative.values)
        lat_list = list(lat_negative.values)
        lon_list = list(lon_negative.values)
        val_list = list(val_negative)
        
        dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}
        
        for i in range(len(val_list)):
            dict["Time"].append(time_list[i])
            dict["Lat"].append(lat_list[i])
            dict["Lon"].append(lon_list[i])
            dict["Value"].append(val_list[i])
        
        negatives_df = pd.DataFrame(dict)
        
        negatives_df_list.append(negatives_df)
        
    negatives_df_dict = {has_negatives_vars[i]: negatives_df_list[i] for i in range(len(has_negatives_vars))}
    
    return negatives_df_dict


#%% calculate relative humidity and return where rh over 100%
# expect rh to be between 0 and 1, but small deviations above 1 are allowed
def rel_humidity(ds, ds_variables):
    
    if "T2" in ds_variables and "PSFC" in ds_variables and "Q2" in ds_variables: 
    
        es = 6.112 * np.exp(17.67 * (ds["T2"] - 273.15) / (ds["T2"] - 29.65))
        qvs = 0.622 * es / (0.01 * ds["PSFC"] - (1.0 - 0.622) * es)
        rh = ds["Q2"] / qvs
        
        ds["RH"] = rh
        ds_variables.append("RH")


def rh_over100(ds, ds_variables, RH_threshold=1.15):
    
    if "RH" in ds_variables:
        
        rh_over100_ds = ds["RH"].where(ds["RH"] > RH_threshold)
        
        return rh_over100_ds


def RH_storage(ds, ds_variables, rh_over100_ds):
    
    if "RH" in ds_variables:
    
        RH_dict = {"RH": np.where(rh_over100_ds.notnull())}
            
        val_RH = ds["RH"].values[tuple(RH_dict["RH"])]
        time_RH = ds["RH"].time[tuple(RH_dict["RH"])[0]]
        lat_RH = ds["RH"].south_north[tuple(RH_dict["RH"])[1]]
        lon_RH = ds["RH"].west_east[tuple(RH_dict["RH"])[2]]
        
        time_list = list(time_RH.values)
        lat_list = list(lat_RH.values)
        lon_list = list(lon_RH.values)
        val_list = list(val_RH)
        
        dict = {"Time": [], "Lat": [], "Lon": [], "Value": []}
        
        for i in range(len(val_list)):
            dict["Time"].append(time_list[i])
            dict["Lat"].append(lat_list[i])
            dict["Lon"].append(lon_list[i])
            dict["Value"].append(val_list[i])
        
        RH_df = pd.DataFrame(dict)
            
        RH_df_dict = {"RH": RH_df}
        
        return RH_df_dict

