"""
JAXA数据可视化 - 连续14天
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import json

def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

def plot_jaxa_sst(nc_file, date_str, save_path):
    """绘制JAXA SST"""
    with nc.Dataset(nc_file, 'r') as ds:
        sst = ds.variables['sea_surface_temperature'][:]
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
