#!/usr/bin/env python3
"""
5-panel comparison with styled visualization
JAXA Raw -> Temporal Fill -> Lowpass Filter -> KNN Fill -> Post Filter
"""
import numpy as np
import h5py
import netCDF4 as nc
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap

# 配置
TARGET_TIME = '2017-07-09T00:00:00'
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
TEMPORAL_DIR = Path('/data/chla_data_imputation_data_260125/sst_temperal_data')
LOWPASS_DIR = Path('/data/chla_data_imputation_data_260125/sst_filtered')
KNN_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled_optimized')
POST_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# 颜色配置
land_color = '#D3D3D3'
cloud_color = '#E8E8E8'
cmap_sst = 'RdYlBu_r'

def format_lon(x, pos):
    return f'{x:.0f}°E'

def format_lat(y, pos):
    return f'{y:.0f}°N'

print(f'Loading data for {TARGET_TIME}...')

# 1. 读取JAXA原始数据
jaxa_file = JAXA_RAW_DIR / '201707/09/20170709000000.nc'
with nc.Dataset(jaxa_file) as f:
    jaxa_raw = f.variables['sea_surface_temperature'][:].squeeze()
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]

# 2-5. 读取处理后的数据
with h5py.File(TEMPORAL_DIR / 'jaxa_weighted_series_00.h5', 'r') as f:
    timestamps = f['timestamps'][:]
    target_idx = None
    for i, ts in enumerate(timestamps):
        ts_str = ts.decode() if isinstance(ts, bytes) else ts
        if ts_str == TARGET_TIME:
            target_idx = i
            break
    if target_idx is None:
        target_idx = 72

    temporal_fill = f['sst_data'][target_idx]
    lat = f['latitude'][:]
    lon = f['longitude'][:]

with h5py.File(LOWPASS_DIR / 'jaxa_filtered_00.h5', 'r') as f:
    lowpass_filtered = f['sst_data'][target_idx]

with h5py.File(KNN_DIR / 'jaxa_knn_filled_00.h5', 'r') as f:
    knn_filled = f['sst_data'][target_idx]
    land_mask = f['land_mask'][:]

with h5py.File(POST_DIR / 'jaxa_filtered_00.h5', 'r') as f:
    post_filtered = f['sst_data'][target_idx]

print(f'Using frame index: {target_idx}')

# 转换为摄氏度
jaxa_raw_c = jaxa_raw - 273.15
temporal_fill_c = temporal_fill - 273.15
lowpass_filtered_c = lowpass_filtered - 273.15
knn_filled_c = knn_filled - 273.15
post_filtered_c = post_filtered - 273.15

# 创建网格
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 色标范围
vmin_sst, vmax_sst = 26, 32

# 创建图形
fig = plt.figure(figsize=(30, 6))

# 布局: [图1, 图2, 图3, 图4, 图5, cbar]
gs = gridspec.GridSpec(1, 6, figure=fig,
                      width_ratios=[1, 1, 1, 1, 1, 0.05],
                      wspace=0.15, hspace=0.1,
                      left=0.02, right=0.98, top=0.88, bottom=0.12)

axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
cbar_ax = fig.add_subplot(gs[0, 5])

# 陆地显示
land_display = np.ma.masked_where(land_mask == 0, land_mask)

titles = [
    'JAXA Raw\n(Original Observation)',
    'Temporal Fill\n(Time-weighted)',
    'Lowpass Filter\n(Fill regions only)',
    'KNN Fill\n(3D Spatiotemporal)',
    'Post Filter\n(Final Result)'
]

data_list = [jaxa_raw_c, temporal_fill_c, lowpass_filtered_c, knn_filled_c, post_filtered_c]

for idx, (ax, data, title) in enumerate(zip(axes, data_list, titles)):
    # 设置背景色
    ax.set_facecolor(cloud_color)

    # 绘制陆地
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 绘制SST数据
    data_plot = np.ma.masked_where((land_mask > 0) | np.isnan(data), data)
    im = ax.pcolormesh(lon_grid, lat_grid, data_plot,
                       cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')

    # 格式化坐标轴
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    if idx == 0:
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        ax.set_ylabel('Latitude', fontsize=11)
    else:
        ax.set_yticks([])

    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)

# 添加colorbar
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('SST (°C)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# 添加总标题
fig.suptitle(f'SST Data Processing Pipeline Comparison\n{TARGET_TIME}',
             fontsize=14, fontweight='bold', y=0.98)

# 保存
output_file = OUTPUT_DIR / f'5panel_styled_frame{target_idx}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'\nSaved to: {output_file}')
plt.close()

print('Done!')
