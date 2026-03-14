#!/usr/bin/env python3
"""
5-panel comparison: JAXA Raw -> Temporal Fill -> Lowpass Filter -> KNN Fill -> Post Filter
"""
import numpy as np
import h5py
import netCDF4 as nc
import matplotlib.pyplot as plt
from pathlib import Path

# 选择一个时间点进行可视化（2017年7月15日12:00）
TARGET_TIME = '2017-07-15T12:00:00'

# 数据路径
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
TEMPORAL_DIR = Path('/data/chla_data_imputation_data_260125/sst_temperal_data')
LOWPASS_DIR = Path('/data/chla_data_imputation_data_260125/sst_filtered')
KNN_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled_optimized')
POST_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f'Loading data for {TARGET_TIME}...')

# 1. 读取JAXA原始数据
jaxa_file = JAXA_RAW_DIR / '201707/15/20170715120000.nc'
with nc.Dataset(jaxa_file) as f:
    jaxa_raw = f.variables['sea_surface_temperature'][:].squeeze()
    lat = f.variables['lat'][:]
    lon = f.variables['lon'][:]

# 2-5. 读取处理后的数据（从Series 0中找到对应时间点）
with h5py.File(TEMPORAL_DIR / 'jaxa_weighted_series_00.h5', 'r') as f:
    timestamps = f['timestamps'][:]
    # 找到对应时间点的索引
    target_idx = None
    for i, ts in enumerate(timestamps):
        ts_str = ts.decode() if isinstance(ts, bytes) else ts
        if ts_str == TARGET_TIME:
            target_idx = i
            break

    if target_idx is None:
        print(f'Time {TARGET_TIME} not found, using frame 100')
        target_idx = 100

    temporal_fill = f['sst_data'][target_idx]
    lat = f['latitude'][:]
    lon = f['longitude'][:]

with h5py.File(LOWPASS_DIR / 'jaxa_filtered_00.h5', 'r') as f:
    lowpass_filtered = f['sst_data'][target_idx]

with h5py.File(KNN_DIR / 'jaxa_knn_filled_00.h5', 'r') as f:
    knn_filled = f['sst_data'][target_idx]

with h5py.File(POST_DIR / 'jaxa_filtered_00.h5', 'r') as f:
    post_filtered = f['sst_data'][target_idx]

print(f'Using frame index: {target_idx}')

# 创建5连图
fig = plt.figure(figsize=(25, 5))

titles = [
    '1. JAXA Raw',
    '2. Temporal Fill',
    '3. Lowpass Filter',
    '4. KNN Fill',
    '5. Post Filter'
]

data_list = [jaxa_raw, temporal_fill, lowpass_filtered, knn_filled, post_filtered]

# 统一色标范围
vmin, vmax = 285, 310

for i, (data, title) in enumerate(zip(data_list, titles)):
    ax = fig.add_subplot(1, 5, i+1)

    # 转换为摄氏度
    data_celsius = data - 273.15

    im = ax.imshow(data_celsius, cmap='RdYlBu_r', vmin=vmin-273.15, vmax=vmax-273.15,
                   origin='lower', aspect='auto')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('SST (°C)', fontsize=10)

plt.tight_layout()
output_file = OUTPUT_DIR / f'5panel_comparison_frame{target_idx}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'\nSaved to: {output_file}')
plt.close()

print('Done!')
