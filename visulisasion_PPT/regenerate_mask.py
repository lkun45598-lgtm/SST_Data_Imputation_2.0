"""只重新生成mask图 - 不需要GPU"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.ostia_dataset import SSTDatasetTemporal

def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

def plot_mask(mask, land_mask, lon_coords, lat_coords, date_str, save_path):
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('white')

    # 陆地
    land_color = '#D2B48C'
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # Mask - 新配色：蓝色(观测) -> 橙红色(缺失)
    ocean_mask = 1 - land_mask
    mask_display = mask * ocean_mask
    mask_display = np.ma.masked_where(land_mask > 0, mask_display)

    colors = ['#2E86AB', '#A23B72']
    cmap_custom = LinearSegmentedColormap.from_list('custom', colors, N=100)
    im = ax.pcolormesh(lon_grid, lat_grid, mask_display,
                       cmap=cmap_custom, vmin=0, vmax=1, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'OSTIA Missing Mask\n{date_str}', fontsize=13, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.set_label('Mask', fontsize=11)
    cbar.ax.set_yticklabels(['Observed', 'Missing'])

    missing_ratio = np.sum(mask * ocean_mask) / np.sum(ocean_mask) * 100
    ax.text(0.02, 0.98, f'Missing: {missing_ratio:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 主程序
data_dir = '/data/sst_data/sst_missing_value_imputation/processed_data'
output_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT/Ostia_mask')

with h5py.File(f'{data_dir}/processed_sst_valid.h5', 'r') as f:
    lon_coords = f['longitude'][:]
    lat_coords = f['latitude'][:]

train_dataset = SSTDatasetTemporal(hdf5_path=f'{data_dir}/processed_sst_train.h5', normalize=True)
valid_dataset = SSTDatasetTemporal(hdf5_path=f'{data_dir}/processed_sst_valid.h5', normalize=True,
                                  mean=train_dataset.mean, std=train_dataset.std)

base_date = datetime(2015, 1, 1)
for day_idx in range(14):
    sample = valid_dataset[day_idx]
    mask = sample['mask_seq'][-1]
    land_mask = sample['land_mask']
    date_str = (base_date + timedelta(days=day_idx)).strftime('%Y-%m-%d')

    save_path = output_dir / f'ostia_mask_{date_str}.png'
    plot_mask(mask, land_mask, lon_coords, lat_coords, date_str, save_path)
    print(f'✓ {date_str}')

print('\n完成！新配色mask图已生成。')
