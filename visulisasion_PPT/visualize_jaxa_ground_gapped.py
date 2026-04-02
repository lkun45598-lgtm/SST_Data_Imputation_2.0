#!/usr/bin/env python3
"""
JAXA Ground Truth可视化 - 基于低缺失率14天搜索结果
读取 jaxa_best_14days_gapped.json 中的文件列表生成可视化
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import json

def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

def plot_jaxa_ground(nc_file, date_str, save_path, gap_from_prev=0):
    """绘制JAXA Ground Truth SST"""
    with nc.Dataset(nc_file, 'r') as ds:
        sst = ds.variables['sea_surface_temperature'][:]
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]

        sst = np.squeeze(sst)
        sst_celsius = sst - 273.15

        valid_data = sst_celsius[~np.isnan(sst_celsius)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 20, 30

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor('lightgray')

        sst_masked = np.ma.masked_where(np.isnan(sst_celsius), sst_celsius)
        im = ax.pcolormesh(lon_grid, lat_grid, sst_masked,
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')

        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(f'JAXA Ground Truth SST\n{date_str}',
                    fontsize=13, fontweight='bold', pad=10)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SST (°C)', fontsize=11)

        missing_rate = np.isnan(sst).sum() / sst.size * 100
        ax.text(0.02, 0.98, f'Missing: {missing_rate:.1f}%',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return missing_rate

def main():
    base_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT')
    json_file = base_dir / 'jaxa_best_14days_gapped.json'
    output_dir = base_dir / 'Jaxa_ground'
    output_dir.mkdir(exist_ok=True)

    with open(json_file, 'r') as f:
        days = json.load(f)

    print("=" * 60)
    print(f"JAXA Ground Truth 可视化 - {len(days)}天")
    print("=" * 60)

    # 先清除旧图片
    for old_png in output_dir.glob('jaxa_ground_*.png'):
        old_png.unlink()
        print(f"  删除旧图: {old_png.name}")

    rates = []
    for i, day_info in enumerate(days):
        date_str = day_info['date']
        nc_file = day_info['file']
        gap = day_info.get('gap_from_prev', 0)

        save_path = output_dir / f'jaxa_ground_{date_str}.png'
        rate = plot_jaxa_ground(nc_file, date_str, save_path, gap)
        rates.append(rate)

        gap_str = f"  (间隔{gap}天)" if gap > 1 else ""
        print(f"  {i+1:2d}/{len(days)}: {date_str} → 缺失率 {rate:.1f}%{gap_str}")

    print(f"\n平均缺失率: {np.mean(rates):.1f}%")
    print(f"图片保存在: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
