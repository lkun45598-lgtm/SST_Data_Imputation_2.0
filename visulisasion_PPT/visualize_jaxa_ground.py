"""
JAXA Ground Truth可视化 - 连续14天
使用2015-07-23到2015-08-05（缺失率较低的时间段）
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
from tqdm import tqdm

def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"

def get_best_hour_for_day(day_dir):
    """获取某天缺失率最低的小时文件"""
    hour_files = sorted(day_dir.glob('*.nc'))
    best_file = None
    best_rate = 100.0

    for f in hour_files:
        try:
            with nc.Dataset(f, 'r') as ds:
                sst = ds.variables['sea_surface_temperature'][:]
                missing = np.isnan(sst)
                rate = missing.sum() / sst.size * 100
                if rate < best_rate:
                    best_rate = rate
                    best_file = f
        except:
            continue

    return best_file, best_rate

def plot_jaxa_ground(nc_file, date_str, save_path):
    """绘制JAXA Ground Truth SST"""
    with nc.Dataset(nc_file, 'r') as ds:
        sst = ds.variables['sea_surface_temperature'][:]
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]

        # 转换为摄氏度
        sst_celsius = sst - 273.15

        # 计算有效数据范围
        valid_data = sst_celsius[~np.isnan(sst_celsius)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 20, 30

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor('lightgray')

        # 绘制SST（缺失区域显示为灰色背景）
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

        # 统计缺失率
        missing_rate = np.isnan(sst).sum() / sst.size * 100
        ax.text(0.02, 0.98, f'Missing: {missing_rate:.1f}%',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    base_dir = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
    output_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT/Jaxa_ground')
    output_dir.mkdir(exist_ok=True)

    # 使用2015-07-23到2015-08-05（缺失率较低）
    start_date = datetime(2015, 7, 23)
    num_days = 14

    print("="*60)
    print("JAXA Ground Truth 14天可视化")
    print(f"时间段: {start_date.strftime('%Y-%m-%d')} 到 {(start_date + timedelta(days=13)).strftime('%Y-%m-%d')}")
    print("="*60)

    for day_idx in tqdm(range(num_days), desc="生成可视化"):
        current_date = start_date + timedelta(days=day_idx)
        year_month = current_date.strftime('%Y%m')
        day = current_date.strftime('%d')
        date_str = current_date.strftime('%Y-%m-%d')

        day_dir = base_dir / year_month / day

        if not day_dir.exists():
            print(f"\n⚠ {date_str}: 目录不存在")
            continue

        # 获取该天缺失率最低的小时
        best_file, missing_rate = get_best_hour_for_day(day_dir)

        if best_file is None:
            print(f"\n⚠ {date_str}: 无有效数据")
            continue

        # 生成可视化
        save_path = output_dir / f'jaxa_ground_{date_str}.png'
        plot_jaxa_ground(best_file, date_str, save_path)

    print("\n" + "="*60)
    print(f"✓ 完成！图片保存在: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
