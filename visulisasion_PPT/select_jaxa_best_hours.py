"""
筛选JAXA数据：连续14天中每天缺失率最低的小时
"""
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
import json

def calculate_missing_rate(nc_file):
    """计算缺失率"""
    try:
        with nc.Dataset(nc_file, 'r') as ds:
            sst = ds.variables['sea_surface_temperature'][:]

            # 缺失值 (NaN或填充值)
            if hasattr(ds.variables['sea_surface_temperature'], '_FillValue'):
                fill_value = ds.variables['sea_surface_temperature']._FillValue
                missing = np.isnan(sst) | (sst == fill_value)
            else:
                missing = np.isnan(sst)

            total_pixels = sst.size
            missing_rate = missing.sum() / total_pixels * 100
            return missing_rate
    except Exception as e:
        print(f"Error reading {nc_file}: {e}")
        return 100.0

def main():
    base_dir = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')

    # 从2015-07-07开始连续14天
    start_date = datetime(2015, 7, 7)
    num_days = 14

    results = []

    print("开始筛选JAXA数据...")
    print("="*60)

    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)
        date_str = current_date.strftime('%Y%m%d')
        year_month = current_date.strftime('%Y%m')
        day = current_date.strftime('%d')

        day_dir = base_dir / year_month / day

        if not day_dir.exists():
            print(f"⚠ {date_str}: 目录不存在")
            continue

        # 获取该天所有小时文件
        hour_files = sorted(day_dir.glob(f'{date_str}*.nc'))

        if not hour_files:
            print(f"⚠ {date_str}: 无数据文件")
            continue

        # 计算每小时的缺失率
        best_hour = None
        best_rate = 100.0

        for hour_file in hour_files:
            rate = calculate_missing_rate(hour_file)
            if rate < best_rate:
                best_rate = rate
                best_hour = hour_file

        if best_hour is None:
            print(f"⚠ {date_str}: 所有文件读取失败")
            continue

        results.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'file': str(best_hour),
            'missing_rate': best_rate
        })

        print(f"{current_date.strftime('%Y-%m-%d')}: {best_hour.name} (缺失率: {best_rate:.2f}%)")

    # 保存结果
    output_file = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT/jaxa_selected_hours.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*60)
    print(f"✓ 筛选完成！结果保存至: {output_file}")
    print(f"平均缺失率: {np.mean([r['missing_rate'] for r in results]):.2f}%")

if __name__ == '__main__':
    main()
