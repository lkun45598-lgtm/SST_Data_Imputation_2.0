"""
在所有JAXA数据中搜索连续14天缺失率最低的时间段
"""
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json

def calculate_missing_rate(nc_file):
    """计算缺失率"""
    try:
        with nc.Dataset(nc_file, 'r') as ds:
            sst = ds.variables['sea_surface_temperature'][:]
            if hasattr(ds.variables['sea_surface_temperature'], '_FillValue'):
                fill_value = ds.variables['sea_surface_temperature']._FillValue
                missing = np.isnan(sst) | (sst == fill_value)
            else:
                missing = np.isnan(sst)
            return missing.sum() / sst.size * 100
    except:
        return 100.0

def get_best_hour_for_day(day_dir):
    """获取某天缺失率最低的小时"""
    hour_files = sorted(day_dir.glob('*.nc'))
    if not hour_files:
        return None, 100.0

    best_file = None
    best_rate = 100.0
    for f in hour_files:
        rate = calculate_missing_rate(f)
        if rate < best_rate:
            best_rate = rate
            best_file = f
    return best_file, best_rate

def main():
    base_dir = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')

    # 收集所有日期的最佳小时数据
    print("扫描所有JAXA数据...")
    daily_data = {}

    for year_month_dir in sorted(base_dir.iterdir()):
        if not year_month_dir.is_dir():
            continue
        year_month = year_month_dir.name

        for day_dir in sorted(year_month_dir.iterdir()):
            if not day_dir.is_dir():
                continue

            date_str = year_month + day_dir.name
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
            except:
                continue

            best_file, best_rate = get_best_hour_for_day(day_dir)
            if best_file:
                daily_data[date_obj] = {'file': str(best_file), 'rate': best_rate}
                print(f"{date_obj.strftime('%Y-%m-%d')}: {best_rate:.2f}%")

    # 查找连续14天缺失率最低的时间段
    print("\n搜索连续14天最佳时间段...")
    sorted_dates = sorted(daily_data.keys())

    best_start = None
    best_avg_rate = 100.0

    for i in range(len(sorted_dates) - 13):
        # 检查是否连续14天
        is_continuous = True
        for j in range(13):
            if (sorted_dates[i+j+1] - sorted_dates[i+j]).days != 1:
                is_continuous = False
                break

        if not is_continuous:
            continue

        # 计算这14天的平均缺失率
        rates = [daily_data[sorted_dates[i+j]]['rate'] for j in range(14)]
        avg_rate = np.mean(rates)

        if avg_rate < best_avg_rate:
            best_avg_rate = avg_rate
            best_start = sorted_dates[i]

    if best_start is None:
        print("未找到连续14天的数据")
        return

    # 保存结果
    results = []
    for j in range(14):
        date = best_start + timedelta(days=j)
        results.append({
            'date': date.strftime('%Y-%m-%d'),
            'file': daily_data[date]['file'],
            'missing_rate': daily_data[date]['rate']
        })

    output_file = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT/jaxa_best_14days.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 找到最佳连续14天:")
    print(f"起始日期: {best_start.strftime('%Y-%m-%d')}")
    print(f"平均缺失率: {best_avg_rate:.2f}%")
    print(f"结果保存至: {output_file}")

if __name__ == '__main__':
    main()
