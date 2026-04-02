#!/usr/bin/env python3
"""
在所有JAXA数据中搜索14天缺失率最低的时间段
允许相邻选择日之间隔1-2天（不要求严格连续）

策略：
1. 扫描所有日期，每天选最低缺失率的小时
2. 用滑动窗口搜索：在一个~30天的窗口内选14天，允许跳天
3. 用动态规划在窗口内选最优14天组合
"""
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
import json
import multiprocessing as mp
from functools import partial

BASE_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
OUTPUT_FILE = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT/jaxa_best_14days_gapped.json')

NUM_DAYS_TO_SELECT = 14
MAX_GAP = 2            # 相邻选中日之间最多跳几天（1=可隔1天, 2=可隔2天）
MAX_SPAN = 30           # 搜索窗口跨度（天）
NUM_WORKERS = 32


def calculate_missing_rate(nc_file):
    """计算单个NC文件的缺失率"""
    try:
        with nc.Dataset(str(nc_file), 'r') as ds:
            sst = ds.variables['sea_surface_temperature'][:]
            if hasattr(ds.variables['sea_surface_temperature'], '_FillValue'):
                fill_value = ds.variables['sea_surface_temperature']._FillValue
                missing = np.isnan(sst) | (sst == fill_value)
            else:
                missing = np.isnan(sst)
            return missing.sum() / sst.size * 100
    except:
        return 100.0


def process_one_day(date_obj):
    """处理一天：找到缺失率最低的小时"""
    year_month = date_obj.strftime('%Y%m')
    day = date_obj.strftime('%d')
    day_dir = BASE_DIR / year_month / day

    if not day_dir.exists():
        return None

    hour_files = sorted(day_dir.glob('*.nc'))
    if not hour_files:
        return None

    best_file = None
    best_rate = 100.0

    for f in hour_files:
        rate = calculate_missing_rate(f)
        if rate < best_rate:
            best_rate = rate
            best_file = f

    if best_file is None:
        return None

    return {
        'date': date_obj,
        'date_str': date_obj.strftime('%Y-%m-%d'),
        'file': str(best_file),
        'rate': best_rate
    }


def find_best_subset(rates, n_select, max_gap):
    """
    在rates数组中选n_select个元素，使总和最小
    约束：相邻选中索引之间的距离 <= max_gap+1

    用动态规划：
    dp[i][j] = 前i个日期中选j个，且第j个是第i个时的最小平均缺失率之和
    """
    n = len(rates)
    if n < n_select:
        return None, float('inf')

    INF = float('inf')
    # dp[i][j] = 以rates[i]结尾，选了j个时的最小sum
    dp = [[INF] * (n_select + 1) for _ in range(n)]
    parent = [[-1] * (n_select + 1) for _ in range(n)]

    # 初始化：选第1个
    for i in range(n):
        dp[i][1] = rates[i]
        parent[i][1] = -1

    # 填表
    for j in range(2, n_select + 1):
        for i in range(j - 1, n):
            # 上一个选的位置 prev 必须满足 i - prev <= max_gap + 1
            for prev in range(max(0, i - max_gap - 1), i):
                if dp[prev][j - 1] + rates[i] < dp[i][j]:
                    dp[i][j] = dp[prev][j - 1] + rates[i]
                    parent[i][j] = prev

    # 找最优结尾
    best_end = -1
    best_sum = INF
    for i in range(n_select - 1, n):
        if dp[i][n_select] < best_sum:
            best_sum = dp[i][n_select]
            best_end = i

    if best_end == -1:
        return None, INF

    # 回溯路径
    path = []
    cur = best_end
    for j in range(n_select, 0, -1):
        path.append(cur)
        cur = parent[cur][j]
    path.reverse()

    return path, best_sum


def main():
    print("=" * 70)
    print("JAXA 低缺失率14天搜索（允许隔天）")
    print(f"选择天数: {NUM_DAYS_TO_SELECT}，最大间隔: {MAX_GAP} 天")
    print("=" * 70)

    # Step 1: 收集所有可用日期
    print("\n[1/3] 收集所有可用日期...")
    all_dates = []
    for ym_dir in sorted(BASE_DIR.iterdir()):
        if not ym_dir.is_dir() or not ym_dir.name.isdigit():
            continue
        for d_dir in sorted(ym_dir.iterdir()):
            if not d_dir.is_dir():
                continue
            try:
                date_obj = datetime.strptime(ym_dir.name + d_dir.name, '%Y%m%d')
                all_dates.append(date_obj)
            except:
                continue

    print(f"  共 {len(all_dates)} 天数据，时间范围: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")

    # Step 2: 并行计算每天的最低缺失率
    print(f"\n[2/3] 并行计算每天最低缺失率（{NUM_WORKERS} workers）...")
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_one_day, all_dates)

    # 过滤有效结果
    daily_data = {}
    for r in results:
        if r is not None:
            daily_data[r['date']] = r

    valid_dates = sorted(daily_data.keys())
    print(f"  有效天数: {len(valid_dates)}")

    # 打印整体统计
    all_rates = [daily_data[d]['rate'] for d in valid_dates]
    print(f"  缺失率范围: {min(all_rates):.1f}% ~ {max(all_rates):.1f}%")
    print(f"  中位数: {np.median(all_rates):.1f}%")

    # 显示缺失率最低的30天
    sorted_by_rate = sorted(valid_dates, key=lambda d: daily_data[d]['rate'])
    print(f"\n  缺失率最低的30天:")
    for d in sorted_by_rate[:30]:
        print(f"    {d.strftime('%Y-%m-%d')}: {daily_data[d]['rate']:.1f}%")

    # Step 3: 滑动窗口 + 动态规划搜索最优14天组合
    print(f"\n[3/3] 搜索最优{NUM_DAYS_TO_SELECT}天组合（窗口{MAX_SPAN}天，最大间隔{MAX_GAP}天）...")

    best_combo = None
    best_avg = float('inf')

    # 对每个可能的起始日期，取窗口内的日期做DP
    for start_idx in range(len(valid_dates)):
        start_date = valid_dates[start_idx]

        # 找窗口内的所有日期
        window_dates = []
        for idx in range(start_idx, len(valid_dates)):
            d = valid_dates[idx]
            span = (d - start_date).days
            if span > MAX_SPAN:
                break
            window_dates.append(d)

        if len(window_dates) < NUM_DAYS_TO_SELECT:
            continue

        # 构建天数间距数组（用于DP约束）
        # 我们需要考虑实际日历天间距，不仅是索引间距
        # 重新编码：把日期转为从start_date的天偏移
        day_offsets = [(d - start_date).days for d in window_dates]
        rates = [daily_data[d]['rate'] for d in window_dates]

        # DP: 在window_dates中选14个，相邻选中日的日历天间距 <= MAX_GAP+1
        n = len(window_dates)
        INF = float('inf')
        dp = [[INF] * (NUM_DAYS_TO_SELECT + 1) for _ in range(n)]
        parent = [[-1] * (NUM_DAYS_TO_SELECT + 1) for _ in range(n)]

        for i in range(n):
            dp[i][1] = rates[i]

        for j in range(2, NUM_DAYS_TO_SELECT + 1):
            for i in range(j - 1, n):
                for prev in range(i - 1, -1, -1):
                    # 日历天间距约束
                    gap_days = day_offsets[i] - day_offsets[prev]
                    if gap_days > MAX_GAP + 1:
                        break
                    if dp[prev][j - 1] + rates[i] < dp[i][j]:
                        dp[i][j] = dp[prev][j - 1] + rates[i]
                        parent[i][j] = prev

        # 找最优结尾
        for i in range(NUM_DAYS_TO_SELECT - 1, n):
            if dp[i][NUM_DAYS_TO_SELECT] < best_avg * NUM_DAYS_TO_SELECT:
                cur_sum = dp[i][NUM_DAYS_TO_SELECT]
                cur_avg = cur_sum / NUM_DAYS_TO_SELECT

                # 回溯
                path = []
                cur = i
                for j in range(NUM_DAYS_TO_SELECT, 0, -1):
                    path.append(cur)
                    cur = parent[cur][j]
                path.reverse()

                combo_dates = [window_dates[p] for p in path]
                best_combo = combo_dates
                best_avg = cur_avg

    if best_combo is None:
        print("未找到满足条件的组合！")
        return

    # 输出结果
    print(f"\n{'=' * 70}")
    print(f"找到最优{NUM_DAYS_TO_SELECT}天组合!")
    print(f"平均缺失率: {best_avg:.1f}%")
    print(f"时间跨度: {best_combo[0].strftime('%Y-%m-%d')} ~ {best_combo[-1].strftime('%Y-%m-%d')}")
    print(f"{'=' * 70}")

    results_list = []
    for i, d in enumerate(best_combo):
        info = daily_data[d]
        gap_str = ""
        if i > 0:
            gap = (d - best_combo[i - 1]).days
            gap_str = f"  (间隔 {gap} 天)"
        print(f"  {i + 1:2d}. {d.strftime('%Y-%m-%d')}: {info['rate']:.1f}%{gap_str}")
        results_list.append({
            'date': d.strftime('%Y-%m-%d'),
            'file': info['file'],
            'missing_rate': info['rate'],
            'gap_from_prev': (d - best_combo[i - 1]).days if i > 0 else 0
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存至: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
