#!/usr/bin/env python3
"""
Daily Dataset Missing Rate Analysis & Visualization

分析间隔24h的JAXA数据集:
1. 各daily帧的缺失率变化（原始 → 时序填充 → KNN填充）
2. 可视化sample帧的3阶段对比
3. 统计24小时合并覆盖率效果

NPY cache中的mask含义:
  obs=1: 直接卫星观测
  obs=0, miss=0: 时序反距离加权填充成功
  miss=1: 时序填充后仍缺失（由KNN补齐）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import h5py

NPY_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache')
KNN_H5_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visualization/output')
SERIES_ID = 8


def load_data(series_id):
    sst = np.load(str(NPY_DIR / f'sst_{series_id:02d}.npy'), mmap_mode='r')
    obs = np.load(str(NPY_DIR / f'obs_{series_id:02d}.npy'), mmap_mode='r')
    miss = np.load(str(NPY_DIR / f'miss_{series_id:02d}.npy'), mmap_mode='r')
    land = np.load(str(NPY_DIR / f'land_{series_id:02d}.npy'), mmap_mode='r')
    h5_path = KNN_H5_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'
    with h5py.File(str(h5_path), 'r') as f:
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in f['timestamps'][:]]
    return sst, obs, miss, land, lat, lon, timestamps


def compute_daily_stats(obs, miss, land, stride=24):
    T = obs.shape[0]
    ocean = (land == 0)
    oc = ocean.sum()
    daily_indices = list(range(0, T, stride))

    orig_miss = []
    after_temp = []
    combined_miss = []

    for h in daily_indices:
        orig_miss.append((1 - obs[h][ocean].sum() / oc) * 100)
        after_temp.append(miss[h][ocean].sum() / oc * 100)

        start = max(0, h - 23)
        comb = np.zeros_like(ocean, dtype=bool)
        for f in range(start, min(h + 1, T)):
            comb |= (obs[f] == 1)
        comb &= ocean
        combined_miss.append((1 - comb.sum() / oc) * 100)

    return (np.array(daily_indices), np.array(orig_miss),
            np.array(after_temp), np.array(combined_miss))


def plot_missing_rate(daily_indices, orig, after_temp, combined, timestamps, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    N = len(daily_indices)
    x = np.arange(N)

    dates = []
    for h in daily_indices:
        ts = timestamps[h] if h < len(timestamps) else f'h={h}'
        dates.append(ts.split('T')[0] if 'T' in ts else ts[:10])

    ax = axes[0]
    ax.fill_between(x, orig, alpha=0.25, color='#e74c3c')
    ax.plot(x, orig, '-', color='#e74c3c', lw=1.5, alpha=0.8,
            label=f'Single-hour missing (avg: {orig.mean():.1f}%)')
    ax.fill_between(x, combined, alpha=0.25, color='#f39c12')
    ax.plot(x, combined, '-', color='#f39c12', lw=1.5, alpha=0.8,
            label=f'Combined 24h remaining (avg: {combined.mean():.1f}%)')
    ax.fill_between(x, after_temp, alpha=0.25, color='#27ae60')
    ax.plot(x, after_temp, '-', color='#27ae60', lw=1.5, alpha=0.8,
            label=f'After temporal fill (avg: {after_temp.mean():.1f}%)')

    ax.set_ylabel('Missing Rate (%)')
    ax.set_title(f'JAXA Daily Dataset (stride=24h) Missing Rate — Series {SERIES_ID}\n'
                 f'({dates[0]} to {dates[-1]}, {N} daily frames)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)

    step = max(1, N // 8)
    ticks = list(range(0, N, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks], rotation=30, ha='right')
    ax.set_xlim(-1, N)

    ax2 = axes[1]
    improvement = orig - after_temp
    ax2.bar(x, improvement, width=1.0, color='#3498db', alpha=0.7)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_xlabel('Date')
    ax2.set_title(f'Temporal Fill Improvement (avg: {improvement.mean():.1f}%)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([dates[i] for i in ticks], rotation=30, ha='right')
    ax2.set_xlim(-1, N)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_frame_comparison(sst, obs, miss, land, lat, lon, timestamps,
                          d_idx, h, save_path):
    H, W = land.shape
    ocean = (land == 0)
    oc = ocean.sum()

    frame_sst = sst[h]
    frame_obs = obs[h]
    frame_miss = miss[h]

    observed = (frame_obs == 1) & ocean
    temporal = (frame_obs == 0) & (frame_miss == 0) & ocean
    knn = (frame_miss == 1) & ocean

    obs_pct = observed.sum() / oc * 100
    temp_pct = temporal.sum() / oc * 100
    knn_pct = knn.sum() / oc * 100

    ts = timestamps[h] if h < len(timestamps) else f'h={h}'
    date_str = ts.split('T')[0] if 'T' in ts else ts[:10]

    valid = frame_sst[ocean & ~np.isnan(frame_sst)]
    vmin, vmax = np.percentile(valid, [2, 98])

    lon_g, lat_g = np.meshgrid(lon, lat)
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='#d0d0d0')

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # (a) Original observation only
    ax = axes[0]
    d = np.full((H, W), np.nan)
    d[observed] = frame_sst[observed]
    im = ax.pcolormesh(lon_g, lat_g, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f'(a) Single-hour Obs\nCoverage: {obs_pct:.1f}%', fontweight='bold')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SST (K)')

    # (b) After temporal fill
    ax = axes[1]
    d = np.full((H, W), np.nan)
    d[observed | temporal] = frame_sst[observed | temporal]
    im = ax.pcolormesh(lon_g, lat_g, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f'(b) After Temporal Fill\nCoverage: {obs_pct+temp_pct:.1f}% (+{temp_pct:.1f}%)',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SST (K)')

    # (c) After KNN fill
    ax = axes[2]
    d = frame_sst.copy()
    d[land == 1] = np.nan
    im = ax.pcolormesh(lon_g, lat_g, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f'(c) After KNN Fill\n100% (+{knn_pct:.1f}% by KNN)', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='SST (K)')

    # (d) Source map
    ax = axes[3]
    m = np.full((H, W), np.nan)
    m[observed] = 0
    m[temporal] = 1
    m[knn] = 2
    m[land == 1] = 3
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
    mcmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], mcmap.N)
    ax.pcolormesh(lon_g, lat_g, m, cmap=mcmap, norm=norm, shading='auto')
    ax.set_title(f'(d) Data Source Map\n{date_str}', fontweight='bold')
    ax.legend(handles=[
        Patch(facecolor='#3498db', label=f'Observed ({obs_pct:.1f}%)'),
        Patch(facecolor='#2ecc71', label=f'Temporal ({temp_pct:.1f}%)'),
        Patch(facecolor='#e74c3c', label=f'KNN ({knn_pct:.1f}%)'),
        Patch(facecolor='#95a5a6', label='Land'),
    ], loc='lower left', fontsize=8, framealpha=0.9)

    for a in axes:
        a.set_xlabel('Longitude')

    fig.suptitle(f'Day {d_idx} ({date_str}, h={h})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Frame saved: {save_path} (obs={obs_pct:.1f}%, temp={temp_pct:.1f}%, knn={knn_pct:.1f}%)')


def main():
    print('=' * 70)
    print('Daily Dataset Missing Rate Analysis')
    print('=' * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'\nLoading series {SERIES_ID}...')
    sst, obs, miss, land, lat, lon, timestamps = load_data(SERIES_ID)
    T = sst.shape[0]
    ocean = (land == 0)
    print(f'  Shape: {sst.shape}, ocean: {ocean.sum()} px')

    print(f'\nComputing daily statistics...')
    daily_idx, orig, after_temp, combined = compute_daily_stats(obs, miss, land)
    N = len(daily_idx)
    print(f'  {N} daily frames')
    print(f'  Original missing:     avg={orig.mean():.1f}%')
    print(f'  Combined 24h remain:  avg={combined.mean():.1f}%')
    print(f'  After temporal fill:  avg={after_temp.mean():.1f}%')
    print(f'  Improvement:          avg={np.mean(orig - after_temp):.1f}%')

    print(f'\nPlotting missing rate...')
    plot_missing_rate(daily_idx, orig, after_temp, combined, timestamps,
                      OUTPUT_DIR / 'daily_missing_rate_series08.png')

    print(f'\nPlotting frame comparisons...')
    sorted_idx = np.argsort(orig)
    samples = [
        sorted_idx[N // 10],
        sorted_idx[N // 4],
        sorted_idx[N // 2],
        sorted_idx[3 * N // 4],
        sorted_idx[9 * N // 10],
    ]
    for i, di in enumerate(samples):
        h = daily_idx[di]
        plot_frame_comparison(sst, obs, miss, land, lat, lon, timestamps,
                              di, h, OUTPUT_DIR / f'daily_frame_{i:02d}_day{di}.png')

    print(f'\nDone! Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
