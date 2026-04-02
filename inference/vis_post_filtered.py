#!/usr/bin/env python3
"""
使用post-filtered数据进行推理和可视化
对比：原始观测 vs KNN+post-filter vs 模型预测
"""
import os
import sys
import torch
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

# ============================================================================
# 配置
# ============================================================================
POST_FILTERED_DIR = Path('/data/chla_data_imputation_data_260125/sst_post_filtered')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth'
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')

SERIES_ID = 0
WINDOW_SIZE = 30
STRIDE = 24
NORM_MEAN = 299.9221
NORM_STD = 2.6919
GPU_ID = 5
NUM_FRAMES = 5

# ============================================================================
# 工具函数
# ============================================================================
def fmt_lon(x, pos):
    return f'{x:.0f}°E' if x >= 0 else f'{-x:.0f}°W'

def fmt_lat(x, pos):
    return f'{x:.0f}°N' if x >= 0 else f'{-x:.0f}°S'

land_cmap = ListedColormap(['#D2B48C'])

# ============================================================================
# 加载数据
# ============================================================================
print("Loading post-filtered data...")
h5_path = POST_FILTERED_DIR / f'jaxa_knn_filled_{SERIES_ID:02d}.h5'
with h5py.File(h5_path, 'r') as f:
    sst_data = f['sst_data'][:]
    obs_mask = f['original_obs_mask'][:]
    land_mask = f['land_mask'][:]
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    timestamps = [ts.decode() if isinstance(ts, bytes) else ts for ts in f['timestamps'][:]]

T, H, W = sst_data.shape
ocean = (land_mask == 0)
print(f"Data shape: {sst_data.shape}")
print(f"Ocean pixels: {ocean.sum()}")

# ============================================================================
# 加载模型
# ============================================================================
print(f"\nLoading model: {MODEL_PATH}")
device = torch.device(f'cuda:{GPU_ID}')
model = FNO_CBAM_SST_Temporal(
    out_size=(H, W),
    modes1=80,
    modes2=64,
    width=64,
    depth=6
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("Model loaded")

# ============================================================================
# 推理
# ============================================================================
print("\nRunning inference...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 选择几个有代表性的帧
frame_indices = np.linspace(WINDOW_SIZE * STRIDE, T - 1, NUM_FRAMES, dtype=int)

for frame_idx in frame_indices:
    print(f"\nProcessing frame {frame_idx}...")

    # 构建输入窗口
    start_idx = frame_idx - WINDOW_SIZE * STRIDE
    if start_idx < 0:
        continue

    window_indices = [start_idx + i * STRIDE for i in range(WINDOW_SIZE)]
    sst_window = sst_data[window_indices]  # (30, H, W)
    obs_window = obs_mask[window_indices]

    # 归一化
    sst_norm = (sst_window - NORM_MEAN) / NORM_STD

    # 构建mask序列（1=缺失，0=观测）
    mask_window = 1 - obs_window

    # 转为tensor
    sst_tensor = torch.from_numpy(sst_norm).float().unsqueeze(0).to(device)  # (1, 30, H, W)
    mask_tensor = torch.from_numpy(mask_window).float().unsqueeze(0).to(device)  # (1, 30, H, W)

    # 推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)  # (1, 1, H, W)

    # 反归一化
    pred_sst = pred.squeeze().cpu().numpy() * NORM_STD + NORM_MEAN

    # Ground truth
    gt_sst = sst_data[frame_idx]

    # 原始观测（挖空未观测区域）
    raw_obs = gt_sst.copy()
    raw_obs[obs_mask[frame_idx] == 0] = np.nan

    # 误差（只在海洋区域计算）
    error = np.abs(pred_sst - gt_sst)
    error[land_mask == 1] = np.nan

    # ========================================================================
    # 可视化
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Post-Filtered Data Inference (Frame {frame_idx}, {timestamps[frame_idx]})',
                 fontsize=14, fontweight='bold')

    vmin, vmax = np.nanpercentile(gt_sst[ocean], [2, 98])
    vmin_c, vmax_c = vmin - 273.15, vmax - 273.15

    titles = [
        'Original Observations (with clouds)',
        'Post-Filtered Ground Truth',
        'FNO-CBAM Prediction',
        'Absolute Error (K)'
    ]

    data_list = [
        raw_obs - 273.15,
        gt_sst - 273.15,
        pred_sst - 273.15,
        error
    ]

    for ax, title, data in zip(axes.flat, titles, data_list):
        data_plot = data.copy()
        data_plot[land_mask == 1] = np.nan

        if 'Error' in title:
            im = ax.pcolormesh(lon, lat, data_plot, cmap='hot_r',
                               vmin=0, vmax=2, shading='auto')
            cbar_label = 'Error (K)'
        else:
            im = ax.pcolormesh(lon, lat, data_plot, cmap='RdYlBu_r',
                               vmin=vmin_c, vmax=vmax_c, shading='auto')
            cbar_label = 'SST (°C)'

        # Land overlay
        land_disp = np.where(land_mask == 1, 1.0, np.nan)
        ax.pcolormesh(lon, lat, land_disp, cmap=land_cmap, vmin=0, vmax=1, shading='auto')

        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
        plt.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)

    # 统计信息
    mae = np.nanmean(error[ocean])
    rmse = np.sqrt(np.nanmean(error[ocean]**2))
    obs_rate = obs_mask[frame_idx][ocean].mean() * 100

    fig.text(0.5, 0.02,
             f'MAE: {mae:.3f}K | RMSE: {rmse:.3f}K | Obs Rate: {obs_rate:.1f}%',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = OUTPUT_DIR / f'inference_post_filtered_frame{frame_idx}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  MAE: {mae:.3f}K, RMSE: {rmse:.3f}K, Obs: {obs_rate:.1f}%")

print("\nDone!")
