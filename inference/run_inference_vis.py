#!/usr/bin/env python3
"""
FNO-CBAM SST重建可视化 - 5连图 (参考 plot_reconstruction_5panel.py 风格)

1. KNN填充后的完整图 (Ground Truth)
2. KNN填充后经过mask的图（显示人工挖空区域）
3. JAXA原始数据（真实云缺失）
4. FNO预测+高斯滤波后的图
5. 误差图 (仅人工挖空区域)
"""

import os
import sys
import torch
import numpy as np
import h5py
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datetime import datetime, timedelta
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal
from training.train_jaxa import output_composition

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
NPY_DIR = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/sst_knn_npy_cache'
KNN_H5_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth'
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/inference_results')

SERIES_ID = 8
WINDOW_SIZE = 30
STRIDE = 24
NORM_MEAN = 299.9221
NORM_STD = 2.6919
GPU_ID = 7
GAUSSIAN_SIGMA = 1.0
NUM_SAMPLES = 20
MASK_RATIO = 0.2


# ============================================================================
# matplotlib 设置
# ============================================================================
def setup_matplotlib():
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=1.5, labelsize=12)
    plt.rc('lines', linewidth=1.5)
    plt.rcParams.update({
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.pad': 5, 'ytick.major.pad': 5,
    })


def format_lon(x, pos):
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"

def format_lat(y, pos):
    return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"


# ============================================================================
# 数据加载
# ============================================================================
def load_npy_data(npy_dir, series_id):
    sst = np.load(f'{npy_dir}/sst_{series_id:02d}.npy', mmap_mode='r')
    obs = np.load(f'{npy_dir}/obs_{series_id:02d}.npy', mmap_mode='r')
    miss = np.load(f'{npy_dir}/miss_{series_id:02d}.npy', mmap_mode='r')
    land = np.load(f'{npy_dir}/land_{series_id:02d}.npy', mmap_mode='r')
    return sst, obs, miss, land


def load_h5_metadata(h5_dir, series_id):
    """从H5加载经纬度坐标和时间戳"""
    h5_path = h5_dir / f'jaxa_knn_filled_{series_id:02d}.h5'
    with h5py.File(h5_path, 'r') as f:
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts
                      for ts in f['timestamps'][:]]
        land_mask = f['land_mask'][:]
    return lat, lon, timestamps, land_mask


def load_jaxa_raw(date_str, hour=0):
    """加载JAXA原始L3数据"""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    day_dir = JAXA_RAW_DIR / f'{date.year:04d}{date.month:02d}' / f'{date.day:02d}'
    if not day_dir.exists():
        return None, None

    # 尝试指定小时，不存在则找最近的
    for h in [hour] + list(range(24)):
        fp = day_dir / f'{date.year:04d}{date.month:02d}{date.day:02d}{h:02d}0000.nc'
        if fp.exists():
            with nc.Dataset(fp, 'r') as f:
                sst = f.variables['sea_surface_temperature'][0, :, :]
            return sst, h
    return None, None


def apply_gaussian_filter(sst, land_mask, sigma=1.0):
    """高斯滤波 (只对海洋区域)"""
    result = sst.copy()
    valid = ~np.isnan(result) & (land_mask == 0)
    if valid.sum() == 0:
        return result
    mean_val = np.nanmean(result[valid])
    tmp = result.copy()
    tmp[~valid] = mean_val
    filtered = gaussian_filter(tmp, sigma=sigma)
    result = np.where(land_mask == 0, filtered, np.nan)
    return result


# ============================================================================
# Mask 生成
# ============================================================================
def generate_block_mask(obs_mask, land_mask, ratio=0.2, min_size=10, max_size=50, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    valid = (obs_mask * (1 - land_mask)).astype(bool)
    target = int(valid.sum() * ratio)
    if target < 10:
        return np.zeros_like(obs_mask, dtype=np.float32)
    H, W = obs_mask.shape
    mask = np.zeros((H, W), dtype=np.float32)
    count = 0
    for _ in range(500):
        if count >= target:
            break
        bh, bw = rng.randint(min_size, max_size+1), rng.randint(min_size, max_size+1)
        y, x = rng.randint(0, H-bh+1), rng.randint(0, W-bw+1)
        if valid[y:y+bh, x:x+bw].sum() > 0:
            mask[y:y+bh, x:x+bw] = 1.0
            count = int((mask * valid).sum())
    return mask


# ============================================================================
# 模型推理
# ============================================================================
def run_inference(model, sst_seq, mask_seq, device):
    sst_norm = (sst_seq - NORM_MEAN) / NORM_STD
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)
    sst_t = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(sst_t, mask_t)
    pred_composed = output_composition(pred, sst_t, mask_t)
    return pred_composed.squeeze().cpu().numpy() * NORM_STD + NORM_MEAN


# ============================================================================
# 5面板可视化 (参考 plot_reconstruction_5panel.py)
# ============================================================================
def create_5panel_plot(knn_filled, artificial_mask, jaxa_raw, fno_gaussian,
                       land_mask, lon_coords, lat_coords, date_str,
                       save_path, jaxa_hour=None):
    """
    1. KNN Filled (Complete) - Ground Truth
    2. KNN + Artificial Mask (显示挖空)
    3. JAXA Raw (真实云缺失)
    4. FNO + Gaussian (模型预测)
    5. |Error| (人工挖空区域)
    """
    setup_matplotlib()

    # 转摄氏度
    knn_c = knn_filled - 273.15
    fno_c = fno_gaussian - 273.15
    jaxa_c = jaxa_raw - 273.15 if jaxa_raw is not None else None

    ocean = land_mask == 0
    masked_region = (artificial_mask > 0) & ocean

    # 误差
    error = np.abs(fno_c - knn_c)
    if masked_region.sum() > 0:
        mae = np.nanmean(error[masked_region])
        rmse = np.sqrt(np.nanmean(error[masked_region]**2))
        max_err = np.nanmax(error[masked_region])
    else:
        mae = rmse = max_err = 0

    mask_ratio = masked_region.sum() / ocean.sum() * 100

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 颜色
    cmap_sst = 'RdYlBu_r'
    cmap_error = 'hot_r'
    land_color = '#D2B48C'
    cloud_color = '#E8E8E8'

    valid_sst = knn_c[ocean & ~np.isnan(knn_c)]
    vmin_sst = np.percentile(valid_sst, 2) if len(valid_sst) > 0 else 26
    vmax_sst = np.percentile(valid_sst, 98) if len(valid_sst) > 0 else 32

    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 布局: [图1, 图2, 图3, 图4, cbar_sst, 图5, cbar_err]
    fig = plt.figure(figsize=(36, 7))
    gs = gridspec.GridSpec(1, 8, figure=fig,
                           width_ratios=[1, 1, 1, 1, 0.06, 1, 0.06, 0.02],
                           wspace=0.12, left=0.02, right=0.99, top=0.85, bottom=0.12)

    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 3]]
    axes.append(fig.add_subplot(gs[0, 5]))

    # ===== 1. KNN Filled (Complete) =====
    ax = axes[0]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    im1 = ax.pcolormesh(lon_grid, lat_grid,
                        np.ma.masked_where(land_mask > 0, knn_c),
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4); ax.locator_params(axis='y', nbins=5)
    ax.set_title('KNN Filled (Complete)\nGround Truth', fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('Latitude', fontsize=11); ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)

    # ===== 2. KNN + Artificial Mask =====
    ax = axes[1]
    ax.set_facecolor(cloud_color)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    knn_with_mask = knn_c.copy()
    knn_with_mask[artificial_mask > 0] = np.nan
    ax.pcolormesh(lon_grid, lat_grid,
                  np.ma.masked_where((land_mask > 0) | np.isnan(knn_with_mask), knn_with_mask),
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'KNN + Mask on Observed\n(Masked: {mask_ratio:.1f}% of ocean)',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1); ax.set_yticks([])

    # ===== 3. JAXA Raw =====
    ax = axes[2]
    ax.set_facecolor(cloud_color)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    if jaxa_c is not None:
        jaxa_plot = np.ma.masked_where((land_mask > 0) | np.isnan(jaxa_c), jaxa_c)
        ax.pcolormesh(lon_grid, lat_grid, jaxa_plot,
                      cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
        jaxa_valid_rate = ((~np.isnan(jaxa_c)) & ocean).sum() / ocean.sum() * 100
        hour_str = f'{jaxa_hour:02d}:00 UTC' if jaxa_hour is not None else ''
        title = f'JAXA Raw ({hour_str})\n(Valid: {jaxa_valid_rate:.1f}%)'
    else:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        title = 'JAXA Raw\n(No Data)'
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1); ax.set_yticks([])

    # ===== 4. FNO + Gaussian =====
    ax = axes[3]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    im4 = ax.pcolormesh(lon_grid, lat_grid,
                        np.ma.masked_where(land_mask > 0, fno_c),
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'FNO-CBAM + Gaussian\n(σ={GAUSSIAN_SIGMA})',
                 fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1); ax.set_yticks([])

    # SST Colorbar
    cax_sst = fig.add_subplot(gs[0, 4])
    cbar_sst = plt.colorbar(im4, cax=cax_sst)
    cbar_sst.set_label('SST (°C)', fontsize=11)

    # ===== 5. Error (人工挖空区域) =====
    ax = axes[4]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')
    error_display = error.copy()
    error_display[artificial_mask == 0] = np.nan
    im5 = ax.pcolormesh(lon_grid, lat_grid,
                        np.ma.masked_where((land_mask > 0) | (artificial_mask == 0), error_display),
                        cmap=cmap_error, vmin=0, vmax=1.0, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('|FNO - KNN| Error\n(Masked Region)', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1); ax.set_yticks([])
    ax.text(0.02, 0.98, f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nMax: {max_err:.3f}°C',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    # Error Colorbar
    cax_err = fig.add_subplot(gs[0, 6])
    cbar_err = plt.colorbar(im5, cax=cax_err)
    cbar_err.set_label('|Error| (°C)', fontsize=11)

    # 图例
    fig.legend(handles=[
        Patch(facecolor=land_color, edgecolor='black', label='Land'),
        Patch(facecolor=cloud_color, edgecolor='black', label='Cloud/Missing'),
    ], loc='upper right', bbox_to_anchor=(0.995, 0.98), fontsize=10, framealpha=0.9)

    fig.text(0.5, 0.98, f'JAXA SST Reconstruction: {date_str}',
             ha='center', va='top', fontsize=16, fontweight='bold')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return mae, rmse, max_err


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("FNO-CBAM SST Reconstruction - 5 Panel Visualization")
    print("=" * 70)

    device = torch.device(f'cuda:{GPU_ID}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = FNO_CBAM_SST_Temporal(out_size=(451, 351), modes1=80, modes2=64,
                                   width=64, depth=6).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Epoch {ckpt['epoch']+1}, val_mae={ckpt.get('val_mae','N/A')}")

    # 加载npy数据
    print(f"\n加载 Series {SERIES_ID} npy数据...")
    sst, obs, miss, land = load_npy_data(NPY_DIR, SERIES_ID)
    print(f"  {sst.shape[0]} frames, shape={sst.shape[1:]}")

    # 加载H5元数据 (经纬度, 时间戳)
    print("加载H5元数据 (lat/lon/timestamps)...")
    lat, lon, timestamps, land_mask = load_h5_metadata(KNN_H5_DIR, SERIES_ID)
    print(f"  lat: [{lat[0]:.2f}, {lat[-1]:.2f}], lon: [{lon[0]:.2f}, {lon[-1]:.2f}]")
    print(f"  时间范围: {timestamps[0]} → {timestamps[-1]} ({len(timestamps)} days)")

    # 选取均匀分布的日期索引 (H5 daily index → npy hourly index 映射: day_idx * 24)
    n_days = len(timestamps)
    day_indices = np.linspace(WINDOW_SIZE, n_days - 1, NUM_SAMPLES, dtype=int)

    print(f"\n开始推理 ({NUM_SAMPLES} 个样本)...")
    rng = np.random.RandomState(42)
    all_mae, all_rmse = [], []

    for i, day_idx in enumerate(day_indices):
        date_str = timestamps[day_idx][:10]  # YYYY-MM-DD
        hourly_idx = day_idx * STRIDE  # H5 daily → npy hourly

        # 构建30天窗口 (与训练一致)
        frame_indices = [max(0, hourly_idx - t * STRIDE)
                         for t in range(WINDOW_SIZE - 1, -1, -1)]
        sst_seq = sst[frame_indices].copy()
        mask_seq = miss[frame_indices].copy()

        # 目标帧
        gt_sst = sst[hourly_idx].copy()
        obs_mask = obs[hourly_idx].copy()

        # 人工挖空 (与训练一致)
        artificial_mask = generate_block_mask(obs_mask, land_mask, MASK_RATIO, rng=rng)
        loss_mask = (artificial_mask * obs_mask * (1 - land_mask)).astype(np.float32)

        # 修改第30天输入 (与训练一致)
        sst_seq_input = sst_seq.copy()
        sst_seq_input[-1] = np.where(artificial_mask == 1, NORM_MEAN, sst_seq[-1])
        mask_seq_input = mask_seq.copy()
        mask_seq_input[-1] = artificial_mask

        # 模型推理
        pred_sst = run_inference(model, sst_seq_input, mask_seq_input, device)

        # Output composition + Gaussian
        fno_composed = np.where(artificial_mask > 0, pred_sst, gt_sst)
        fno_gaussian = apply_gaussian_filter(fno_composed, land_mask, sigma=GAUSSIAN_SIGMA)

        # 加载JAXA原始数据
        jaxa_raw, jaxa_hour = load_jaxa_raw(date_str, hour=0)

        # 计算指标
        valid = loss_mask > 0
        if valid.sum() > 0:
            err = np.abs(fno_gaussian[valid] - gt_sst[valid])
            mae = err.mean()
            rmse = np.sqrt((err**2).mean())
            all_mae.append(mae)
            all_rmse.append(rmse)
        else:
            mae = rmse = 0

        # 5面板可视化
        save_path = OUTPUT_DIR / f'recon_5panel_{date_str.replace("-","")}.png'
        create_5panel_plot(
            knn_filled=gt_sst,
            artificial_mask=artificial_mask,
            jaxa_raw=jaxa_raw,
            fno_gaussian=fno_gaussian,
            land_mask=land_mask,
            lon_coords=lon,
            lat_coords=lat,
            date_str=date_str,
            save_path=save_path,
            jaxa_hour=jaxa_hour
        )

        print(f"  [{i+1}/{NUM_SAMPLES}] {date_str}: MAE={mae:.4f}°C, RMSE={rmse:.4f}°C, Points={int(valid.sum())}")

    # 汇总
    print(f"\n{'='*70}")
    print(f"推理完成! 图片保存在: {OUTPUT_DIR}")
    print(f"{'='*70}")
    print(f"  样本数: {NUM_SAMPLES}")
    print(f"  平均 MAE:  {np.mean(all_mae):.4f}°C")
    print(f"  平均 RMSE: {np.mean(all_rmse):.4f}°C")
    print(f"  MAE 范围:  [{np.min(all_mae):.4f}, {np.max(all_mae):.4f}]°C")
    print(f"  最好: {np.min(all_mae):.4f}°C | 最差: {np.max(all_mae):.4f}°C")


if __name__ == '__main__':
    main()
