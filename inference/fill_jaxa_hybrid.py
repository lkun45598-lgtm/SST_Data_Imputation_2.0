#!/usr/bin/env python3
"""
JAXA SST缺失值填充 - 混合输入方案

输入方案:
- 前29天: KNN填充的完整数据 (提供时序信息)
- 第30天: 时序加权数据 + 高斯预处理滤波

输出:
- 模型填充后的SST数据
- 高斯后处理滤波

作者: Claude Code
日期: 2026-01-24
"""

import os
import sys
import torch
import numpy as np
import h5py
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# 数据路径
KNN_FILLED_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
WEIGHTED_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_weighted_aligned')
OUTPUT_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_hybrid_output')
VIS_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_hybrid_visualization')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth'

# 模型参数
WINDOW_SIZE = 30  # 30天输入序列
GPU_ID = 0  # 使用的GPU

# 高斯滤波参数
PREPROCESS_SIGMA = 1.0  # 第30天输入预处理高斯滤波sigma
POSTPROCESS_SIGMA = 1.0  # 输出后处理高斯滤波sigma

# 处理配置
SERIES_IDS = [0]  # 要处理的序列ID
NUM_TEST_SAMPLES = 30  # 测试样本数 (设为None则处理全部)
VIS_INTERVAL = 3  # 可视化间隔


# ============================================================================
# Gaussian Filter Functions
# ============================================================================

def apply_gaussian_filter(sst_data, mask, sigma=1.0):
    """
    对SST数据应用高斯滤波

    Args:
        sst_data: [H, W] SST数据 (Kelvin)
        mask: [H, W] 缺失掩码 (1=缺失, 0=有效) 或陆地掩码
        sigma: 高斯滤波的标准差

    Returns:
        filtered_sst: [H, W] 滤波后的SST数据
    """
    sst = sst_data.copy()

    # 有效区域
    mask_valid = ~np.isnan(sst) & (mask == 0)

    if mask_valid.sum() == 0:
        return sst

    # 用均值临时填充无效区域
    mean_val = np.nanmean(sst)
    sst_for_filter = sst.copy()
    sst_for_filter[~mask_valid] = mean_val

    # 应用高斯滤波
    filtered = gaussian_filter(sst_for_filter, sigma=sigma)

    # 只在有效区域保留滤波结果，无效区域保持原值
    result = np.where(mask_valid, filtered, sst)

    return result


# ============================================================================
# Visualization Functions
# ============================================================================

def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=14)
    plt.rc('axes', linewidth=1.5, labelsize=14)
    plt.rc('lines', linewidth=1.5)
    params = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
    }
    plt.rcParams.update(params)


def format_lon(x, pos):
    return f"{abs(x):.1f}{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.1f}{'N' if y >= 0 else 'S'}"


def create_visualization(original_sst, knn_sst, model_filled_sst, smoothed_sst,
                         missing_mask, land_mask, lon_coords, lat_coords,
                         timestamp, save_path):
    """
    创建5面板可视化

    1. Original (时序加权+高斯预处理) - 有缺失
    2. KNN Filled - 粗填充
    3. Model Filled - 模型填充
    4. Gaussian Smoothed - 高斯后处理
    5. Difference (Smoothed - KNN) in missing regions
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(32, 7))
    gs = gridspec.GridSpec(1, 7, figure=fig,
                          width_ratios=[1, 1, 1, 1, 0.08, 1, 0.08],
                          wspace=0.12, hspace=0.1,
                          left=0.03, right=0.98, top=0.85, bottom=0.15)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_knn = fig.add_subplot(gs[0, 1])
    ax_model = fig.add_subplot(gs[0, 2])
    ax_smooth = fig.add_subplot(gs[0, 3])
    ax_diff = fig.add_subplot(gs[0, 5])

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 转换为摄氏度
    original_celsius = original_sst - 273.15
    knn_celsius = knn_sst - 273.15
    model_celsius = model_filled_sst - 273.15
    smooth_celsius = smoothed_sst - 273.15

    # 颜色设置
    cmap_sst = 'RdYlBu_r'
    cmap_diff = 'RdBu_r'
    land_color = '#D2B48C'

    # 数据范围
    ocean_mask = land_mask == 0
    valid_data = knn_celsius[ocean_mask & ~np.isnan(knn_celsius)]
    if len(valid_data) > 0:
        vmin_sst = np.percentile(valid_data, 2)
        vmax_sst = np.percentile(valid_data, 98)
    else:
        vmin_sst, vmax_sst = 20, 32

    # 陆地显示
    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # 计算缺失率
    ocean_pixels = np.sum(ocean_mask)
    missing_pixels = np.sum((missing_mask > 0) & ocean_mask)
    missing_rate = missing_pixels / ocean_pixels * 100 if ocean_pixels > 0 else 0

    # ===== 1. Original SST (时序加权+高斯预处理，有缺失) =====
    ax = ax_orig
    ax.set_facecolor('skyblue')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    orig_display = original_celsius.copy()
    orig_display[missing_mask > 0] = np.nan
    orig_masked = np.ma.masked_where((land_mask > 0) | np.isnan(orig_display), orig_display)

    ax.pcolormesh(lon_grid, lat_grid, orig_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title(f'Weighted+Gaussian Input\n(Missing: {missing_rate:.1f}%)',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_box_aspect(1)

    # ===== 2. KNN Filled SST =====
    ax = ax_knn
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    knn_masked = np.ma.masked_where((land_mask > 0) | np.isnan(knn_celsius), knn_celsius)
    ax.pcolormesh(lon_grid, lat_grid, knn_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('KNN Filled\n(Day 30)',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. Model Filled SST =====
    ax = ax_model
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    model_masked = np.ma.masked_where((land_mask > 0) | np.isnan(model_celsius), model_celsius)
    ax.pcolormesh(lon_grid, lat_grid, model_masked,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('FNO-CBAM Filled\n(Raw Output)',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 4. Gaussian Smoothed SST =====
    ax = ax_smooth
    ax.set_facecolor('lightgray')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    smooth_masked = np.ma.masked_where((land_mask > 0) | np.isnan(smooth_celsius), smooth_celsius)
    im4 = ax.pcolormesh(lon_grid, lat_grid, smooth_masked,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'Gaussian Post-processed\n(sigma={POSTPROCESS_SIGMA})',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # SST Colorbar
    cax4_container = fig.add_subplot(gs[0, 4])
    cax4_container.axis('off')
    cax4 = inset_axes(cax4_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax4_container.transAxes)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.set_label('SST (C)', fontsize=11)

    # ===== 5. Difference in Missing Regions =====
    ax = ax_diff
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    diff = smooth_celsius - knn_celsius
    diff_display = diff.copy()
    diff_display[(missing_mask == 0) | (land_mask > 0)] = np.nan
    diff_masked = np.ma.masked_where(np.isnan(diff_display), diff_display)

    vmax_diff = 2.0
    im5 = ax.pcolormesh(lon_grid, lat_grid, diff_masked,
                        cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff, shading='auto')
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('Smoothed - KNN\n(Missing Regions)',
                fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 差异统计
    if np.sum(~np.isnan(diff_display)) > 0:
        mean_diff = np.nanmean(diff_display)
        std_diff = np.nanstd(diff_display)
        ax.text(0.02, 0.98, f'Mean: {mean_diff:.3f}C\nStd: {std_diff:.3f}C',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Diff Colorbar
    cax5_container = fig.add_subplot(gs[0, 6])
    cax5_container.axis('off')
    cax5 = inset_axes(cax5_container, width="50%", height="90%", loc='center',
                      bbox_to_anchor=(-0.5, 0, 1, 1), bbox_transform=cax5_container.transAxes)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar5.set_label('Diff (C)', fontsize=11)

    # 标题
    fig.text(0.5, 0.98, f'JAXA SST Hybrid Filling: {timestamp[:10]}',
            ha='center', va='top', fontsize=16, fontweight='bold')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Model Functions
# ============================================================================

def load_model(model_path: str, device: torch.device):
    """加载模型并返回归一化参数"""
    print(f"加载模型: {model_path}")

    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_mean = checkpoint.get('norm_mean', 299.9221)
    norm_std = checkpoint.get('norm_std', 2.6919)

    print(f"  Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, MAE: {checkpoint.get('val_mae', 'N/A'):.4f}K)")
    print(f"  Normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")

    return model, norm_mean, norm_std


def fill_sst_hybrid(model, sst_knn_seq, sst_weighted_day30, mask_seq,
                    land_mask, norm_mean, norm_std, device,
                    preprocess_sigma=1.0):
    """
    混合输入方案填充SST缺失值

    Args:
        model: FNO-CBAM模型
        sst_knn_seq: [29, H, W] 前29天KNN填充后的SST序列 (Kelvin)
        sst_weighted_day30: [H, W] 第30天时序加权SST (有NaN)
        mask_seq: [30, H, W] 30天的缺失掩码 (1=缺失, 0=有效)
        land_mask: [H, W] 陆地掩码
        norm_mean, norm_std: 归一化参数
        device: 计算设备
        preprocess_sigma: 第30天输入的高斯预处理sigma

    Returns:
        filled_sst: [H, W] 模型填充后的SST (Kelvin)
        preprocessed_day30: [H, W] 预处理后的第30天输入
    """
    H, W = sst_weighted_day30.shape

    # 构建30天输入序列
    sst_seq = np.zeros((WINDOW_SIZE, H, W), dtype=np.float32)

    # 前29天: KNN填充数据
    sst_seq[:WINDOW_SIZE-1] = sst_knn_seq

    # 第30天: 时序加权数据 + 高斯预处理
    sst_day30 = sst_weighted_day30.copy()

    # 对第30天有效区域应用高斯滤波预处理
    missing_day30 = mask_seq[-1]
    sst_day30_filtered = apply_gaussian_filter(sst_day30, missing_day30, sigma=preprocess_sigma)

    # 缺失区域用均值填充
    sst_day30_filtered = np.where(np.isnan(sst_day30_filtered), norm_mean, sst_day30_filtered)
    sst_seq[-1] = sst_day30_filtered

    preprocessed_day30 = sst_day30_filtered.copy()

    # 归一化
    sst_norm = (sst_seq - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    # 转为tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    # 反归一化
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    # 输出组合：观测区域保留预处理后的输入值，缺失区域用模型预测
    last_day_mask = mask_seq[-1]
    filled_sst = preprocessed_day30.copy()
    filled_sst = np.where(last_day_mask > 0, pred_kelvin, filled_sst)
    filled_sst = np.where(np.isnan(filled_sst), pred_kelvin, filled_sst)
    filled_sst[land_mask > 0] = np.nan

    return filled_sst, preprocessed_day30


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_knn_data(h5_path):
    """加载KNN填充后的H5文件数据"""
    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        original_missing_mask = f['original_missing_mask'][:]
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    return {
        'sst_data': sst_data,
        'original_missing_mask': original_missing_mask,
        'land_mask': land_mask,
        'lat': lat,
        'lon': lon,
        'timestamps': timestamps
    }


def load_weighted_data(h5_path):
    """加载时序加权的H5文件数据"""
    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        missing_mask = f['missing_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    return {
        'sst_data': sst_data,
        'missing_mask': missing_mask,
        'lat': lat,
        'lon': lon,
        'timestamps': timestamps
    }


def get_hybrid_30day_sequence(sst_knn, mask_knn, sst_weighted, mask_weighted, idx, window_size=30):
    """
    获取混合的30天序列

    前29天: KNN填充数据
    第30天: 时序加权数据
    """
    H, W = sst_knn.shape[1], sst_knn.shape[2]

    sst_knn_seq = np.zeros((window_size - 1, H, W), dtype=np.float32)
    mask_seq = np.zeros((window_size, H, W), dtype=np.float32)

    # 前29天
    for t in range(window_size - 1):
        src_idx = idx - (window_size - 1) + t
        if src_idx < 0:
            src_idx = 0
        sst_knn_seq[t] = sst_knn[src_idx]
        mask_seq[t] = mask_knn[src_idx]

    # 第30天
    sst_weighted_day30 = sst_weighted[idx].copy()
    mask_seq[-1] = mask_weighted[idx]

    return sst_knn_seq, sst_weighted_day30, mask_seq


# ============================================================================
# Output Functions
# ============================================================================

def save_result_nc(output_path: Path, smoothed_sst: np.ndarray, model_sst: np.ndarray,
                   knn_sst: np.ndarray, weighted_sst: np.ndarray,
                   lat: np.ndarray, lon: np.ndarray,
                   timestamp: str, missing_mask: np.ndarray):
    """保存结果为NetCDF文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(output_path, 'w', format='NETCDF4') as f:
        f.createDimension('lat', len(lat))
        f.createDimension('lon', len(lon))
        f.createDimension('time', 1)

        lat_var = f.createVariable('lat', 'f4', ('lat',))
        lon_var = f.createVariable('lon', 'f4', ('lon',))
        time_var = f.createVariable('time', 'S32', ('time',))
        sst_smoothed_var = f.createVariable('sst_smoothed', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_model_var = f.createVariable('sst_model', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_knn_var = f.createVariable('sst_knn', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        sst_weighted_var = f.createVariable('sst_weighted', 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
        mask_var = f.createVariable('missing_mask', 'u1', ('time', 'lat', 'lon'))

        lat_var[:] = lat
        lon_var[:] = lon
        time_var[:] = np.array([timestamp], dtype='S32')
        sst_smoothed_var[0, :, :] = smoothed_sst
        sst_model_var[0, :, :] = model_sst
        sst_knn_var[0, :, :] = knn_sst
        sst_weighted_var[0, :, :] = weighted_sst
        mask_var[0, :, :] = missing_mask

        f.title = 'JAXA SST Hybrid Filling (KNN[1-29] + Weighted+Gaussian[30])'
        f.source = f'FNO-CBAM model with preprocess_sigma={PREPROCESS_SIGMA}, postprocess_sigma={POSTPROCESS_SIGMA}'
        f.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        sst_smoothed_var.units = 'Kelvin'
        sst_smoothed_var.long_name = 'Sea Surface Temperature (Gaussian Post-processed)'
        sst_model_var.units = 'Kelvin'
        sst_model_var.long_name = 'Sea Surface Temperature (Model Raw Output)'
        sst_knn_var.units = 'Kelvin'
        sst_knn_var.long_name = 'Sea Surface Temperature (KNN Filled)'
        sst_weighted_var.units = 'Kelvin'
        sst_weighted_var.long_name = 'Sea Surface Temperature (Temporal Weighted Input)'


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("JAXA SST Hybrid Filling")
    print("Input: KNN[Day 1-29] + Weighted+Gaussian[Day 30]")
    print("Output: Model Filled + Gaussian Post-processing")
    print("=" * 70)

    # 设备
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Preprocess sigma: {PREPROCESS_SIGMA}")
    print(f"Postprocess sigma: {POSTPROCESS_SIGMA}")

    # 加载模型
    model, norm_mean, norm_std = load_model(MODEL_PATH, device)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 处理统计
    total_processed = 0
    total_failed = 0

    # 遍历每个序列
    for series_id in SERIES_IDS:
        knn_path = KNN_FILLED_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'
        weighted_path = WEIGHTED_DIR / f'jaxa_weighted_series_{series_id:02d}.h5'

        if not knn_path.exists():
            print(f"\nSkipping - KNN file not found: {knn_path}")
            continue
        if not weighted_path.exists():
            print(f"\nSkipping - Weighted file not found: {weighted_path}")
            continue

        print(f"\nProcessing Series {series_id}")

        # 加载数据
        knn_data = load_knn_data(knn_path)
        weighted_data = load_weighted_data(weighted_path)

        sst_knn = knn_data['sst_data']
        mask_knn = knn_data['original_missing_mask']
        land_mask = knn_data['land_mask']
        lat = knn_data['lat']
        lon = knn_data['lon']
        timestamps = knn_data['timestamps']

        sst_weighted = weighted_data['sst_data']
        mask_weighted = weighted_data['missing_mask']

        num_frames = min(sst_knn.shape[0], sst_weighted.shape[0])
        print(f"  Total frames: {num_frames}")

        # 确定处理范围
        start_idx = WINDOW_SIZE - 1
        if NUM_TEST_SAMPLES is not None:
            indices = list(range(start_idx, min(start_idx + NUM_TEST_SAMPLES, num_frames)))
        else:
            indices = list(range(start_idx, num_frames))

        print(f"  Processing frames: {len(indices)}")

        for i, idx in enumerate(tqdm(indices, desc=f"Series {series_id}")):
            try:
                # 获取混合30天序列
                sst_knn_seq, sst_weighted_day30, mask_seq = get_hybrid_30day_sequence(
                    sst_knn, mask_knn, sst_weighted, mask_weighted, idx, WINDOW_SIZE
                )

                # 模型填充 (包含第30天高斯预处理)
                filled_sst, preprocessed_day30 = fill_sst_hybrid(
                    model, sst_knn_seq, sst_weighted_day30, mask_seq,
                    land_mask, norm_mean, norm_std, device,
                    preprocess_sigma=PREPROCESS_SIGMA
                )

                # 高斯后处理
                smoothed_sst = apply_gaussian_filter(filled_sst, land_mask, sigma=POSTPROCESS_SIGMA)
                smoothed_sst[land_mask > 0] = np.nan

                # 保存NC文件
                timestamp = timestamps[idx]
                output_file = OUTPUT_DIR / f'series_{series_id:02d}' / f'jaxa_hybrid_{timestamp.replace(":", "").replace("-", "")}.nc'

                # 准备保存的数据
                knn_sst_day = sst_knn[idx].copy()
                knn_sst_day[land_mask > 0] = np.nan

                weighted_sst_day = sst_weighted_day30.copy()
                weighted_sst_day[land_mask > 0] = np.nan

                save_result_nc(
                    output_file,
                    smoothed_sst=smoothed_sst,
                    model_sst=filled_sst,
                    knn_sst=knn_sst_day,
                    weighted_sst=weighted_sst_day,
                    lat=lat,
                    lon=lon,
                    timestamp=timestamp,
                    missing_mask=mask_seq[-1].astype(np.uint8)
                )

                # 可视化
                if i % VIS_INTERVAL == 0:
                    vis_path = VIS_DIR / f'series_{series_id:02d}_{timestamp.replace(":", "").replace("-", "")}.png'
                    create_visualization(
                        original_sst=preprocessed_day30,
                        knn_sst=knn_sst_day,
                        model_filled_sst=filled_sst,
                        smoothed_sst=smoothed_sst,
                        missing_mask=mask_seq[-1],
                        land_mask=land_mask,
                        lon_coords=lon,
                        lat_coords=lat,
                        timestamp=timestamp,
                        save_path=vis_path
                    )

                total_processed += 1

            except Exception as e:
                print(f"\n  Failed at frame {idx}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
                continue

    # 统计
    print(f"\n" + "=" * 70)
    print("Filling Complete!")
    print("=" * 70)
    print(f"Successfully processed: {total_processed} frames")
    print(f"Failed: {total_failed} frames")
    print(f"NC output: {OUTPUT_DIR}")
    print(f"Visualization: {VIS_DIR}")


if __name__ == '__main__':
    main()
