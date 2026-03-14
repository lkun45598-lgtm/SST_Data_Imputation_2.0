#!/usr/bin/env python3
"""
FNO-CBAM SST重建可视化 - 5连图 (使用post-filtered数据)

1. Post-filtered完整图 (Ground Truth)
2. 人工挖空后的图（显示mask区域）
3. JAXA原始数据（真实云缺失）
4. FNO预测+高斯滤波后的图
5. 误差图 (仅人工挖空区域)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import netCDF4 as nc
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from datetime import datetime
from scipy.ndimage import gaussian_filter
import warnings
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
POST_FILTERED_DIR = Path('/data/chla_data_imputation_data_260125/sst_knn_filled_optimized')
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/run004_jaxa_3dknn_progressive_stride1_lr0.0005/best_model.pth'
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output/optimized')

SERIES_ID = 0
WINDOW_SIZE = 30  # 恢复模型训练配置
STRIDE = 24  # 恢复模型训练配置
GAUSSIAN_SIGMA = 1.0
GPU_ID = 5
MASK_RATIO = 0.2
SEED = 42

# ============================================================================
# 随机方形挖空生成器
# ============================================================================
class RandomSquareMaskGenerator:
    def __init__(self, mask_ratio=0.2, min_size=10, max_size=50, seed=None):
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_obs_mask):
        H, W = valid_obs_mask.shape
        artificial_mask = np.zeros((H, W), dtype=np.float32)

        valid_count = valid_obs_mask.sum()
        if valid_count == 0:
            return artificial_mask

        target_masked = int(valid_count * self.mask_ratio)
        current_masked = 0

        valid_y, valid_x = np.where(valid_obs_mask == 1)
        if len(valid_y) == 0:
            return artificial_mask

        y_min, y_max = valid_y.min(), valid_y.max()
        x_min, x_max = valid_x.min(), valid_x.max()

        max_attempts = 1000
        attempts = 0

        while current_masked < target_masked and attempts < max_attempts:
            size = self.rng.integers(self.min_size, self.max_size + 1)

            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue

            y_start = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x_start = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))

            y_end = min(y_start + size, H)
            x_end = min(x_start + size, W)

            region = valid_obs_mask[y_start:y_end, x_start:x_end].copy()
            new_masked = region.sum() - (artificial_mask[y_start:y_end, x_start:x_end] * region).sum()

            if new_masked > 0:
                artificial_mask[y_start:y_end, x_start:x_end] = np.where(
                    region == 1, 1.0, artificial_mask[y_start:y_end, x_start:x_end]
                )
                current_masked = (artificial_mask * valid_obs_mask).sum()

            attempts += 1

        return artificial_mask

# ============================================================================
# 工具函数
# ============================================================================
def setup_matplotlib():
    plt.rc('font', size=11)
    plt.rc('axes', linewidth=1.5, labelsize=11)
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

def apply_gaussian_filter(sst_data, land_mask, sigma=1.0):
    sst = sst_data.copy()
    valid_mask = ~np.isnan(sst) & (land_mask == 0)
    if valid_mask.sum() == 0:
        return sst

    mean_val = np.nanmean(sst[valid_mask])
    sst_for_filter = sst.copy()
    sst_for_filter[~valid_mask] = mean_val

    filtered = gaussian_filter(sst_for_filter, sigma=sigma)
    result = np.where(land_mask == 0, filtered, np.nan)

    return result

# ============================================================================
# 数据加载
# ============================================================================
def load_post_filtered_data(series_id):
    h5_path = POST_FILTERED_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'

    with h5py.File(h5_path, 'r') as f:
        sst_data = f['sst_data'][:]
        obs_mask = f['original_obs_mask'][:]
        land_mask = f['land_mask'][:]
        lat = f['latitude'][:]
        lon = f['longitude'][:]
        timestamps = f['timestamps'][:]
        timestamps = [ts.decode('utf-8') if isinstance(ts, bytes) else ts for ts in timestamps]

    return {
        'sst_data': sst_data,
        'obs_mask': obs_mask,
        'land_mask': land_mask,
        'lat': lat,
        'lon': lon,
        'timestamps': timestamps
    }

def load_jaxa_raw(date_str, hour=0):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year, month, day = date.year, date.month, date.day

    day_dir = JAXA_RAW_DIR / f'{year:04d}{month:02d}' / f'{day:02d}'
    if not day_dir.exists():
        return None, None, None, None

    filename = f'{year:04d}{month:02d}{day:02d}{hour:02d}0000.nc'
    file_path = day_dir / filename

    if not file_path.exists():
        for h in range(24):
            filename = f'{year:04d}{month:02d}{day:02d}{h:02d}0000.nc'
            file_path = day_dir / filename
            if file_path.exists():
                hour = h
                break
        else:
            return None, None, None, None

    with nc.Dataset(file_path, 'r') as f:
        sst = f.variables['sea_surface_temperature'][0, :, :]
        lat = f.variables['lat'][:]
        lon = f.variables['lon'][:]

    return sst, lat, lon, hour

def load_model(model_path, device, H, W):
    model = FNO_CBAM_SST_Temporal(
        out_size=(H, W),
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

    print(f"Model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"Normalization: mean={norm_mean:.4f}K, std={norm_std:.4f}K")

    return model, norm_mean, norm_std

# ============================================================================
# 推理
# ============================================================================
def run_fno_inference(model, sst_seq, mask_seq, norm_mean, norm_std, device):
    sst_norm = (sst_seq - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    pred_sst = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    return pred_sst

# ============================================================================
# 可视化
# ============================================================================
def plot_5panel(gt_full, gt_masked, jaxa_raw, fno_pred, error,
                artificial_mask, land_mask, lat, lon, timestamp,
                obs_rate, mae, rmse, output_path):

    setup_matplotlib()

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.25,
                           left=0.05, right=0.95, top=0.92, bottom=0.08)

    # 标题
    fig.suptitle(f'FNO-CBAM SST Reconstruction (Post-Filtered Data)\n{timestamp} | Obs: {obs_rate:.1f}% | MAE: {mae:.3f}K | RMSE: {rmse:.3f}K',
                 fontsize=14, fontweight='bold')

    # 颜色映射
    land_cmap = ListedColormap(['#D2B48C'])
    mask_cmap = ListedColormap(['#FF6B6B'])

    # 温度范围
    ocean = (land_mask == 0)
    vmin, vmax = np.nanpercentile(gt_full[ocean], [2, 98])
    vmin_c, vmax_c = vmin - 273.15, vmax - 273.15

    titles = [
        '1. Post-Filtered Ground Truth',
        '2. Artificially Masked Input',
        '3. JAXA Raw Observation',
        '4. FNO-CBAM Prediction + Gaussian Filter',
        '5. Absolute Error (masked region only)'
    ]

    data_list = [gt_full, gt_masked, jaxa_raw, fno_pred, error]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for idx, (title, data, pos) in enumerate(zip(titles, data_list, positions)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])

        if idx == 4:  # Error panel - 不转换单位
            data_plot = data.copy()
            data_plot[land_mask == 1] = np.nan
            im = ax.pcolormesh(lon, lat, data_plot, cmap='hot_r',
                               vmin=0, vmax=2, shading='auto')
            cbar_label = 'Error (K)'
        else:  # SST panels - 转换为摄氏度
            data_plot = data.copy() - 273.15
            data_plot[land_mask == 1] = np.nan
            im = ax.pcolormesh(lon, lat, data_plot, cmap='RdYlBu_r',
                               vmin=vmin_c, vmax=vmax_c, shading='auto')
            cbar_label = 'SST (°C)'

        # Land overlay
        land_disp = np.where(land_mask == 1, 1.0, np.nan)
        ax.pcolormesh(lon, lat, land_disp, cmap=land_cmap, vmin=0, vmax=1, shading='auto')

        # Mask overlay for panel 2
        if idx == 1:
            mask_disp = np.where((artificial_mask == 1) & (land_mask == 0), 1.0, np.nan)
            ax.pcolormesh(lon, lat, mask_disp, cmap=mask_cmap, vmin=0, vmax=1,
                          shading='auto', alpha=0.4)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)

    # Legend
    legend_elements = [
        Patch(facecolor='#D2B48C', label='Land'),
        Patch(facecolor='#FF6B6B', alpha=0.4, label='Artificial Mask')
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(0.98, 0.05), fontsize=11, frameon=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("FNO-CBAM Inference with Post-Filtered Data")
    print("="*70)

    # 加载数据
    print("\nLoading post-filtered data...")
    data = load_post_filtered_data(SERIES_ID)
    sst_data = data['sst_data']
    obs_mask = data['obs_mask']
    land_mask = data['land_mask']
    lat = data['lat']
    lon = data['lon']
    timestamps = data['timestamps']

    T, H, W = sst_data.shape
    ocean = (land_mask == 0)
    print(f"Data shape: {sst_data.shape}, Ocean: {ocean.sum()} px")

    # 加载模型
    print("\nLoading model...")
    device = torch.device(f'cuda:{GPU_ID}')
    model, norm_mean, norm_std = load_model(MODEL_PATH, device, H, W)

    # 初始化mask生成器
    mask_gen = RandomSquareMaskGenerator(mask_ratio=MASK_RATIO, seed=SEED)

    # 选择几个代表性帧（需要>=720帧历史数据）
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_indices = [750, 800, 900]

    print("\nRunning inference...")
    for frame_idx in frame_indices:
        if frame_idx < WINDOW_SIZE * STRIDE:
            continue

        print(f"\n  Frame {frame_idx}...")

        # 构建输入窗口
        start_idx = frame_idx - WINDOW_SIZE * STRIDE
        window_indices = [start_idx + i * STRIDE for i in range(WINDOW_SIZE)]
        sst_window = sst_data[window_indices]
        obs_window = obs_mask[window_indices]

        # Ground truth
        gt_full = sst_data[frame_idx]
        obs_frame = obs_mask[frame_idx]

        # 生成人工mask（只在真实观测区域）
        valid_for_mask = obs_frame * (1 - land_mask)
        artificial_mask = mask_gen.generate(valid_for_mask)

        # 挖空后的输入（用全局均值填充，与训练一致）
        gt_masked = gt_full.copy()
        gt_masked[artificial_mask == 1] = norm_mean

        # 调试信息
        mask_count = (artificial_mask == 1).sum()
        ocean_count = ocean.sum()
        print(f"    Mask: {mask_count} pixels ({mask_count/ocean_count*100:.1f}% of ocean)")
        print(f"    GT range: {gt_full[ocean].min():.2f}-{gt_full[ocean].max():.2f}K")
        print(f"    Fill value: {norm_mean:.2f}K ({norm_mean-273.15:.2f}°C)")
        print(f"    Masked region temp before: {gt_full[artificial_mask==1].mean():.2f}K")
        print(f"    Masked region temp after: {gt_masked[artificial_mask==1].mean():.2f}K")

        # 构建mask序列
        mask_window = 1 - obs_window
        mask_window[-1] = artificial_mask  # 最后一帧用人工mask

        # 构建SST输入序列（最后一帧替换为挖空后的）
        sst_window_input = sst_window.copy()
        sst_window_input[-1] = gt_masked

        # 推理
        pred_sst = run_fno_inference(model, sst_window_input, mask_window,
                                      norm_mean, norm_std, device)

        # Output composition: 非挖空区域用输入值，挖空区域用预测值
        composed_sst = gt_masked.copy()
        composed_sst[artificial_mask == 1] = pred_sst[artificial_mask == 1]

        # 高斯滤波
        pred_filtered = apply_gaussian_filter(composed_sst, land_mask, sigma=GAUSSIAN_SIGMA)

        # 加载JAXA原始数据
        date_str = timestamps[frame_idx][:10]
        jaxa_raw, _, _, _ = load_jaxa_raw(date_str, hour=0)
        if jaxa_raw is None:
            jaxa_raw = np.full_like(gt_full, np.nan)

        # 计算误差（只在人工mask区域）
        error = np.abs(pred_filtered - gt_full)
        error[artificial_mask == 0] = np.nan
        error[land_mask == 1] = np.nan

        # 统计
        mask_ocean = (artificial_mask == 1) & (land_mask == 0)
        mae = np.nanmean(error[mask_ocean])
        rmse = np.sqrt(np.nanmean(error[mask_ocean]**2))
        obs_rate = obs_frame[ocean].sum() / ocean.sum() * 100

        # 可视化
        output_path = OUTPUT_DIR / f'reconstruction_5panel_postfiltered_frame{frame_idx}.png'
        plot_5panel(gt_full, gt_masked, jaxa_raw, pred_filtered, error,
                    artificial_mask, land_mask, lat, lon, timestamps[frame_idx],
                    obs_rate, mae, rmse, output_path)

        print(f"    MAE: {mae:.3f}K, RMSE: {rmse:.3f}K, Obs: {obs_rate:.1f}%")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)

if __name__ == '__main__':
    main()
