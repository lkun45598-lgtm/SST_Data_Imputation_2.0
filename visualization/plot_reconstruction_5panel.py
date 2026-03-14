"""
FNO-CBAM SST重建可视化 - 5连图

1. KNN填充后的完整图
2. KNN填充后经过mask的图（显示缺失区域）
3. JAXA原始数据（有云缺失）
4. FNO预测+高斯滤波后的图
5. 误差图
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import netCDF4 as nc
from pathlib import Path
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import warnings
import sys

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# 数据路径
KNN_DATA_DIR = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/jaxa_knn_filled')
JAXA_RAW_DIR = Path('/data/sst_data/sst_missing_value_imputation/jaxa_data/jaxa_extract_L3')
LAND_MASK_FILE = Path('/home/lz/Data_Imputation/visualization/jaxa_land_mask.npz')
MODEL_PATH = '/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/jaxa_finetune_8years/best_model.pth'
OUTPUT_DIR = Path('/home/lz/Data_Imputation/visualization/output')

# 高斯滤波参数
GAUSSIAN_SIGMA = 1.0


class RandomSquareMaskGenerator:
    """随机方形挖空生成器 - 只在观测区域生成mask"""

    def __init__(self, mask_ratio=0.2, min_size=10, max_size=50, seed=None):
        """
        Args:
            mask_ratio: 目标挖空比例（相对于有效观测区域）
            min_size: 方形最小边长
            max_size: 方形最大边长
            seed: 随机种子
        """
        self.mask_ratio = mask_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def generate(self, valid_obs_mask):
        """
        在valid_obs_mask区域内生成随机方形挖空

        Args:
            valid_obs_mask: 有效观测区域 (H, W), 1=可挖空(真实观测), 0=不可挖空

        Returns:
            artificial_mask: 挖空掩码 (H, W), 1=被挖空, 0=保留
        """
        H, W = valid_obs_mask.shape
        artificial_mask = np.zeros((H, W), dtype=np.float32)

        valid_count = valid_obs_mask.sum()
        if valid_count == 0:
            return artificial_mask

        target_masked = int(valid_count * self.mask_ratio)
        current_masked = 0

        # 获取有效区域的边界
        valid_y, valid_x = np.where(valid_obs_mask == 1)
        if len(valid_y) == 0:
            return artificial_mask

        y_min, y_max = valid_y.min(), valid_y.max()
        x_min, x_max = valid_x.min(), valid_x.max()

        max_attempts = 1000
        attempts = 0

        while current_masked < target_masked and attempts < max_attempts:
            # 随机方形大小
            size = self.rng.integers(self.min_size, self.max_size + 1)

            # 随机位置（在有效区域范围内）
            if y_max - size < y_min or x_max - size < x_min:
                attempts += 1
                continue

            y_start = self.rng.integers(y_min, max(y_min + 1, y_max - size + 1))
            x_start = self.rng.integers(x_min, max(x_min + 1, x_max - size + 1))

            y_end = min(y_start + size, H)
            x_end = min(x_start + size, W)

            # 只挖空valid_obs_mask区域内的像素
            region = valid_obs_mask[y_start:y_end, x_start:x_end].copy()
            new_masked = region.sum() - (artificial_mask[y_start:y_end, x_start:x_end] * region).sum()

            if new_masked > 0:
                artificial_mask[y_start:y_end, x_start:x_end] = np.where(
                    region == 1, 1.0, artificial_mask[y_start:y_end, x_start:x_end]
                )
                current_masked = (artificial_mask * valid_obs_mask).sum()

            attempts += 1

        return artificial_mask


def setup_matplotlib():
    """配置matplotlib高质量绘图"""
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=1.5, labelsize=12)
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
    return f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.0f}°{'N' if y >= 0 else 'S'}"


def load_land_mask():
    """加载预计算的陆地掩码"""
    if LAND_MASK_FILE.exists():
        data = np.load(LAND_MASK_FILE)
        return data['land_mask']
    return None


def load_model(model_path, device):
    """加载FNO-CBAM模型"""
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

    print(f"Model loaded (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"Normalization: mean={norm_mean:.4f}K, std={norm_std:.4f}K")

    return model, norm_mean, norm_std


def load_knn_data(series_id=0):
    """加载KNN填充后的数据"""
    h5_path = KNN_DATA_DIR / f'jaxa_knn_filled_{series_id:02d}.h5'

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


def load_jaxa_raw(date_str, hour=0):
    """
    加载JAXA原始数据（某个具体时刻）

    Args:
        date_str: 日期字符串 YYYY-MM-DD
        hour: 小时 (0-23)，默认0点UTC（与时间加权数据时刻一致）

    Returns:
        sst: [H, W] 原始SST (Kelvin)，有NaN
        lat, lon: 坐标
        actual_hour: 实际加载的小时
    """
    # 解析日期
    date = datetime.strptime(date_str, '%Y-%m-%d')
    year = date.year
    month = date.month
    day = date.day

    day_dir = JAXA_RAW_DIR / f'{year:04d}{month:02d}' / f'{day:02d}'

    if not day_dir.exists():
        print(f"Warning: JAXA raw data not found for {date_str}")
        return None, None, None, None

    # 尝试加载指定小时的数据
    filename = f'{year:04d}{month:02d}{day:02d}{hour:02d}0000.nc'
    file_path = day_dir / filename

    if not file_path.exists():
        # 如果指定小时不存在，找一个存在的
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


def apply_gaussian_filter(sst_data, land_mask, sigma=1.0):
    """对SST数据应用高斯滤波"""
    sst = sst_data.copy()

    # 用均值临时填充NaN和陆地
    valid_mask = ~np.isnan(sst) & (land_mask == 0)
    if valid_mask.sum() == 0:
        return sst

    mean_val = np.nanmean(sst[valid_mask])
    sst_for_filter = sst.copy()
    sst_for_filter[~valid_mask] = mean_val

    # 高斯滤波
    filtered = gaussian_filter(sst_for_filter, sigma=sigma)

    # 只保留海洋区域的滤波结果
    result = np.where(land_mask == 0, filtered, np.nan)

    return result


def run_fno_inference(model, sst_seq, mask_seq, norm_mean, norm_std, device):
    """
    运行FNO模型推理

    Args:
        model: FNO-CBAM模型
        sst_seq: [30, H, W] SST序列 (Kelvin)
        mask_seq: [30, H, W] mask序列
        norm_mean, norm_std: 归一化参数
        device: 设备

    Returns:
        pred: [H, W] 预测结果 (Kelvin)
    """
    # 归一化
    sst_norm = (sst_seq - norm_mean) / norm_std
    sst_norm = np.nan_to_num(sst_norm, nan=0.0)

    # 转tensor
    sst_tensor = torch.from_numpy(sst_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask_seq.astype(np.float32)).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        pred = model(sst_tensor, mask_tensor)

    # 反归一化
    pred_kelvin = pred.squeeze().cpu().numpy() * norm_std + norm_mean

    return pred_kelvin


def create_five_panel_plot(knn_filled, knn_masked, jaxa_raw, fno_gaussian,
                           artificial_mask, land_mask, lon_coords, lat_coords,
                           date_str, save_path, jaxa_hour=None):
    """
    创建5连图

    Args:
        knn_filled: [H, W] KNN填充后的完整SST (Kelvin) - 作为Ground Truth
        knn_masked: [H, W] (unused, 内部处理)
        jaxa_raw: [H, W] JAXA原始数据 (Kelvin)，有云缺失
        fno_gaussian: [H, W] FNO预测+高斯滤波后的SST (Kelvin)
        artificial_mask: [H, W] 人工生成的mask (1=被挖空, 0=保留) - 只在观测区域
        land_mask: [H, W] 陆地掩码 (1=陆地, 0=海洋)
        lon_coords, lat_coords: 坐标
        date_str: 日期字符串
        save_path: 保存路径
        jaxa_hour: JAXA数据的小时
    """
    setup_matplotlib()

    # 转换为摄氏度
    knn_filled_c = knn_filled - 273.15
    knn_masked_c = knn_masked - 273.15
    jaxa_raw_c = jaxa_raw - 273.15 if jaxa_raw is not None else None
    fno_gaussian_c = fno_gaussian - 273.15

    # 计算误差（FNO vs KNN在人工挖空区域）
    error = np.abs(fno_gaussian_c - knn_filled_c)

    # 海洋掩码
    ocean_mask = (land_mask == 0)
    # 误差只计算在人工挖空的区域（这些区域我们知道KNN的真值）
    masked_region = (artificial_mask > 0) & ocean_mask

    # 统计
    if masked_region.sum() > 0:
        mae = np.nanmean(error[masked_region])
        rmse = np.sqrt(np.nanmean(error[masked_region]**2))
        max_err = np.nanmax(error[masked_region])
    else:
        mae = rmse = max_err = 0

    mask_ratio = masked_region.sum() / ocean_mask.sum() * 100 if ocean_mask.sum() > 0 else 0

    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    # 颜色设置
    cmap_sst = 'RdYlBu_r'
    cmap_error = 'hot_r'
    land_color = '#D2B48C'
    cloud_color = '#E8E8E8'

    # SST颜色范围
    valid_sst = knn_filled_c[ocean_mask & ~np.isnan(knn_filled_c)]
    if len(valid_sst) > 0:
        vmin_sst = np.percentile(valid_sst, 2)
        vmax_sst = np.percentile(valid_sst, 98)
    else:
        vmin_sst, vmax_sst = 26, 32

    # 创建图形 - 5连图布局
    fig = plt.figure(figsize=(36, 7))

    # [图1, 图2, 图3, 图4, cbar_sst, 图5, cbar_err]
    gs = gridspec.GridSpec(1, 8, figure=fig,
                          width_ratios=[1, 1, 1, 1, 0.06, 1, 0.06, 0.02],
                          wspace=0.12, hspace=0.1,
                          left=0.02, right=0.99, top=0.85, bottom=0.12)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 5]),
    ]

    # 陆地显示
    land_display = np.ma.masked_where(land_mask == 0, land_mask)

    # ===== 1. KNN填充后完整图 =====
    ax = axes[0]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    knn_masked_plot = np.ma.masked_where(land_mask > 0, knn_filled_c)
    im1 = ax.pcolormesh(lon_grid, lat_grid, knn_masked_plot,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.set_title('KNN Filled (Complete)\nDay 30', fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)

    # ===== 2. KNN填充后+Mask图（显示人工挖空区域）=====
    ax = axes[1]
    ax.set_facecolor(cloud_color)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 显示人工挖空区域为灰色
    knn_with_mask = knn_filled_c.copy()
    knn_with_mask[artificial_mask > 0] = np.nan
    knn_with_mask_plot = np.ma.masked_where((land_mask > 0) | np.isnan(knn_with_mask), knn_with_mask)

    ax.pcolormesh(lon_grid, lat_grid, knn_with_mask_plot,
                  cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'KNN + Mask on Observed\n(Masked: {mask_ratio:.1f}% of obs)', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 3. JAXA原始数据 =====
    ax = axes[2]
    ax.set_facecolor(cloud_color)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    if jaxa_raw_c is not None:
        jaxa_plot = np.ma.masked_where((land_mask > 0) | np.isnan(jaxa_raw_c), jaxa_raw_c)
        ax.pcolormesh(lon_grid, lat_grid, jaxa_plot,
                      cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')
        # 计算JAXA原始数据的有效率
        jaxa_valid = ~np.isnan(jaxa_raw_c) & ocean_mask
        jaxa_valid_rate = jaxa_valid.sum() / ocean_mask.sum() * 100 if ocean_mask.sum() > 0 else 0
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
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # ===== 4. FNO预测+高斯滤波 =====
    ax = axes[3]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    fno_plot = np.ma.masked_where(land_mask > 0, fno_gaussian_c)
    im4 = ax.pcolormesh(lon_grid, lat_grid, fno_plot,
                        cmap=cmap_sst, vmin=vmin_sst, vmax=vmax_sst, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title(f'FNO + Gaussian\n(sigma={GAUSSIAN_SIGMA})', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # SST Colorbar
    cax_sst = fig.add_subplot(gs[0, 4])
    cbar_sst = plt.colorbar(im4, cax=cax_sst)
    cbar_sst.set_label('SST (°C)', fontsize=11)

    # ===== 5. 误差图（只显示人工挖空区域的误差）=====
    ax = axes[4]
    ax.set_facecolor('white')
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 只显示人工挖空区域的误差
    error_display = error.copy()
    error_display[artificial_mask == 0] = np.nan
    error_plot = np.ma.masked_where((land_mask > 0) | (artificial_mask == 0), error_display)

    im5 = ax.pcolormesh(lon_grid, lat_grid, error_plot,
                        cmap=cmap_error, vmin=0, vmax=1.0, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.locator_params(axis='x', nbins=4)
    ax.set_title('|FNO - KNN| Error\n(Masked Region)', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_box_aspect(1)
    ax.set_yticks([])

    # 添加统计信息
    stats_text = f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nMax: {max_err:.3f}°C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Error Colorbar
    cax_err = fig.add_subplot(gs[0, 6])
    cbar_err = plt.colorbar(im5, cax=cax_err)
    cbar_err.set_label('|Error| (°C)', fontsize=11)

    # 图例
    legend_elements = [
        Patch(facecolor=land_color, edgecolor='black', label='Land'),
        Patch(facecolor=cloud_color, edgecolor='black', label='Cloud/Missing'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.995, 0.98), fontsize=10, framealpha=0.9)

    # 总标题
    fig.text(0.5, 0.98, f'JAXA SST Reconstruction: {date_str}',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return mae, rmse, max_err


def main():
    import argparse

    parser = argparse.ArgumentParser(description='FNO-CBAM 5-Panel Visualization')
    parser.add_argument('--date', type=str, default='2017-08-17',
                       help='Target date (Day 30) in YYYY-MM-DD format')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--hour', type=int, default=0,
                       help='Hour (0-23) for JAXA raw data (default: 0, matching weighted data)')
    parser.add_argument('--mask-ratio', type=float, default=0.2,
                       help='Random mask ratio on observed regions (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for mask generation (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')

    args = parser.parse_args()

    print("=" * 70)
    print("FNO-CBAM SST Reconstruction - 5 Panel Visualization")
    print("=" * 70)

    # 设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print("\nLoading model...")
    model, norm_mean, norm_std = load_model(MODEL_PATH, device)

    # 加载KNN数据
    print("\nLoading KNN data...")
    knn_data = load_knn_data(series_id=0)

    # 找到目标日期的索引
    target_date = args.date
    target_idx = None
    for i, ts in enumerate(knn_data['timestamps']):
        if target_date in ts:
            target_idx = i
            break

    if target_idx is None:
        print(f"Error: Date {target_date} not found in KNN data")
        return

    if target_idx < 29:
        print(f"Error: Not enough data before {target_date}. Need at least 30 days.")
        return

    print(f"Target date: {target_date} (index {target_idx})")

    # 加载陆地掩码
    land_mask = load_land_mask()
    if land_mask is None:
        land_mask = knn_data['land_mask']

    # 准备30天数据
    start_idx = target_idx - 29
    sst_seq = knn_data['sst_data'][start_idx:target_idx+1]  # [30, H, W]
    mask_seq = knn_data['original_missing_mask'][start_idx:target_idx+1]  # [30, H, W]

    # 第30天的数据
    knn_filled = knn_data['sst_data'][target_idx]
    original_missing_mask = knn_data['original_missing_mask'][target_idx]
    lat = knn_data['lat']
    lon = knn_data['lon']

    # 先加载JAXA原始数据（单个时刻）- 用于确定真实观测区域
    print(f"\nLoading JAXA raw data for {target_date}...")
    jaxa_raw, _, _, jaxa_hour = load_jaxa_raw(target_date, hour=args.hour)
    if jaxa_raw is not None:
        print(f"  Loaded hour: {jaxa_hour:02d}:00 UTC")
        jaxa_valid = ~np.isnan(jaxa_raw)
        print(f"  Valid pixels: {jaxa_valid.sum()} ({jaxa_valid.sum()/jaxa_raw.size*100:.1f}%)")
    else:
        print("Error: Cannot load JAXA raw data")
        return

    # 确定真实观测区域：基于JAXA原始数据的有效观测位置（非NaN 且 非陆地）
    observed_region = (~np.isnan(jaxa_raw)) & (land_mask == 0)
    print(f"\nJAXA observed ocean region: {observed_region.sum()} pixels ({observed_region.sum()/((land_mask==0).sum())*100:.1f}%)")

    # 在JAXA真实观测区域上生成随机mask
    print(f"Generating random mask (ratio={args.mask_ratio}, seed={args.seed})...")
    mask_generator = RandomSquareMaskGenerator(
        mask_ratio=args.mask_ratio,
        min_size=10,
        max_size=50,
        seed=args.seed
    )
    artificial_mask = mask_generator.generate(observed_region.astype(np.float32))

    actual_mask_ratio = artificial_mask.sum() / observed_region.sum() * 100 if observed_region.sum() > 0 else 0
    print(f"  Artificial mask pixels: {int(artificial_mask.sum())} ({actual_mask_ratio:.1f}% of JAXA observed)")

    # 合并原始缺失和人工mask，得到最终的mask_seq
    # 对于前29天，使用原始缺失mask
    # 对于第30天，使用JAXA缺失(~observed_region) + 人工mask
    jaxa_missing_mask = (~observed_region & (land_mask == 0)).astype(np.float32)
    combined_mask = np.maximum(jaxa_missing_mask, artificial_mask).astype(np.float32)

    # 运行FNO推理
    print("\nRunning FNO inference...")
    # 更新mask_seq的第30天为combined_mask
    mask_seq_for_inference = mask_seq.copy()
    mask_seq_for_inference[-1] = combined_mask

    # 关键：在输入的第30天，所有mask区域（JAXA缺失+人工挖空）都需要填充均值
    sst_seq_for_inference = sst_seq.copy()
    sst_seq_for_inference[-1] = np.where(combined_mask > 0, norm_mean, sst_seq[-1])

    fno_pred = run_fno_inference(model, sst_seq_for_inference, mask_seq_for_inference, norm_mean, norm_std, device)

    # Output composition: 观测区域用KNN真值，人工mask区域用FNO预测
    fno_composed = np.where(artificial_mask > 0, fno_pred, knn_filled)

    # 计算不加高斯滤波的原始误差
    raw_error = np.abs(fno_composed - knn_filled)
    raw_mae = np.mean(raw_error[artificial_mask > 0])
    print(f"  Raw FNO MAE (before Gaussian): {raw_mae:.4f}K = {raw_mae:.4f}°C")

    # 高斯滤波
    print(f"Applying Gaussian filter (sigma={GAUSSIAN_SIGMA})...")
    fno_gaussian = apply_gaussian_filter(fno_composed, land_mask, sigma=GAUSSIAN_SIGMA)

    # 创建可视化
    if args.output:
        save_path = args.output
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = OUTPUT_DIR / f'reconstruction_5panel_{target_date.replace("-", "")}.png'

    print(f"\nCreating 5-panel visualization...")
    mae, rmse, max_err = create_five_panel_plot(
        knn_filled=knn_filled,
        knn_masked=knn_filled,  # 会在函数内部处理mask
        jaxa_raw=jaxa_raw,
        fno_gaussian=fno_gaussian,
        artificial_mask=artificial_mask,  # 人工生成的mask（在观测区域）
        land_mask=land_mask,
        lon_coords=lon,
        lat_coords=lat,
        date_str=target_date,
        save_path=save_path,
        jaxa_hour=jaxa_hour
    )

    print(f"\nResults:")
    print(f"  MAE:  {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  Max:  {max_err:.4f}°C")
    print(f"\nSaved: {save_path}")
    print("Done!")


if __name__ == '__main__':
    main()
