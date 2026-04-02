"""
OSTIA数据可视化 - 连续14天
生成3种图：原始ground truth、mask、FNO重建
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from datetime import datetime, timedelta
import sys

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.ostia_dataset import SSTDatasetTemporal
from models.fno_cbam_temporal import FNO_CBAM_SST_Temporal


def format_lon(x, pos):
    return f"{abs(x):.1f}°{'E' if x >= 0 else 'W'}"


def format_lat(y, pos):
    return f"{abs(y):.1f}°{'N' if y >= 0 else 'S'}"


def setup_matplotlib():
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=1.5)


def get_sample_date(base_date, sample_idx):
    """获取样本日期"""
    sample_date = base_date + timedelta(days=sample_idx)
    return sample_date.strftime('%Y-%m-%d')


def plot_ground_truth(sst, land_mask, lon_coords, lat_coords, date_str, save_path, mean, std):
    """绘制Ground Truth SST"""
    # 反归一化到摄氏度
    sst_kelvin = sst * std + mean
    sst_celsius = sst_kelvin - 273.15

    ocean_mask = 1 - land_mask
    vmin = np.percentile(sst_celsius[ocean_mask > 0], 2)
    vmax = np.percentile(sst_celsius[ocean_mask > 0], 98)

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('lightgray')

    # 陆地
    land_color = '#D2B48C'
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # SST
    sst_masked = np.ma.masked_where(land_mask > 0, sst_celsius)
    im = ax.pcolormesh(lon_grid, lat_grid, sst_masked,
                       cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'OSTIA Ground Truth SST\n{date_str}', fontsize=13, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SST (°C)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mask(sst, mask, land_mask, lon_coords, lat_coords, date_str, save_path, mean, std):
    """绘制Masked SST可视化 - 显示应用mask后的SST"""
    # 反归一化
    sst_kelvin = sst * std + mean
    sst_celsius = sst_kelvin - 273.15

    # 应用mask: 缺失区域设为NaN
    masked_sst = sst_celsius.copy()
    masked_sst[mask > 0] = np.nan

    ocean_mask = 1 - land_mask
    valid_data = sst_celsius[(ocean_mask > 0) & (mask == 0)]
    vmin = np.percentile(valid_data, 2)
    vmax = np.percentile(valid_data, 98)

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('lightgray')

    # 陆地
    land_color = '#D2B48C'
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    # 缺失区域显示为深灰色
    missing_display = np.ma.masked_where((land_mask > 0) | (mask == 0), mask)
    ax.pcolormesh(lon_grid, lat_grid, missing_display,
                  cmap=ListedColormap(['#404040']), vmin=0, vmax=1, shading='auto')

    # SST (仅观测区域)
    sst_display = np.ma.masked_where((land_mask > 0) | (mask > 0), sst_celsius)
    im = ax.pcolormesh(lon_grid, lat_grid, sst_display,
                       cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'OSTIA Masked SST (Observed Only)\n{date_str}', fontsize=13, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SST (°C)', fontsize=11)

    # 统计
    ocean_pixels = np.sum(ocean_mask)
    missing_pixels = np.sum(mask * ocean_mask)
    missing_ratio = missing_pixels / ocean_pixels * 100

    ax.text(0.02, 0.98, f'Missing: {missing_ratio:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fno_reconstruction(sst, land_mask, lon_coords, lat_coords, date_str, save_path, mean, std):
    """绘制FNO重建后的SST"""
    sst_kelvin = sst * std + mean
    sst_celsius = sst_kelvin - 273.15

    ocean_mask = 1 - land_mask
    vmin = np.percentile(sst_celsius[ocean_mask > 0], 2)
    vmax = np.percentile(sst_celsius[ocean_mask > 0], 98)

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('lightgray')

    land_color = '#D2B48C'
    land_display = np.ma.masked_where(land_mask == 0, land_mask)
    ax.pcolormesh(lon_grid, lat_grid, land_display,
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1, shading='auto')

    sst_masked = np.ma.masked_where(land_mask > 0, sst_celsius)
    im = ax.pcolormesh(lon_grid, lat_grid, sst_masked,
                       cmap='RdYlBu_r', vmin=vmin, vmax=vmax, shading='auto')

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'FNO-CBAM Reconstructed SST\n{date_str}', fontsize=13, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SST (°C)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_model(checkpoint_path, device):
    """加载模型"""
    model = FNO_CBAM_SST_Temporal(
        out_size=(451, 351),
        modes1=80,
        modes2=64,
        width=64,
        depth=6,
        cbam_reduction_ratio=16
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功 (Epoch: {checkpoint['epoch']})")
    return model


def main():
    print("="*60)
    print("OSTIA 14天可视化 - Ground Truth + Mask + FNO重建")
    print("="*60)

    setup_matplotlib()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 路径配置
    checkpoint_path = '/home/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/experiments/temporal_30days_composition/best_model.pth'
    data_dir = '/data/sst_data/sst_missing_value_imputation/processed_data'
    base_dir = Path('/data1/user/lz/FNO_CBAM/data_for_agent_FNO_CBAM_H20/FNO_CBAM/visulisasion_PPT')
    
    output_dirs = {
        'ground': base_dir / 'Ostia_ground',
        'mask': base_dir / 'Ostia_mask',
        'fno': base_dir / 'Ostia_fno_constructed'
    }

    # 加载模型
    model = load_model(checkpoint_path, device)

    # 加载坐标
    with h5py.File(f'{data_dir}/processed_sst_valid.h5', 'r') as f:
        lon_coords = f['longitude'][:]
        lat_coords = f['latitude'][:]
        print(f"经度范围: {lon_coords.min():.2f}° ~ {lon_coords.max():.2f}°")
        print(f"纬度范围: {lat_coords.min():.2f}° ~ {lat_coords.max():.2f}°\n")

    # 加载数据集
    print("加载数据集...")
    train_dataset = SSTDatasetTemporal(
        hdf5_path=f'{data_dir}/processed_sst_train.h5',
        normalize=True
    )
    valid_dataset = SSTDatasetTemporal(
        hdf5_path=f'{data_dir}/processed_sst_valid.h5',
        normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    print(f"验证集样本数: {len(valid_dataset)}")
    print(f"归一化参数: mean={train_dataset.mean:.2f}K, std={train_dataset.std:.2f}K\n")

    # 处理连续14天
    num_days = 14
    base_date = datetime(2015, 1, 1)
    
    print(f"开始处理连续{num_days}天...")
    for day_idx in tqdm(range(num_days), desc="生成可视化"):
        if day_idx >= len(valid_dataset):
            print(f"警告: 样本{day_idx}超出数据集范围")
            continue

        # 加载样本
        sample = valid_dataset[day_idx]
        sst_seq = torch.from_numpy(sample['input_sst_seq']).unsqueeze(0).to(device)
        mask_seq = torch.from_numpy(sample['mask_seq']).unsqueeze(0).to(device)
        gt_sst = sample['ground_truth_sst']
        land_mask = sample['land_mask']

        # 模型推理
        with torch.no_grad():
            pred = model(sst_seq, mask_seq)
            
            # Output Composition
            last_input = sst_seq[:, -1:, :, :]
            last_mask = mask_seq[:, -1:, :, :]
            pred = last_input * (1 - last_mask) + pred * last_mask

        # 转numpy
        pred_np = pred[0, 0].cpu().numpy()
        mask_np = mask_seq[0, -1].cpu().numpy()

        # 获取日期
        date_str = get_sample_date(base_date, day_idx)

        # 生成3种图
        plot_ground_truth(
            gt_sst, land_mask, lon_coords, lat_coords, date_str,
            output_dirs['ground'] / f'ostia_ground_{date_str}.png',
            train_dataset.mean, train_dataset.std
        )

        plot_mask(
            gt_sst, mask_np, land_mask, lon_coords, lat_coords, date_str,
            output_dirs['mask'] / f'ostia_mask_{date_str}.png',
            train_dataset.mean, train_dataset.std
        )

        plot_fno_reconstruction(
            pred_np, land_mask, lon_coords, lat_coords, date_str,
            output_dirs['fno'] / f'ostia_fno_{date_str}.png',
            train_dataset.mean, train_dataset.std
        )

    print("\n" + "="*60)
    print("✓ 完成！")
    print(f"Ground Truth 保存在: {output_dirs['ground']}")
    print(f"Mask 保存在: {output_dirs['mask']}")
    print(f"FNO重建 保存在: {output_dirs['fno']}")
    print("="*60)


if __name__ == '__main__':
    main()
